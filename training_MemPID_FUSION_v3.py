# training_MemPID_FUSION_v3.py
# ═══════════════════════════════════════════════════════════════════
#  🧠 MemPID_FUSION v3 - MULTI-HEAD IMPORTANCE POOL + ADAPTIVE DECAY
#
#  A novel language model using PID controllers instead of attention.
#  No O(n²) - just O(n)!
#
#  New in v3:
#    - Multi-Head Importance Pool (4 heads, content-selective)
#    - Adaptive Decay (time-selective forgetting)
#    - Larger architecture (1024 dim, 2048 context)
#
#  The Problem with Mean Pooling:
#    [King, uh, the, well, daughter] → all weighted equally
#    → "uh" dilutes "King" → fuzzy context
#
#  The Solution - Importance Pool:
#    [King, uh, the, well, daughter]
#       ↓     ↓    ↓    ↓      ↓
#     0.35  0.02 0.08 0.03   0.32  ← Learned weights!
#    → Important tokens dominate, noise is ignored
#
#  Multi-Head: 4 different "editors"
#    Head 1 → Subjects (King)
#    Head 2 → Negations (not)
#    Head 3 → Verbs (spoke)
#    Head 4 → Noise-Filter
#
#  Still O(n) - linear! No O(n²) like Attention!
#
#  Author: Arthur-KI
#  License: MIT
#  GitHub: https://github.com/Arthur-KI/MemPID_FUSION
# ═══════════════════════════════════════════════════════════════════

import os
import random
import math
import time
import contextlib
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ═══════════════════════════════════════════════════════════════════
# 0. SETTINGS
# ═══════════════════════════════════════════════════════════════════

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 2
GRAD_ACCUM = 1
BLOCK_SIZE = 2048
MAX_ITERS = 150000
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000
LEARNING_RATE = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modell Architektur
VOCAB_SIZE = 16000
DIM = 1024
LAYERS_PER_STACK = 6
KERNEL_SIZE = 64
MAX_DILATION = 32
EXPANSION_FACTOR = 2
GATE_RANK = 64

# v3 NEW: Multi-Head Importance
IMPORTANCE_HEADS = 4            # 4 "editors" for different aspects
USE_IMPORTANCE_POOL = True      # Instead of CausalMeanPool

# Regularisierung
DROPOUT_RATE = 0.2
EMB_DROPOUT = 0.1
NOISE_LEVEL = 0.02

# PID Controller
PID_INIT_KP = 0.5
PID_INIT_KI = 0.3
PID_INIT_KD = 0.2

# Training Control
VAL_SPLIT = 0.1
PATIENCE = 12
MIN_DELTA = 0.005

PRECISION_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# ═══════════════════════════════════════════════════════════════════
# 1. DATEN-ORDNER
# ═══════════════════════════════════════════════════════════════════
BASE_DATA_DIR = "training_data"
MANUAL_FOLDERS = {}

def scan_data_folders(base_dir=BASE_DATA_DIR, manual_folders=MANUAL_FOLDERS):
    data_folders = {}
    data_folders.update(manual_folders)
    
    if os.path.exists(base_dir):
        print(f"🔍 Scanne {base_dir}/ für Unterordner...")
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                token = f"<{item.upper()}>"
                if token not in data_folders:
                    data_folders[token] = item_path
    
    if data_folders:
        print(f"\n📁 Gefundene Daten-Kategorien:")
        for token, path in data_folders.items():
            files = glob(os.path.join(path, "*.txt"))
            print(f"   {token}: {path}/ ({len(files)} Dateien)")
    
    return data_folders


# ═══════════════════════════════════════════════════════════════════
# 2. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


# ═══════════════════════════════════════════════════════════════════
# v9 NEU: MULTI-HEAD IMPORTANCE POOL
# ═══════════════════════════════════════════════════════════════════

class MultiHeadImportancePool(nn.Module):
    """
    Multi-Head Importance Pooling - ECHTES O(n)!
    
    Danke Gemini für den Fix! 🙏
    
    Vorher (FALSCH): [T × T] Matrix = O(n²) = versteckte Attention!
    Jetzt (RICHTIG): cumsum Trick = O(n) = echtes lineares Processing!
    
    Der "Eimer-Trick":
      - Statt bei Token 100 auf Token 1,2,3...99 zurückzuschauen
      - Einfach alles in einen Eimer werfen (cumsum)
      - Bei Token 100 nur in den Eimer schauen!
    
    Heads spezialisieren auf verschiedene Aspekte:
      - Head 1: Subjekte/Nomen
      - Head 2: Verben/Aktionen  
      - Head 3: Negationen/Modifikatoren
      - Head 4: Rauschen-Filter
    """
    def __init__(self, dim, heads=4, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        
        # Importance: SIGMOID statt Softmax!
        # Wir brauchen absolute Wichtigkeit (0-1), keine relative Verteilung
        self.importance = nn.Linear(dim, heads, bias=False)
        
        # Value Projection: Pro Head eine Transformation
        self.value_proj = nn.Linear(dim, dim, bias=False)
        
        # Output Projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Leichte Initialisierung
        nn.init.normal_(self.importance.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        
        # Mix-Faktor (wie stark fließt Importance ein)
        self.mix = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        """
        ECHTER O(n) Forward Pass mit cumsum!
        
        Args:
            x: [B, T, D]
        Returns:
            x + importance_context: [B, T, D]
        """
        B, T, D = x.shape
        
        # 1. Berechne Values & Importance Scores
        # Values: [B, T, heads, head_dim]
        values = self.value_proj(x).view(B, T, self.heads, self.head_dim)
        
        # Scores: [B, T, heads, 1]
        # SIGMOID: "Ist dieses Token wichtig? 0=nein, 1=ja" (unabhängig!)
        scores = torch.sigmoid(self.importance(x)).unsqueeze(-1)
        
        # 2. Gewichte die Values ("Rein in den Eimer")
        # weighted_values: [B, T, heads, head_dim]
        weighted_values = values * scores
        
        # 3. DER WASSER-TRICK (cumsum = O(n)!)
        # Statt [T × T] Matrix: Einfach akkumulieren!
        # cumsum ist automatisch KAUSAL (summiert von links nach rechts)
        
        # PRÄZISIONS-FIX: cumsum in float32 für Stabilität bei langen Sequenzen
        # (BF16 kann bei großen Summen Präzision verlieren)
        orig_dtype = weighted_values.dtype
        
        # Zähler: Summe der gewichteten Inhalte
        numerator = torch.cumsum(weighted_values.float(), dim=1).to(orig_dtype)  # O(n)!
        
        # Nenner: Summe der Wichtigkeiten (für Durchschnitt)
        denominator = (torch.cumsum(scores.float(), dim=1) + self.eps).to(orig_dtype)  # O(n)!
        
        # 4. Normalisierung ("Was ist im Eimer?")
        # Division = gewichteter Durchschnitt bis Position t
        context = numerator / denominator  # [B, T, heads, head_dim]
        
        # 5. Zusammenfügen und Output
        context = context.reshape(B, T, D)  # [B, T, D]
        out = self.out_proj(context)
        
        return x + self.mix * out
    
    def get_importance_stats(self):
        """Für Monitoring"""
        return {
            'mix': self.mix.item(),
            'importance_mean': self.importance.weight.mean().item()
        }


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE DECAY (von v8)
# ═══════════════════════════════════════════════════════════════════

class AdaptiveDecay(nn.Module):
    """Content-abhängiges Vergessen (von v8)"""
    def __init__(self, dim):
        super().__init__()
        self.decay_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.decay_proj.weight)
        self.base_decay = nn.Parameter(torch.ones(1, 1, dim) * 2.2)
        self.content_scale = nn.Parameter(torch.ones(1, 1, dim) * 0.1)
    
    def forward(self, state, new_input, content):
        content_signal = self.decay_proj(content)
        decay = torch.sigmoid(self.base_decay + self.content_scale * content_signal)
        new_state = decay * state + (1 - decay) * new_input
        return new_state, decay.mean().item()


# ═══════════════════════════════════════════════════════════════════
# PID MEMORY (von v8)
# ═══════════════════════════════════════════════════════════════════

class AdaptivePIDMemory(nn.Module):
    def __init__(self, dim, batch_size=1, device='cuda'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.batch_size = batch_size
        
        self.register_buffer('memory_p', torch.zeros(batch_size, dim))
        self.register_buffer('memory_i', torch.zeros(batch_size, dim))
        self.register_buffer('step', torch.ones(1))
        
        self.kp = nn.Parameter(torch.ones(dim) * 0.5)
        self.ki = nn.Parameter(torch.ones(dim) * 0.3)
        self.kd = nn.Parameter(torch.ones(dim) * 0.2)
        
        self.adaptive_decay = AdaptiveDecay(dim)
        
        self.project = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.project.weight)
        self.norm = RMSNorm(dim)
        
        self.last_decay = 0.9

    def inject(self, x):
        B = x.shape[0]
        mem_p_slice = self.memory_p[:B].clone()
        mem_i_slice = self.memory_i[:B].clone()
        
        p_term = self.kp * mem_p_slice
        i_term = self.ki * (mem_i_slice / max(self.step.item(), 1))
        d_term = self.kd * (mem_p_slice - mem_i_slice / max(self.step.item(), 1))
        pid_signal = p_term + i_term + d_term
        pid_proj = self.project(pid_signal).unsqueeze(1)
        return self.norm(x + pid_proj)

    def update(self, current_state):
        B = current_state.shape[0]
        curr = current_state.detach()
        
        with torch.no_grad():
            alpha = 0.9
            new_i = alpha * self.memory_i[:B] + (1 - alpha) * curr
            self.memory_i[:B].copy_(new_i)
            self.memory_p[:B].copy_(curr)
            self.step.add_(1)

    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        self.memory_p.zero_()
        self.memory_i.zero_()
        self.step.fill_(1)
    
    def get_decay(self):
        return self.last_decay


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE PID GATE (von v8)
# ═══════════════════════════════════════════════════════════════════

class AdaptivePIDGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kp = nn.Parameter(torch.ones(1, 1, dim) * PID_INIT_KP)
        self.ki = nn.Parameter(torch.ones(1, 1, dim) * PID_INIT_KI)
        self.kd = nn.Parameter(torch.ones(1, 1, dim) * PID_INIT_KD)
        self.norm = RMSNorm(dim)
        self.adaptive_decay = AdaptiveDecay(dim)
        self.last_decay = 0.9

    def forward(self, curr, integ, prev, step, content_for_decay):
        p_term = self.kp * curr
        new_integ, decay = self.adaptive_decay(integ, curr, content_for_decay)
        self.last_decay = decay
        i_term = self.ki * (new_integ / max(step, 1))
        d_term = self.kd * (curr - prev)
        raw_gate = p_term + i_term + d_term
        return F.silu(self.norm(raw_gate)), new_integ

    def get_gains(self):
        return {
            'Kp': self.kp.mean().item(),
            'Ki': self.ki.mean().item(),
            'Kd': self.kd.mean().item(),
            'Decay': self.last_decay
        }


# ═══════════════════════════════════════════════════════════════════
# TOKEN MIXER
# ═══════════════════════════════════════════════════════════════════

class TokenMixer(nn.Module):
    def __init__(self, dim, dilation, kernel_size):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim, bias=False)
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x_normed = self.norm(x)
        x_t = x_normed.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation
        x_t = F.pad(x_t, (pad, 0))
        x_t = self.conv(x_t)
        x_t = F.silu(x_t)
        x_mixed = x_t.transpose(1, 2)
        out = self.pointwise(x_mixed)
        out = self.drop(out)
        return out


# ═══════════════════════════════════════════════════════════════════
# DIMENSION MIXER
# ═══════════════════════════════════════════════════════════════════

class DimensionMixer(nn.Module):
    def __init__(self, dim, expansion=EXPANSION_FACTOR):
        super().__init__()
        hidden = int(dim * expansion)
        hidden = ((hidden + 63) // 64) * 64
        
        self.norm = RMSNorm(dim)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_value = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x_normed = self.norm(x)
        gate = F.silu(self.w_gate(x_normed))
        value = self.w_value(x_normed)
        hidden = gate * value
        out = self.w_out(hidden)
        out = self.drop(out)
        return out


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE MIXER BLOCK
# ═══════════════════════════════════════════════════════════════════

class AdaptiveMixerBlock(nn.Module):
    def __init__(self, dim, dilation, kernel_size, layer_idx):
        super().__init__()
        self.token_mixer = TokenMixer(dim, dilation, kernel_size)
        self.dim_mixer = DimensionMixer(dim)
        self.pid_gate = AdaptivePIDGate(dim)
        self.layer_idx = layer_idx
        
        self.register_buffer('integ', None)
        self.register_buffer('prev', None)

    def forward(self, x):
        B, T, D = x.shape
        
        if self.integ is None or self.integ.shape[0] != B or self.integ.shape[1] != T:
            self.integ = torch.zeros(B, T, D, device=x.device)
            self.prev = torch.zeros(B, T, D, device=x.device)
        
        token_out = self.token_mixer(x)
        
        gate_mod, new_integ = self.pid_gate(
            curr=token_out,
            integ=self.integ,
            prev=self.prev,
            step=self.layer_idx + 1,
            content_for_decay=token_out
        )
        gate_val = torch.sigmoid(gate_mod)
        
        self.integ = new_integ.detach()
        self.prev = token_out.detach()
        
        x = x + (token_out * gate_val)
        dim_out = self.dim_mixer(x)
        x = x + dim_out
        
        return x
    
    def reset_state(self):
        self.integ = None
        self.prev = None


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE MIXER STACK mit Importance Pool
# ═══════════════════════════════════════════════════════════════════

class AdaptiveMixerStack(nn.Module):
    def __init__(self, dim, num_layers, kernel_size, reverse=False):
        super().__init__()
        self.layers = nn.ModuleList()
        
        dilations = [min(2 ** i, MAX_DILATION) for i in range(num_layers)]
        if reverse:
            dilations = dilations[::-1]
        
        for i, d in enumerate(dilations):
            self.layers.append(AdaptiveMixerBlock(dim, d, kernel_size, i))
        
        # v9 NEU: Multi-Head Importance Pool statt CausalMeanPool!
        if USE_IMPORTANCE_POOL:
            self.context_pool = MultiHeadImportancePool(dim, heads=IMPORTANCE_HEADS)
        else:
            self.context_pool = None
        
        self.num_layers = num_layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # v9: Importance Pool für selektiven Kontext
        if self.context_pool is not None:
            x = self.context_pool(x)
        
        return x
    
    def reset_states(self):
        for layer in self.layers:
            layer.reset_state()
    
    def get_importance_stats(self):
        if self.context_pool is not None:
            return self.context_pool.get_importance_stats()
        return {}


class StackGate(nn.Module):
    def __init__(self, dim, rank=GATE_RANK):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.norm = RMSNorm(dim)
        nn.init.zeros_(self.up.weight)

    def forward(self, stack_out, residual):
        gate = torch.sigmoid(self.up(F.silu(self.down(stack_out))))
        return self.norm(gate * stack_out + (1 - gate) * residual)


# ═══════════════════════════════════════════════════════════════════
# 3. HAUPTMODELL
# ═══════════════════════════════════════════════════════════════════

class MemPIDModel(nn.Module):
    def __init__(self, vocab_size, dim=DIM, layers=LAYERS_PER_STACK, kernel_size=KERNEL_SIZE, batch_size=1):
        super().__init__()
        self.dim = dim
        
        self.mem = AdaptivePIDMemory(dim, batch_size, DEVICE)
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.02)
        self.emb_norm = RMSNorm(dim)
        self.emb_drop = nn.Dropout(EMB_DROPOUT)
        
        # Up-Down-Up Highway mit Importance Pool
        self.up_stack = AdaptiveMixerStack(dim, layers, kernel_size, False)
        self.down_stack = AdaptiveMixerStack(dim, layers, kernel_size, True)
        
        self.gate1 = StackGate(dim)
        self.gate2 = StackGate(dim)
        self.gate3 = StackGate(dim)
        
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pad_id=None):
        B, L = idx.shape
        x = self.emb(idx) + self.pos[:, :L, :]
        
        if self.training and NOISE_LEVEL > 0:
            x = x + torch.randn_like(x) * NOISE_LEVEL
        
        x = self.emb_norm(x)
        x = self.emb_drop(x)
        x = self.mem.inject(x)
        
        initial = x
        
        # Up-Down-Up Highway
        up1 = self.up_stack(x)
        gated1 = self.gate1(up1, initial)
        
        down = self.down_stack(gated1)
        gated2 = self.gate2(down, initial)
        
        up2 = self.up_stack(gated2)
        output = self.gate3(up2, gated2)
        
        self.mem.update(output[:, -1, :])
        output = self.final_norm(output)
        logits = self.head(output)
        
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_id)
        
        return logits, loss

    def reset_states(self):
        self.mem.reset(self.mem.batch_size)
        self.up_stack.reset_states()
        self.down_stack.reset_states()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.9, eos_id=None):
        self.eval()
        self.reset_states()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            self.reset_states()  # Reset für jeden Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_id is not None and idx_next.item() == eos_id:
                break
        
        return idx


# ═══════════════════════════════════════════════════════════════════
# 4. DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def setup_data():
    DATA_FOLDERS = scan_data_folders(BASE_DATA_DIR, MANUAL_FOLDERS)
    
    if not DATA_FOLDERS:
        print("\n❌ FEHLER: Keine Daten gefunden!")
        raise FileNotFoundError(f"Kein Ordner in {BASE_DATA_DIR}/ gefunden")
    
    dilations = [min(2 ** i, MAX_DILATION) for i in range(LAYERS_PER_STACK)]
    receptive_field = sum((KERNEL_SIZE - 1) * d for d in dilations) + 1
    
    print("\n" + "═" * 60)
    print("  🧠 MemPID v9 - MULTI-HEAD IMPORTANCE")
    print("═" * 60)
    print(f"  DIM={DIM}, LAYERS={LAYERS_PER_STACK}×3 = {LAYERS_PER_STACK*3} Denkschritte")
    print(f"  KERNEL={KERNEL_SIZE}, MaxDilation={MAX_DILATION}")
    print(f"  → Dilations: {dilations}")
    print(f"  → Reichweite: {receptive_field} Tokens ({receptive_field/BLOCK_SIZE*100:.0f}%)")
    print(f"  → Adaptive Decay (ZEIT-selektiv)")
    print(f"  → Multi-Head Importance Pool: {IMPORTANCE_HEADS} Heads (CONTENT-selektiv)")
    print(f"  → Dropout={DROPOUT_RATE}, Patience={PATIENCE}")
    print("═" * 60 + "\n")
    
    token_files = {}
    special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<SEP>", "<META>"] + list(DATA_FOLDERS.keys())
    
    all_filepaths = []
    for token, folder in DATA_FOLDERS.items():
        if os.path.exists(folder):
            files = glob(os.path.join(folder, "*.txt"))
            if files:
                token_files[token] = files
                all_filepaths.extend(files)
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    print("Trainiere Tokenizer...")
    
    def get_training_corpus():
        for path in all_filepaths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(1_000_000)
                        if not chunk:
                            break
                        yield chunk
            except:
                pass
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    
    token_ids = {t: tokenizer.token_to_id(t) for t in DATA_FOLDERS.keys() if t in token_files}
    
    train_data, val_data = {}, {}
    
    print("Lade und verarbeite Daten...")
    for category_token, filepaths in token_files.items():
        print(f"  Verarbeite {category_token}...")
        cat_id = tokenizer.token_to_id(category_token)
        all_tokens = []
        
        for fpath in filepaths:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
                encoded = tokenizer.encode(text).ids
                all_tokens.extend(encoded)
            except:
                pass
        
        if len(all_tokens) < BLOCK_SIZE + 1:
            continue
        
        split_idx = int(len(all_tokens) * (1 - VAL_SPLIT))
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]
        
        train_chunks = []
        for i in range(0, len(train_tokens) - BLOCK_SIZE, BLOCK_SIZE):
            chunk = train_tokens[i:i + BLOCK_SIZE + 1]
            if len(chunk) == BLOCK_SIZE + 1:
                train_chunks.append((cat_id, chunk))
        
        val_chunks = []
        for i in range(0, len(val_tokens) - BLOCK_SIZE, BLOCK_SIZE):
            chunk = val_tokens[i:i + BLOCK_SIZE + 1]
            if len(chunk) == BLOCK_SIZE + 1:
                val_chunks.append((cat_id, chunk))
        
        if train_chunks:
            train_data[category_token] = train_chunks
        if val_chunks:
            val_data[category_token] = val_chunks
        
        print(f"    {category_token}: {len(train_tokens):,} train / {len(val_tokens):,} val tokens ({len(train_chunks)} chunks)")
    
    return tokenizer, train_data, val_data, token_ids


class BalancedSampler:
    def __init__(self, data_dict, device='cuda'):
        self.data = data_dict
        self.device = device
        self.categories = list(data_dict.keys())
        
        counts = {cat: len(chunks) for cat, chunks in data_dict.items()}
        total = sum(counts.values())
        
        raw_weights = {cat: math.sqrt(c / total) for cat, c in counts.items()}
        weight_sum = sum(raw_weights.values())
        self.weights = {cat: w / weight_sum for cat, w in raw_weights.items()}
        
        print("📊 Sampling-Gewichtung:")
        for cat, w in self.weights.items():
            print(f"    {cat}: {w*100:.1f}%")

    def get_batch(self, batch_size, pad_id):
        xs, ys = [], []
        
        for _ in range(batch_size):
            cat = random.choices(self.categories, weights=[self.weights[c] for c in self.categories])[0]
            cat_id, chunk = random.choice(self.data[cat])
            
            chunk_used = chunk[:BLOCK_SIZE]
            full = [cat_id] + chunk_used
            
            x = full[:-1]
            y = full[1:]
            
            xs.append(torch.tensor(x, dtype=torch.long))
            ys.append(torch.tensor(y, dtype=torch.long))
        
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)


# ═══════════════════════════════════════════════════════════════════
# 5. TRAINING
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_step = 0
        self.should_stop = False

    def __call__(self, val_loss, step):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


@torch.no_grad()
def estimate_loss(model, train_sampler, val_sampler, pad_id, eval_iters=20):
    model.eval()
    out = {}
    
    for name, sampler in [('train', train_sampler), ('val', val_sampler)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = sampler.get_batch(BATCH_SIZE, pad_id)
            model.reset_states()
            _, loss = model(xb, yb, pad_id)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    
    model.train()
    return out


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def train():
    tokenizer, train_data, val_data, token_ids = setup_data()
    
    PAD_ID = tokenizer.token_to_id("<PAD>")
    EOS_ID = tokenizer.token_to_id("<EOS>")
    
    train_sampler = BalancedSampler(train_data, DEVICE)
    val_sampler = BalancedSampler(val_data, DEVICE)
    
    model = MemPIDModel(
        vocab_size=tokenizer.get_vocab_size(),
        dim=DIM,
        layers=LAYERS_PER_STACK,
        kernel_size=KERNEL_SIZE,
        batch_size=BATCH_SIZE
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Modell: {num_params/1e6:.1f}M Parameter ({trainable_params/1e6:.1f}M trainierbar)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping()
    
    print(f"\n🚀 Training startet...")
    print(f"   Batch: {BATCH_SIZE}")
    print(f"   🎯 Adaptive Decay + Multi-Head Importance Pool!")
    
    progress = tqdm(range(1, MAX_ITERS + 1), desc="Training")
    
    for step in progress:
        model.train()
        model.reset_states()
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        xb, yb = train_sampler.get_batch(BATCH_SIZE, PAD_ID)
        
        with torch.amp.autocast(device_type='cuda', dtype=PRECISION_DTYPE):
            logits, loss = model(xb, yb, PAD_ID)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_sampler, val_sampler, PAD_ID)
            
            first_block = model.up_stack.layers[0]
            gains = first_block.pid_gate.get_gains()
            imp_stats = model.up_stack.get_importance_stats()
            
            tqdm.write(f"\n📊 Step {step}: Train={losses['train']:.4f}, Val={losses['val']:.4f}")
            tqdm.write(f"   PID: Kp={gains['Kp']:.3f}, Ki={gains['Ki']:.3f}, Kd={gains['Kd']:.3f}")
            tqdm.write(f"   🎯 Decay={gains['Decay']:.3f}, Importance Mix={imp_stats.get('mix', 0):.3f}")
            
            is_best = early_stopping(losses['val'], step)
            
            if is_best:
                tqdm.write(f"   💾 Neuer Best! Val Loss: {losses['val']:.4f}")
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': losses['val'],
                    'config': {
                        'dim': DIM, 'layers': LAYERS_PER_STACK,
                        'kernel': KERNEL_SIZE, 'dilation': MAX_DILATION,
                        'vocab_size': tokenizer.get_vocab_size(),
                        'adaptive_decay': True,
                        'importance_heads': IMPORTANCE_HEADS
                    }
                }, 'best_model_v3.pt')
                tokenizer.save('tokenizer_v3.json')
            
            if early_stopping.should_stop:
                tqdm.write(f"\n⏹️ Early Stopping bei Step {step}")
                tqdm.write(f"   Bester Val Loss: {early_stopping.best_loss:.4f} @ Step {early_stopping.best_step}")
                break
        
        if step % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'checkpoint_v3_step{step}.pt')
    
    print("\n" + "═" * 50)
    print("  ✅ Training abgeschlossen!")
    print(f"  🎯 Adaptive Decay + Multi-Head Importance!")
    print(f"  Bester Val Loss: {early_stopping.best_loss:.4f} @ Step {early_stopping.best_step}")
    print("═" * 50)
    
    # Demo Generation
    print("\n📝 Demo-Generierung:")
    model.eval()
    
    for cat_token in list(train_data.keys())[:3]:
        cat_id = tokenizer.token_to_id(cat_token)
        start = torch.tensor([[cat_id]], device=DEVICE)
        
        try:
            generated = model.generate(start, max_new_tokens=100, temperature=0.8, eos_id=EOS_ID)
            text = tokenizer.decode(generated[0].tolist())
            
            print(f"\n{cat_token}:")
            print("-" * 40)
            print(text[:300] + "..." if len(text) > 300 else text)
        except Exception as e:
            print(f"{cat_token}: Fehler bei Generierung - {e}")


if __name__ == "__main__":
    train()
