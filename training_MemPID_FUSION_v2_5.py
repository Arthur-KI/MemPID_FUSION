# training_MemPID_FUSION_v2_5.py
# ═══════════════════════════════════════════════════════════════════
#  🧠 MemPID FUSION v2.5
#  Gleiche Größe wie v2 (~28.7M) + Alle modernen Verbesserungen
#  
#  Änderungen:
#  ✓ LayerNorm → RMSNorm
#  ✓ GLU → SwiGLU  
#  ✓ GELU → SiLU
#  ✓ Pre-Norm statt Post-Norm
#  ✓ TokenShift in Memory
#  
#  BEHALTEN (Größe wie v2):
#  ✓ DIM = 512
#  ✓ 6 Layer pro Stack
#  ✓ Up-Down-Up Highway
#  ✓ CausalMeanPool
#  ✓ PIDGates, StackGates, PIDMemory
# ═══════════════════════════════════════════════════════════════════

import os
import random
import math
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
BATCH_SIZE = 8               # Wie v2
BLOCK_SIZE = 2048
MAX_ITERS = 150000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modell Architektur - GLEICHE GRÖßE WIE V2
VOCAB_SIZE = 16000
DIM = 512                    # Behalten!
LAYERS_PER_STACK = 6         # Behalten
KERNEL_SIZE = 15             # Behalten
USE_CAUSAL_MEAN = True
CAUSAL_MEAN_WEIGHT = 0.1
GATE_RANK = 64               # Behalten

# SwiGLU Hidden Dim (kleiner für ~gleiche Params)
SWIGLU_MULT = 2              # Statt 8/3, kleiner halten

# Regularisierung
DROPOUT_RATE = 0.1
EMB_DROPOUT = 0.1
NOISE_LEVEL = 0.02

# PID Controller
PID_INIT_KP = 0.5
PID_INIT_KI = 0.3
PID_INIT_KD = 0.2
EMA_ALPHA = 0.90

# Training Control
VAL_SPLIT = 0.1
PATIENCE = 20
MIN_DELTA = 0.005

PRECISION_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════
# 1. DATEN-ORDNER
# ═══════════════════════════════════════════════════════════════════
# PFADE ANPASSEN! Nutze die gleichen wie im Original-FUSION:
DATA_FOLDERS = {
    "<KLASSIKER>": "buecher/klassiker",
    "<PHILOSOPHIE>": "buecher/philosophie",
    "<LYRIK>": "buecher/lyrik",
    "<DRAMA>": "buecher/drama",
    "<WISSEN>": "buecher/wissen",
}

# ═══════════════════════════════════════════════════════════════════
# 2. MODEL COMPONENTS - MODERNIZED
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm (LLaMA-Style) - ersetzt LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalMeanPool(nn.Module):
    """Kausaler Mittelwert-Pool für globalen Kontext"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        cumsum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x + self.weight * (cumsum / counts)


class PIDMemoryModule(nn.Module):
    """Rekurrenter PID-Speicher mit TokenShift"""
    def __init__(self, dim, batch_size, device, ema_alpha=EMA_ALPHA):
        super().__init__()
        self.dim = dim
        self.ema_alpha = ema_alpha
        
        # Memory Buffers
        self.register_buffer('memory_p', torch.zeros(batch_size, dim, device=device))
        self.register_buffer('memory_i', torch.zeros(batch_size, dim, device=device))
        self.register_buffer('memory_d', torch.zeros(batch_size, dim, device=device))
        
        # TokenShift für Memory Injection
        self.token_shift = nn.Parameter(torch.ones(dim) * 0.5)
        
        # Memory Gate
        self.mem_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.mem_gate.weight)
        
        self.reset()

    def reset(self):
        self.memory_p.zero_()
        self.memory_i.zero_()
        self.memory_d.zero_()

    def update(self, current_state):
        B = current_state.shape[0]
        new_d = current_state.detach() - self.memory_p[:B].detach()
        self.memory_d[:B].copy_(new_d)
        new_i = (self.ema_alpha * self.memory_i[:B].detach()) + ((1 - self.ema_alpha) * current_state.detach())
        self.memory_i[:B].copy_(new_i)
        self.memory_p[:B].copy_(current_state.detach())

    def inject(self, embeddings):
        B, T, D = embeddings.shape
        
        # TokenShift
        shift = self.token_shift.view(1, 1, -1)
        shifted = torch.zeros_like(embeddings)
        shifted[:, 0, :] = embeddings[:, 0, :]
        shifted[:, 1:, :] = shift * embeddings[:, :-1, :] + (1 - shift) * embeddings[:, 1:, :]
        
        # Memory Injection
        mem_p = torch.sigmoid(self.mem_gate(self.memory_p[:B]))
        return shifted + mem_p.unsqueeze(1).expand_as(embeddings)

    def get_state_info(self):
        return {
            'P': self.memory_p.abs().mean().item(),
            'I': self.memory_i.abs().mean().item(),
            'D': self.memory_d.abs().mean().item(),
        }


class PIDGate(nn.Module):
    """Lernbarer PID-Gate mit SiLU + RMSNorm"""
    def __init__(self, dim, kp=PID_INIT_KP, ki=PID_INIT_KI, kd=PID_INIT_KD):
        super().__init__()
        self.kp = nn.Parameter(torch.ones(dim) * kp)
        self.ki = nn.Parameter(torch.ones(dim) * ki)
        self.kd = nn.Parameter(torch.ones(dim) * kd)
        self.norm = RMSNorm(dim)

    def forward(self, curr, integ, prev, step):
        p_term = self.kp * curr
        i_term = self.ki * (integ / max(step, 1))
        d_term = self.kd * (curr - prev)
        return self.norm(F.silu(p_term + i_term + d_term))

    def get_gains(self):
        return {
            'Kp': self.kp.mean().item(),
            'Ki': self.ki.mean().item(),
            'Kd': self.kd.mean().item()
        }


class SwiGLU(nn.Module):
    """SwiGLU - kompakte Version für kleine Modelle"""
    def __init__(self, dim, mult=SWIGLU_MULT):
        super().__init__()
        hidden_dim = int(dim * mult)
        # Auf 64 aufrunden
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalDilatedConv(nn.Module):
    """Kausale dilatierte Convolution mit SwiGLU + Pre-Norm"""
    def __init__(self, dim, dilation, kernel_size):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Pre-Norm
        self.norm = RMSNorm(dim)
        
        # Depthwise Conv
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        
        # SwiGLU (kompakt)
        self.swiglu = SwiGLU(dim)
        
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        residual = x
        
        # Pre-Norm
        x = self.norm(x)
        
        # Conv
        x = x.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        
        # SwiGLU
        x = self.swiglu(x)
        x = self.drop(x)
        
        return x + residual


class PIDHighwayStack(nn.Module):
    """PID-gesteuerter Highway Stack"""
    def __init__(self, dim, layers, kernel_size, reverse=False):
        super().__init__()
        if reverse:
            dilations = [2 ** i for i in reversed(range(layers))]
        else:
            dilations = [2 ** i for i in range(layers)]
        
        self.layers = nn.ModuleList([
            CausalDilatedConv(dim, d, kernel_size) for d in dilations
        ])
        self.gates = nn.ModuleList([PIDGate(dim) for _ in range(layers - 1)])

    def forward(self, x):
        integral = x.clone()
        previous = x.clone()
        current = x
        
        for i, layer in enumerate(self.layers):
            if i > 0:
                current = layer(self.gates[i - 1](current, integral, previous, i))
            else:
                current = layer(current)
            previous = current.clone()
            integral = integral + current
        
        return current


class StackGate(nn.Module):
    """Gate für Stack-Übergang"""
    def __init__(self, dim, rank=GATE_RANK):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)

    def forward(self, new_info, old_info):
        new_normed = self.norm(new_info)
        g = torch.sigmoid(self.up(F.silu(self.down(new_normed))))
        return g * new_info + (1.0 - g) * old_info


class MemPIDModel(nn.Module):
    """MemPID FUSION v2.5 - Kleine Größe, Moderne Tricks"""
    def __init__(self, vocab_size, dim=DIM, layers=LAYERS_PER_STACK, kernel_size=KERNEL_SIZE, batch_size=BATCH_SIZE):
        super().__init__()
        self.dim = dim
        
        # Memory
        self.mem = PIDMemoryModule(dim, batch_size, DEVICE)
        
        # Embeddings
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.02)
        self.emb_norm = RMSNorm(dim)
        self.emb_drop = nn.Dropout(EMB_DROPOUT)

        # Up-Down-Up Highway
        self.up_stack = PIDHighwayStack(dim, layers, kernel_size, False)
        self.down_stack = PIDHighwayStack(dim, layers, kernel_size, True)

        # Stack Gates
        self.gate1 = StackGate(dim)
        self.gate2 = StackGate(dim)
        self.gate3 = StackGate(dim)

        # Causal Mean Pool
        self.mix = CausalMeanPool(CAUSAL_MEAN_WEIGHT) if USE_CAUSAL_MEAN else None
        
        # Output
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight Tying
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
        
        # Embeddings
        x = self.emb(idx) + self.pos[:, :L, :]

        # Noise (Training)
        if self.training and NOISE_LEVEL > 0:
            x = x + torch.randn_like(x) * NOISE_LEVEL

        # Pre-Norm + Dropout
        x = self.emb_norm(x)
        x = self.emb_drop(x)
        
        # Memory Injection
        x = self.mem.inject(x)
        initial = x

        # === UP-DOWN-UP ===
        
        # UP 1
        up1 = self.up_stack(x)
        if self.mix:
            up1 = self.mix(up1)
        gated1 = self.gate1(up1, initial)

        # DOWN
        down = self.down_stack(gated1)
        if self.mix:
            down = self.mix(down)
        gated2 = self.gate2(down, initial)

        # UP 2
        up2 = self.up_stack(gated2)
        if self.mix:
            up2 = self.mix(up2)
        output = self.gate3(up2, gated2)

        # Memory Update
        self.mem.update(output[:, -1, :])

        # Output
        output = self.final_norm(output)
        logits = self.head(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_id
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        ctx = torch.amp.autocast(device_type='cuda', dtype=PRECISION_DTYPE) if DEVICE == 'cuda' else contextlib.nullcontext()
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            with ctx:
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        
        return idx


# ═══════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def setup_data():
    """Lade Daten und trainiere Tokenizer"""
    token_files = {}
    special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<SEP>", "<META>"] + list(DATA_FOLDERS.keys())

    print("═" * 60)
    print("  🧠 MemPID FUSION v2.5")
    print("  Gleiche Größe + Moderne Verbesserungen")
    print("═" * 60)
    print(f"Device: {DEVICE} | Precision: {PRECISION_DTYPE}")
    print(f"Upgrades:")
    print(f"  ✓ RMSNorm statt LayerNorm")
    print(f"  ✓ SwiGLU statt GLU")
    print(f"  ✓ SiLU statt GELU")
    print(f"  ✓ Pre-Norm statt Post-Norm")
    print(f"  ✓ TokenShift in Memory")
    print(f"Architektur: DIM={DIM}, LAYERS={LAYERS_PER_STACK}, BLOCK={BLOCK_SIZE}")
    print("═" * 60 + "\n")

    print("Lade Texte...")
    all_filepaths = []
    total_chars = 0

    for token, folder in DATA_FOLDERS.items():
        if os.path.exists(folder):
            files = glob(os.path.join(folder, "*.txt"))
            if files:
                token_files[token] = files
                all_filepaths.extend(files)
                chars = sum(os.path.getsize(f) for f in files)
                total_chars += chars
                print(f"  {token}: {chars:,} Zeichen ({len(files)} Dateien)")

    print(f"\n📊 Gesamt: {total_chars:,} Zeichen ({total_chars/1e6:.2f} MB)\n")

    print(f"Trainiere Tokenizer (Vocab: {VOCAB_SIZE})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    def get_training_corpus():
        for path in all_filepaths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    while True:
                        chunk = f.read(1_000_000)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                print(f"  ⚠️ Fehler bei {path}: {e}")

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=special_tokens
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    token_ids = {t: tokenizer.token_to_id(t) for t in DATA_FOLDERS.keys() if t in token_files}

    # Encode
    train_data = {}
    val_data = {}
    print("\nEncodiere und splitte Daten...")
    
    CHUNK_SIZE = 5_000_000

    for token, files in token_files.items():
        train_tensors = []
        val_tensors = []

        for path in tqdm(files, desc=f"  {token}", leave=False):
            try:
                file_size = os.path.getsize(path)
                
                if file_size > CHUNK_SIZE * 2:
                    with open(path, 'r', encoding='utf-8') as f:
                        chunk_num = 0
                        while True:
                            text = f.read(CHUNK_SIZE)
                            if not text:
                                break
                            
                            ids = tokenizer.encode(text + " <EOS>").ids
                            if len(ids) > BLOCK_SIZE:
                                tensor = torch.tensor(ids, dtype=torch.long)
                                if chunk_num % 10 == 0:
                                    val_tensors.append(tensor)
                                else:
                                    train_tensors.append(tensor)
                                chunk_num += 1
                            
                            del ids
                            if chunk_num % 50 == 0:
                                import gc
                                gc.collect()
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    ids = tokenizer.encode(text + " <EOS>").ids
                    if len(ids) > BLOCK_SIZE:
                        tensor = torch.tensor(ids, dtype=torch.long)
                        split_idx = int(len(tensor) * (1 - VAL_SPLIT))
                        train_tensors.append(tensor[:split_idx])
                        val_tensors.append(tensor[split_idx:])
                        
            except Exception as e:
                tqdm.write(f"  ⚠️ Fehler: {e}")
                continue

        train_data[token] = train_tensors
        val_data[token] = val_tensors

        train_tokens = sum(len(t) for t in train_tensors)
        val_tokens = sum(len(t) for t in val_tensors)
        print(f"  {token}: {train_tokens:,} train / {val_tokens:,} val tokens")

    return train_data, val_data, token_ids, tokenizer


# ═══════════════════════════════════════════════════════════════════
# 4. SEQUENTIAL SAMPLING
# ═══════════════════════════════════════════════════════════════════

class SequentialSampler:
    """Sequenzielles Sampling mit AUTO-GEWICHTUNG"""

    def __init__(self, data_map, token_ids, balance_mode='sqrt'):
        self.data_map = data_map
        self.token_ids = token_ids
        self.categories = [c for c in data_map.keys() if data_map[c]]
        
        token_counts = {}
        for cat in self.categories:
            token_counts[cat] = sum(len(t) for t in data_map[cat])
        
        if balance_mode == 'sqrt':
            raw_weights = {cat: math.sqrt(count) for cat, count in token_counts.items()}
        elif balance_mode == 'equal':
            raw_weights = {cat: 1.0 for cat in self.categories}
        else:
            raw_weights = {cat: count for cat, count in token_counts.items()}
        
        weight_sum = sum(raw_weights.values())
        self.weights = [raw_weights[cat] / weight_sum for cat in self.categories]
        
        print("\n📊 Auto-Gewichtung (sqrt-balanced):")
        for cat, w in zip(self.categories, self.weights):
            tokens = token_counts[cat]
            print(f"    {cat}: {tokens:,} tokens → {w*100:.1f}% sampling")
        print()

        self.cursors = {}
        for cat in self.categories:
            self.cursors[cat] = {'file': 0, 'pos': 0}

    def get_batch(self, batch_size, pad_id):
        xs, ys = [], []

        for _ in range(batch_size):
            cat = random.choices(self.categories, weights=self.weights, k=1)[0]
            files = self.data_map[cat]
            if not files:
                continue

            cur = self.cursors[cat]
            file_idx = cur['file'] % len(files)
            pos = cur['pos']
            source = files[file_idx]

            if pos + BLOCK_SIZE + 1 > len(source):
                cur['file'] = (cur['file'] + 1) % len(files)
                file_idx = cur['file']
                source = files[file_idx]

                max_offset = min(BLOCK_SIZE, len(source) - BLOCK_SIZE - 1)
                if max_offset > 0:
                    cur['pos'] = random.randint(0, max_offset)
                else:
                    cur['pos'] = 0
                pos = cur['pos']

            chunk = source[pos: pos + BLOCK_SIZE]
            cur['pos'] += BLOCK_SIZE

            if len(chunk) < BLOCK_SIZE:
                pad_tensor = torch.full((BLOCK_SIZE - len(chunk),), pad_id, dtype=torch.long)
                chunk = torch.cat([chunk, pad_tensor])

            cat_id = torch.tensor([self.token_ids[cat]], dtype=torch.long)
            full = torch.cat([cat_id, chunk])

            xs.append(full[:-1])
            ys.append(full[1:])

        if not xs:
            return None, None

        return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)

    def reset(self):
        for cat in self.cursors:
            self.cursors[cat] = {'file': 0, 'pos': 0}


# ═══════════════════════════════════════════════════════════════════
# 5. TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.best_step = 0

    def __call__(self, val_loss, step):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_step = step
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def status(self):
        return f"Best: {self.best_loss:.4f} @ Step {self.best_step} | Patience: {self.counter}/{self.patience}"


def get_lr(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    if step > MAX_ITERS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def evaluate(model, sampler, pad_id, num_batches=20):
    model.eval()
    model.mem.reset()
    total_loss = 0
    count = 0

    ctx = torch.amp.autocast(device_type='cuda', dtype=PRECISION_DTYPE) if DEVICE == 'cuda' else contextlib.nullcontext()

    for _ in range(num_batches):
        xb, yb = sampler.get_batch(BATCH_SIZE, pad_id)
        if xb is None:
            continue
        with ctx:
            _, loss = model(xb, yb, pad_id)
        if loss is not None:
            total_loss += loss.item()
            count += 1

    model.mem.reset()
    model.train()
    return total_loss / max(count, 1)


@torch.no_grad()
def generate_sample(model, tokenizer, prompt_token, max_tokens=100):
    model.eval()
    model.mem.reset()

    token_id = tokenizer.token_to_id(prompt_token)
    if token_id is None:
        return "[Unknown token]"

    idx = torch.tensor([[token_id]], dtype=torch.long, device=DEVICE)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)

    text = tokenizer.decode(output[0].tolist())
    model.mem.reset()
    model.train()
    return text[:200]


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Setup
    train_data, val_data, token_ids, tokenizer = setup_data()
    vocab_size = tokenizer.get_vocab_size()
    PAD_ID = tokenizer.token_to_id("<PAD>")

    print(f"\nVokabular: {vocab_size} Tokens")

    # Sampler
    train_sampler = SequentialSampler(train_data, token_ids, balance_mode='sqrt')
    val_sampler = SequentialSampler(val_data, token_ids, balance_mode='sqrt')

    # Model
    model = MemPIDModel(vocab_size, batch_size=BATCH_SIZE).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameter: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Vergleich
    print(f"   (v2 hatte ~17M, Ziel: ähnliche Größe)")
    
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"💾 VRAM nach Init: {allocated:.2f} GB")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.amp.GradScaler() if DEVICE == 'cuda' else None

    # Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    # Resume?
    start_step = 0
    if os.path.exists("mempid_fusion_v2_5_best.pt"):
        print("♻️ Lade Checkpoint...")
        checkpoint = torch.load("mempid_fusion_v2_5_best.pt", map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            start_step = checkpoint.get('step', 0)
        else:
            model.load_state_dict(checkpoint)

    print("\n🚀 Training startet...")
    print("─" * 60)

    model.train()
    ctx = torch.amp.autocast(device_type='cuda', dtype=PRECISION_DTYPE) if DEVICE == 'cuda' else contextlib.nullcontext()

    pbar = tqdm(range(start_step, MAX_ITERS), initial=start_step, total=MAX_ITERS)

    for step in pbar:
        # LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Batch
        xb, yb = train_sampler.get_batch(BATCH_SIZE, PAD_ID)
        if xb is None:
            continue

        # Forward
        with ctx:
            logits, loss = model(xb, yb, PAD_ID)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss = loss.item()
        if step % 10 == 0:
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'lr': f'{lr:.2e}'})

        # Eval
        if step > 0 and step % EVAL_INTERVAL == 0:
            mem_info = model.mem.get_state_info()
            val_loss = evaluate(model, val_sampler, PAD_ID)
            is_best = early_stopping(val_loss, step)
            
            if DEVICE == 'cuda':
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
            else:
                peak_mem = 0

            tqdm.write("\n" + "─" * 60)
            tqdm.write(f"Step {step}:")
            tqdm.write(f"  Train Loss: {train_loss:.4f}")
            tqdm.write(f"  Val Loss:   {val_loss:.4f} {'🏆 BEST!' if is_best else ''}")
            tqdm.write(f"  {early_stopping.status()}")
            tqdm.write(f"  Peak VRAM: {peak_mem:.2f} GB")

            # PID Info
            gains = model.up_stack.gates[0].get_gains()
            mem_info = model.mem.get_state_info()
            tqdm.write(f"  PID Gate: Kp={gains['Kp']:.3f}, Ki={gains['Ki']:.3f}, Kd={gains['Kd']:.3f}")
            tqdm.write(f"  Memory: P={mem_info['P']:.3f}, I={mem_info['I']:.3f}, D={mem_info['D']:.3f}")

            if is_best:
                sample = generate_sample(model, tokenizer, "<KLASSIKER>", max_tokens=80)
                tqdm.write(f"  Sample: {sample[:150]}...")

                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'val_loss': val_loss,
                    'config': {
                        'dim': DIM,
                        'layers': LAYERS_PER_STACK,
                        'vocab_size': vocab_size,
                        'block_size': BLOCK_SIZE,
                        'version': 'v2.5'
                    }
                }, "mempid_fusion_v2_5_best.pt")
                tqdm.write("  💾 Modell gespeichert!")

            tqdm.write("─" * 60 + "\n")

            if early_stopping.should_stop:
                tqdm.write("🛑 EARLY STOPPING!")
                break

    # Final
    print("\nSpeichere finale Modelle...")
    torch.save(model.state_dict(), "mempid_fusion_v2_5_final.pt")
    tokenizer.save("tokenizer_fusion_v2_5.json")

    print("\n" + "═" * 60)
    print("  🧠 MemPID FUSION v2.5 TRAINING ABGESCHLOSSEN")
    print("═" * 60)
    print(f"  Bester Val Loss: {early_stopping.best_loss:.4f} @ Step {early_stopping.best_step}")
    print(f"  Architektur: DIM={DIM}, Layers={LAYERS_PER_STACK}×3")
    print(f"  Upgrades: RMSNorm, SwiGLU, SiLU, Pre-Norm, TokenShift")
    print("  Dateien:")
    print("    - mempid_fusion_v2_5_best.pt")
    print("    - mempid_fusion_v2_5_final.pt")
    print("    - tokenizer_fusion_v2_5.json")
    print("═" * 60)
