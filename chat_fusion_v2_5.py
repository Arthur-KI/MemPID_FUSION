# chat_fusion_v2_5.py
# ═══════════════════════════════════════════════════════════════════
#  🧠 MemPID FUSION v2.5 - CHAT MODUS
#  Interaktiver Chat mit deinem trainierten Modell
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import contextlib

# ═══════════════════════════════════════════════════════════════════
# EINSTELLUNGEN
# ═══════════════════════════════════════════════════════════════════

# Pfade - ANPASSEN falls nötig!
MODEL_PATH = "mempid_fusion_v2_5_best.pt"
TOKENIZER_PATH = "tokenizer_fusion_v2_5.json"

# Generierung
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9

# Architektur (muss mit Training übereinstimmen!)
DIM = 512
LAYERS_PER_STACK = 6
KERNEL_SIZE = 15
BLOCK_SIZE = 2048
GATE_RANK = 64
SWIGLU_MULT = 2
DROPOUT_RATE = 0.1
EMB_DROPOUT = 0.1
USE_CAUSAL_MEAN = True
CAUSAL_MEAN_WEIGHT = 0.1

# PID
PID_INIT_KP = 0.5
PID_INIT_KI = 0.3
PID_INIT_KD = 0.2
EMA_ALPHA = 0.90

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRECISION_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

# ═══════════════════════════════════════════════════════════════════
# MODEL COMPONENTS (identisch zum Training)
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalMeanPool(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        cumsum = torch.cumsum(x, dim=1)
        counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x + self.weight * (cumsum / counts)


class PIDMemoryModule(nn.Module):
    def __init__(self, dim, batch_size, device, ema_alpha=EMA_ALPHA):
        super().__init__()
        self.dim = dim
        self.ema_alpha = ema_alpha
        
        self.register_buffer('memory_p', torch.zeros(batch_size, dim, device=device))
        self.register_buffer('memory_i', torch.zeros(batch_size, dim, device=device))
        self.register_buffer('memory_d', torch.zeros(batch_size, dim, device=device))
        
        self.token_shift = nn.Parameter(torch.ones(dim) * 0.5)
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
        
        shift = self.token_shift.view(1, 1, -1)
        shifted = torch.zeros_like(embeddings)
        shifted[:, 0, :] = embeddings[:, 0, :]
        shifted[:, 1:, :] = shift * embeddings[:, :-1, :] + (1 - shift) * embeddings[:, 1:, :]
        
        mem_p = torch.sigmoid(self.mem_gate(self.memory_p[:B]))
        return shifted + mem_p.unsqueeze(1).expand_as(embeddings)

    def get_state_info(self):
        return {
            'P': self.memory_p.abs().mean().item(),
            'I': self.memory_i.abs().mean().item(),
            'D': self.memory_d.abs().mean().item(),
        }


class PIDGate(nn.Module):
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


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=SWIGLU_MULT):
        super().__init__()
        hidden_dim = int(dim * mult)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalDilatedConv(nn.Module):
    def __init__(self, dim, dilation, kernel_size):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.swiglu = SwiGLU(dim)
        self.drop = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = F.silu(x)
        x = x.transpose(1, 2)
        x = self.swiglu(x)
        x = self.drop(x)
        return x + residual


class PIDHighwayStack(nn.Module):
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
    def __init__(self, vocab_size, dim=DIM, layers=LAYERS_PER_STACK, kernel_size=KERNEL_SIZE, batch_size=1):
        super().__init__()
        self.dim = dim
        
        self.mem = PIDMemoryModule(dim, batch_size, DEVICE)
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, BLOCK_SIZE, dim) * 0.02)
        self.emb_norm = RMSNorm(dim)
        self.emb_drop = nn.Dropout(EMB_DROPOUT)

        self.up_stack = PIDHighwayStack(dim, layers, kernel_size, False)
        self.down_stack = PIDHighwayStack(dim, layers, kernel_size, True)

        self.gate1 = StackGate(dim)
        self.gate2 = StackGate(dim)
        self.gate3 = StackGate(dim)

        self.mix = CausalMeanPool(CAUSAL_MEAN_WEIGHT) if USE_CAUSAL_MEAN else None
        
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, idx, targets=None, pad_id=None):
        B, L = idx.shape
        
        x = self.emb(idx) + self.pos[:, :L, :]
        x = self.emb_norm(x)
        x = self.emb_drop(x)
        
        x = self.mem.inject(x)
        initial = x

        up1 = self.up_stack(x)
        if self.mix:
            up1 = self.mix(up1)
        gated1 = self.gate1(up1, initial)

        down = self.down_stack(gated1)
        if self.mix:
            down = self.mix(down)
        gated2 = self.gate2(down, initial)

        up2 = self.up_stack(gated2)
        if self.mix:
            up2 = self.mix(up2)
        output = self.gate3(up2, gated2)

        self.mem.update(output[:, -1, :])

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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.9, 
                 stop_tokens=None, callback=None):
        """
        Generiert Text mit Top-K und Top-P Sampling
        
        Args:
            idx: Start-Tokens [1, seq_len]
            max_new_tokens: Maximale Anzahl neuer Tokens
            temperature: Kreativität (0.1=konservativ, 1.0=kreativ)
            top_k: Nur die K wahrscheinlichsten Tokens
            top_p: Nucleus Sampling Schwelle
            stop_tokens: Liste von Token-IDs zum Stoppen
            callback: Funktion die bei jedem Token aufgerufen wird
        """
        self.eval()
        ctx = torch.amp.autocast(device_type='cuda', dtype=PRECISION_DTYPE) if DEVICE == 'cuda' else contextlib.nullcontext()
        
        if stop_tokens is None:
            stop_tokens = []
        
        for i in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            with ctx:
                logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
            
            # Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-P (Nucleus) Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat([idx, next_idx], dim=1)
            
            # Callback für Streaming
            if callback:
                callback(next_idx.item())
            
            # Stop bei EOS oder anderen Stop-Tokens
            if next_idx.item() in stop_tokens:
                break
        
        return idx


# ═══════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════

class ChatInterface:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        print("═" * 60)
        print("  🧠 MemPID FUSION v2.5 - CHAT MODUS")
        print("═" * 60)
        print(f"  Device: {DEVICE}")
        print(f"  Precision: {PRECISION_DTYPE}")
        print("═" * 60)
        
        # Lade Tokenizer
        print("\n📚 Lade Tokenizer...")
        if not os.path.exists(tokenizer_path):
            print(f"❌ FEHLER: Tokenizer nicht gefunden: {tokenizer_path}")
            sys.exit(1)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"   Vokabular: {self.vocab_size} Tokens")
        
        # Special Tokens
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")
        self.unk_id = self.tokenizer.token_to_id("<UNK>")
        
        # Lade Modell
        print("\n🔧 Lade Modell...")
        if not os.path.exists(model_path):
            print(f"❌ FEHLER: Modell nicht gefunden: {model_path}")
            sys.exit(1)
        
        self.model = MemPIDModel(self.vocab_size, batch_size=1).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            # Memory Buffers haben andere Batch-Size - ignorieren und neu initialisieren
            keys_to_remove = [k for k in state_dict.keys() if 'memory_p' in k or 'memory_i' in k or 'memory_d' in k]
            for k in keys_to_remove:
                del state_dict[k]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"   Checkpoint Step: {checkpoint.get('step', 'unknown')}")
            print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        else:
            state_dict = checkpoint
            keys_to_remove = [k for k in state_dict.keys() if 'memory_p' in k or 'memory_i' in k or 'memory_d' in k]
            for k in keys_to_remove:
                del state_dict[k]
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
        
        # Parameter zählen
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameter: {num_params:,} ({num_params/1e6:.1f}M)")
        
        if DEVICE == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"   VRAM: {allocated:.2f} GB")
        
        # Settings
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_k = DEFAULT_TOP_K
        self.top_p = DEFAULT_TOP_P
        self.stream = True
        
        print("\n✅ Modell geladen!")
        print("─" * 60)
    
    def encode(self, text):
        """Text zu Token-IDs"""
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids):
        """Token-IDs zu Text"""
        return self.tokenizer.decode(ids)
    
    def generate(self, prompt, max_tokens=None, temperature=None, top_k=None, 
                 top_p=None, stream=None):
        """Generiert Text basierend auf Prompt"""
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        top_k = top_k if top_k is not None else self.top_k
        top_p = top_p if top_p is not None else self.top_p
        stream = stream if stream is not None else self.stream
        
        # Encode prompt
        tokens = self.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        
        # Stop tokens
        stop_tokens = [self.eos_id] if self.eos_id else []
        
        # Streaming callback
        generated_tokens = []
        def stream_callback(token_id):
            generated_tokens.append(token_id)
            if stream:
                token_text = self.decode([token_id])
                print(token_text, end='', flush=True)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_tokens,
                callback=stream_callback if stream else None
            )
        
        if stream:
            print()  # Newline nach Streaming
        
        # Decode nur neue Tokens
        new_tokens = output[0, len(tokens):].tolist()
        return self.decode(new_tokens)
    
    def reset_memory(self):
        """Setzt das PID-Memory zurück"""
        self.model.mem.reset()
        print("🔄 Memory zurückgesetzt!")
    
    def show_settings(self):
        """Zeigt aktuelle Einstellungen"""
        print("\n⚙️  AKTUELLE EINSTELLUNGEN:")
        print(f"   Max Tokens:   {self.max_tokens}")
        print(f"   Temperature:  {self.temperature}")
        print(f"   Top-K:        {self.top_k}")
        print(f"   Top-P:        {self.top_p}")
        print(f"   Streaming:    {self.stream}")
        
        mem_info = self.model.mem.get_state_info()
        print(f"\n📊 MEMORY STATUS:")
        print(f"   P-Term: {mem_info['P']:.4f}")
        print(f"   I-Term: {mem_info['I']:.4f}")
        print(f"   D-Term: {mem_info['D']:.4f}")
    
    def show_help(self):
        """Zeigt Hilfe"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  🧠 FUSION v2.5 CHAT - BEFEHLE                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CHAT:                                                       ║
║    Einfach Text eingeben und Enter drücken                   ║
║                                                              ║
║  BEFEHLE:                                                    ║
║    /help          - Diese Hilfe anzeigen                     ║
║    /settings      - Aktuelle Einstellungen                   ║
║    /reset         - Memory zurücksetzen                      ║
║    /temp <wert>   - Temperature setzen (0.1-2.0)             ║
║    /tokens <n>    - Max Tokens setzen                        ║
║    /topk <n>      - Top-K setzen (0=aus)                     ║
║    /topp <wert>   - Top-P setzen (0.0-1.0)                   ║
║    /stream        - Streaming an/aus                         ║
║    /quit          - Beenden                                  ║
║                                                              ║
║  KATEGORIEN (als Prompt-Start):                              ║
║    <KLASSIKER>    - Klassische Literatur                     ║
║    <PHILOSOPHIE>  - Philosophischer Stil                     ║
║    <LYRIK>        - Gedichte/Poesie                          ║
║    <WISSEN>       - Wikipedia/Faktenwissen                   ║
║                                                              ║
║  BEISPIELE:                                                  ║
║    > <LYRIK> Die Sonne scheint                               ║
║    > <KLASSIKER> Es war einmal                               ║
║    > <WISSEN> Deutschland ist                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    def run(self):
        """Hauptschleife des Chat-Interface"""
        self.show_help()
        
        print("\n🚀 Chat bereit! Tippe '/help' für Hilfe.\n")
        
        while True:
            try:
                user_input = input("Du: ").strip()
                
                if not user_input:
                    continue
                
                # Befehle
                if user_input.startswith('/'):
                    cmd = user_input.lower().split()
                    
                    if cmd[0] == '/quit' or cmd[0] == '/exit':
                        print("\n👋 Auf Wiedersehen!")
                        break
                    
                    elif cmd[0] == '/help':
                        self.show_help()
                    
                    elif cmd[0] == '/settings':
                        self.show_settings()
                    
                    elif cmd[0] == '/reset':
                        self.reset_memory()
                    
                    elif cmd[0] == '/temp' and len(cmd) > 1:
                        try:
                            self.temperature = float(cmd[1])
                            print(f"✅ Temperature: {self.temperature}")
                        except:
                            print("❌ Ungültiger Wert!")
                    
                    elif cmd[0] == '/tokens' and len(cmd) > 1:
                        try:
                            self.max_tokens = int(cmd[1])
                            print(f"✅ Max Tokens: {self.max_tokens}")
                        except:
                            print("❌ Ungültiger Wert!")
                    
                    elif cmd[0] == '/topk' and len(cmd) > 1:
                        try:
                            self.top_k = int(cmd[1])
                            print(f"✅ Top-K: {self.top_k}")
                        except:
                            print("❌ Ungültiger Wert!")
                    
                    elif cmd[0] == '/topp' and len(cmd) > 1:
                        try:
                            self.top_p = float(cmd[1])
                            print(f"✅ Top-P: {self.top_p}")
                        except:
                            print("❌ Ungültiger Wert!")
                    
                    elif cmd[0] == '/stream':
                        self.stream = not self.stream
                        print(f"✅ Streaming: {'AN' if self.stream else 'AUS'}")
                    
                    else:
                        print("❌ Unbekannter Befehl. Tippe '/help' für Hilfe.")
                    
                    continue
                
                # Generiere Antwort
                print("\n🧠: ", end='')
                response = self.generate(user_input)
                if not self.stream:
                    print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Auf Wiedersehen!")
                break
            except Exception as e:
                print(f"\n❌ Fehler: {e}")
                continue


# ═══════════════════════════════════════════════════════════════════
# QUICK GENERATION (ohne Chat-Interface)
# ═══════════════════════════════════════════════════════════════════

def quick_generate(prompt, max_tokens=200, temperature=0.8):
    """Schnelle Generierung ohne Chat-Interface"""
    chat = ChatInterface()
    chat.stream = False
    return chat.generate(prompt, max_tokens=max_tokens, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='🧠 MemPID FUSION v2.5 Chat')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Pfad zum Modell')
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_PATH, help='Pfad zum Tokenizer')
    parser.add_argument('--prompt', type=str, default=None, help='Einzelner Prompt (kein Chat)')
    parser.add_argument('--tokens', type=int, default=DEFAULT_MAX_TOKENS, help='Max Tokens')
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE, help='Temperature')
    
    args = parser.parse_args()
    
    # Update paths
    MODEL_PATH = args.model
    TOKENIZER_PATH = args.tokenizer
    
    if args.prompt:
        # Einmal generieren
        result = quick_generate(args.prompt, max_tokens=args.tokens, temperature=args.temp)
        print(f"\n{result}")
    else:
        # Chat starten
        chat = ChatInterface(args.model, args.tokenizer)
        chat.max_tokens = args.tokens
        chat.temperature = args.temp
        chat.run()