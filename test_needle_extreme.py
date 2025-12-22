import torch
import torch.nn as nn
import torch.optim as optim
import sys
import gc

# Importiere dein Modell
try:
    from training_MemPID_FUSION_v3 import MemPIDModel
except ImportError:
    print("❌ Fehler: 'training_MemPID_FUSION_v3.py' nicht gefunden!")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# 🧪 THE NEEDLE IN THE HAYSTACK - EXTREME EDITION (FIXED)
# ═══════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config für den Test
TEST_DIM = 256          
TEST_LAYERS = 4         
TEST_VOCAB = 1000       
NEEDLE_ID = 42          
TRIGGER_ID = 999        

def generate_haystack(length):
    data = torch.randint(0, TEST_VOCAB-2, (length,)).tolist()
    data[5] = NEEDLE_ID         # Nadel am Anfang
    data[-1] = TRIGGER_ID       # Trigger am Ende
    
    x = torch.tensor([data], dtype=torch.long).to(DEVICE)
    y = torch.tensor([NEEDLE_ID], dtype=torch.long).to(DEVICE)
    return x, y

def run_test():
    print(f"\n🚀 Starte MemPID v3 'EXTREME' Test auf {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("="*60)
    
    # Die "Extreme" Stufen
    lengths = [16384, 32768, 65536, 131072, 262144, 524288, 1048576] 
    
    for seq_len in lengths:
        # Speicherbereinigung vor jeder Runde
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n🔥 Teste Kontext-Länge: {seq_len:,} Tokens...")
        
        try:
            # FIX: Scaler HIER erstellen, damit er für jede Runde frisch ist
            scaler = torch.amp.GradScaler('cuda')
            
            # Modell init
            model = MemPIDModel(
                vocab_size=TEST_VOCAB, 
                dim=TEST_DIM, 
                layers=TEST_LAYERS, 
                batch_size=1
            ).to(DEVICE)
            
            # Positional Embeddings erweitern
            if model.pos.shape[1] < seq_len:
                new_pos = torch.randn(1, seq_len, TEST_DIM).to(DEVICE) * 0.02
                model.pos = nn.Parameter(new_pos)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            success = False
            steps = 200 
            
            for step in range(steps):
                optimizer.zero_grad()
                
                x, target = generate_haystack(seq_len)
                
                # Mixed Precision Context
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits, _ = model(x) 
                    last_token_logits = logits[0, -1, :]
                    loss = criterion(last_token_logits.unsqueeze(0), target)
                
                # Scaled Backward Pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                pred = torch.argmax(last_token_logits).item()
                
                if step % 10 == 0:
                    stats = model.up_stack.get_importance_stats()
                    print(f"   Step {step:03d} | Loss: {loss.item():.4f} | Pred: {pred:03d} | Mix: {stats.get('mix', 0):.3f}")
                
                if pred == NEEDLE_ID and loss.item() < 0.1:
                    print(f"✅ SUCCESS bei {seq_len:,}! Gefunden nach {step} Steps.")
                    success = True
                    break
            
            if not success:
                print(f"❌ FAILED bei {seq_len:,}. (Lernen nicht konvergiert)")

            # Aufräumen
            del model
            del optimizer
            del scaler # Jetzt ist das sicher, weil er oben neu erstellt wird
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"\n💀 OUT OF MEMORY bei {seq_len:,} Tokens!")
            print(f"   Das Limit deiner 4080 ist erreicht. Aber bis {lengths[lengths.index(seq_len)-1]:,} lief es!")
            torch.cuda.empty_cache()
            break 
        except Exception as e:
            print(f"\n⚠️ Unerwarteter Fehler: {e}")
            break

if __name__ == "__main__":
    run_test()