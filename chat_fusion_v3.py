# chat_fusion_v3.py
# ═══════════════════════════════════════════════════════════════════
#  🧠 MemPID_FUSION v2 - Interactive Chat & Test
#
#  Author: Arthur-KI
#  License: MIT
#  GitHub: https://github.com/Arthur-KI/MemPID_FUSION
# ═══════════════════════════════════════════════════════════════════

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Import model class from training script
from training_MemPID_FUSION_v2 import MemPIDModel

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model_v2.pt'
TOKENIZER_PATH = 'tokenizer_v2.json'

print("═" * 60)
print("  🧠 MemPID_FUSION v2 - Chat & Test")
print("═" * 60)
print(f"\n📂 Lade Modell von {MODEL_PATH}...")

# Tokenizer laden
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# Checkpoint laden
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint['config']

print(f"   Config: DIM={config['dim']}, LAYERS={config['layers']}")
print(f"   Vocab: {config['vocab_size']}")
print(f"   Val Loss: {checkpoint['val_loss']:.4f}")

# Modell erstellen
model = MemPIDModel(
    vocab_size=config['vocab_size'],
    dim=config['dim'],
    layers=config['layers'],
    kernel_size=config['kernel'],
    batch_size=1
).to(DEVICE)

# State Dict bereinigen (Trainings-Buffer entfernen)
state_dict = checkpoint['model']
clean_state_dict = {}

for key, value in state_dict.items():
    # Überspringe Trainings-Buffer
    if "integ" in key or "prev" in key:
        continue
    if "memory_p" in key or "memory_i" in key or "step" in key:
        continue
    clean_state_dict[key] = value

model.load_state_dict(clean_state_dict, strict=False)
model.eval()

print(f"\n✅ Modell geladen!")
print("═" * 60)

# ═══════════════════════════════════════════════════════════════════
# GENERATION FUNKTION
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    prompt: str = "",
    category: str = None,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    show_progress: bool = True
):
    """
    Generiert Text basierend auf Prompt und/oder Kategorie.
    
    Args:
        prompt: Starttext (optional)
        category: Eine von <KLASSIKER>, <GESETZE>, <LYRIK>, <PHILOSOPHIE>, <WISSEN>
        max_tokens: Maximale Anzahl neuer Tokens
        temperature: Kreativität (0.1=konservativ, 1.0=kreativ)
        top_k: Nur die k wahrscheinlichsten Tokens
        top_p: Nucleus Sampling Schwelle
        show_progress: Zeige Fortschritt
    
    Returns:
        Generierter Text
    """
    model.eval()
    model.reset_states()
    
    # Starte mit Kategorie-Token und/oder Prompt
    tokens = []
    
    if category:
        cat_id = tokenizer.token_to_id(category)
        if cat_id is not None:
            tokens.append(cat_id)
        else:
            print(f"⚠️ Kategorie {category} nicht gefunden!")
    
    if prompt:
        encoded = tokenizer.encode(prompt).ids
        tokens.extend(encoded)
    
    if not tokens:
        # Fallback: Starte mit <KLASSIKER>
        tokens = [tokenizer.token_to_id("<KLASSIKER>")]
    
    idx = torch.tensor([tokens], device=DEVICE)
    
    # Generation Loop
    iterator = range(max_tokens)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Generiere", leave=False)
    
    for _ in iterator:
        model.reset_states()
        
        # Kontext begrenzen
        idx_cond = idx if idx.size(1) <= 2048 else idx[:, -2048:]
        
        # Forward
        logits, _ = model(idx_cond)
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
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop bei EOS
        eos_id = tokenizer.token_to_id("<EOS>")
        if eos_id and idx_next.item() == eos_id:
            break
    
    # Dekodieren
    text = tokenizer.decode(idx[0].tolist())
    return text


# ═══════════════════════════════════════════════════════════════════
# INTERAKTIVER CHAT MODUS
# ═══════════════════════════════════════════════════════════════════

def interactive_chat():
    """Interaktiver Chat-Modus"""
    
    print("\n" + "═" * 60)
    print("  💬 INTERAKTIVER MODUS")
    print("═" * 60)
    print("""
Befehle:
  /klassiker  - Generiere im Klassiker-Stil
  /gesetze    - Generiere im Gesetzes-Stil
  /lyrik      - Generiere im Lyrik-Stil
  /philosophie- Generiere im Philosophie-Stil
  /wissen     - Generiere im Wissens-Stil
  
  /temp X     - Setze Temperature (z.B. /temp 0.7)
  /tokens X   - Setze Max Tokens (z.B. /tokens 300)
  /lang       - Generiere langen Text (500 Tokens)
  
  /quit       - Beenden
  
Oder einfach Text eingeben als Prompt!
""")
    
    # Defaults
    temperature = 0.8
    max_tokens = 2000
    current_category = "<KLASSIKER>"
    
    while True:
        try:
            user_input = input("\n🎤 Du: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Auf Wiedersehen!")
            break
        
        if not user_input:
            continue
        
        # Befehle verarbeiten
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd == "/quit" or cmd == "/exit":
                print("\n👋 Auf Wiedersehen!")
                break
            
            elif cmd == "/klassiker":
                current_category = "<KLASSIKER>"
                print(f"📚 Kategorie: {current_category}")
                text = generate(category=current_category, max_tokens=max_tokens, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd == "/gesetze":
                current_category = "<GESETZE>"
                print(f"⚖️ Kategorie: {current_category}")
                text = generate(category=current_category, max_tokens=max_tokens, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd == "/lyrik":
                current_category = "<LYRIK>"
                print(f"🎭 Kategorie: {current_category}")
                text = generate(category=current_category, max_tokens=max_tokens, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd == "/philosophie":
                current_category = "<PHILOSOPHIE>"
                print(f"🤔 Kategorie: {current_category}")
                text = generate(category=current_category, max_tokens=max_tokens, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd == "/wissen":
                current_category = "<WISSEN>"
                print(f"📖 Kategorie: {current_category}")
                text = generate(category=current_category, max_tokens=max_tokens, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd.startswith("/temp "):
                try:
                    temperature = float(cmd.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"🌡️ Temperature: {temperature}")
                except:
                    print("❌ Ungültige Temperature! Beispiel: /temp 0.7")
            
            elif cmd.startswith("/tokens "):
                try:
                    max_tokens = int(cmd.split()[1])
                    max_tokens = max(10, min(1000, max_tokens))
                    print(f"📏 Max Tokens: {max_tokens}")
                except:
                    print("❌ Ungültige Anzahl! Beispiel: /tokens 300")
            
            elif cmd == "/lang":
                print(f"📜 Generiere langen Text (500 Tokens)...")
                text = generate(category=current_category, max_tokens=500, temperature=temperature)
                print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")
            
            elif cmd == "/help":
                print(__doc__)
            
            else:
                print(f"❌ Unbekannter Befehl: {cmd}")
        
        else:
            # User Input als Prompt verwenden
            print(f"\n⏳ Generiere Fortsetzung...")
            text = generate(
                prompt=user_input,
                category=current_category,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(f"\n🤖 Modell:\n{'-'*40}\n{text}\n{'-'*40}")


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST MODUS
# ═══════════════════════════════════════════════════════════════════

def quick_test():
    """Schneller Test aller Kategorien"""
    
    print("\n" + "═" * 60)
    print("  🧪 QUICK TEST - Alle Kategorien")
    print("═" * 60)
    
    categories = [
        ("<KLASSIKER>", "📚"),
        ("<GESETZE>", "⚖️"),
        ("<LYRIK>", "🎭"),
        ("<PHILOSOPHIE>", "🤔"),
        ("<WISSEN>", "📖"),
    ]
    
    for cat, emoji in categories:
        print(f"\n{emoji} {cat}:")
        print("-" * 50)
        
        text = generate(category=cat, max_tokens=150, temperature=0.8, show_progress=False)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 50)


# ═══════════════════════════════════════════════════════════════════
# LANGER KONTEXT TEST
# ═══════════════════════════════════════════════════════════════════

def long_context_test():
    """Test ob Modell Kontext über lange Texte hält"""
    
    print("\n" + "═" * 60)
    print("  📜 LANGER KONTEXT TEST (500 Tokens)")
    print("═" * 60)
    
    print("\nGeneriere langen Klassiker-Text...")
    text = generate(category="<KLASSIKER>", max_tokens=500, temperature=0.7)
    
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)
    
    # Einfache Analyse
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    print(f"\n📊 Statistik:")
    print(f"   Wörter: {len(words)}")
    print(f"   Sätze: ~{sentences}")
    print(f"   Ø Wörter/Satz: {len(words)/max(sentences,1):.1f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WÄHLE MODUS:")
    print("═" * 60)
    print("""
  1. 💬 Interaktiver Chat
  2. 🧪 Quick Test (alle Kategorien)
  3. 📜 Langer Kontext Test (500 Tokens)
  
  (oder 'q' zum Beenden)
""")
    
    while True:
        choice = input("Deine Wahl [1/2/3/q]: ").strip()
        
        if choice == '1':
            interactive_chat()
            break
        elif choice == '2':
            quick_test()
            break
        elif choice == '3':
            long_context_test()
            break
        elif choice.lower() == 'q':
            print("👋 Auf Wiedersehen!")
            break
        else:
            print("❌ Bitte 1, 2, 3 oder q eingeben!")
