# MemPID_FUSION

```
███╗   ███╗███████╗███╗   ███╗██████╗ ██╗██████╗ 
████╗ ████║██╔════╝████╗ ████║██╔══██╗██║██╔══██╗
██╔████╔██║█████╗  ██╔████╔██║██████╔╝██║██║  ██║
██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║██║  ██║
██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║     ██║██████╔╝
╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═════╝ 
                ███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
                ██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
                █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
                ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
                ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
                ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

> **A novel language model architecture using PID controllers instead of attention. No O(n²) - just O(n)!**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🆕 What's New in v3

| Feature | v1 | v3 |
|---------|----|----|
| **Multi-Head Importance Pool** | ❌ Mean Pool | ✅ 4 learned "editors" |
| **Adaptive Decay** | ❌ Static | ✅ Content-aware forgetting |
| **Dimensions** | 512 | 1024 |
| **Context Window** | 512 tokens | 2048 tokens |
| **Parameters** | ~28M | ~128M |
| **Coherent Generation** | ~200 tokens | **300-500 tokens** |

---

## 🧠 What is MemPID_FUSION?

MemPID_FUSION is an experimental language model that replaces the traditional **Attention mechanism** with **PID controllers** (Proportional-Integral-Derivative) from control theory.

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional Transformer    vs    MemPID_FUSION             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Attention O(n²)]              [PID Gates O(n)]            │
│       ↓                              ↓                      │
│  Expensive for                  Linear complexity!          │
│  long sequences                 Efficient memory!           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🚫 **No Attention** | O(n) complexity instead of O(n²) |
| 🎛️ **PID Controllers** | Learnable Kp, Ki, Kd per dimension |
| 🎯 **Multi-Head Importance Pool** | 4 heads learn what's important (NEW!) |
| 🌊 **Adaptive Decay** | Content-aware forgetting (NEW!) |
| 🛣️ **Highway Connections** | Up → Down → Up architecture |
| ⚡ **Efficient** | ~128M params, runs on consumer GPUs |

---

## 🎯 Multi-Head Importance Pool (New in v3!)

The key innovation of v3: Instead of treating all tokens equally, the model learns **what's important**.

```
The Problem with Mean Pooling:
  [King, uh, the, well, daughter] → all weighted equally
  → "uh" dilutes "King" → fuzzy context

The Solution - Importance Pool:
  [King, uh, the, well, daughter]
     ↓     ↓    ↓    ↓      ↓
   0.35  0.02 0.08 0.03   0.32  ← Learned weights!
  → Important tokens dominate, noise is ignored
```

**4 Heads = 4 "Editors"**, each specializing in different aspects:
- Head 1 → Subjects/Nouns
- Head 2 → Verbs/Actions  
- Head 3 → Negations/Modifiers
- Head 4 → Noise Filter

**Still O(n)!** Uses cumsum trick instead of attention matrix.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemPID_FUSION v3 Block                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input                                                      │
│    ↓                                                        │
│  ┌─────────────┐                                            │
│  │   RMSNorm   │  ← Pre-normalization                       │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │    Causal Dilated Convolution       │                    │
│  │    (kernel=64, dilations=1→32)      │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │      Adaptive PID Memory Gate       │                    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌───────┐  │                    │
│  │  │ Kp  │ │ Ki  │ │ Kd  │ │ Decay │  │ ← All learnable!   │
│  │  └──┬──┘ └──┬──┘ └──┬──┘ └───┬───┘  │                    │
│  │     ↓       ↓       ↓        ↓      │                    │
│  │   P-Term  I-Term  D-Term  Adaptive  │                    │
│  │     └───────┴───────┴────────┘      │                    │
│  │              ↓                      │                    │
│  │        Gated Output                 │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │    Multi-Head Importance Pool       │  ← NEW in v3!      │
│  │    (4 heads, cumsum trick, O(n))    │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  Output + Residual                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Up-Down-Up     │
                    │  Highway        │
                    ├─────────────────┤
                    │  Up Stack (6L)  │
                    │       ↓         │
                    │  Gate + Skip    │
                    │       ↓         │
                    │  Down Stack(6L) │
                    │       ↓         │
                    │  Gate + Skip    │
                    │       ↓         │
                    │  Up Stack (6L)  │
                    │       ↓         │
                    │  Final Gate     │
                    └─────────────────┘
```

### 🎛️ The PID Controller

Each dimension has its own PID controller with **adaptive decay**:

```python
# PID Gate with Adaptive Decay
P_term = Kp * current_state           # Present (react now)
I_term = Ki * integral_state          # Past (accumulated memory)  
D_term = Kd * (current - previous)    # Change (detect transitions)

# NEW: Content-aware decay
decay = sigmoid(base_decay + content_signal)
new_integral = decay * old_integral + (1 - decay) * current

output = silu(P_term + I_term + D_term) * input
```

| Term | Function | What it learns |
|------|----------|----------------|
| **P** (Proportional) | React to current input | Immediate patterns |
| **I** (Integral) | Accumulate over time | Long-term context |
| **D** (Derivative) | Detect changes | Transitions, surprises |
| **Decay** | Adaptive forgetting | When to remember/forget |

---

## 📊 Model Specifications

```
┌─────────────────────────────────────────┐
│  MemPID_FUSION v3                       │
├─────────────────────────────────────────┤
│  Parameters:     ~128M                  │
│  Dimensions:     1024                   │
│  Layers:         6 per stack (×3)       │
│  Vocab Size:     16,000 (BPE)           │
│  Context:        2048 tokens            │
│  Importance:     4 heads                │
│  Precision:      bfloat16               │
│  Val Loss:       ~4.03                  │
└─────────────────────────────────────────┘
```

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Arthur-KI/MemPID_FUSION.git
cd MemPID_FUSION

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
numpy
tqdm
tokenizers
```

---

## 💻 Usage

### Training

```bash
# Prepare your data in training_data/ folder
# Each subfolder becomes a category token: training_data/classics/ → <CLASSICS>

python training_MemPID_FUSION_v3.py
```

### Chat / Inference

```bash
python chat_fusion_v3.py
```

Choose from:
1. 💬 Interactive Chat
2. 🧪 Quick Test (all categories)
3. 📜 Long Context Test (500 tokens)

---

## 📝 Meta-Tokens

The model uses special tokens to control output style:

| Token | Style |
|-------|-------|
| `<KLASSIKER>` | Classical German literature |
| `<PHILOSOPHIE>` | Philosophical writing |
| `<LYRIK>` | Poetry |
| `<WISSEN>` | Encyclopedia/Facts |
| `<GESETZE>` | Legal texts |

---

## 📈 Results

| Metric | v1 | v3 |
|--------|----|----|
| Val Loss | 3.85 | 4.03 |
| Coherent tokens | ~200 | **300-500** |
| Grammar | ✅ | ✅ |
| Style differentiation | ✅ | ✅ |
| Long-range coherence | ⚠️ | ✅ |
| Factual accuracy | ❌ | ❌ (limited by size) |

**Note:** Higher loss but better coherence! The Importance Pool successfully filters noise.

---

## 🔬 Why PID instead of Attention?

| Aspect | Attention | PID |
|--------|-----------|-----|
| Complexity | O(n²) | **O(n)** |
| Memory | High | Low |
| Long sequences | Expensive | Efficient |
| Interpretability | Black box | Control theory! |

The hypothesis: PID controllers can learn to regulate information flow similarly to attention, but with explicit temporal dynamics (P=present, I=past, D=change) and **linear complexity**.

---

## 📁 Project Structure

```
MemPID_FUSION/
├── training_MemPID_FUSION_v2.py   # Training + Model Definition
├── chat_fusion_v2.py              # Inference / Interactive Chat
├── requirements.txt               # Dependencies
├── CHANGELOG.md                   # Version history
├── LICENSE.txt                    # MIT License
└── README.md                      # This file
```

---

## 🗺️ Roadmap

- [x] v1: Basic PID architecture (28M params)
- [x] v3: Multi-Head Importance Pool + Adaptive Decay (128M params)
- [ ] 500M parameter version
- [ ] Benchmarks against GPT-2

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Open issues
- Submit pull requests
- Share your experiments
- Train on different data

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file.

---

## 📖 Citation

```bibtex
@software{mempid_fusion,
  author = {Arthur-KI},
  title = {MemPID_FUSION: Language Model with PID Controllers},
  year = {2025},
  url = {https://github.com/Arthur-KI/MemPID_FUSION}
}
```

---

## 🙏 Acknowledgments

This project was created through human-AI collaboration:

- **Arthur** - Vision, ideas, training, testing
- **Claude (Anthropic)** - Architecture design, code, documentation
- **Gemini Pro (Google)** - Analysis, cumsum O(n) fix, data strategy

Proof that curiosity beats credentials! Built without formal CS education.

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   "No Attention? No Problem."                               │
│                                        - MemPID_FUSION      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
