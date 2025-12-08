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

> **A novel language model architecture using PID controllers instead of attention.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

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
| 🌊 **Dilated Convolutions** | Captures local and global patterns |
| 🛣️ **Highway Connections** | Smooth gradient flow |
| ⚡ **Efficient** | 28M params, 0.23 GB VRAM |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemPID_FUSION Block                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input                                                      │
│    ↓                                                        │
│  ┌─────────────┐                                            │
│  │  TokenShift │  ← Temporal mixing                         │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │   RMSNorm   │  ← Pre-normalization                       │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │    Causal Dilated Convolution       │                    │
│  │    (kernel=4, dilations=1,2,4,8)    │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │   SwiGLU    │  ← Activation                              │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │         PID Memory Gate             │                    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐            │                    │
│  │  │ Kp  │ │ Ki  │ │ Kd  │  ← Learnable                    │
│  │  └──┬──┘ └──┬──┘ └──┬──┘            │                    │
│  │     ↓       ↓       ↓               │                    │
│  │   P-Term  I-Term  D-Term            │                    │
│  │     └───────┴───────┘               │                    │
│  │              ↓                      │                    │
│  │        Gate Output                  │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  ┌─────────────────────────────────────┐                    │
│  │     Highway (Up → Down → Up)        │                    │
│  └──────┬──────────────────────────────┘                    │
│         ↓                                                   │
│  Output + Residual                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🎛️ The PID Controller

The core innovation: Each dimension has its own PID controller.

```python
# Simplified PID Gate
P_term = Kp * current_state      # Present (now)
I_term = Ki * integral_state     # Past (memory)  
D_term = Kd * (current - prev)   # Change (derivative)

output = sigmoid(P_term + I_term + D_term) * input
```

| Term | Function | What it learns |
|------|----------|----------------|
| **P** (Proportional) | React to current input | Immediate patterns |
| **I** (Integral) | Accumulate over time | Long-term context |
| **D** (Derivative) | Detect changes | Transitions, surprises |

---

## 📊 Model Specifications

```
┌─────────────────────────────────────────┐
│  MemPID_FUSION v2.5                     │
├─────────────────────────────────────────┤
│  Parameters:     28,703,936 (28.7M)     │
│  Dimensions:     512                    │
│  Layers:         16                     │
│  Vocab Size:     16,000 (BPE)           │
│  Context:        512 tokens             │
│  VRAM Usage:     ~0.23 GB               │
│  Precision:      bfloat16               │
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
python training_MemPID_FUSION_v2_5.py
```

### Chat / Inference

```bash
python chat_fusion_v2_5.py
```

---

## 📝 Meta-Tokens

The model uses special tokens to control output style:

| Token | Style |
|-------|-------|
| `<KLASSIKER>` | Classical German literature |
| `<PHILOSOPHIE>` | Philosophical writing |
| `<LYRIK>` | Poetry |
| `<WISSEN>` | Encyclopedia/Facts |
| `<DRAMA>` | Theater/Dialogue |

### Example Outputs

**Input:** `<KLASSIKER> Goethe, Faust:`
```
KAISER:
Wie es drüben ist mir's gut,
Der Witze nicht wieder zurück.
Es hebt sich an. Ich bin euch lieb.
```

**Input:** `<PHILOSOPHIE> Zarathustra`
```
Also sprachst du nach, dass du nicht mehr sehst, 
sondern darum, wie du willst; du schliebst dich in dir,
als ob du es besser ist als du, so stolz bist...
```

---

## 📈 Results

Training on German literature and Wikipedia:

| Metric | Value |
|--------|-------|
| Final Loss | 3.85 |
| Coherent sentences | ✅ |
| Style differentiation | ✅ |
| Factual accuracy | ❌ (limited by size) |

---

## 🔬 Why PID instead of Attention?

| Aspect | Attention | PID |
|--------|-----------|-----|
| Complexity | O(n²) | O(n) |
| Memory | High | Low |
| Long sequences | Expensive | Efficient |
| Interpretability | Black box | Control theory! |

The hypothesis: PID controllers can learn to regulate information flow similarly to attention, but with explicit temporal dynamics (P=present, I=past, D=change).

---

## 📁 Project Structure

```
MemPID_FUSION/
├── training_MemPID_FUSION_v2_5.py   # Training + Model
├── chat_fusion_v2_5.py              # Inference / Chat
├── requirements.txt                  # Dependencies
├── LICENSE
└── README.md
```

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Open issues
- Submit pull requests
- Share your experiments

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

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
- **Gemini Pro (Google)** - Analysis, data strategy suggestions

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
