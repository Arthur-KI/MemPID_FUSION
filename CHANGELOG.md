# Changelog

All notable changes to MemPID_FUSION will be documented in this file.

---

## [v2.0] - 2025

### 🆕 Added
- **Multi-Head Importance Pool**: 4 learned heads that determine token importance
  - Uses sigmoid (absolute importance) instead of softmax (relative)
  - O(n) complexity via cumsum trick
  - Each head specializes: subjects, verbs, negations, noise-filter
- **Adaptive Decay**: Content-aware forgetting in PID integral term
  - Learns when to remember vs. forget based on content
  - Replaces static decay factor
- **Larger Architecture**:
  - DIM: 512 → 1024
  - Context: 512 → 2048 tokens
  - Parameters: ~28M → ~128M

### 🔧 Changed
- Kernel size: 4 → 64
- Max dilation: 8 → 32
- Layers: 16 total → 6 per stack (Up-Down-Up = 18 effective)
- PID initialization refined

### 📈 Improved
- Coherent generation length: ~200 → 300-500 tokens
- Better long-range context retention
- More stable generation over long sequences

### 📊 Metrics
- Val Loss: ~4.03 (similar to v1, but better coherence)

---

## [v1.0] - 2025

### Initial Release
- Basic PID-based language model architecture
- Replaces attention with PID controllers (O(n) vs O(n²))
- Components:
  - PID Memory Gate (Kp, Ki, Kd learnable)
  - Dilated Causal Convolutions
  - SwiGLU activation
  - Highway connections
- 28M parameters
- 512 context length
- Trained on German literature + Wikipedia

### 📊 Metrics
- Val Loss: 3.85
- Coherent generation: ~200 tokens

---

## Future Plans

- [ ] **v3**: English training data for broader accessibility
- [ ] **500M version**: Scaling experiment
- [ ] Benchmarks against GPT-2 (124M, 355M)
- [ ] Community feedback integration
