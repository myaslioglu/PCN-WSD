# PCN-WSD: Predictive Coding Network for Word Sense Disambiguation

**7,700 parameters · Hebbian learning · No backpropagation · 100% held-out accuracy**

A biologically-plausible hierarchical neural network that achieves perfect generalization on Turkish word sense disambiguation using only local synaptic learning rules — 16,000× smaller than GPT-2 Small.

## Quick Summary

| | k-NN (64D) | PCN-WSD (16D) |
|---|---|---|
| **Clean** | 100% | 92% |
| **σ=0.7 noise** | 35% | **54% 🧠** |
| **Hold-out** | 100% | **100%** |
| **Parameters** | — | **7,680** |

**Key finding:** At moderate noise (σ=0.7), PCN overtakes k-NN by 19 points. The hierarchical inference dynamics filter noise by reconstructing signals from learned priors — exactly as predictive coding theory predicts.

## Architecture

```
Input (64D PCA) → Layer 0 (64→32) → Layer 1 (32→16) → Output (16D)
```

Each layer has 3 weight matrices updated via local Hebbian rules:
- **W_fwd**: forward projection (input → hidden state)
- **W_bwd**: backward prediction (hidden state → input reconstruction)  
- **W_down**: downward prediction to the layer below

**Inference**: 20-30 iterative cycles. Weights are frozen, only internal states (`μ`) update: `dμ/dt = W_fwd · ε − μ`

**Learning**: `ΔW = clamp(η · μ · ε^T, ±0.05)` — purely local, no backprop.

## Development Iterations

| v | Architecture | Embedding | Accuracy | Key Insight |
|---|---|---|---|---|
| v1 | 3L: 32→16→8→4 | 32D random | 84.6% | Feasibility proof |
| v2 | 3-5L same dim | 384D MiniLM | 8.3% | Weight explosion (NaN) |
| v3 | 5L fixed 64D | 64D PCA | 41.3% | Gradient clipping fixes NaN |
| **Hybrid** | **3L: 64→32→16** | **64D PCA** | **97.1%** | **Dimensional reduction + clipping** |

## Key Results

- **100% hold-out accuracy** on 26 unseen test samples (trained on 104, 13 senses)
- **Noise crossover at σ=0.7**: kNN collapses (35%), PCN maintains (54%)
- **Clean > Noisy training**: PCN learns best from clean templates, filters noise at inference time
- **Gradient clipping** (γ=10) + weight norm clipping (ω=3) essential for stability

## Files

| File | Description |
|---|---|
| `pcn_wsd.py` | v1 prototype — random embeddings, first success |
| `pcn_benchmark.py` | v2 extended benchmark |
| `pcn_v3_clipped.py` | v3 clipped PCN with residual ablation |
| `pcn_v3_residual.py` | v3 residual + precision experiments |
| `residual_ablation.py` | Residual connection ablation study |
| `knn_vs_pcn_noise.py` | k-NN vs PCN noise robustness comparison |

## Why This Matters

1. **Energy efficiency**: 7.7K params vs GPT-3's 175B. The brain runs on 20W.
2. **Biologically plausible**: Local Hebbian updates, no weight transport problem.
3. **Dynamic inference**: Iterative state updates — the model "thinks" about ambiguous input.
4. **Noise filtering**: Hierarchical predictions reconstruct corrupted signals — like the brain recognizing degraded stimuli.

## Theoretical Foundations

- **Predictive Coding** (Rao & Ballard, 1999; Friston, 2005): Brain as hierarchical prediction machine
- **Hebbian Learning** (Hebb, 1949): "Neurons that fire together wire together"
- **Tolman** (1932): Behavior as function of multiple interacting variables
- **Spearman** (1923): Noegenetic principles — experience, eduction of relations, eduction of correlates

## Dependencies

```
pip install torch sentence-transformers scikit-learn numpy
```

## Citation

Yaşlıoğlu, M. (2026). Hierarchical Predictive Coding Networks for Word Sense Disambiguation: A Biologically-Plausible Alternative to Backpropagation-Based NLP. Technical Report, İstanbul University.

## License

MIT
