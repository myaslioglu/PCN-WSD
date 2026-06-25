# PCN-WSD: Predictive Coding Network for Word Sense Disambiguation

**Hebbian learning · No backpropagation · Biologically-plausible WSD**

A biologically-plausible hierarchical neural network for Turkish word sense disambiguation using only local synaptic learning rules. Demonstrates that predictive coding dynamics can disambiguate word meanings from context without gradient-based optimization.

## Quick Summary

| | k-NN | PCN-WSD |
|---|---|---|
| **Clean (MiniLM 384D)** | 90-100% | 60-85% |
| **σ=0.7 noise** | 35% | **54% 🧠** |
| **Parameters (v1, 32D)** | — | **~6K** |

**Key finding:** At moderate noise (σ=0.7), PCN overtakes k-NN by 19 points. The hierarchical inference dynamics filter noise by reconstructing signals from learned priors — exactly as predictive coding theory predicts. PCN shines when embedding quality degrades; with strong embeddings, simple k-NN suffices.

## Architecture

All layers operate at the same dimensionality (fixed-dim design). Hierarchy emerges from information flow direction (sensory → local → global), not from dimensional compression:

```
Input (D) → Layer 0 (D→D) → Layer 1 (D→D) → ... → Layer N (D→D)
```

Each layer has 2 weight matrices updated via local Hebbian rules:
- **W_fwd**: forward projection (input → hidden state)
- **W_bwd**: backward prediction (hidden state → input reconstruction)

**Inference**: 20-30 iterative cycles. Weights are frozen, only internal states (`μ`) update: `dμ/dt = W_fwd · ε − μ`

**Learning**: `ΔW = clamp(η · μ · ε^T, ±0.05)` — purely local, no backprop.

## Development Iterations

| v | Architecture | Embedding | Accuracy | Key Insight |
|---|---|---|---|---|
| v1 | 3L fixed 32D | 32D random | 84.6% | Feasibility proof |
| v2 | 4L fixed 64D | 64D random | ~81% | Precision-weighting, k-fold CV |
| v3 | 5L fixed 64D | 64D synthetic | 41-60% | Residual connections + gradient clipping fix NaN |
| v4 | 5L fixed 384D | 384D MiniLM | 60-85% | Real multilingual embeddings + EMA precision |
| Noise | 5L fixed 384D | 384D MiniLM | — | k-NN vs PCN noise robustness (crossover at σ=0.7) |

> **Note:** A hybrid 64→32→16 dimensional-reduction variant (`pcn_hybrid.py`) with W_down was prototyped but underperformed (~10% hold-out), confirming that fixed-dim hierarchy with iterative inference is more stable for this task scale.

## Key Results

- **5-fold cross-validation** with 4-layer PCN shows consistent generalization
- **Noise crossover at σ=0.7**: kNN collapses (35%), PCN maintains (54%)
- **Clean > Noisy training**: PCN learns best from clean templates, filters noise at inference time
- **Gradient clipping** (γ=10) + weight norm clipping (ω=3) essential for stability
- **Residual connections** improve deep model stability (3L→5L→7L)

## Files

| File | Description |
|---|---|
| `pcn_wsd.py` | v1 prototype — random embeddings, first success (84.6%) |
| `pcn_benchmark.py` | v2 extended benchmark with k-fold CV |
| `pcn_v3_clipped.py` | v3 clipped PCN with residual ablation |
| `pcn_v3_residual.py` | v4: 384D MiniLM + residual + EMA precision |
| `residual_ablation.py` | Residual connection ablation study |
| `knn_vs_pcn_noise.py` | k-NN vs PCN noise robustness comparison |
| `pcn_hybrid.py` | 🧪 Experimental: 64→32→16 dim-reduction + W_down + PCA + hold-out (underperforms) |

## Why This Matters

1. **Energy efficiency**: ~6K-1.5M params vs GPT-3's 175B. The brain runs on 20W.
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
