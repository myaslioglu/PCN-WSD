#!/usr/bin/env python3
"""Hızlı residual ablation testi — embedding'siz, sadece mimari karşılaştırması."""
import torch, torch.nn.functional as F, numpy as np, time
from collections import defaultdict

torch.manual_seed(42); np.random.seed(42)
DIM = 64
N_SAMPLES = 90

# ===== Basit veri seti (rastgele embedding) =====
VOCAB = {}
for sense_name in [f"{w}_{s}" for w in ['yuz','cay','bas','dil','at'] 
                   for s in ['a','b','c']]:
    for word in [f"{sense_name}_{i}" for i in range(6)]:
        VOCAB[word] = torch.randn(DIM) * 0.1

def make_samples(n=90):
    samples = []
    for sense in [f"{w}_{s}" for w in ['yuz','cay','bas','dil','at'] for s in ['a','b','c']]:
        for _ in range(n // 15):
            words = [f"{sense}_{i}" for i in range(4)]
            vec = torch.stack([VOCAB[w] for w in words]).mean(dim=0)
            samples.append((vec, sense))
    return samples

# ===== Residual PCN Layer =====
class ResLayer:
    def __init__(self, dim, use_residual=True, alpha=0.3):
        self.W_fwd = torch.randn(dim, dim) * 0.1
        self.W_bwd = torch.randn(dim, dim) * 0.1
        self.mu = torch.zeros(dim, 1)
        self.prev_mu = torch.zeros(dim, 1)
        self.use_residual = use_residual
        self.alpha = alpha
        self.error_ema = 1.0
    
    def predict(self): return torch.tanh(self.W_bwd @ self.mu)
    
    def step(self, x, pred_above=None, dt=0.1):
        if x.dim() == 1: x = x.unsqueeze(1)
        tgt = x + pred_above if pred_above is not None else x
        err = tgt - self.predict()
        em = err.norm().item()
        self.error_ema = 0.9*self.error_ema + 0.1*em
        prec = max(0.1, min(10.0, 1.0/(self.error_ema+1e-6)))
        self.prev_mu = self.mu.clone()
        grad = self.W_fwd @ err
        if self.use_residual:
            self.mu = self.mu + 0.1*(prec*grad - self.mu) + self.alpha*(self.prev_mu - self.mu)
        else:
            self.mu = self.mu + 0.1*(prec*grad - self.mu)
        return em
    
    def learn(self, x, pred_above=None, lr=0.02):
        if x.dim() == 1: x = x.unsqueeze(1)
        tgt = x + pred_above if pred_above is not None else x
        err = tgt - self.predict()
        self.W_fwd += lr * err @ self.mu.T
        self.W_bwd += lr * self.mu @ err.T
    
    def reset(self):
        self.mu = torch.zeros(DIM, 1)
        self.prev_mu = torch.zeros(DIM, 1)
        self.error_ema = 1.0

class ResPCN:
    def __init__(self, n_layers=5, use_residual=True, alpha=0.3):
        self.layers = [ResLayer(DIM, use_residual, alpha) for _ in range(n_layers)]
        self.n = n_layers
    
    def infer(self, x, n_iter=25, dt=0.1):
        total = 0.0
        for _ in range(n_iter):
            preds = [l.predict() for l in self.layers]
            e0 = self.layers[0].step(x, preds[0] if self.n>1 else None, dt)
            total += e0
            for i in range(1, self.n-1):
                total += self.layers[i].step(self.layers[i-1].mu.squeeze(), preds[i], dt)
            if self.n > 1:
                total += self.layers[-1].step(self.layers[-2].mu.squeeze(), None, dt)
        return total
    
    def learn(self, x, lr=0.02):
        self.infer(x, n_iter=20)
        preds = [l.predict() for l in self.layers]
        self.layers[0].learn(x, preds[0] if self.n>1 else None, lr)
        for i in range(1, self.n-1):
            self.layers[i].learn(self.layers[i-1].mu.squeeze(), preds[i], lr)
        if self.n > 1:
            self.layers[-1].learn(self.layers[-2].mu.squeeze(), None, lr)
    
    def global_state(self): return self.layers[-1].mu.squeeze()
    def reset(self):
        for l in self.layers: l.reset()

def eval_model(model, embs, samples):
    correct = 0
    for x, true_s in samples:
        model.reset(); model.infer(x, 30)
        mu = model.global_state()
        best_s, best_sim = None, -1
        for sn, el in embs.items():
            avg = torch.stack(el[-6:]).mean(0)
            sim = F.cosine_similarity(mu.unsqueeze(0), avg.unsqueeze(0)).item()
            if sim > best_sim: best_sim, best_s = sim, sn
        if best_s == true_s: correct += 1
    return correct/len(samples)*100

# ===== MAIN =====
samples = make_samples(N_SAMPLES)
print(f"🧪 Residual Ablation Test | {N_SAMPLES} örnek, {DIM}D rastgele embedding\n")

configs = [
    ("3-layer (no residual)", 3, False),
    ("3-layer + Residual",    3, True),
    ("5-layer (no residual)", 5, False),
    ("5-layer + Residual",    5, True),
    ("7-layer (no residual)", 7, False),
    ("7-layer + Residual",    7, True),
]

results = {}
for name, n_layers, use_res in configs:
    model = ResPCN(n_layers, use_res)
    embs = defaultdict(list)
    t0 = time.time()
    
    for ep in range(30):
        np.random.shuffle(samples)
        for x, s in samples:
            model.reset(); model.infer(x, 25); model.learn(x, 0.02)
            embs[s].append(model.global_state())
    
    acc = eval_model(model, embs, samples)
    elapsed = time.time() - t0
    results[name] = acc
    print(f"  {name:<25}  {acc:5.1f}%  ({elapsed:.1f}s)")

print(f"\n{'═' * 50}")
print(f"📊 RESIDUAL ETKİSİ:")
print(f"  3-layer:  {results.get('3-layer (no residual)',0):.1f}% → {results.get('3-layer + Residual',0):.1f}%")
print(f"  5-layer:  {results.get('5-layer (no residual)',0):.1f}% → {results.get('5-layer + Residual',0):.1f}%")
print(f"  7-layer:  {results.get('7-layer (no residual)',0):.1f}% → {results.get('7-layer + Residual',0):.1f}%")
