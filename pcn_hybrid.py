#!/usr/bin/env python3
"""
🧠 PCN-WSD Hybrid: 64→32→16 Boyut Azaltan + W_down + PCA + Hold-out
======================================================================
README'de iddia edilen hibrit mimarinin gerçek implementasyonu.

Mimari:
  Input (64D PCA) → Layer 0 (64→32) → Layer 1 (32→16) → Layer 2 (16D output)

Her katmanda 3 ağırlık matrisi:
  - W_fwd: ileri projeksiyon (input → hidden state)
  - W_bwd: geri tahmin (hidden state → input rekonstrüksiyonu)
  - W_down: bir üst katmandan bu katmana aşağı yönlü tahmin

Öğrenme: Hebbian (ΔW = η · μ · εᵀ), tamamen lokal
Inference: 20-30 iteratif döngü, sadece μ güncellenir
Veri: sentence-transformers (MiniLM 384D) → PCA 64D
Test: Hold-out (80/20 stratified split)

Hacı 🧿 — 25 Haziran 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sentence_transformers import SentenceTransformer
import time
import sys

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 0. VERİ SETİ: Doğal cümleler (pcn_v3_residual.py'dan)
# ============================================================

TRAINING_SAMPLES = [
    # === YÜZ (sayı, anatomi, yüzme) ===
    ("yüz elli iki milyon lira bütçe ayrıldı", "yuz_sayi"),
    ("bu projeye tam yüz bin dolar harcadık", "yuz_sayi"),
    ("sınavdan yüz üzerinden doksan beş aldı", "yuz_sayi"),
    ("toplam yüz kişilik bir ekip çalıştı", "yuz_sayi"),
    ("yüzde yüz başarı oranı yakaladık", "yuz_sayi"),
    ("hesapta tam yüz lira kalmış", "yuz_sayi"),
    ("yüz binlerce insan meydanda toplandı", "yuz_sayi"),
    ("maaşına yüzde otuz zam yapıldı", "yuz_sayi"),

    ("yüzündeki gülümseme her şeye değerdi", "yuz_anatomi"),
    ("gözleri ve yüz hatları annesine benziyor", "yuz_anatomi"),
    ("yüzünü yıkamak için lavaboya gitti", "yuz_anatomi"),
    ("yüz ifadesinden ne düşündüğü belli olmuyordu", "yuz_anatomi"),
    ("soğuktan yüzü kızarmıştı", "yuz_anatomi"),
    ("yüz estetiği ameliyatı olmayı düşünüyor", "yuz_anatomi"),
    ("yüzünde kocaman bir sivilce çıkmıştı", "yuz_anatomi"),
    ("yüz hatları keskin ve belirgindi", "yuz_anatomi"),

    ("yazın her gün denizde yüzüyoruz", "yuz_yuzme"),
    ("havuzda yüzmek serinlemek için birebir", "yuz_yuzme"),
    ("kulaç atarak yüz metre yüzdü", "yuz_yuzme"),
    ("dalgalı denizde yüzmek tehlikeli", "yuz_yuzme"),
    ("yüzme bilmeyenler sığ suda kalsın", "yuz_yuzme"),
    ("profesyonel yüzücü gibi yüzüyor", "yuz_yuzme"),
    ("sabah erkenden kalkıp denize yüzmeye gitti", "yuz_yuzme"),
    ("sırt üstü yüzmek en rahatıdır", "yuz_yuzme"),

    # === ÇAY (içecek, dere) ===
    ("demli bir çay içmeden güne başlayamam", "cay_icecek"),
    ("çay bardağında ince belli olanı tercih ederim", "cay_icecek"),
    ("sıcak çay eşliğinde sohbet çok keyifliydi", "cay_icecek"),
    ("çay demlemek için suyu kaynattı", "cay_icecek"),
    ("şekerli çay içmeyi pek sevmez", "cay_icecek"),
    ("kahvaltıda çayın yanında simit yedik", "cay_icecek"),
    ("çay ocağından taze çay istedi", "cay_icecek"),
    ("kaçak çay Osmanlı'dan beri sevilir", "cay_icecek"),

    ("köprüden geçen çayın kenarında piknik yaptık", "cay_dere"),
    ("bu çay yazın kurur kışın coşar", "cay_dere"),
    ("çay boyunca yürüyüş yapmak çok huzurlu", "cay_dere"),
    ("taş köprünün altından akan çay şırıl şırıldı", "cay_dere"),
    ("çay kenarındaki çınar ağacının gölgesi harika", "cay_dere"),
    ("dağdan gelen çay vadiyi besliyor", "cay_dere"),
    ("çayın şırıltısı kuş seslerine karışıyordu", "cay_dere"),
    ("çay yatağı taşlarla dolmuştu", "cay_dere"),

    # === BAŞ (anatomi, lider, başlangıç) ===
    ("baş ağrısı yüzünden bütün gün yataktan çıkamadı", "bas_anatomi"),
    ("kafasını vurunca başı çok acıdı", "bas_anatomi"),
    ("başına şapka takmadan dışarı çıkma", "bas_anatomi"),
    ("beynini kullanmayıp başını taşlara vurdu", "bas_anatomi"),
    ("baş dönmesi şikayetiyle doktora gitti", "bas_anatomi"),
    ("omuzdan yukarısı baş bölgesidir", "bas_anatomi"),
    ("başının belaya gireceğini hissetmişti", "bas_anatomi"),
    ("baş parmağını çekiçle ezdi", "bas_anatomi"),

    ("şirketin başına yeni bir müdür atandı", "bas_lider"),
    ("ekibin başında deneyimli bir şef var", "bas_lider"),
    ("ordunun başkomutanı ziyarete geldi", "bas_lider"),
    ("projenin baş sorumlusu kim acaba", "bas_lider"),
    ("takımın başındaki isim istifa etti", "bas_lider"),
    ("bölüm başkanı toplantıya katılamadı", "bas_lider"),
    ("başbakan yeni kararnameyi imzaladı", "bas_lider"),
    ("şirkette baş mühendis olarak çalışıyor", "bas_lider"),

    ("başlangıçta her şey çok zor görünüyordu", "bas_baslangic"),
    ("her işin başı sağlık derler", "bas_baslangic"),
    ("sıfırdan başlayıp zirveye çıktı", "bas_baslangic"),
    ("yolun başında olduğumuzu unutmayalım", "bas_baslangic"),
    ("kitabın baş kısmında önsöz var", "bas_baslangic"),
    ("baştan sona kadar dikkatle dinledi", "bas_baslangic"),
    ("başlangıç seviyesi İngilizce kursuna yazıldı", "bas_baslangic"),
    ("her şeyin bir başı ve sonu vardır", "bas_baslangic"),

    # === DİL (organ, lisan, coğrafya) ===
    ("dilini çaydanlığa değdirince yandı", "dil_organ"),
    ("yemek çok acıydı dili damağı şişti", "dil_organ"),
    ("dil sağlığı için ağız bakımı şart", "dil_organ"),
    ("dilinin altındaki yara canını sıkıyordu", "dil_organ"),
    ("tat alma tomurcukları dilin üzerindedir", "dil_organ"),
    ("dilini çıkarıp doktora gösterdi", "dil_organ"),
    ("dili damağı kurumuş su istiyordu", "dil_organ"),
    ("dil piercingi yaptırmak istiyor", "dil_organ"),

    ("ana dili Türkçe olanlar için gramer kolaydır", "dil_lisan"),
    ("yabancı dil öğrenmek beyni geliştirir", "dil_lisan"),
    ("Türk dil kurumu yeni kelimeler türetiyor", "dil_lisan"),
    ("dil bilgisi kurallarına dikkat etmelisin", "dil_lisan"),
    ("ikinci dil olarak İngilizce şart oldu", "dil_lisan"),
    ("programlama dilleri arasında Python en kolayı", "dil_lisan"),
    ("dil bariyeri uluslararası işlerde sorun oluyor", "dil_lisan"),
    ("ölü diller arasında Latince en bilinenidir", "dil_lisan"),

    ("yarımadanın ucunda ince bir dil oluşmuş", "dil_cografya"),
    ("denize doğru uzanan kara diline burun denir", "dil_cografya"),
    ("coğrafyada dil şeklindeki çıkıntılara dikkat et", "dil_cografya"),
    ("körfezin ağzındaki dil balıkçıları koruyor", "dil_cografya"),
    ("kıyıdaki kum dili zamanla şekil değiştirir", "dil_cografya"),
    ("ada ile anakara arasında dar bir dil var", "dil_cografya"),
    ("buzul dili vadi boyunca uzanıyordu", "dil_cografya"),
    ("haritada ince bir dil gibi görünüyor", "dil_cografya"),

    # === AT (hayvan, fırlatma) ===
    ("çiftlikteki atlar dörtnala koşuyordu", "at_hayvan"),
    ("ata binmeyi küçük yaşta öğrenmişti", "at_hayvan"),
    ("nalbant atın nalını değiştirirken dikkatliydi", "at_hayvan"),
    ("eyeri düzgün bağlamazsan attan düşersin", "at_hayvan"),
    ("yarış atları milyonlarca liraya satılıyor", "at_hayvan"),
    ("tay büyüyüp güçlü bir at oldu", "at_hayvan"),
    ("at arabasıyla köy yolunda ilerlediler", "at_hayvan"),
    ("vahşi atları evcilleştirmek zordur", "at_hayvan"),

    ("topu en uzağa o atar", "at_firlatma"),
    ("çöpü çöp kutusuna atmayı unutma", "at_firlatma"),
    ("adımını dikkatli atmazsan düşersin", "at_firlatma"),
    ("nişan alıp hedefi tam on ikiden vurdu", "at_firlatma"),
    ("taşı suya atınca halkalar oluştu", "at_firlatma"),
    ("okunu tam hedefe isabet ettirdi", "at_firlatma"),
    ("imzayı atıp belgeyi teslim etti", "at_firlatma"),
    ("topu potaya doğru fırlattı", "at_firlatma"),
]

ALL_SENSES = sorted(set(s for _, s in TRAINING_SAMPLES))
print(f"📊 Veri seti: {len(TRAINING_SAMPLES)} doğal cümle, {len(ALL_SENSES)} anlam")
print(f"   Sense'ler: {ALL_SENSES}")

# ============================================================
# 1. EMBEDDING: sentence-transformers + PCA
# ============================================================

print("\n📥 MiniLM yükleniyor...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
FULL_DIM = embed_model.get_sentence_embedding_dimension()  # 384
print(f"   Ham embedding boyutu: {FULL_DIM}D")

# Tüm embedding'leri hesapla
print("⏳ Embedding'ler hesaplanıyor...")
all_texts = [t for t, _ in TRAINING_SAMPLES]
all_labels = [s for _, s in TRAINING_SAMPLES]
all_embeddings_raw = embed_model.encode(all_texts, show_progress_bar=True)

# PCA ile 64D'ye indir
TARGET_DIM = 64
print(f"   PCA: {FULL_DIM}D → {TARGET_DIM}D")
pca = PCA(n_components=TARGET_DIM, random_state=42)
all_embeddings_64d = pca.fit_transform(all_embeddings_raw)
explained_var = pca.explained_variance_ratio_.sum()
print(f"   Açıklanan varyans: {explained_var*100:.1f}%")

# PyTorch tensor'a çevir
embeddings_tensor = torch.from_numpy(all_embeddings_64d).float()

# ============================================================
# 2. HOLD-OUT SPLIT (stratified)
# ============================================================

print("\n📊 Hold-out split (80/20 stratified)...")
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = list(splitter.split(all_embeddings_64d, all_labels))[0]

train_x = embeddings_tensor[train_idx]
train_y = [all_labels[i] for i in train_idx]
test_x = embeddings_tensor[test_idx]
test_y = [all_labels[i] for i in test_idx]

print(f"   Train: {len(train_x)} örnek, Test: {len(test_x)} örnek (hold-out)")
print(f"   Test sense dağılımı:")
for s in ALL_SENSES:
    count = sum(1 for y in test_y if y == s)
    print(f"     {s}: {count}")

# ============================================================
# 3. HİBRİT PCN KATMANI (W_fwd + W_bwd + W_down)
# ============================================================

class HybridPCNLayer:
    """
    3 ağırlık matrisli hibrit PCN katmanı.
    
    - W_fwd [out_dim × in_dim]: ileri projeksiyon
    - W_bwd [in_dim × out_dim]: geri tahmin (reconstruction)
    - W_down [downward_dim × out_dim]: bu katmandan BİR ALT katmanın INPUT'una tahmin
      (downward_dim = bir alt katmanın in_dim'i; en alt katmanda None)
    
    Her katman boyutu bir öncekinden küçük (dimensional reduction).
    """
    
    def __init__(self, in_dim: int, out_dim: int, downward_dim: int = None, name: str = "layer"):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downward_dim = downward_dim
        
        # 3 ağırlık matrisi
        self.W_fwd = torch.randn(out_dim, in_dim, requires_grad=False) * 0.1
        self.W_bwd = torch.randn(in_dim, out_dim, requires_grad=False) * 0.1
        if downward_dim is not None:
            self.W_down = torch.randn(downward_dim, out_dim, requires_grad=False) * 0.1
        else:
            self.W_down = None  # en alt katman
        
        # İç durum
        self.mu = torch.zeros(out_dim, 1, requires_grad=False)
        
        # Aktivasyon
        self.activation = torch.tanh
        
        # Clipping parametreleri
        self.grad_clip = 10.0
        self.weight_norm_max = 5.0
    
    def predict_downward(self) -> torch.Tensor:
        """Bu katmanın iç durumundan bir alt katmana tahmin.
        W_down [downward_dim × out_dim] @ mu [out_dim × 1] = [downward_dim × 1]
        En alt katmanda None döner."""
        if self.W_down is None:
            return None
        return self.activation(torch.matmul(self.W_down, self.mu))
    
    def predict_upward(self) -> torch.Tensor:
        """Bu katmanın iç durumundan girdiyi rekonstrükte et."""
        return self.activation(torch.matmul(self.W_bwd, self.mu))
    
    def inference_step(self, input_signal: torch.Tensor,
                       prediction_from_above: torch.Tensor = None,
                       dt: float = 0.1):
        """
        Bir inference adımı: μ güncelle.
        Returns: (error_norm, error_vector)
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(1)
        
        # Hedef: girdi + üst katmandan gelen tahmin
        target = input_signal
        if prediction_from_above is not None:
            target = target + prediction_from_above
        
        # Kendi rekonstrüksiyonu
        self_prediction = self.predict_upward()
        error = target - self_prediction
        
        # Gradyan + clipping
        grad = torch.matmul(self.W_fwd, error)
        gn = grad.norm()
        if gn > self.grad_clip:
            grad = grad * (self.grad_clip / gn)
        
        # Durum güncelleme
        d_mu = dt * (grad - self.mu)
        self.mu = self.mu + d_mu
        
        return error.norm().item(), error.detach().clone()
    
    def learning_step(self, input_signal: torch.Tensor,
                      prediction_from_above: torch.Tensor = None,
                      lr: float = 0.01,
                      error_from_below: torch.Tensor = None):
        """
        Hebbian öğrenme: W_fwd, W_bwd güncellenir.
        W_down: bir alt katmanın error'u (error_from_below) ile güncellenir.
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(1)
        
        target = input_signal
        if prediction_from_above is not None:
            target = target + prediction_from_above
        
        self_prediction = self.predict_upward()
        error = target - self_prediction
        
        # Hebbian güncelleme (clipping'li)
        clamp_val = 0.05
        
        dW_fwd = torch.clamp(lr * torch.matmul(self.mu, error.T), -clamp_val, clamp_val)
        self.W_fwd = self.W_fwd + dW_fwd
        
        dW_bwd = torch.clamp(lr * torch.matmul(error, self.mu.T), -clamp_val, clamp_val)
        self.W_bwd = self.W_bwd + dW_bwd
        
        # W_down: bir alt katmanın error'una göre güncelle
        # (bu katmanın mu'sundan alt katmanın input'una tahmin → alt katmanın error'unu azalt)
        if self.W_down is not None and error_from_below is not None:
            dW_down = torch.clamp(
                lr * torch.matmul(error_from_below, self.mu.T),
                -clamp_val, clamp_val
            )
            self.W_down = self.W_down + dW_down
        
        # Weight norm clipping
        weights = [self.W_fwd, self.W_bwd]
        if self.W_down is not None:
            weights.append(self.W_down)
        for W in weights:
            wn = W.norm()
            if wn > self.weight_norm_max:
                W.data = W.data * (self.weight_norm_max / wn)
    
    def param_count(self) -> int:
        """Bu katmandaki toplam parametre sayısı."""
        count = self.W_fwd.numel() + self.W_bwd.numel()
        if self.W_down is not None:
            count += self.W_down.numel()
        return count
    
    def reset_state(self):
        self.mu = torch.zeros(self.out_dim, 1, requires_grad=False)


# ============================================================
# 4. HİBRİT PCN (3 katman: 64→32→16)
# ============================================================

class HybridPCN:
    """
    3 katmanlı boyut azaltan hibrit PCN:
    Layer 0: 64 → 32 (sensory)
    Layer 1: 32 → 16 (local context)
    Layer 2: 16 → 16 (global, output)
    """
    
    def __init__(self, input_dim: int = 64):
        # L0: en alt katman, W_down yok (altında katman yok)
        self.layer0 = HybridPCNLayer(input_dim, 32, downward_dim=None, name="Sensory (L0)")
        # L1: downward_dim=64 (L0'ın input boyutu)
        self.layer1 = HybridPCNLayer(32, 16, downward_dim=64, name="Local (L1)")
        # L2: downward_dim=32 (L1'in input boyutu)
        self.layer2 = HybridPCNLayer(16, 16, downward_dim=32, name="Global (L2)")
        self.layers = [self.layer0, self.layer1, self.layer2]
    
    def total_params(self) -> int:
        return sum(l.param_count() for l in self.layers)
    
    def inference(self, input_vec: torch.Tensor,
                  n_iterations: int = 25, dt: float = 0.1):
        """
        Hiyerarşik PCN inference.
        Returns: (total_error, error_L0, error_L1, error_L2)
        """
        total_error = 0.0
        last_err0, last_err1, last_err2 = None, None, None
        
        for _ in range(n_iterations):
            # Yukarıdan aşağıya tahmin zinciri
            pred_l2 = self.layer2.predict_downward()  # L2 → L1 (32D)
            pred_l1 = self.layer1.predict_downward()  # L1 → L0 (64D)
            
            # Aşağıdan yukarıya güncelleme
            _, last_err0 = self.layer0.inference_step(input_vec, pred_l1, dt)
            _, last_err1 = self.layer1.inference_step(self.layer0.mu, pred_l2, dt)
            e2, last_err2 = self.layer2.inference_step(self.layer1.mu, None, dt)
            
            total_error += last_err0.norm().item() + last_err1.norm().item() + e2
        
        return total_error, last_err0, last_err1, last_err2
    
    def learn(self, input_vec: torch.Tensor, lr: float = 0.01):
        """Tüm katmanlarda Hebbian öğrenme.
        W_down'lar alt katmanların error'unu kullanır."""
        _, err_l0, err_l1, err_l2 = self.inference(input_vec, n_iterations=20)
        
        # Tahmin zinciri
        pred_l2 = self.layer2.predict_downward()
        pred_l1 = self.layer1.predict_downward()
        
        # L0: prediction_from_above = L1'den gelen, error_from_below yok (en alt)
        self.layer0.learning_step(input_vec, pred_l1, lr, error_from_below=None)
        # L1: prediction_from_above = L2'den gelen, error_from_below = L0'ın error'u
        self.layer1.learning_step(self.layer0.mu, pred_l2, lr, error_from_below=err_l0)
        # L2: en üst katman, error_from_below = L1'in error'u
        self.layer2.learning_step(self.layer1.mu, None, lr, error_from_below=err_l1)
    
    def get_global_state(self) -> torch.Tensor:
        """Layer 2'nin iç durumu (16D global temsil)."""
        return self.layer2.mu.squeeze().clone()
    
    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()


# ============================================================
# 5. EĞİTİM VE DEĞERLENDİRME
# ============================================================

def compute_centroids(model, data_x, data_y, n_inf=30):
    """Eğitim sonrası train set'te tek geçişle sabit sense centroid'leri hesapla.
    Epoch sırasına duyarlı değil, tekrarlanabilir."""
    centroids = defaultdict(list)
    for x, s in zip(data_x, data_y):
        model.reset_state()
        model.inference(x, n_iterations=n_inf)
        mu = model.get_global_state()
        if mu.norm() > 0 and not torch.isnan(mu).any():
            centroids[s].append(mu)
    return {s: torch.stack(vecs).mean(dim=0) for s, vecs in centroids.items() if vecs}


def evaluate_with_centroids(model, centroids, data_x, data_y, n_inf=30):
    """Sabit centroid'lerle değerlendirme (epoch sırasından bağımsız)."""
    correct = 0
    for x, true_sense in zip(data_x, data_y):
        model.reset_state()
        model.inference(x, n_iterations=n_inf)
        mu = model.get_global_state()
        
        if mu.norm() == 0 or torch.isnan(mu).any():
            continue
        
        best_sense, best_sim = None, -1
        for sn, centroid in centroids.items():
            sim = F.cosine_similarity(mu.unsqueeze(0), centroid.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_sense = sim, sn
        
        if best_sense == true_sense:
            correct += 1
    
    return correct / len(data_y) * 100 if len(data_y) > 0 else 0


def evaluate(model, sense_embs, data_x, data_y, n_inf=30):
    """Eğitim sırası ara değerlendirme: son 6 epoch embedding ortalaması."""
    correct = 0
    for x, true_sense in zip(data_x, data_y):
        model.reset_state()
        model.inference(x, n_iterations=n_inf)
        mu = model.get_global_state()
        
        if mu.norm() == 0 or torch.isnan(mu).any():
            continue
        
        best_sense, best_sim = None, -1
        for sn, emb_list in sense_embs.items():
            if not emb_list:
                continue
            avg = torch.stack(emb_list[-6:]).mean(dim=0)
            if avg.norm() == 0:
                continue
            sim = F.cosine_similarity(mu.unsqueeze(0), avg.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_sense = sim, sn
        
        if best_sense == true_sense:
            correct += 1
    
    return correct / len(data_y) * 100 if len(data_y) > 0 else 0


def train_and_evaluate():
    print("\n" + "=" * 65)
    print("🧠 PCN-WSD HYBRID: 64→32→16 + W_down + PCA + Hold-out")
    print("=" * 65)
    
    model = HybridPCN(input_dim=TARGET_DIM)
    n_params = model.total_params()
    
    print(f"\n   Mimari:    Input(64D) → L0(64→32) → L1(32→16) → L2(16D)")
    print(f"   Parametre: {n_params} (W_fwd + W_bwd + W_down)")
    print(f"   Embedding: MiniLM 384D → PCA {TARGET_DIM}D ({explained_var*100:.1f}% var)")
    print(f"   Test:      Hold-out {len(test_x)}/{len(train_x)} (stratified)")
    print(f"   Öğrenme:   Hebbian (lokal, backprop yok)")
    
    # ============================================================
    # BASELINE: kNN (PCA 64D uzayında)
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"📌 BASELINE: kNN (PCA 64D, cosine similarity)")
    print(f"{'─' * 65}")
    
    # Her sense için centroid hesapla
    from collections import Counter
    sense_centroids_knn = {}
    for x, s in zip(train_x, train_y):
        if s not in sense_centroids_knn:
            sense_centroids_knn[s] = []
        sense_centroids_knn[s].append(x)
    for s in sense_centroids_knn:
        sense_centroids_knn[s] = torch.stack(sense_centroids_knn[s]).mean(dim=0)
    
    knn_correct = 0
    for x, true_sense in zip(test_x, test_y):
        best_sense, best_sim = None, -1
        for s, c in sense_centroids_knn.items():
            sim = F.cosine_similarity(x.unsqueeze(0), c.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_sense = sim, s
        if best_sense == true_sense:
            knn_correct += 1
    
    knn_acc = knn_correct / len(test_y) * 100
    print(f"   kNN hold-out doğruluğu: {knn_correct}/{len(test_y)} ({knn_acc:.1f}%)")
    
    # ============================================================
    # PCN EĞİTİMİ
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"🔄 PCN EĞİTİMİ ({len(train_x)} train örneği)")
    print(f"{'─' * 65}")
    
    n_epochs = 50
    lr = 0.02
    sense_embs = defaultdict(list)
    
    train_acc_history = []
    holdout_acc_history = []
    best_holdout = 0
    best_epoch = 0
    
    t0 = time.time()
    
    for epoch in range(n_epochs):
        # Shuffle
        indices = np.random.permutation(len(train_x))
        
        epoch_loss = 0
        for idx in indices:
            x = train_x[idx]
            s = train_y[idx]
            
            model.reset_state()
            err, _, _, _ = model.inference(x, n_iterations=20)
            model.learn(x, lr=lr)
            epoch_loss += err
            
            gs = model.get_global_state()
            if not torch.isnan(gs).any():
                sense_embs[s].append(gs)
        
        # Her 5 epoch'ta değerlendirme
        if (epoch + 1) % 5 == 0:
            train_acc = evaluate(model, sense_embs, train_x, train_y)
            holdout_acc = evaluate(model, sense_embs, test_x, test_y)
            
            train_acc_history.append(train_acc)
            holdout_acc_history.append(holdout_acc)
            
            if holdout_acc > best_holdout:
                best_holdout = holdout_acc
                best_epoch = epoch + 1
            
            marker = " ⭐" if holdout_acc == best_holdout else ""
            print(f"   Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_x):.4f} | "
                  f"Train: {train_acc:.1f}% | Hold-out: {holdout_acc:.1f}%{marker}")
    
    elapsed = time.time() - t0
    
    # ============================================================
    # FINAL DEĞERLENDİRME (centroid-tabanlı, epoch sırasından bağımsız)
    # ============================================================
    print(f"\n{'═' * 65}")
    print(f"📊 FİNAL SONUÇLAR (centroid-tabanlı)")
    print(f"{'═' * 65}")
    
    # Eğitim sonrası train set'te tek geçişle centroid hesapla
    centroids = compute_centroids(model, train_x, train_y, n_inf=30)
    
    final_train = evaluate_with_centroids(model, centroids, train_x, train_y, n_inf=30)
    final_holdout = evaluate_with_centroids(model, centroids, test_x, test_y, n_inf=30)
    
    print(f"\n   {'Metrik':<30} {'Değer'}")
    print(f"   {'─' * 50}")
    print(f"   {'Mimari':<30} 3L: 64→32→16 + W_down")
    print(f"   {'Parametre':<30} {n_params}")
    print(f"   {'Embedding':<30} MiniLM 384D → PCA {TARGET_DIM}D")
    print(f"   {'Train örnek':<30} {len(train_x)}")
    print(f"   {'Test örnek (hold-out)':<30} {len(test_x)}")
    print(f"   {'Epoch':<30} {n_epochs}")
    print(f"   {'Süre':<30} {elapsed:.1f}s")
    print(f"   {'─' * 50}")
    print(f"   {'kNN (hold-out baseline)':<30} {knn_acc:.1f}%")
    print(f"   {'PCN Train doğruluğu':<30} {final_train:.1f}%")
    print(f"   {'PCN Hold-out doğruluğu':<30} {final_holdout:.1f}%")
    print(f"   {'En iyi hold-out (ara)':<30} {best_holdout:.1f}% (epoch {best_epoch})")
    
    # ============================================================
    # SENSE BAZINDA ANALİZ (centroid-tabanlı)
    # ============================================================
    print(f"\n{'─' * 65}")
    print(f"🔍 HOLD-OUT SENSE BAZINDA (centroid)")
    print(f"{'─' * 65}")
    
    sense_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for x, true_sense in zip(test_x, test_y):
        model.reset_state()
        model.inference(x, n_iterations=30)
        mu = model.get_global_state()
        
        if mu.norm() == 0 or torch.isnan(mu).any():
            sense_stats[true_sense]["total"] += 1
            continue
        
        best_sense, best_sim = None, -1
        for sn, centroid in centroids.items():
            sim = F.cosine_similarity(mu.unsqueeze(0), centroid.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim, best_sense = sim, sn
        
        sense_stats[true_sense]["total"] += 1
        if best_sense == true_sense:
            sense_stats[true_sense]["correct"] += 1
    
    print(f"\n   {'Sense':<22} {'Test':<6} {'Doğru':<6} {'Başarı':<10}")
    print(f"   {'─' * 48}")
    for sense in sorted(ALL_SENSES):
        s = sense_stats[sense]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
        print(f"   {sense:<22} {s['total']:<6} {s['correct']:<6} {acc:5.1f}%  {bar}")
    
    # ============================================================
    # ÖZET
    # ============================================================
    print(f"\n{'═' * 65}")
    print(f"🏆 HİBRİT PCN ÖZETİ")
    print(f"{'═' * 65}")
    
    print(f"\n   Baseline (kNN hold-out):       {knn_acc:.1f}%")
    print(f"   PCN Hybrid train:              {final_train:.1f}%")
    print(f"   PCN Hybrid hold-out:           {final_holdout:.1f}%")
    print(f"   vs README iddiası (97.1%):     {'✅' if final_holdout >= 97.1 else '❌'}")
    print(f"   vs README hold-out (100%):     {'✅' if final_holdout >= 100 else '❌'}")
    
    delta = final_holdout - knn_acc
    print(f"\n   💡 PCN vs kNN farkı: {delta:+.1f}%")
    if delta > 0:
        print(f"      PCN, kNN'den {delta:.1f} puan daha iyi!")
    else:
        print(f"      kNN daha iyi. Embedding'ler bu görev için fazla güçlü.")
    
    return {
        "architecture": "3L: 64→32→16 + W_down",
        "parameters": n_params,
        "embedding": f"MiniLM 384D → PCA {TARGET_DIM}D",
        "train_samples": len(train_x),
        "test_samples": len(test_x),
        "epochs": n_epochs,
        "knn_holdout": knn_acc,
        "pcn_train": final_train,
        "pcn_holdout": final_holdout,
        "best_holdout": best_holdout,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    
    # JSON özet (CI/log için)
    print(f"\n📋 JSON ÖZET:")
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # results_hybrid.json'a kaydet (tekrarlanabilirlik)
    import os
    out_path = os.path.join(os.path.dirname(__file__), 'results_hybrid.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Kaydedildi: {out_path}")
