#!/usr/bin/env python3
"""
🧠 PCN-WSD: Predictive Coding Network for Word Sense Disambiguation
=====================================================================
3 katmanlı hiyerarşik Predictive Coding mimarisi ile Türkçe eşsesli
kelimelerin anlamını bağlamdan çıkarma (WSD) prototipi.

Mimari:
  Katman 0 (Sensory):   Kelime embedding'leri (girdi)
  Katman 1 (Local):     Cümle içi bağlam entegrasyonu
  Katman 2 (Global):    Soyut kavram/niyet çıkarımı

Her katman:
  - İleri yön: Tahmin hatasını (prediction error) yukarı taşır
  - Geri yön: Üst katmandan alt katmana tahmin gönderir
  - Dinamik durum (mu): Inference sırasında iteratif güncellenir
  - Hebbian öğrenme: ΔW = error × muᵀ (yerel, backprop yok)

Hacı 🧿 — 24 Haziran 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import json

# ============================================================
# 1. BASİT EMBEDDING (pretrained yerine hızlı prototip)
# ============================================================

# Türkçe eşsesli kelimeler ve bağlam kelimeleri için mini embedding
# Gerçek projede word2vec/fastText kullanılır
MINI_VOCAB = {
    # "yüz" kelimesinin 3 anlamı ve bağlam kelimeleri
    "yüz_sayi":     ["yüz", "sayı", "rakam", "iki", "elli", "bin", "kırk", "otuz", "yetmiş", "doksan"],
    "yüz_anatomi":  ["yüz", "göz", "burun", "ağız", "çene", "yanak", "alın", "dudak", "kirpik", "kaş"],
    "yüz_yüzme":    ["yüz", "yüzmek", "su", "deniz", "havuz", "kulaç", "dalga", "kıyı", "plaj", "sahil"],
    # "çay" kelimesinin 2 anlamı
    "çay_içecek":   ["çay", "demli", "bardak", "içmek", "sıcak", "şeker", "kahve", "fincan", "demlik", "bitki"],
    "çay_dere":     ["çay", "dere", "ırmak", "su", "akarsu", "köprü", "kenar", "şelale", "vadi", "taş"],
    # "baş" kelimesinin 3 anlamı
    "baş_anatomi":  ["baş", "kafa", "beyin", "saç", "şapka", "düşünmek", "akıl", "zihin", "kafatası", "ense"],
    "baş_lider":    ["baş", "lider", "yönetici", "müdür", "şef", "komutan", "reis", "amir", "patron", "önder"],
    "baş_başlangıç":["baş", "başlangıç", "ilk", "önce", "sıfır", "start", "öncü", "ön", "uç", "tepe"],
    # "dil" kelimesinin 3 anlamı
    "dil_organ":    ["dil", "ağız", "tat", "konuşmak", "damak", "yutmak", "çiğnemek", "ısırmak", "ses", "hece"],
    "dil_lisan":    ["dil", "lisan", "konuşmak", "sözcük", "gramer", "cümle", "alfabe", "tercüme", "anadil", "lehçe"],
    "dil_coğrafya": ["dil", "burun", "kara", "deniz", "yarımada", "koy", "körfez", "kıyı", "sahil", "toprak"],
    # "at" kelimesinin 2 anlamı
    "at_hayvan":    ["at", "hayvan", "binek", "nal", "dörtnala", "koşmak", "binici", "eyer", "tay", "midilli"],
    "at_fırlatma":  ["at", "atmak", "fırlatmak", "top", "taş", "hedef", "isabet", "nişan", "vurmak", "savurmak"],
}

# Her kelime için 32 boyutlu rastgele embedding (gerçekte pretrained kullanılır)
torch.manual_seed(42)
embedding_dim = 32
word_to_idx = {}
embeddings = []

for sense_name, words in MINI_VOCAB.items():
    for w in words:
        if w not in word_to_idx:
            word_to_idx[w] = len(word_to_idx)
            embeddings.append(torch.randn(embedding_dim) * 0.1)

EMBEDDING_MATRIX = torch.stack(embeddings)  # [vocab_size, 32]

def get_embedding(word: str) -> torch.Tensor:
    """Kelime embedding'ini döndür."""
    if word in word_to_idx:
        return EMBEDDING_MATRIX[word_to_idx[word]]
    return torch.zeros(embedding_dim)

def build_context_vector(sentence: List[str]) -> torch.Tensor:
    """Cümledeki kelimelerin ortalama embedding'i."""
    vecs = [get_embedding(w) for w in sentence]
    return torch.stack(vecs).mean(dim=0)  # [32]

# ============================================================
# 2. PCN KATMANI
# ============================================================

class PCNLayer:
    """
    Tek bir Predictive Coding katmanı.
    
    - W_fwd: İleri yön ağırlıkları (input → hidden)
    - W_bwd: Geri yön ağırlıkları (hidden → input, tahmin üretimi)
    - mu: Dinamik iç durum (inference sırasında güncellenir)
    
    Tüm katmanlar aynı boyutta çalışır (dim=32).
    Hiyerarşi boyut sıkıştırmasından değil, katman pozisyonundan gelir.
    """
    
    def __init__(self, dim: int, name: str = "layer"):
        self.name = name
        self.dim = dim
        
        # Ağırlıklar (başlangıçta küçük rastgele)
        self.W_fwd = torch.randn(dim, dim, requires_grad=False) * 0.1
        self.W_bwd = torch.randn(dim, dim, requires_grad=False) * 0.1
        
        # İç durum
        self.mu = torch.zeros(dim, 1, requires_grad=False)
        
        # Aktivasyon
        self.activation = torch.tanh
    
    def predict_downward(self) -> torch.Tensor:
        """
        Kendi iç durumundan (mu) aşağı katmana tahmin üret.
        prediction = tanh(W_bwd × mu)
        """
        return self.activation(torch.matmul(self.W_bwd, self.mu))  # [dim, 1]
    
    def inference_step(self, input_signal: torch.Tensor,
                       prediction_from_above: torch.Tensor = None,
                       dt: float = 0.1) -> float:
        """
        Bir inference adımı: iç durumu (mu) güncelle.
        
        d(mu)/dt = W_fwd × error − mu
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(1)
        
        if prediction_from_above is not None:
            target = input_signal + prediction_from_above
        else:
            target = input_signal
        
        self_prediction = self.predict_downward()
        error = target - self_prediction
        
        d_mu = torch.matmul(self.W_fwd, error) - self.mu
        self.mu = self.mu + dt * d_mu
        
        return error.norm().item()
    
    def learning_step(self, input_signal: torch.Tensor,
                      prediction_from_above: torch.Tensor = None,
                      lr: float = 0.01):
        """
        Hebbian öğrenme: ΔW = error × muᵀ (yerel, backprop yok)
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(1)
        
        if prediction_from_above is not None:
            target = input_signal + prediction_from_above
        else:
            target = input_signal
        
        self_prediction = self.predict_downward()
        error = target - self_prediction
        
        dW_fwd = lr * torch.matmul(error, self.mu.T)
        self.W_fwd = self.W_fwd + dW_fwd
        
        dW_bwd = lr * torch.matmul(self.mu, error.T)
        self.W_bwd = self.W_bwd + dW_bwd
    
    def reset_state(self):
        """İç durumu sıfırla."""
        self.mu = torch.zeros(self.dim, 1, requires_grad=False)


# ============================================================
# 3. 3-KATMANLI HİYERARŞİK PCN
# ============================================================

class PredictiveCodingWSD:
    """
    3 katmanlı Predictive Coding ağı:
    - Layer 0 (Sensory): Kelime embedding'lerini işler
    - Layer 1 (Local): Lokal bağlam entegrasyonu
    - Layer 2 (Global): Soyut kavram/niyet çıkarımı
    
    Her katman 32 boyutlu, aynı uzayda çalışır.
    Hiyerarşi, bilgi akış yönünden gelir (sensory → local → global).
    """
    
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.layer0 = PCNLayer(dim, "Sensory (L0)")
        self.layer1 = PCNLayer(dim, "Local (L1)")
        self.layer2 = PCNLayer(dim, "Global (L2)")
        
        self.layers = [self.layer0, self.layer1, self.layer2]
    
    def inference(self, context_vec: torch.Tensor,
                  n_iterations: int = 30, dt: float = 0.1) -> float:
        """
        Hiyerarşik PCN inference:
        - L2: L1'in mu'sunu "girdi" olarak alır (en üst katman)
        - L1: L0'ın mu'sunu girdi olarak alır, L2'den tahmin bekler
        - L0: Dış context_vec'i girdi olarak alır, L1'den tahmin bekler
        """
        total_error = 0.0
        
        for _ in range(n_iterations):
            # Her katman kendi tahminini üretir (yukarıdan aşağı)
            pred_l2 = self.layer2.predict_downward()  # L2 → L1 tahmin
            pred_l1 = self.layer1.predict_downward()  # L1 → L0 tahmin
            
            # Aşağıdan yukarıya güncelleme
            # L0: dış veri + L1'den tahmin
            e0 = self.layer0.inference_step(context_vec, pred_l1, dt)
            # L1: L0'ın durumu + L2'den tahmin
            e1 = self.layer1.inference_step(self.layer0.mu.squeeze(), pred_l2, dt)
            # L2: sadece L1'in durumu (en üst katman, üstten tahmin yok)
            e2 = self.layer2.inference_step(self.layer1.mu.squeeze(), None, dt)
            
            total_error = e0 + e1 + e2
        
        return total_error
    
    def learn(self, context_vec: torch.Tensor, lr: float = 0.01):
        """Tüm katmanlarda Hebbian öğrenme."""
        self.inference(context_vec, n_iterations=20)
        
        pred_l2 = self.layer2.predict_downward()
        pred_l1 = self.layer1.predict_downward()
        
        self.layer0.learning_step(context_vec, pred_l1, lr)
        self.layer1.learning_step(self.layer0.mu.squeeze(), pred_l2, lr)
        self.layer2.learning_step(self.layer1.mu.squeeze(), None, lr)
    
    def get_global_state(self) -> torch.Tensor:
        """Layer 2'nin iç durumu (global kavram temsili)."""
        return self.layer2.mu.squeeze().clone()
    
    def reset_state(self):
        """Tüm katmanların iç durumunu sıfırla."""
        for layer in self.layers:
            layer.reset_state()


# ============================================================
# 4. WSD VERİ SETİ
# ============================================================

def generate_wsd_data() -> List[Tuple[List[str], str]]:
    """
    Eşsesli kelimeler için bağlam cümleleri ve doğru anlam etiketleri.
    Her örnek: (kelime_listesi, sense_label)
    """
    data = []
    
    # "yüz" için örnekler
    data.append((["yüz", "sayı", "elli", "iki"], "yüz_sayi"))
    data.append((["yüz", "rakam", "bin", "doksan"], "yüz_sayi"))
    data.append((["yüz", "göz", "burun", "ağız"], "yüz_anatomi"))
    data.append((["yüz", "yanak", "alın", "dudak"], "yüz_anatomi"))
    data.append((["yüz", "su", "deniz", "kulaç"], "yüz_yüzme"))
    data.append((["yüz", "havuz", "plaj", "dalga"], "yüz_yüzme"))
    
    # "çay" için örnekler
    data.append((["çay", "demli", "bardak", "içmek"], "çay_içecek"))
    data.append((["çay", "sıcak", "şeker", "fincan"], "çay_içecek"))
    data.append((["çay", "dere", "ırmak", "köprü"], "çay_dere"))
    data.append((["çay", "akarsu", "vadi", "şelale"], "çay_dere"))
    
    # "baş" için örnekler
    data.append((["baş", "kafa", "beyin", "saç"], "baş_anatomi"))
    data.append((["baş", "şapka", "ense", "alın"], "baş_anatomi"))
    data.append((["baş", "lider", "müdür", "şef"], "baş_lider"))
    data.append((["baş", "komutan", "reis", "amir"], "baş_lider"))
    data.append((["baş", "başlangıç", "ilk", "önce"], "baş_başlangıç"))
    data.append((["baş", "sıfır", "start", "tepe"], "baş_başlangıç"))
    
    # "dil" için örnekler
    data.append((["dil", "ağız", "tat", "konuşmak"], "dil_organ"))
    data.append((["dil", "damak", "çiğnemek", "ısırmak"], "dil_organ"))
    data.append((["dil", "lisan", "gramer", "cümle"], "dil_lisan"))
    data.append((["dil", "alfabe", "tercüme", "anadil"], "dil_lisan"))
    data.append((["dil", "burun", "kara", "deniz"], "dil_coğrafya"))
    data.append((["dil", "yarımada", "körfez", "kıyı"], "dil_coğrafya"))
    
    # "at" için örnekler
    data.append((["at", "hayvan", "nal", "binek"], "at_hayvan"))
    data.append((["at", "dörtnala", "koşmak", "tay"], "at_hayvan"))
    data.append((["at", "atmak", "fırlatmak", "top"], "at_fırlatma"))
    data.append((["at", "hedef", "nişan", "taş"], "at_fırlatma"))
    
    return data


def get_sense_id(sense_name: str) -> int:
    """Sense isminden ID'ye çevir."""
    senses = ["yüz_sayi", "yüz_anatomi", "yüz_yüzme", 
              "çay_içecek", "çay_dere",
              "baş_anatomi", "baş_lider", "baş_başlangıç",
              "dil_organ", "dil_lisan", "dil_coğrafya",
              "at_hayvan", "at_fırlatma"]
    return senses.index(sense_name) if sense_name in senses else -1


# ============================================================
# 5. EĞİTİM VE TEST
# ============================================================

def train_and_evaluate():
    print("=" * 60)
    print("🧠 PCN-WSD: Predictive Coding ile Kelime Anlamı Giderme")
    print("=" * 60)
    
    model = PredictiveCodingWSD()
    data = generate_wsd_data()
    
    # Her sense için embedding hesapla
    sense_embeddings = {}
    
    print(f"\n📊 Veri seti: {len(data)} örnek, {len(set(s for _, s in data))} farklı anlam")
    print(f"   Embedding boyutu: {embedding_dim}")
    print(f"   Katmanlar: L0(32) → L1(32) → L2(32) [hiyerarşik, aynı boyut]")
    print(f"   Öğrenme: Hebbian (ΔW = error × muᵀ)")
    
    print("\n" + "─" * 60)
    print("🔄 EĞİTİM BAŞLIYOR")
    print("─" * 60)
    
    n_epochs = 30
    correct_history = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        np.random.shuffle(data)
        
        for words, true_sense in data:
            # Bağlam vektörü
            context = build_context_vector(words)  # [32]
            
            # Inference + Learning
            model.reset_state()
            error = model.inference(context, n_iterations=25)
            model.learn(context, lr=0.05)
            epoch_loss += error
            
            # Bu örneğin sense embedding'ini kaydet
            if true_sense not in sense_embeddings:
                sense_embeddings[true_sense] = []
            sense_embeddings[true_sense].append(model.layer2.mu.clone())
        
        # Her 5 epoch'ta bir test
        if (epoch + 1) % 5 == 0:
            correct = 0
            total = 0
            
            for words, true_sense in data:
                context = build_context_vector(words)
                model.reset_state()
                model.inference(context, n_iterations=30)
                
                # Layer 2'nin mu'sunu sense embedding'lerle karşılaştır
                mu = model.layer2.mu.squeeze()
                
                # En yakın sense embedding'ini bul (cosine similarity)
                best_sense = None
                best_sim = -1
                
                for sense_name, emb_list in sense_embeddings.items():
                    if emb_list:
                        avg_emb = torch.stack(emb_list).mean(dim=0).squeeze()
                        sim = F.cosine_similarity(mu.unsqueeze(0), avg_emb.unsqueeze(0))
                        if sim.item() > best_sim:
                            best_sim = sim
                            best_sense = sense_name
                
                if best_sense == true_sense:
                    correct += 1
                total += 1
            
            accuracy = correct / total * 100
            correct_history.append(accuracy)
            
            print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss/len(data):.4f} | "
                  f"Doğruluk: {correct}/{total} ({accuracy:.1f}%)")
    
    # ============================================================
    # 6. SONUÇLAR
    # ============================================================
    print("\n" + "═" * 60)
    print("📊 SON TEST SONUÇLARI")
    print("═" * 60)
    
    # Detaylı test
    print(f"\n{'Örnek (bağlam)':<35} {'Tahmin':<20} {'Gerçek':<20} {'✓/✗'}")
    print("─" * 80)
    
    correct = 0
    for words, true_sense in data:
        context = build_context_vector(words)
        model.reset_state()
        model.inference(context, n_iterations=30)
        mu = model.layer2.mu.squeeze()
        
        best_sense = None
        best_sim = -1
        for sense_name, emb_list in sense_embeddings.items():
            if emb_list:
                avg_emb = torch.stack(emb_list).mean(dim=0).squeeze()
                sim = F.cosine_similarity(mu.unsqueeze(0), avg_emb.unsqueeze(0))
                if sim.item() > best_sim:
                    best_sim = sim
                    best_sense = sense_name
        
        is_correct = "✓" if best_sense == true_sense else "✗"
        if best_sense == true_sense:
            correct += 1
        
        context_str = " ".join(words[:4])
        print(f"  {context_str:<35} {best_sense:<20} {true_sense:<20} {is_correct}")
    
    final_accuracy = correct / len(data) * 100
    print(f"\n{'═' * 60}")
    print(f"🎯 Final Doğruluk: {correct}/{len(data)} ({final_accuracy:.1f}%)")
    print(f"   Baseline (rastgele): ~{100/len(set(s for _, s in data)):.1f}% "
          f"(13 farklı anlam → %7.7)")
    
    # Eğitim eğrisi
    if correct_history:
        print(f"\n📈 Doğruluk eğrisi: {' → '.join(f'{a:.0f}%' for a in correct_history)}")
    
    print(f"\n🧿 Model Özeti:")
    print(f"   Mimari: 3 katmanlı hiyerarşik PCN")
    print(f"   Parametre sayısı: ~{3 * 32 * 32 * 2} (~6K)")
    print(f"   Öğrenme kuralı: Hebbian (yerel, backprop yok)")
    print(f"   Inference: Iteratif durum güncellemesi (dinamik adaptasyon)")
    
    return model, sense_embeddings, final_accuracy


if __name__ == "__main__":
    model, sense_embeddings, accuracy = train_and_evaluate()
