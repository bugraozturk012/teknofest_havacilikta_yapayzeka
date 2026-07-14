"""
TEKNOFEST 2026 - Model Eğitim Modülü

Şartname sınıflarına uygun YOLO eğitimi:
- Sınıf 0: Taşıt (tüm motorlu araçlar, raylı taşıtlar, deniz taşıtları)
- Sınıf 1: İnsan
- Sınıf 2: UAP (Uçan Araba Park alanı)
- Sınıf 3: UAİ (Uçan Ambulans İniş alanı)

Veri seti yapısı (data.yaml):
    train: dataset/train/images
    val: dataset/val/images
    nc: 4
    names: ['tasit', 'insan', 'uap', 'uai']
"""

from ultralytics import YOLO
import torch
import os
import sys
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================
# AYARLAR (config.py içinden okunur — taşınabilirlik için)
# ============================================================
MODEL_TIPI = config.EGITIM_MODEL_TIPI       # n=hızlı, s=dengeli, m=doğru
DATA_YAML = config.EGITIM_DATA_YAML         # Proje köküne göre relatif yol; kendi veri setinize göre config.py'de güncelleyin
EPOCH = config.EGITIM_EPOCH
IMG_BOYUTU = config.EGITIM_IMG_BOYUTU
BATCH = config.EGITIM_BATCH                 # GPU VRAM'inize göre config.py'de ayarlayın
PROJE_ADI = config.EGITIM_PROJE_ADI
CALISTIRMA_ADI = config.EGITIM_CALISTIRMA_ADI


def data_yaml_olustur():
    """Eğer data.yaml yoksa oluştur."""
    if os.path.exists(DATA_YAML):
        return True
    
    os.makedirs("dataset", exist_ok=True)
    
    data = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['tasit', 'insan', 'uap', 'uai']
    }
    
    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"[EGITIM] {DATA_YAML} oluşturuldu.")
    print("[EGITIM] dataset/train/images ve dataset/valid/images klasörlerine")
    print("         görüntüleri ve YOLO formatında etiketleri koyun.")
    return True


def egitimi_baslat():
    print("\n" + "=" * 50)
    print("  TEKNOFEST 2026 - MODEL EĞİTİMİ")
    print("=" * 50)

    # GPU kontrolü
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu} ({vram:.1f} GB)")
        device = 0
    else:
        print("[GPU] Bulunamadı, CPU kullanılacak (yavaş olacak)")
        device = "cpu"

    # Data.yaml kontrolü
    if not os.path.exists(DATA_YAML):
        data_yaml_olustur()
        print("\n[HATA] Eğitim verisi henüz hazır değil!")
        print("Adımlar:")
        print("1. Hava görüntülerini toplayın (VisDrone, DOTA, kendi veriler)")
        print("2. 4 sınıfı etiketleyin (CVAT veya LabelImg ile)")
        print("3. YOLO formatında dataset/train ve dataset/val'a koyun")
        print("4. Bu scripti tekrar çalıştırın")
        return

    # Yarım kalan bir eğitim varsa (çökme/kapanma sonrası last.pt diskte kaldıysa) ondan devam et
    son_checkpoint = f"runs/detect/modeller/{PROJE_ADI}/{CALISTIRMA_ADI}/weights/last.pt"
    devam_ediyor = os.path.exists(son_checkpoint)

    def taze_egitimi_baslat():
        print(f"[EGITIM] Model: {MODEL_TIPI}")
        taze_model = YOLO(MODEL_TIPI)
        return taze_model.train(
            data=DATA_YAML,
            epochs=EPOCH,
            imgsz=IMG_BOYUTU,
            batch=BATCH,
            name=CALISTIRMA_ADI,
            device=device,
            project=f"modeller/{PROJE_ADI}",
            exist_ok=True,
            verbose=True,
            workers=8,
            patience=30,        # 30 epoch iyileşme yoksa dur
            save_period=10,     # Her 10 epoch'ta kaydet
            augment=True,       # Veri artırma
            hsv_h=0.015,        # Renk augmentasyonu
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,         # Rotasyon
            translate=0.1,
            scale=0.5,
            flipud=0.5,         # Dikey flip (hava görüntüsü için önemli)
            fliplr=0.5,         # Yatay flip
            mosaic=1.0,         # Mozaik augmentasyonu
            mixup=0.1,          # MixUp
        )

    try:
        if devam_ediyor:
            print(f"[EGITIM] Yarım kalan eğitim bulundu, kaldığı yerden devam ediliyor: {son_checkpoint}")
            try:
                model = YOLO(son_checkpoint)
                results = model.train(resume=True)
            except Exception as devam_hatasi:
                # Eğitim zaten tamamlanmışsa Ultralytics resume'u reddeder — bu durumda sıfırdan başla
                print(f"[EGITIM] Kaldığı yerden devam edilemedi ({devam_hatasi}), sıfırdan başlanıyor.")
                results = taze_egitimi_baslat()
        else:
            results = taze_egitimi_baslat()

        print(f"\n[EGITIM] Sonuçlar kaydedildi: {results.save_dir}")

        # En iyi modeli kopyala
        best_model = f"runs/detect/modeller/{PROJE_ADI}/{CALISTIRMA_ADI}/weights/best.pt"
        if os.path.exists(best_model):
            import shutil
            shutil.copy2(best_model, "modeller/best.pt")
            print("\n[BASARI] En iyi model: modeller/best.pt")

        print("[BASARI] Eğitim tamamlandı!")
        
    except Exception as e:
        print(f"\n[HATA] {e}")
        if "Out of memory" in str(e) or "CUDA" in str(e):
            print("[IPUCU] BATCH değerini düşürün (8 veya 4)")


if __name__ == '__main__':
    egitimi_baslat()
