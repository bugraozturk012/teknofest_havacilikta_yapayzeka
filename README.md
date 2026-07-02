# TEKNOFEST 2026 — Havacılıkta Yapay Zeka

Bu proje, TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması'nın üç görevini
(Nesne Tespiti, Pozisyon Kestirimi, Görüntü Eşleştirme) gerçekleştiren
yarışmacı yazılımıdır.

## Klasör Yapısı

```
Teknofest_2026_Proje/
├── config.py                  # Tüm ayarlar tek dosyada (yarışma günü burayı güncelleyin)
├── main.py                     # Yarışma günü çalıştırılacak ana program
├── egitim.py                   # YOLO model eğitim betiği
├── dinov2_indir.py              # DINOv2 ağırlıklarını indirir (tek seferlik, offline kurulum için)
├── requirements.txt
├── .gitignore
│
├── kaynak_kodlar/               # Asıl algoritmalar
│   ├── sunucu_istemci.py        # Yarışma sunucusuyla iletişim (auth, rate-limit, JSON formatı)
│   ├── nesne_tespiti.py         # Görev 1: Taşıt/İnsan/UAP/UAİ tespiti
│   ├── pozisyon_kestirimi.py    # Görev 2: Kalman + Optical Flow ile pozisyon kestirimi
│   └── goruntu_eslestirme.py    # Görev 3: DINOv2 + SIFT hibrit referans nesne eşleştirme
│
├── test/                        # Yarışma sunucusu olmadan test araçları
│   ├── sahte_sunucu.py          # Flask tabanlı sahte yarışma sunucusu (python test/sahte_sunucu.py)
│   ├── test_lokal.py            # Webcam/video ile canlı görselleştirme testi
│   └── video.mp4                # (siz ekleyin) test videosu
│
├── modeller/                    # Model ağırlıkları
│   ├── best.pt                  # Eğitilmiş YOLO modeli (config.NESNE_TESPIT_MODEL)
│   ├── yolov8s.pt                # Eğitim için temel model (egitim.py)
│   ├── dinov2_vits14.pth         # DINOv2 ağırlıkları (Görev 3)
│   └── dinov2_repo/              # DINOv2 kaynak kodu (offline çalışma için)
│
├── dataset/                     # Eğitim veri seti (bkz. dataset/README.md)
├── referanslar/                  # Görev 3 referans nesne görselleri (oturum başında)
├── runs/                         # Ultralytics eğitim çıktıları (otomatik üretilir)
├── venv312/                      # Proje sanal ortamı (bkz. Kurulum)
└── resmi_arayuz/                 # TEKNOFEST'in resmi referans bağlantı arayüzü (değiştirilmez)
```

## Kurulum

Proje kendi sanal ortamını (`venv312`) kullanır — CUDA'lı PyTorch zaten
kurulu (RTX 4070 ile test edildi).

```powershell
cd Teknofest_2026_Proje
.\venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

VS Code'da sağ alttan Python yorumlayıcısını `venv312` olarak seçin
(`Ctrl+Shift+P` → "Python: Select Interpreter").

GPU/CUDA durumunu doğrulamak için:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Kullanım

| Amaç | Komut |
|---|---|
| Model eğitimi | `python egitim.py` (önce `dataset/` klasörünü doldurun) |
| Yarışma günü | `python main.py` (önce `config.py`'deki sunucu bilgilerini güncelleyin) |
| Lokal test sunucusu | `python test/sahte_sunucu.py` (sonra başka bir terminalde `python main.py`) |
| Webcam/video görselleştirme testi | `python test/test_lokal.py` |

## Yarışma Günü Kontrol Listesi (`config.py`)

- [ ] `SUNUCU_URL`, `KULLANICI_URL` — yarışma sunucusu bilgileri
- [ ] `API_AUTH_PATH`, `API_FRAME_PATH`, `API_RESULT_PATH`, `API_REFERANS_PATH` — gerçek API dokümanına göre teyit edin
- [ ] `OTURUM_ACMA_AKTIF`, `TAKIM_ADI`, `TAKIM_SIFRE` — sunucu token istiyorsa doldurun
- [ ] `SINIF_URL_FORMATINDA_GONDER` — `cls` alanının gerçek formatını teyit edin
- [ ] `modeller/best.pt` — güncel eğitilmiş model burada olmalı

## Notlar

- `resmi_arayuz/` klasörü TEKNOFEST'in GitHub'dan paylaştığı referans
  istemcidir; kendi kodumuz onu birebir kullanmaz ama API davranışını
  anlamak için referans alınmıştır.
- Algoritmanın çalışma hızı şartnameye göre puanlama kriteri değildir.
