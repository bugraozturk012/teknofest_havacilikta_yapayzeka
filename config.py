"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Merkezi konfigürasyon. Yarışma günü sadece burası güncellenecek.
"""

# --- Sunucu ---
SUNUCU_URL = "http://127.0.0.1:5000"  # yarışma günü değişecek
KULLANICI_URL = "http://localhost/users/1/"  # yarışma günü değişecek

# API yolları - resmi_arayuz/TAKIM_BAGLANTI_ARAYUZU/src/connection_handler.py'den (v2.1.0, dokunma)
# v2.1.0'dan beri frames/translation sıradaki TEK kareyi dönüyor, tümünü değil (bkz. sunucu_istemci.py)
API_AUTH_PATH = "auth/"
API_FRAME_PATH = "frames/"
API_TRANSLATION_PATH = "translation/"
API_RESULT_PATH = "prediction/"
API_PROGRESS_PATH = "progress/"   # kaldığı yerden devam için
API_REFERANS_PATH = "reference/"  # Görev 3 referansları

# --- Kimlik doğrulama ---
OTURUM_ACMA_AKTIF = False   # sunucu token istiyorsa True
TAKIM_ADI = ""
TAKIM_SIFRE = ""

# --- Hız sınırı ---
# Resmi örnek kendi hızını MIN_FRAME_INTERVAL ile kısıyor, biz de aynısını yapıyoruz -
# sunucuya limiti zorlatmak yerine kendimiz frenliyoruz.
MIN_KARE_ARALIGI = 0.25   # iki kare isteği arası min bekleme (sn)

# --- Sonuç formatı ---
# "cls" düz sayı değil URL olarak gidiyor (classes/1/). Yarışma günü teyit et.
SINIF_URL_FORMATINDA_GONDER = True
SINIF_URL_OFSET = 1   # classes/1/ = Taşıt

# --- Model ---
NESNE_TESPIT_MODEL = "modeller/best.pt"
NESNE_TESPIT_FALLBACK = "yolov8n.pt"
NESNE_TESPIT_CONF = 0.25
NESNE_TESPIT_IOU = 0.5

# Bir nesne raporlanmadan önce aynı takip_id ile en az bu kadar ardışık karede
# görülmeli. Tek karelik gürültü tespitlerini eler, gerçek nesneleri etkilemez.
NESNE_TESPIT_TAKIP_ONAY_ESIGI = 3

# --- Sınıflar (Şartname Tablo 2-5) ---
SINIF_TASIT = 0
SINIF_INSAN = 1
SINIF_UAP = 2
SINIF_UAI = 3

HAREKET_YOK = -1       # taşıt değilse
HAREKET_HAREKETSIZ = 0
HAREKET_HAREKETLI = 1

INIS_YOK = -1          # iniş alanı değilse
INIS_UYGUN_DEGIL = 0
INIS_UYGUN = 1

# --- Pozisyon kestirimi ---
BASLANGIC_X = 0.0
BASLANGIC_Y = 0.0
BASLANGIC_Z = 0.0

# kamera parametreleri, yarışmada paylaşılacak - şimdilik tahmini
KAMERA_FOV_YATAY = 80.0
KAMERA_FOV_DIKEY = 60.0

OF_WIN_SIZE = (21, 21)
OF_MAX_LEVEL = 3
OF_MAX_CORNERS = 200
OF_QUALITY_LEVEL = 0.01
OF_MIN_DISTANCE = 10

HAREKET_PIKSEL_ESIGI = 5.0  # ego-motion kompanzasyonu sonrası hareket eşiği

# iniş alanı çevresine pay ekler, yakın ama üstte olmayan cisimleri de yakalasın diye (Şekil 11)
INIS_KONTROL_GENISLETME_ORANI = 0.15

# --- Görüntü eşleştirme ---
ESLESTIRME_MIN_ESLESME = 20   # 10'da rastgele doku eşleşmeleri de geçiyordu, 20 temiz
ESLESTIRME_RATIO_TEST = 0.75  # Lowe's ratio test

# SIFT'i tam çözünürlükte çalıştırmak GPU kullanımından sonra (DINOv2) laptopta
# çok yavaşlıyordu (CPU/GPU güç paylaşımı). 0.33'te aynı keypoint sayısı çok daha
# hızlı çıkıyor, koordinatlar sonra orijinal karaya geri ölçekleniyor.
ESLESTIRME_SIFT_OLCEK = 0.33

# --- Sunucu iletişimi ---
ISTEK_TIMEOUT = 10
ISTEK_MAX_RETRY = 3
ISTEK_RETRY_BEKLEME = 0.5

# --- Eğitim (egitim.py) ---
EGITIM_MODEL_TIPI = "modeller/yolov8s.pt"
EGITIM_DATA_YAML = "dataset/data.yaml"
EGITIM_EPOCH = 150
EGITIM_IMG_BOYUTU = 640
EGITIM_BATCH = 32
EGITIM_PROJE_ADI = "teknofest_2026"
EGITIM_CALISTIRMA_ADI = "egitim_v1"
