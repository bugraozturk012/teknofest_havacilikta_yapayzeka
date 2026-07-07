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
# DİKKAT: best.pt bulunamazsa buraya SADECE yerelde zaten var olan bir dosya
# yazın. Yarışma ağında internet olmayacağı için nesne_tespiti.py, bu dosya da
# yoksa YOLO()'nun otomatik indirmeyi denemesini (donma/çökme riski) engelleyip
# hata fırlatır. Ayrıca bu model COCO sınıflarıyla eğitili - yarışma sınıflarıyla
# (taşıt/insan/UAP/UAİ) uyumlu DEĞİL, sadece geliştirme/test amaçlıdır.
NESNE_TESPIT_FALLBACK = "modeller/yolov8s.pt"
NESNE_TESPIT_CONF = 0.25
NESNE_TESPIT_IOU = 0.5

# İniş alanı (UAP/UAİ) üzerinde bilinen 4 sınıfın (taşıt/insan/UAP/UAİ) dışında
# tespit edilemeyen bir "yabancı obje" (mont, kutu vb. - Şartname Şekil 10/11)
# olup olmadığını kontrol etmek için ikinci, genel (COCO) bir model kullanılır.
# Sadece ped bölgesi kırpılıp bu modele veriliyor, her karede tam kare taranmıyor.
# Yerelde yoksa bu kontrol sessizce pasif kalır (sadece bilinen 4 sınıfla devam edilir).
INIS_YABANCI_OBJE_MODEL = "modeller/yolov8s.pt"
INIS_YABANCI_OBJE_CONF = 0.4
# Bulunan kutunun pedin kendi alanına oranı bunun altındaysa "küçük/yerel bir
# cisim" sayılır (pedin işaretinin/dairenin kendisinin yanlışlıkla yeniden
# tespit edilip "yabancı obje" sanılmasını önler).
INIS_YABANCI_OBJE_ALAN_ORANI_ESIK = 0.5

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

# Gerçek 2026 kamera kalibrasyonu (resmi_arayuz/Kamera_Kalibrasyon/
# Kamera_Kalibrasyon_Parametreleri_2026.txt). Anahtar: kare (genislik, yukseklik)
# piksel - deger: (fx, fy, cx, cy). Kare çözünürlüğüne göre otomatik seçilir
# (pozisyon_kestirimi.py:_odak_uzakligi_bul), tabloda olmayan bir çözünürlük
# gelirse ampirik ölçek faktörüne geri düşülür.
KAMERA_KALIBRASYON = {
    (640, 512): (731.7965, 732.0172, 319.2367, 251.2424),      # Termal
    (4000, 3000): (2792.2, 2795.2, 1988.0, 1562.2),            # RGB 4K
    (1920, 1080): (1389.7, 1387.1, 954.007, 558.896),          # RGB 1080p (test videomuz bu)
}

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
# Resmi örnek 5 deneme + üstel artan bekleme kullanıyor (0.1->1.6s), biz de aynı
# stratejiye geçtik - geçici ağ dalgalanmasını "oturum bitti" sanıp erken çıkma riskini azaltır.
ISTEK_TIMEOUT = 10
ISTEK_MAX_RETRY = 5
ISTEK_RETRY_BEKLEME = 0.5  # ilk bekleme, her denemede 2 katına çıkar

# --- Eğitim (egitim.py) ---
EGITIM_MODEL_TIPI = "modeller/yolov8s.pt"
EGITIM_DATA_YAML = "dataset/data.yaml"
EGITIM_EPOCH = 150
EGITIM_IMG_BOYUTU = 640
EGITIM_BATCH = 32
EGITIM_PROJE_ADI = "teknofest_2026"
EGITIM_CALISTIRMA_ADI = "egitim_v1"
