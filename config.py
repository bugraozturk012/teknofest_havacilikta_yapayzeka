"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Merkezi Konfigürasyon Dosyası

Tüm ayarlar burada toplanır. Yarışma günü sadece bu dosyayı
güncellemek yeterli olacaktır.
"""

# ============================================================
# SUNUCU AYARLARI (Yarışma günü güncellenecek)
# ============================================================
SUNUCU_URL = "http://127.0.0.1:5000"  # Yarışma günü verilecek
KULLANICI_URL = "http://localhost/users/1/"  # Yarışma günü verilecek

# ============================================================
# SUNUCU API YOLLARI
# Kaynak: resmi_arayuz/TAKIM_BAGLANTI_ARAYUZU/src/connection_handler.py
# (v2.1.0, 26.06.2026 - bu yollar "modifiye etmeyin" denen sunucu haberleşme
# kodundan alındı, tahmini değil).
# NOT (2026-07-02): v2.1.0 ile /frames/ ve /translation/ artık TÜM kareleri
# değil, sıradaki TEK kareyi döner (tahmin gönderilmeden ilerlemez). Bkz.
# kaynak_kodlar/sunucu_istemci.py.
# ============================================================
API_AUTH_PATH = "auth/"
API_FRAME_PATH = "frames/"
API_TRANSLATION_PATH = "translation/"
API_RESULT_PATH = "prediction/"
API_PROGRESS_PATH = "progress/"   # Kaldığı yerden devam (frame_index/total_frames/completed)
API_REFERANS_PATH = "reference/"  # Görev 3 - {url, session, image_url, frame_start_image_url, frame_end_image_url, order}

# ============================================================
# KİMLİK DOĞRULAMA (Resmi bağlantı arayüzü Token tabanlı auth kullanıyor)
# ============================================================
OTURUM_ACMA_AKTIF = False   # Sunucu token istiyorsa True yapın
TAKIM_ADI = ""              # KYS'ye kayıtlı takım kullanıcı adı
TAKIM_SIFRE = ""            # KYS'ye kayıtlı takım şifresi

# ============================================================
# HIZ SINIRI (RATE LIMIT) - v2.1.0 resmi örneğine göre güncellendi (2026-07-02)
# Eski: get_frame 5/dk, post_result sabit 80/dk (tüm-liste mimarisi içindi).
# Yeni: /frames/ artık tek kare döndüğü için ~300/dk'ya kadar kaldırıyor;
# post_result limiti artık sunucu tarafında dinamik/yüksek (sabit değil).
# Resmi main.py kendi hızını MIN_FRAME_INTERVAL ile kısıyor, biz de aynısını
# yapıyoruz - limiti sunucuya zorlatmak yerine kendimiz kısıyoruz.
# ============================================================
MIN_KARE_ARALIGI = 0.25   # Saniye - iki kare isteği arasında en az bu kadar bekle (~4 kare/sn tavan)

# ============================================================
# SONUÇ GÖNDERİM FORMATI
# ============================================================
# Şartname Şekil 17 örneğinde ve resmi arayüzde "cls" alanı düz sayı değil,
# bir URL olarak gönderiliyor (örn: "http://.../classes/1/"). Yarışma günü
# gerçek API dokümanı gelince bu bayrağı teyit edip gerekirse False yapın.
SINIF_URL_FORMATINDA_GONDER = True
SINIF_URL_OFSET = 1   # Resmi örnekte sınıf ID'sine +1 uygulanıyor (classes/1/ = Taşıt)

# ============================================================
# MODEL AYARLARI
# ============================================================
NESNE_TESPIT_MODEL = "modeller/best.pt"  # Eğitilmiş YOLO modeli
NESNE_TESPIT_FALLBACK = "yolov8n.pt"     # Yedek model
NESNE_TESPIT_CONF = 0.25                 # Minimum güven eşiği
NESNE_TESPIT_IOU = 0.5                   # NMS IoU eşiği

# ============================================================
# SINIF TANIMLARI (Şartname Tablo 2, 3, 4, 5)
# ============================================================
# Sınıf ID'leri
SINIF_TASIT = 0
SINIF_INSAN = 1
SINIF_UAP = 2
SINIF_UAI = 3

# Hareket durumu (sadece taşıt için geçerli)
HAREKET_YOK = -1       # Taşıt değilse
HAREKET_HAREKETSIZ = 0
HAREKET_HAREKETLI = 1

# İniş durumu (sadece UAP/UAİ için geçerli)
INIS_YOK = -1          # İniş alanı değilse
INIS_UYGUN_DEGIL = 0
INIS_UYGUN = 1

# ============================================================
# POZİSYON KESTİRİMİ AYARLARI
# ============================================================
# Başlangıç pozisyonu (Şartname: x0=0, y0=0, z0=0)
BASLANGIC_X = 0.0
BASLANGIC_Y = 0.0
BASLANGIC_Z = 0.0

# Kamera parametreleri (yarışmada paylaşılacak, şimdilik tahmini)
KAMERA_FOV_YATAY = 80.0   # derece
KAMERA_FOV_DIKEY = 60.0   # derece

# Optical Flow parametreleri
OF_WIN_SIZE = (21, 21)
OF_MAX_LEVEL = 3
OF_MAX_CORNERS = 200
OF_QUALITY_LEVEL = 0.01
OF_MIN_DISTANCE = 10

# Hareketlilik tespiti için piksel eşiği
# (Kamera ego-motion kompanze edildikten sonra kalan hareket)
HAREKET_PIKSEL_ESIGI = 5.0

# UAP/UAİ iniş kontrolünde alan etrafına eklenen pay oranı.
# Şartname (Şekil 11): çekim açısı nedeniyle alana yakın ama üzerinde
# olmayan cisimler de "inişe uygun değil" sayılmalı. Bu oran, iniş alanı
# bbox'unu kontrol amaçlı genişletir (raporlanan alan bbox'u değişmez).
INIS_KONTROL_GENISLETME_ORANI = 0.15

# ============================================================
# GÖRÜNTÜ EŞLEŞTIRME AYARLARI
# ============================================================
ESLESTIRME_MIN_ESLESME = 20   # Minimum iyi eşleşme sayısı (gerçek video verisiyle ölçüldü: test/dagilim_tani.py - eşik 10'da rastgele doku eşleşmeleri %16 oranla geçiyordu, 20'de %1.2'ye düşüyor)
ESLESTIRME_RATIO_TEST = 0.75  # Lowe's ratio test eşiği

# ============================================================
# SUNUCU İLETİŞİM AYARLARI
# ============================================================
ISTEK_TIMEOUT = 10       # Saniye
ISTEK_MAX_RETRY = 3      # Maksimum tekrar deneme
ISTEK_RETRY_BEKLEME = 0.5  # Tekrar denemeler arası bekleme (saniye)

# ============================================================
# EĞİTİM AYARLARI (egitim.py tarafından kullanılır)
# ============================================================
EGITIM_MODEL_TIPI = "modeller/yolov8s.pt"   # Lokal kopya varsa tekrar indirmez
EGITIM_DATA_YAML = "dataset/data.yaml"   # Proje köküne göre relatif yol; kendi veri setinize göre güncelleyin
EGITIM_EPOCH = 150
EGITIM_IMG_BOYUTU = 640
EGITIM_BATCH = 32
EGITIM_PROJE_ADI = "teknofest_2026"
EGITIM_CALISTIRMA_ADI = "egitim_v1"
