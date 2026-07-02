"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Ana Yarışma Modülü (main.py)

v2.1.0 resmi bağlantı arayüzüne (26.06.2026) göre yeniden yazıldı: sunucu
kareleri artık TEK TEK verir (bir karenin tahmini gönderilmeden bir sonraki
gelmez). Akış:
1. GET /progress/ ile oturumu tespit et (kaldığı yerden devam).
2. GET /reference/ ile Görev 3 referanslarını (varsa) tek seferde çek.
3. Döngü: sıradaki kareyi al → 3 görevi işle → sonucu gönder → kısa bekle.
"""

import sys
import os
import time

# Windows konsolu eski bir kod sayfası kullanıyorsa veya çıktı yönlendiriliyorsa
# Türkçe/özel karakterler UnicodeEncodeError ile çökebilir. UTF-8'e zorla.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from kaynak_kodlar.sunucu_istemci import SunucuIstemci
from kaynak_kodlar.nesne_tespiti import NesneTespiti
from kaynak_kodlar.pozisyon_kestirimi import PozisyonKestirimi
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme


def _referanslari_yukle(sunucu, eslestirme):
    """
    Görev 3 referanslarını yükler. Önce sunucudan (GET /reference/) dener;
    başarısız olursa lokal 'referanslar/' klasörüne düşer.

    Returns:
        dict: {ref_url: (frame_start_image_url, frame_end_image_url)} —
        lokal fallback durumunda boş dict (pencere bilgisi yok, tüm
        referanslar her karede aranır).
    """
    aktif_ref_pencereleri = {}

    sunucu_referanslar = sunucu.referanslari_al()
    if sunucu_referanslar:
        for ref in sunucu_referanslar:
            ref_url = ref.get("url")
            goruntu_url = ref.get("image_url", "")
            goruntu = sunucu.referans_goruntu_indir(goruntu_url)
            if goruntu is not None:
                eslestirme.referans_yukle(ref_url, goruntu)
                aktif_ref_pencereleri[ref_url] = (
                    ref.get("frame_start_image_url"),
                    ref.get("frame_end_image_url"),
                )
            else:
                print(f"[REFERANS] {ref_url} indirilemedi.")
        return aktif_ref_pencereleri

    referans_klasor = "referanslar"
    if os.path.exists(referans_klasor):
        ref_dosyalar = sorted([f for f in os.listdir(referans_klasor)
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for i, dosya in enumerate(ref_dosyalar):
            yol = os.path.join(referans_klasor, dosya)
            eslestirme.referans_yukle(i + 1, yol)
    else:
        print("[UYARI] Referans klasörü bulunamadı ve sunucudan alınamadı — Görev 3 pasif.")

    return aktif_ref_pencereleri  # boş dict → main döngüsü tüm referansları arar


def _aktif_referanslari_bul(aktif_ref_pencereleri, image_url):
    """
    Karenin image_url'i verilen referans pencerelerinden hangilerinin içine
    düşüyor bulur (v2.1.0 /reference/ mekanizması: frame_start_image_url <=
    image_url <= frame_end_image_url, STRING karşılaştırması).

    Returns:
        None  → pencere bilgisi yok (lokal fallback), eslestir() tüm referansları arasın.
        list  → (boş olabilir) o an aktif referansların anahtarları.
    """
    if not aktif_ref_pencereleri:
        return None
    return [
        url for url, (baslangic, bitis) in aktif_ref_pencereleri.items()
        if baslangic and bitis and baslangic <= image_url <= bitis
    ]


def main():
    print("=" * 60)
    print("  TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI")
    print("  Yarışma Modu Başlatılıyor...")
    print("=" * 60)

    # ============================================================
    # 1. MODÜLLERİ BAŞLAT
    # ============================================================
    sunucu = SunucuIstemci(
        sunucu_url=config.SUNUCU_URL,
        kullanici_url=config.KULLANICI_URL
    )

    nesne_tespit = NesneTespiti(model_yolu=config.NESNE_TESPIT_MODEL)
    pozisyon = PozisyonKestirimi()
    eslestirme = GoruntuEslestirme()

    print("[SISTEM] Tüm modüller hazır.")

    # ============================================================
    # 2. OTURUMU TESPİT ET (kaldığı yerden devam)
    # ============================================================
    bilgi = sunucu.oturumu_baslat()
    if bilgi is None:
        print("[HATA] Sunucuya bağlanılamadı (progress). Program sonlandırılıyor.")
        return
    if not bilgi["oturum_adi"]:
        print("[SISTEM] Aktif oturum yok. Çıkılıyor.")
        return
    if bilgi["tamamlandi"]:
        print(f"[SISTEM] Tüm {bilgi['toplam_kare']} kare zaten gönderilmiş. Yapılacak bir şey yok.")
        return

    toplam_kare = bilgi["toplam_kare"]
    baslangic_index = bilgi["baslangic_index"]
    print(f"[SISTEM] Oturum: {bilgi['oturum_adi']} — "
          f"{baslangic_index}/{toplam_kare}'den devam ediliyor.")

    # ============================================================
    # 3. REFERANS NESNELER (Görev 3)
    # ============================================================
    aktif_ref_pencereleri = _referanslari_yukle(sunucu, eslestirme)

    # ============================================================
    # 4. ANA DÖNGÜ - KARE KARE İŞLE (sunucu tek kare verir, sıralıdır)
    # ============================================================
    islenen = 0
    toplam_sure = 0
    hatalar = 0
    takilma_url = None
    takilma_sayaci = 0
    # Hiç kare işlenmezse (örn. boş kare listesi) özet raporda hata vermesin
    pozisyon_sonuc = {
        "translation_x": config.BASLANGIC_X,
        "translation_y": config.BASLANGIC_Y,
        "translation_z": config.BASLANGIC_Z,
    }

    while True:
        kare_baslangic = time.time()

        # 4a. Sunucunun beklediği tek kareyi al
        kare_verisi = sunucu.siradaki_kareyi_al()
        if kare_verisi is None:
            break

        image_url = kare_verisi["image_url"]
        frame_url = kare_verisi["frame_url"]

        # Aynı kare tahmin gönderilmeden tekrar geliyorsa (gönderim sürekli
        # reddediliyor demektir) sonsuz döngüye girmemek için sonlandır.
        if image_url and image_url == takilma_url:
            takilma_sayaci += 1
            if takilma_sayaci >= 5:
                print(f"[HATA] Kare ilerlemiyor ({image_url}), sonlandırılıyor.")
                break
        else:
            takilma_url = image_url
            takilma_sayaci = 0

        frame = kare_verisi["frame"]
        health = kare_verisi["health_status"]
        ref_x = kare_verisi["translation_x"]
        ref_y = kare_verisi["translation_y"]
        ref_z = kare_verisi["translation_z"]

        # İşlem için temiz kopya (YOLO çizimleri optik akışı bozmasın)
        temiz_kare = frame.copy() if frame is not None else None

        # --------------------------------------------------------
        # 4b. GÖREV 1: NESNE TESPİTİ (%25)
        # --------------------------------------------------------
        detected_objects = nesne_tespit.tespit_et(frame)

        # --------------------------------------------------------
        # 4c. GÖREV 2: POZİSYON KESTİRİMİ (%40)
        # --------------------------------------------------------
        pozisyon_sonuc = pozisyon.guncelle(
            temiz_kare, ref_x, ref_y, ref_z, health
        )
        detected_translations = [pozisyon_sonuc]

        # --------------------------------------------------------
        # 4d. GÖREV 3: GÖRÜNTÜ EŞLEŞTİRME (%25) - sadece bu kare için AKTİF
        # pencerede olan referanslar aranır (v2.1.0 /reference/ mekanizması)
        # --------------------------------------------------------
        aktif_ref_anahtarlari = _aktif_referanslari_bul(aktif_ref_pencereleri, image_url)
        reference_predictions = eslestirme.eslestir(temiz_kare, aktif_ref_anahtarlari)

        # --------------------------------------------------------
        # 4e. SONUCU SUNUCUYA GÖNDER
        # --------------------------------------------------------
        basarili = sunucu.sonuc_gonder(
            frame_url,
            detected_objects,
            detected_translations,
            reference_predictions
        )
        if not basarili:
            print(f"[UYARI] Sonuç gönderilemedi, bu kare değerlendirme dışı kalabilir: {frame_url}")
            hatalar += 1

        islenen += 1
        kare_sure = time.time() - kare_baslangic
        toplam_sure += kare_sure

        # Kendi hız sınırımız (sunucuya yüklenmemek için) - resmi örnekteki
        # MIN_FRAME_INTERVAL ile aynı mantık.
        if kare_sure < config.MIN_KARE_ARALIGI:
            time.sleep(config.MIN_KARE_ARALIGI - kare_sure)

        # İlerleme raporu (her 50 karede bir)
        if islenen % 50 == 0:
            ort_fps = islenen / toplam_sure if toplam_sure > 0 else 0
            print(f"[İLERLEME] {baslangic_index + islenen}/{toplam_kare} kare | "
                  f"FPS: {ort_fps:.1f} | "
                  f"Nesne: {len(detected_objects)} | "
                  f"Poz: ({pozisyon_sonuc['translation_x']:.2f}, "
                  f"{pozisyon_sonuc['translation_y']:.2f}, "
                  f"{pozisyon_sonuc['translation_z']:.2f}) | "
                  f"Health: {health} | "
                  f"Ref.Obj: {len(reference_predictions)}")

    # ============================================================
    # 5. ÖZET RAPOR
    # ============================================================
    print("\n" + "=" * 60)
    print("  OTURUM TAMAMLANDI")
    print("=" * 60)
    print(f"  İşlenen kare   : {islenen}")
    print(f"  Toplam süre     : {toplam_sure:.1f} saniye")
    print(f"  Ortalama FPS    : {islenen/toplam_sure:.2f}" if toplam_sure > 0 else "  Ortalama FPS    : N/A")
    print(f"  Hatalı kare     : {hatalar}")
    print(f"  Son pozisyon    : X={pozisyon_sonuc['translation_x']:.3f}, "
          f"Y={pozisyon_sonuc['translation_y']:.3f}, "
          f"Z={pozisyon_sonuc['translation_z']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
