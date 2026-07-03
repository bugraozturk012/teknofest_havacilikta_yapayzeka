"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Ana yarışma döngüsü.

v2.1.0 API'ye göre: sunucu kareleri tek tek veriyor, biri gönderilmeden diğeri
gelmiyor. Akış: /progress/ ile kaldığı yerden devam -> referansları çek ->
kare al, 3 görevi çalıştır, sonucu gönder, tekrarla.
"""

import sys
import os
import time

# konsol çıktısı yönlendirilince Türkçe karakterler UnicodeEncodeError atabiliyor
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
    Görev 3 referanslarını yükler: önce sunucudan, olmazsa referanslar/ klasöründen.

    Döner: {ref_url: (pencere_baslangic, pencere_bitis)} - lokal fallback'te
    boş dict, bu durumda eslestir() her karede tüm referansları arar.
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
            # anahtar string olmalı - sunucudan gelen referanslarda "reference" alanı URL (string),
            # eslestir() çıktısı bu anahtarı aynen "reference" alanına koyup sunucuya gönderiyor
            eslestirme.referans_yukle(str(i + 1), yol)
    else:
        print("[UYARI] Referans klasörü bulunamadı ve sunucudan alınamadı — Görev 3 pasif.")

    return aktif_ref_pencereleri


def _aktif_referanslari_bul(aktif_ref_pencereleri, image_url):
    """
    Bu karenin hangi referans pencere(ler)ine düştüğünü bulur
    (frame_start_image_url <= image_url <= frame_end_image_url, string karşılaştırma).

    None dönerse pencere bilgisi yok demektir, eslestir() tümünü arasın.
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

    sunucu = SunucuIstemci(
        sunucu_url=config.SUNUCU_URL,
        kullanici_url=config.KULLANICI_URL
    )

    nesne_tespit = NesneTespiti(model_yolu=config.NESNE_TESPIT_MODEL)
    pozisyon = PozisyonKestirimi()
    eslestirme = GoruntuEslestirme()

    print("[SISTEM] Tüm modüller hazır.")

    # kaldığı yerden devam
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

    aktif_ref_pencereleri = _referanslari_yukle(sunucu, eslestirme)

    islenen = 0
    toplam_sure = 0
    hatalar = 0
    takilma_url = None
    takilma_sayaci = 0
    # hiç kare işlenmese bile özet rapor patlamasın diye
    pozisyon_sonuc = {
        "translation_x": config.BASLANGIC_X,
        "translation_y": config.BASLANGIC_Y,
        "translation_z": config.BASLANGIC_Z,
    }

    while True:
        kare_baslangic = time.time()

        kare_verisi = sunucu.siradaki_kareyi_al()
        if kare_verisi is None:
            break

        image_url = kare_verisi["image_url"]
        frame_url = kare_verisi["frame_url"]

        # aynı kare tekrar tekrar geliyorsa gönderim reddediliyor demektir, sonsuz döngüye girme
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

        temiz_kare = frame.copy() if frame is not None else None  # YOLO çizimleri optik akışa karışmasın

        # Görev 1: Nesne Tespiti (%25)
        detected_objects = nesne_tespit.tespit_et(frame)

        # Görev 2: Pozisyon Kestirimi (%40)
        pozisyon_sonuc = pozisyon.guncelle(
            temiz_kare, ref_x, ref_y, ref_z, health
        )
        detected_translations = [pozisyon_sonuc]

        # Görev 3: Görüntü Eşleştirme (%25) - sadece bu karede aktif olan referanslar aranır
        aktif_ref_anahtarlari = _aktif_referanslari_bul(aktif_ref_pencereleri, image_url)
        reference_predictions = eslestirme.eslestir(temiz_kare, aktif_ref_anahtarlari)

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

        # sunucuya yüklenmemek için kendi hızımızı kısıyoruz
        if kare_sure < config.MIN_KARE_ARALIGI:
            time.sleep(config.MIN_KARE_ARALIGI - kare_sure)

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
