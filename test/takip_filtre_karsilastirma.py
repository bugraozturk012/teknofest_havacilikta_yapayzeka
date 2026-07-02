"""
Takip tutarliligi filtresinin (NESNE_TESPIT_TAKIP_ONAY_ESIGI) etkisini olcer.

2250 kareyi video sirasina gore (tracking zamana bagli, paralel calisilamaz)
esik=1 (filtre kapali) ve esik=3 (yeni filtre) ile iki kez isleyip
karsilastirir: toplam tespit sayisi, bilinen supheli FP (frame_000067)
eleniyor mu, bilinen gercek arac hala raporlaniyor mu.
"""

import sys
import os
import json

sys.path.append('..')
import config

# Esigi calisma zamaninda degistirebilmek icin config'i patch'liyoruz
import importlib

KARELER_DIR = "kareler"
CIKTI_DIR = "degerlendirme_ciktilari/takip_filtre_karsilastirma"
os.makedirs(CIKTI_DIR, exist_ok=True)

import cv2


def tam_gecis_yap(esik):
    config.NESNE_TESPIT_TAKIP_ONAY_ESIGI = esik

    # NesneTespiti sinifini her seferinde taze import/instantiate ediyoruz
    # (self.gecmis_konumlar / takip_gorulme_sayisi sifirlansin diye)
    if "kaynak_kodlar.nesne_tespiti" in sys.modules:
        importlib.reload(sys.modules["kaynak_kodlar.nesne_tespiti"])
    from kaynak_kodlar.nesne_tespiti import NesneTespiti

    dedektor = NesneTespiti(model_yolu="../" + config.NESNE_TESPIT_MODEL)

    kare_dosyalari = sorted(os.listdir(KARELER_DIR))
    toplam_tespit = 0
    frame_000067_var_mi = False
    frame_000007_civari_ilk_gorulme = None  # ilk kez Tasit raporlandigi kare index'i (0007-0029 penceresinde)

    for i, dosya in enumerate(kare_dosyalari):
        yol = os.path.join(KARELER_DIR, dosya)
        frame = cv2.imread(yol)
        if frame is None:
            continue

        sonuclar = dedektor.tespit_et(frame)
        toplam_tespit += len(sonuclar)

        if dosya == "frame_000067.jpg" and len(sonuclar) > 0:
            frame_000067_var_mi = True

        if 7 <= i <= 29 and frame_000007_civari_ilk_gorulme is None:
            if any(s["cls"] == str(config.SINIF_TASIT) for s in sonuclar):
                frame_000007_civari_ilk_gorulme = i

        if (i + 1) % 500 == 0:
            print(f"    [esik={esik}] ... {i+1}/{len(kare_dosyalari)} kare islendi")

    return {
        "esik": esik,
        "toplam_tespit": toplam_tespit,
        "frame_000067_fp_hala_var_mi": frame_000067_var_mi,
        "frame_0007_0029_penceresinde_ilk_tasit_karesi": frame_000007_civari_ilk_gorulme,
    }


print("=== GECIS 1: esik=1 (filtre KAPALI - eski davranis) ===")
sonuc_kapali = tam_gecis_yap(esik=1)
print(sonuc_kapali)

print("\n=== GECIS 2: esik=3 (filtre ACIK - yeni davranis) ===")
sonuc_acik = tam_gecis_yap(esik=3)
print(sonuc_acik)

print("\n=== KARSILASTIRMA ===")
print(f"Toplam tespit: {sonuc_kapali['toplam_tespit']} (filtre kapali) -> "
      f"{sonuc_acik['toplam_tespit']} (filtre acik), "
      f"fark: {sonuc_kapali['toplam_tespit'] - sonuc_acik['toplam_tespit']}")
print(f"frame_000067 (supheli FP) raporlaniyor mu? "
      f"Filtre kapali: {sonuc_kapali['frame_000067_fp_hala_var_mi']}, "
      f"Filtre acik: {sonuc_acik['frame_000067_fp_hala_var_mi']}")
print(f"frame 0007-0029 penceresinde ilk Tasit raporu: "
      f"Filtre kapali: kare {sonuc_kapali['frame_0007_0029_penceresinde_ilk_tasit_karesi']}, "
      f"Filtre acik: kare {sonuc_acik['frame_0007_0029_penceresinde_ilk_tasit_karesi']}")

with open(os.path.join(CIKTI_DIR, "karsilastirma.json"), "w", encoding="utf-8") as f:
    json.dump({"filtre_kapali": sonuc_kapali, "filtre_acik": sonuc_acik}, f, ensure_ascii=False, indent=2)
print(f"\nKaydedildi: {CIKTI_DIR}/karsilastirma.json")
