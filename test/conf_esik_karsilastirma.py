"""
NESNE_TESPIT_CONF esigini 0.25 -> 0.35 yukseltmenin etkisini olcer.

Ground-truth yok, o yuzden ayni 2250 karede hem 0.25 hem 0.35 ile cikarim
yapip farki (kaybolan tespitler, sifira dusen kareler) topluyoruz. Kaybolan
tespitlerden ornek kareler manuel inceleme icin kaydedilir.
"""

import sys
import os
import json

sys.path.append('..')
import config

from ultralytics import YOLO
import cv2

KARELER_DIR = "kareler"
CIKTI_DIR = "degerlendirme_ciktilari/conf035_karsilastirma"
os.makedirs(CIKTI_DIR, exist_ok=True)

CONF_ESKI = 0.25
CONF_YENI = 0.35

model = YOLO("../" + config.NESNE_TESPIT_MODEL)
isim_map = model.names

kare_dosyalari = sorted(os.listdir(KARELER_DIR))
print(f"Toplam kare: {len(kare_dosyalari)}")

toplam_025 = 0
toplam_035 = 0
sinif_kaybi = {}
kaybolan_tespit_ornekleri = []  # (dosya, cls, conf)
kareler_035_sifira_duesen = []  # 0.25'te >0 iken 0.35'te 0 olan kareler

for i, dosya in enumerate(kare_dosyalari):
    yol = os.path.join(KARELER_DIR, dosya)
    frame = cv2.imread(yol)
    if frame is None:
        continue

    sonuc_025 = model.predict(frame, conf=CONF_ESKI, iou=config.NESNE_TESPIT_IOU, verbose=False)[0]
    boxes_025 = sonuc_025.boxes
    n_025 = 0 if boxes_025 is None else len(boxes_025)
    toplam_025 += n_025

    sonuc_035 = model.predict(frame, conf=CONF_YENI, iou=config.NESNE_TESPIT_IOU, verbose=False)[0]
    boxes_035 = sonuc_035.boxes
    n_035 = 0 if boxes_035 is None else len(boxes_035)
    toplam_035 += n_035

    if n_025 > 0 and n_035 == 0:
        kareler_035_sifira_duesen.append(dosya)

    if n_025 > n_035 and boxes_025 is not None:
        # 0.35'te kaybolan tespitleri bul: 0.25 conf araligindaki (0.25-0.35) kutular
        confs_025 = boxes_025.conf.cpu().tolist()
        clss_025 = boxes_025.cls.int().cpu().tolist()
        for c, cl in zip(confs_025, clss_025):
            if CONF_ESKI <= c < CONF_YENI:
                ad = isim_map.get(cl, str(cl))
                sinif_kaybi[ad] = sinif_kaybi.get(ad, 0) + 1
                if len(kaybolan_tespit_ornekleri) < 30:
                    kaybolan_tespit_ornekleri.append((dosya, ad, round(c, 3)))

    if (i + 1) % 250 == 0:
        print(f"  ... {i+1}/{len(kare_dosyalari)} kare islendi "
              f"(su ana kadar 0.25={toplam_025}, 0.35={toplam_035})")

print("\n=== SONUC ===")
print(f"Toplam tespit (conf={CONF_ESKI}): {toplam_025}")
print(f"Toplam tespit (conf={CONF_YENI}): {toplam_035}")
print(f"Fark: {toplam_025 - toplam_035} tespit kayboldu (%{100*(toplam_025-toplam_035)/max(toplam_025,1):.1f})")
print(f"\nSinif bazinda kaybolan tespit sayisi (0.25<=conf<0.35 araligi):")
for ad, cnt in sorted(sinif_kaybi.items(), key=lambda x: -x[1]):
    print(f"  {ad}: {cnt}")
print(f"\n0.25'te tespitliyken 0.35'te SIFIR tespite duesen kare sayisi: {len(kareler_035_sifira_duesen)}")
print(f"Bu kareler (ilk 20): {kareler_035_sifira_duesen[:20]}")

ozet = {
    "conf_eski": CONF_ESKI,
    "conf_yeni": CONF_YENI,
    "toplam_tespit_eski": toplam_025,
    "toplam_tespit_yeni": toplam_035,
    "kaybolan_tespit_sayisi": toplam_025 - toplam_035,
    "sinif_bazinda_kayip": sinif_kaybi,
    "sifira_duesen_kare_sayisi": len(kareler_035_sifira_duesen),
    "sifira_duesen_kareler": kareler_035_sifira_duesen,
    "kaybolan_tespit_ornekleri": kaybolan_tespit_ornekleri,
}
with open(os.path.join(CIKTI_DIR, "karsilastirma.json"), "w", encoding="utf-8") as f:
    json.dump(ozet, f, ensure_ascii=False, indent=2)
print(f"\nOzet kaydedildi: {CIKTI_DIR}/karsilastirma.json")

# Gorsel inceleme icin: sifira duesen kareleri hem 0.25 hem 0.35 anotasyonlu kaydet
def kare_kaydet(dosya_adi, conf_esigi, etiket):
    yol = os.path.join(KARELER_DIR, dosya_adi)
    frame = cv2.imread(yol)
    if frame is None:
        return
    sonuc = model.predict(frame, conf=conf_esigi, iou=config.NESNE_TESPIT_IOU, verbose=False)[0]
    anotasyonlu = sonuc.plot()
    cikti_yol = os.path.join(CIKTI_DIR, f"{etiket}_{dosya_adi}")
    cv2.imwrite(cikti_yol, anotasyonlu)

for dosya in kareler_035_sifira_duesen[:10]:
    kare_kaydet(dosya, CONF_ESKI, "eski025")
    kare_kaydet(dosya, CONF_YENI, "yeni035")

print(f"Gorsel karsilastirma ornekleri kaydedildi (ilk 10 sifira-duesen kare, eski025_/yeni035_ etiketli): {CIKTI_DIR}/")
