"""
best.pt model kalitesi degerlendirmesi (etiketli veri olmadan).

Etiketli (ground-truth) bir val seti olmadigi icin klasik mAP hesaplanamiyor.
Bunun yerine test/kareler/ icindeki 2250 gercek TEKNOFEST karesi uzerinde
cikarim yapilip su istatistikler toplanir:
- Kare basina tespit sayisi (sinif bazinda)
- Guven (confidence) skoru dagilimi
- Hic tespit olmayan / asiri tespit olan supheli kareler
- Gorsel inceleme icin ornek (en az / en cok / random) anotasyonlu kareler kaydedilir
"""

import sys
import os
import json
import statistics

sys.path.append('..')
import config

from ultralytics import YOLO
import cv2

KARELER_DIR = "kareler"
CIKTI_DIR = "degerlendirme_ciktilari"
os.makedirs(CIKTI_DIR, exist_ok=True)

model = YOLO("../" + config.NESNE_TESPIT_MODEL)

kare_dosyalari = sorted(os.listdir(KARELER_DIR))
print(f"Toplam kare: {len(kare_dosyalari)}")

per_frame_count = []
per_class_count = {0: 0, 1: 0, 2: 0, 3: 0}
all_confs = []
sifir_tespit_kareler = []
asiri_tespit_kareler = []  # >20 tespit supheli olabilir

for i, dosya in enumerate(kare_dosyalari):
    yol = os.path.join(KARELER_DIR, dosya)
    frame = cv2.imread(yol)
    if frame is None:
        continue

    results = model.predict(
        frame, conf=config.NESNE_TESPIT_CONF, iou=config.NESNE_TESPIT_IOU, verbose=False
    )
    boxes = results[0].boxes
    n = 0 if boxes is None else len(boxes)
    per_frame_count.append(n)

    if n == 0:
        sifir_tespit_kareler.append(dosya)
    if n > 20:
        asiri_tespit_kareler.append((dosya, n))

    if boxes is not None and n > 0:
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        for c in clss:
            per_class_count[c] = per_class_count.get(c, 0) + 1
        all_confs.extend(confs)

    if (i + 1) % 250 == 0:
        print(f"  ... {i+1}/{len(kare_dosyalari)} kare islendi")

print("\n=== SONUC ===")
print(f"Islenen kare sayisi: {len(per_frame_count)}")
print(f"Toplam tespit: {sum(per_frame_count)}")
print(f"Kare basina ortalama tespit: {statistics.mean(per_frame_count):.2f}")
print(f"Kare basina medyan tespit: {statistics.median(per_frame_count)}")
print(f"Sifir tespit olan kare sayisi: {len(sifir_tespit_kareler)} ({100*len(sifir_tespit_kareler)/len(per_frame_count):.1f}%)")
print(f"Asiri tespit (>20) olan kare sayisi: {len(asiri_tespit_kareler)}")
print(f"\nSinif bazinda tespit sayisi: {per_class_count}")
isim_map = model.names
for cid, cnt in per_class_count.items():
    print(f"  {isim_map.get(cid, cid)}: {cnt}")

if all_confs:
    print(f"\nGuven skoru - min: {min(all_confs):.3f}, max: {max(all_confs):.3f}, "
          f"ortalama: {statistics.mean(all_confs):.3f}, medyan: {statistics.median(all_confs):.3f}")
    dusuk_guven = sum(1 for c in all_confs if c < 0.4)
    print(f"0.25-0.40 arasi (sinirda) tespit sayisi: {dusuk_guven} ({100*dusuk_guven/len(all_confs):.1f}%)")

# Ozet JSON kaydet
ozet = {
    "kare_sayisi": len(per_frame_count),
    "toplam_tespit": sum(per_frame_count),
    "ortalama_tespit_kare": statistics.mean(per_frame_count) if per_frame_count else 0,
    "sifir_tespit_kare_sayisi": len(sifir_tespit_kareler),
    "sifir_tespit_ornekleri": sifir_tespit_kareler[:20],
    "asiri_tespit_ornekleri": asiri_tespit_kareler[:20],
    "sinif_dagilimi": {isim_map.get(k, str(k)): v for k, v in per_class_count.items()},
    "guven_istatistik": {
        "min": min(all_confs) if all_confs else None,
        "max": max(all_confs) if all_confs else None,
        "ortalama": statistics.mean(all_confs) if all_confs else None,
        "medyan": statistics.median(all_confs) if all_confs else None,
    }
}
with open(os.path.join(CIKTI_DIR, "ozet.json"), "w", encoding="utf-8") as f:
    json.dump(ozet, f, ensure_ascii=False, indent=2)
print(f"\nOzet kaydedildi: {CIKTI_DIR}/ozet.json")

# Gorsel inceleme icin ornek kareler kaydet: en cok tespitli, en az/sifir tespitli, 5 random
import random
random.seed(42)

def kare_kaydet(dosya_adi, etiket):
    yol = os.path.join(KARELER_DIR, dosya_adi)
    frame = cv2.imread(yol)
    if frame is None:
        return
    results = model.predict(frame, conf=config.NESNE_TESPIT_CONF, iou=config.NESNE_TESPIT_IOU, verbose=False)
    anotasyonlu = results[0].plot()
    cikti_yol = os.path.join(CIKTI_DIR, f"{etiket}_{dosya_adi}")
    cv2.imwrite(cikti_yol, anotasyonlu)

if asiri_tespit_kareler:
    en_cok = max(asiri_tespit_kareler, key=lambda x: x[1])
    kare_kaydet(en_cok[0], "en_cok_tespit")

if sifir_tespit_kareler:
    kare_kaydet(sifir_tespit_kareler[len(sifir_tespit_kareler)//2], "sifir_tespit")

random_ornekler = random.sample(kare_dosyalari, min(5, len(kare_dosyalari)))
for r in random_ornekler:
    kare_kaydet(r, "random")

print(f"Ornek anotasyonlu kareler kaydedildi: {CIKTI_DIR}/")
