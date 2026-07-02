"""
Yeni DINO template cross-correlation metrigi (pencere-ortalama cosine
benzerligi) icin gercek esik degerini olcer. Eski (patch-bazli) metrigin
esigi (0.45) bu yeni metrik icin gecerli degil - pencere ortalamasi
dogasi geregi daha dusuk/farkli bir olcekte olabilir.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
KARE_SINIRI = 300
ISLEM_GENISLIK = 640

e = GoruntuEslestirme()
e.DINO_ESIK = -1.0  # esiksiz, tum max-pencere skorlarini topla

ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

skorlar = []
kare_no = 0
while kare_no < KARE_SINIRI:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * ISLEM_GENISLIK / w)))
    else:
        kucuk = frame
    kare = e._bgr3_yap(kucuk)
    sonuc = e._dino_ile_ara(kare)  # esiksiz oldugu icin hepsi donuyor
    for s in sonuc:
        skorlar.append(s["skor"])
    kare_no += 1
    if kare_no % 50 == 0:
        print(f"... {kare_no}/{KARE_SINIRI}")

cap.release()

arr = np.array(skorlar)
print(f"\nn={len(arr)} (beklenen: {KARE_SINIRI} kare x 12 ref = {KARE_SINIRI*12})")
print(f"min={arr.min():.3f}  p50={np.percentile(arr,50):.3f}  p90={np.percentile(arr,90):.3f}  "
      f"p95={np.percentile(arr,95):.3f}  p99={np.percentile(arr,99):.3f}  max={arr.max():.3f}")
for esik in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    print(f"  esik={esik}: ustunde olan oran = {100*np.mean(arr>esik):.2f}%")
