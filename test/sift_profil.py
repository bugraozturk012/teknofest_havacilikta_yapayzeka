"""
sift_tarama_hizli.py neden beklenenden (22+ dk, 9022 kare) cok yavas oldugunu
bulmak icin 500 kare uzerinde asamalari ayri ayri zamanlar:
1. Sadece video decode + resize
2. + SIFT detectAndCompute (kare basina)
3. + 12 referansla FLANN knnMatch (tam sift_tarama_hizli.py esdegeri)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

import cv2

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
ISLEM_GENISLIK = 640
KARE_SINIRI = 500

# ---- Asama 1: sadece decode + resize ----
cap = cv2.VideoCapture(VIDEO_YOLU)
t0 = time.time()
n = 0
while n < KARE_SINIRI:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        oran = ISLEM_GENISLIK / w
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
    n += 1
cap.release()
t1 = time.time()
print(f"[1] Decode+resize: {n} kare, {t1-t0:.1f}s ({(t1-t0)/n*1000:.1f} ms/kare)")

# ---- Asama 2: + SIFT detectAndCompute ----
sift = cv2.SIFT_create(nfeatures=500)
cap = cv2.VideoCapture(VIDEO_YOLU)
t0 = time.time()
n = 0
while n < KARE_SINIRI:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        oran = ISLEM_GENISLIK / w
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
    else:
        kucuk = frame
    gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gri, None)
    n += 1
cap.release()
t2 = time.time()
print(f"[2] +SIFT detect: {n} kare, {t2-t0:.1f}s ({(t2-t0)/n*1000:.1f} ms/kare)")

# ---- Asama 3: + tum eslestirme._sift_ile_ara (12 referansla FLANN) ----
e = GoruntuEslestirme()
ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)
t0 = time.time()
n = 0
while n < KARE_SINIRI:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        oran = ISLEM_GENISLIK / w
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
    else:
        kucuk = frame
    _ = e._sift_ile_ara(e._bgr3_yap(kucuk))
    n += 1
cap.release()
t3 = time.time()
print(f"[3] +12 ref FLANN match: {n} kare, {t3-t0:.1f}s ({(t3-t0)/n*1000:.1f} ms/kare)")

print(f"\n9022 kareye ekstrapole edilirse ~{(t3-t0)/n*9022/60:.1f} dakika (sadece SIFT asamasi)")
