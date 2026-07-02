"""
eslestir() icindeki _sift_ile_ara ve _dino_ile_ara adimlarini AYRI AYRI
zamanlar - hiz_profili.py'de eslestirme adimi supheli derecede yavas
(2057ms/kare) cikti, kaynagini (SIFT mi DINOv2 mi) netlestirmek icin.
"""

import sys
import os
import time
import statistics

sys.path.append('..')
import config
import cv2

from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

KARELER_DIR = "kareler"
ORNEK_KARE_SAYISI = 30

eslestirme = GoruntuEslestirme(
    dinov2_model_yolu="../modeller/dinov2_vits14.pth",
    dinov2_repo_yolu="../modeller/dinov2_repo",
)

ref_klasor = "../referanslar"
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
ilk_ref = ref_dosyalar[0]
t0 = time.perf_counter()
eslestirme.referans_yukle("test_ref", os.path.join(ref_klasor, ilk_ref))
print(f"[YUKLEME] Referans yukleme (SIFT+DINO template) suresi: {(time.perf_counter()-t0)*1000:.1f} ms")

referans_havuzu = {k: v for k, v in eslestirme.referanslar.items() if k == "test_ref"}

kare_dosyalari = sorted(os.listdir(KARELER_DIR))[:ORNEK_KARE_SAYISI]

sift_sureler = []
dino_sureler = []

for dosya in kare_dosyalari:
    frame = cv2.imread(os.path.join(KARELER_DIR, dosya))
    if frame is None:
        continue
    kare = eslestirme._bgr3_yap(frame)

    t0 = time.perf_counter()
    eslestirme._sift_ile_ara(kare, referans_havuzu)
    t1 = time.perf_counter()
    eslestirme._dino_ile_ara(kare, referans_havuzu)
    t2 = time.perf_counter()

    sift_sureler.append(t1 - t0)
    dino_sureler.append(t2 - t1)

print(f"\n_sift_ile_ara ortalama: {statistics.mean(sift_sureler)*1000:.1f} ms")
print(f"_dino_ile_ara ortalama: {statistics.mean(dino_sureler)*1000:.1f} ms")

# SIFT'in kendi icinde de kirilim: kare uzerinde detectAndCompute mi,
# yoksa FLANN eslestirme mi yavas?
kare_gri_sureler = []
flann_sureler = []
for dosya in kare_dosyalari:
    frame = cv2.imread(os.path.join(KARELER_DIR, dosya))
    if frame is None:
        continue
    kare = eslestirme._bgr3_yap(frame)
    kare_gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    kare_kp, kare_des = eslestirme.sift.detectAndCompute(kare_gri, None)
    t1 = time.perf_counter()
    for ref_data in referans_havuzu.values():
        ref_des = ref_data.get("sift_des")
        if ref_des is not None and kare_des is not None:
            eslestirme.flann.knnMatch(ref_des, kare_des, k=2)
    t2 = time.perf_counter()

    kare_gri_sureler.append(t1 - t0)
    flann_sureler.append(t2 - t1)

print(f"\n  kare uzerinde SIFT detectAndCompute (1920x1080): {statistics.mean(kare_gri_sureler)*1000:.1f} ms")
print(f"  FLANN knnMatch (1 referans): {statistics.mean(flann_sureler)*1000:.1f} ms")
