"""
DINO ve SIFT icin gercek deger dagilimini olcer (esikten ONCE, ham degerler).
Amac: DINO_ESIK ve ESLESTIRME_MIN_ESLESME degerlerini tahminle degil,
gercek video+referans verisindeki ayrim noktasina (sinyal vs gurultu) bakarak secmek.
"""

import sys
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
KARE_SINIRI = 300
ISLEM_GENISLIK = 640

e = GoruntuEslestirme()
ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

dino_max_simler = []   # her (kare, ref) icin en yuksek sim degeri (esiksiz)
sift_iyi_sayilari = []  # her (kare, ref) icin len(iyi) (esiksiz)
sift_inlier_oranlari = []  # MIN_ESLESME gecen adaylar icin gercek inlier orani

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

    # ---- DINO: esiksiz max benzerlik ----
    if e.dinov2_aktif:
        feat_map, grid_boyut, _ = e._dino_patch_features(kare)
        if feat_map is not None:
            feat_norm = F.normalize(feat_map, dim=1)
            for oid, ref_data in e.referanslar.items():
                ref_feat = ref_data.get("dino_feat")
                if ref_feat is None:
                    continue
                ref_expanded = ref_feat.unsqueeze(-1).unsqueeze(-1)
                ref_norm = F.normalize(ref_expanded, dim=1)
                sim_map = (feat_norm * ref_norm).sum(dim=1).squeeze(0)
                dino_max_simler.append(float(sim_map.max().cpu().numpy()))

    # ---- SIFT: esiksiz iyi-eslesme sayisi + (esik gecince) inlier orani ----
    kare_gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    kare_kp, kare_des = e.sift.detectAndCompute(kare_gri, None)
    if kare_des is not None and len(kare_kp) >= 10:
        for oid, ref_data in e.referanslar.items():
            ref_des = ref_data.get("sift_des")
            ref_kp = ref_data.get("sift_kp")
            if ref_des is None or len(ref_kp) < 5:
                continue
            try:
                matches = e.flann.knnMatch(ref_des, kare_des, k=2)
            except Exception:
                continue
            iyi = [p[0] for p in matches if len(p) == 2 and p[0].distance < e.SIFT_RATIO_TEST * p[1].distance]
            sift_iyi_sayilari.append(len(iyi))

            if len(iyi) >= e.SIFT_MIN_ESLESME:
                src = np.float32([ref_kp[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
                dst = np.float32([kare_kp[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)
                try:
                    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    if M is not None and mask is not None:
                        sift_inlier_oranlari.append(float(np.sum(mask) / len(mask)))
                except Exception:
                    pass

    kare_no += 1
    if kare_no % 50 == 0:
        print(f"... {kare_no}/{KARE_SINIRI} kare islendi")

cap.release()

print("\n=== DINO ham benzerlik dagilimi (esiksiz, tum kare x ref ciftleri) ===")
arr = np.array(dino_max_simler)
print(f"n={len(arr)}  min={arr.min():.3f}  p50={np.percentile(arr,50):.3f}  "
      f"p90={np.percentile(arr,90):.3f}  p99={np.percentile(arr,99):.3f}  max={arr.max():.3f}")
print(f"Mevcut esik (0.55) ustunde olan oran: {100*np.mean(arr>0.55):.2f}%")
for esik in [0.30, 0.35, 0.40, 0.45, 0.50]:
    print(f"  esik={esik}: ustunde olan oran = {100*np.mean(arr>esik):.2f}%")

print("\n=== SIFT iyi-eslesme sayisi dagilimi (esiksiz, tum kare x ref ciftleri) ===")
arr2 = np.array(sift_iyi_sayilari)
print(f"n={len(arr2)}  min={arr2.min()}  p50={np.percentile(arr2,50):.1f}  "
      f"p90={np.percentile(arr2,90):.1f}  p99={np.percentile(arr2,99):.1f}  max={arr2.max()}")
print(f"Mevcut esik (10) ustunde olan oran: {100*np.mean(arr2>=10):.2f}%")
for esik in [10, 15, 20, 25, 30, 40]:
    print(f"  esik={esik}: ustunde/esit olan oran = {100*np.mean(arr2>=esik):.2f}%")

print("\n=== SIFT esigini gecen adaylarin RANSAC inlier orani dagilimi ===")
arr3 = np.array(sift_inlier_oranlari)
if len(arr3) > 0:
    print(f"n={len(arr3)}  min={arr3.min():.2f}  p50={np.percentile(arr3,50):.2f}  "
          f"p90={np.percentile(arr3,90):.2f}  max={arr3.max():.2f}")
    print(f"Mevcut esik (0.3) ustunde olan oran: {100*np.mean(arr3>0.3):.2f}%")
else:
    print("Hic homografi kurulamadi.")
