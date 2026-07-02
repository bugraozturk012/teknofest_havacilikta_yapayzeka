"""
Duzeltme sonrasi dogrulama: eslestir() genel API'sini (NMS + ardisik kare
dogrulamasi dahil) 300 kare uzerinde calistirip ayni IOU-ziplama metrigini
tekrar olcer. eslestirme_tani.py ile karsilastirma icin.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

import cv2

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


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


son_bbox = {}
toplam_gorulme = {}
buyuk_ziplama = {}
toplam_kutu_sayisi = 0

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

    sonuc = e.eslestir(kucuk.copy())
    toplam_kutu_sayisi += len(sonuc)

    for t in sonuc:
        oid = t["object_id"]
        bbox = (t["top_left_x"], t["top_left_y"], t["bottom_right_x"], t["bottom_right_y"])
        toplam_gorulme[oid] = toplam_gorulme.get(oid, 0) + 1
        onceki = son_bbox.get(oid)
        if onceki is not None:
            if iou(onceki, bbox) < 0.2:
                buyuk_ziplama[oid] = buyuk_ziplama.get(oid, 0) + 1
        son_bbox[oid] = bbox

    kare_no += 1

cap.release()

print(f"\n=== DUZELTME SONRASI SONUC ({kare_no} kare) ===")
print(f"Toplam onaylanmis kutu sayisi: {toplam_kutu_sayisi} (eskiden 170 idi - SIFT_MIN_ESLESME=10, inlier=0.3, dogrulama yok)")
if not toplam_gorulme:
    print("Hic onaylanmis tespit yok.")
for oid in sorted(toplam_gorulme):
    g = toplam_gorulme[oid]
    z = buyuk_ziplama.get(oid, 0)
    print(f"obj={oid}: gorulme={g}  buyuk_ziplama(IOU<0.2)={z} ({100*z/max(g,1):.0f}%)")
