"""
Goruntu eslestirme (Gorev 3) "stabil degil" sikayetinin kok nedenini
bulmak icin tanilama scripti.

Varsayim: _nms() ayni object_id icin DINO ve SIFT'in bulduklarini
birlestiriyor ama sadece EN YUKSEK skorlu olani tutuyor (goruntu_eslestirme.py:343-350).
DINO ve SIFT farkli karede farkli skor kazaniyorsa, ekrana cizilen kutu
DINO'nun kaba (37x37 grid) kutusu ile SIFT'in dar kutusu arasinda
kare kareye ZIPLAYABILIR -> "stabil degil" hissi.

Bu script DINO ve SIFT sonuclarini AYRI AYRI loglar, hangi kaynagin
"kazandigini" ve ardisik karelerde konum/IOU degisimini olcer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

import cv2

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
KARE_SINIRI = 300  # ilk N kareyi incele

eslestirme = GoruntuEslestirme()

referans_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(referans_klasor)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    eslestirme.referans_yukle(i + 1, os.path.join(referans_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)
if not cap.isOpened():
    print("[HATA] Video acilamadi")
    sys.exit(1)

ISLEM_GENISLIK = 640

# object_id -> son bilinen {"kaynak":..., "bbox":(x1,y1,x2,y2), "kare_no":...}
son_durum = {}
kaynak_degisim_sayisi = {}
toplam_gorulme = {}
buyuk_ziplama_sayisi = {}  # IOU < 0.2 olan ardisik gorulmeler

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

kare_no = 0
while kare_no < KARE_SINIRI:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        oran = ISLEM_GENISLIK / w
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
    else:
        kucuk = frame

    kare_bgr = eslestirme._bgr3_yap(kucuk)
    dino_sonuc = eslestirme._dino_ile_ara(kare_bgr) if eslestirme.dinov2_aktif else []
    sift_sonuc = eslestirme._sift_ile_ara(kare_bgr)

    # Her object_id icin DINO ve SIFT skorlarini karsilastir, kazanani bul
    tum_obj_id = set(d["object_id"] for d in dino_sonuc) | set(s["object_id"] for s in sift_sonuc)
    for oid in tum_obj_id:
        dino_aday = max([d for d in dino_sonuc if d["object_id"] == oid], key=lambda x: x["skor"], default=None)
        sift_aday = max([s for s in sift_sonuc if s["object_id"] == oid], key=lambda x: x["skor"], default=None)

        if dino_aday and sift_aday:
            kazanan = "DINO" if dino_aday["skor"] >= sift_aday["skor"] else "SIFT"
            secilen = dino_aday if kazanan == "DINO" else sift_aday
            # Ayni karede iki kaynak da var mi, IOU'lari ne kadar uyumlu?
            iou_dino_sift = iou(
                (dino_aday["top_left_x"], dino_aday["top_left_y"], dino_aday["bottom_right_x"], dino_aday["bottom_right_y"]),
                (sift_aday["top_left_x"], sift_aday["top_left_y"], sift_aday["bottom_right_x"], sift_aday["bottom_right_y"])
            )
        elif dino_aday:
            kazanan, secilen, iou_dino_sift = "DINO", dino_aday, None
        else:
            kazanan, secilen, iou_dino_sift = "SIFT", sift_aday, None

        bbox = (secilen["top_left_x"], secilen["top_left_y"], secilen["bottom_right_x"], secilen["bottom_right_y"])
        toplam_gorulme[oid] = toplam_gorulme.get(oid, 0) + 1

        onceki = son_durum.get(oid)
        if onceki is not None:
            if onceki["kaynak"] != kazanan:
                kaynak_degisim_sayisi[oid] = kaynak_degisim_sayisi.get(oid, 0) + 1
            ardisik_iou = iou(onceki["bbox"], bbox)
            if ardisik_iou < 0.2:
                buyuk_ziplama_sayisi[oid] = buyuk_ziplama_sayisi.get(oid, 0) + 1

        son_durum[oid] = {"kaynak": kazanan, "bbox": bbox, "kare_no": kare_no}

        if kare_no % 20 == 0:
            extra = f" (DINO-SIFT IOU={iou_dino_sift:.2f})" if iou_dino_sift is not None else ""
            print(f"[{kare_no:3d}] obj={oid} kaynak={kazanan} skor={secilen['skor']:.2f} "
                  f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}){extra}")

    kare_no += 1

cap.release()

print("\n=== TANI SONUCU ===")
print(f"Incelenen kare: {kare_no}")
for oid in sorted(toplam_gorulme):
    gorulme = toplam_gorulme[oid]
    degisim = kaynak_degisim_sayisi.get(oid, 0)
    ziplama = buyuk_ziplama_sayisi.get(oid, 0)
    print(f"obj={oid}: gorulme={gorulme}  kaynak_degisimi={degisim} ({100*degisim/max(gorulme,1):.0f}%)  "
          f"buyuk_konum_ziplamasi(IOU<0.2)={ziplama} ({100*ziplama/max(gorulme,1):.0f}%)")
