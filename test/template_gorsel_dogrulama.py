"""
Yeni patch-template cross-correlation yontemini gorsel olarak dogrular.
eslestir() (NMS + ardisik kare dogrulama dahil tam pipeline) ile bulunan
ilk N onayli tespiti, kutu cizili olarak kaydeder - referans gorseliyle
yan yana karsilastirmak icin.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

import cv2

PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
ISLEM_GENISLIK = 640
CIKTI_DIR = "template_kanit"
os.makedirs(CIKTI_DIR, exist_ok=True)

e = GoruntuEslestirme()
ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

kaydedilen = 0
kare_no = 0
gorulen_obj = set()
while kaydedilen < 12 and kare_no < 9000:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * ISLEM_GENISLIK / w)))
    else:
        kucuk = frame

    sonuc = e.eslestir(kucuk.copy())

    for t in sonuc:
        oid = t["object_id"]
        cizim = kucuk.copy()
        x1, y1 = int(t["top_left_x"]), int(t["top_left_y"])
        x2, y2 = int(t["bottom_right_x"]), int(t["bottom_right_y"])
        cv2.rectangle(cizim, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(cizim, f"obj{oid} kare{kare_no}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cikti_yol = os.path.join(CIKTI_DIR, f"kare_{kare_no:04d}_obj{oid}.jpg")
        cv2.imwrite(cikti_yol, cizim)
        print(f"[{kare_no}] obj{oid} bbox=({x1},{y1},{x2},{y2}) -> {cikti_yol}")
        kaydedilen += 1
        gorulen_obj.add(oid)

    kare_no += 1
    if kare_no % 500 == 0:
        print(f"... {kare_no} kare tarandi, su ana kadar {kaydedilen} tespit")

cap.release()
print(f"\nToplam {kaydedilen} kanit karesi, {len(gorulen_obj)} farkli obj: {sorted(gorulen_obj)}")
print(f"Klasor: {CIKTI_DIR}/")
