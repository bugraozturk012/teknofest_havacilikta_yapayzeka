"""
obj=10 (Referans_Nesne_05.png - dikdortgen yapi) eslesmelerinin GERCEKTEN
dogru yere mi yoksa rastgele mi kutu cizdigini gorsel olarak dogrulamak icin
eslesme bulunan ilk 5 kareyi kutu cizili halde kaydeder. Ayrica kaynagin
(DINO mu SIFT mi) hangisi oldugunu da loglar.
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
CIKTI_DIR = "obj10_kanit"
os.makedirs(CIKTI_DIR, exist_ok=True)

e = GoruntuEslestirme()
ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

kaydedilen = 0
kare_no = 0
while kaydedilen < 6 and kare_no < 2200:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    if w > ISLEM_GENISLIK:
        oran = ISLEM_GENISLIK / w
        kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
    else:
        oran = 1.0
        kucuk = frame

    kare_bgr = e._bgr3_yap(kucuk)
    dino_aday = e._dino_ile_ara(kare_bgr) if e.dinov2_aktif else []
    sift_aday = e._sift_ile_ara(kare_bgr)

    sonuc = e.eslestir(kucuk.copy())

    for t in sonuc:
        if t["object_id"] == 10:
            # Kaynagi belirle (DINO mu SIFT mi bu obje icin bu karede aday uretti)
            dino_var = any(d["object_id"] == 10 for d in dino_aday)
            sift_var = any(s["object_id"] == 10 for s in sift_aday)
            kaynak = []
            if dino_var:
                kaynak.append("DINO")
            if sift_var:
                kaynak.append("SIFT")
            kaynak_str = "+".join(kaynak) if kaynak else "?"

            cizim = kucuk.copy()
            x1, y1 = int(t["top_left_x"]), int(t["top_left_y"])
            x2, y2 = int(t["bottom_right_x"]), int(t["bottom_right_y"])
            cv2.rectangle(cizim, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(cizim, f"obj10 kare{kare_no} src={kaynak_str}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cikti_yol = os.path.join(CIKTI_DIR, f"kare_{kare_no:04d}_{kaynak_str}.jpg")
            cv2.imwrite(cikti_yol, cizim)
            print(f"[{kare_no}] obj10 bulundu, kaynak={kaynak_str}, bbox=({x1},{y1},{x2},{y2}) -> {cikti_yol}")
            kaydedilen += 1

    kare_no += 1

cap.release()
print(f"\nToplam {kaydedilen} kanit karesi kaydedildi: {CIKTI_DIR}/")
if kaydedilen == 0:
    print("Ilk 2200 karede obj=10 hic bulunamadi.")
