"""
Sadece SIFT ile (DINO'suz, ucuz) videonun genelinde her referans nesnenin
hangi karelerde bulundugunu tarar. Amac: cifte_dogrulama_hizli_test.py'nin
DINO+SIFT kontrolunu, SIFT'in zaten guvenilir bulundugu gercek pencerelerde
calistirabilmek icin o pencereleri (kare araliklarini) bulmak.
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
KARE_SINIRI = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

e = GoruntuEslestirme()
ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

bulunma_kareleri = {}

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

    sift_aday = e._sift_ile_ara(e._bgr3_yap(kucuk))
    for s in sift_aday:
        bulunma_kareleri.setdefault(s["object_id"], []).append(kare_no)

    kare_no += 1

cap.release()

print(f"\n=== SIFT-ONLY TARAMA ({kare_no} kare) ===")
if not bulunma_kareleri:
    print("Hicbir referans SIFT ile bulunamadi.")
for oid in sorted(bulunma_kareleri):
    kareler = bulunma_kareleri[oid]
    print(f"obj={oid}: {len(kareler)} kare bulundu, ilk={kareler[0]}, son={kareler[-1]}, ornek_ilk10={kareler[:10]}")
