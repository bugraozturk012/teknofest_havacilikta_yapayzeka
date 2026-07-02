"""
Cifte dogrulamanin (_cifte_dogrulama: DINO ve SIFT ayni bolgeyi BAGIMSIZ
olarak isaret ettiginde onaylama) gercekte hic tetiklenip tetiklenmedigini
olcer. Onceki oturumda tam video taramasi (9000 kare) 20+ dk surup hicbir
eslesme bulamadan durdurulmustu - bu script ayni soruyu sabit 300 karelik
kucuk bir pencerede (obj10_gorsel_dogrulama.py'nin SIFT'i zaten guvenilir
buldugu bolge) hizlica cevaplar.

DINO ve SIFT taramalarini birbirinden BAGIMSIZ calistirir (eslestir()
icindeki cifte_dogrulama/NMS/ardisik-kare filtrelerini atlayarak), her
object_id icin SIFT'in bulma sayisini, DINO'nun bulma sayisini ve
ikisinin AYNI karede AYNI bolgede (IOU>0.3) ortustugu sayiyi raporlar.
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
KARE_SINIRI = 300
ISLEM_GENISLIK = 640
CIKTI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifte_dogrulama_kanit")
os.makedirs(CIKTI_DIR, exist_ok=True)

e = GoruntuEslestirme()
if not e.dinov2_aktif:
    print("[UYARI] DINOv2 yuklenemedi, cifte dogrulama testi anlamsiz (SIFT-only moda dusuldu).")
    sys.exit(1)

ref_klasor = os.path.join(PROJE_KOKU, "referanslar")
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
for i, d in enumerate(ref_dosyalar):
    e.referans_yukle(i + 1, os.path.join(ref_klasor, d))

cap = cv2.VideoCapture(VIDEO_YOLU)

sift_bulunma = {}
dino_bulunma = {}
cifte_onay = {}
kanit_kaydedilen = 0

baslangic = time.time()
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

    kare_bgr = e._bgr3_yap(kucuk)
    sift_aday = e._sift_ile_ara(kare_bgr)
    dino_aday = e._dino_ile_ara(kare_bgr)

    for s in sift_aday:
        sift_bulunma[s["object_id"]] = sift_bulunma.get(s["object_id"], 0) + 1
    for d in dino_aday:
        dino_bulunma[d["object_id"]] = dino_bulunma.get(d["object_id"], 0) + 1

    onaylananlar = e._cifte_dogrulama(dino_aday, sift_aday)
    for o in onaylananlar:
        oid = o["object_id"]
        cifte_onay[oid] = cifte_onay.get(oid, 0) + 1

        if kanit_kaydedilen < 10:
            cizim = kucuk.copy()
            x1, y1 = int(o["top_left_x"]), int(o["top_left_y"])
            x2, y2 = int(o["bottom_right_x"]), int(o["bottom_right_y"])
            cv2.rectangle(cizim, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cizim, f"obj{oid} kare{kare_no} CIFTE-ONAY", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cikti_yol = os.path.join(CIKTI_DIR, f"kare_{kare_no:04d}_obj{oid}.jpg")
            cv2.imwrite(cikti_yol, cizim)
            kanit_kaydedilen += 1

    kare_no += 1

cap.release()
sure = time.time() - baslangic

tum_obj_id = sorted(set(sift_bulunma) | set(dino_bulunma))

print(f"\n=== CIFTE DOGRULAMA HIZLI TEST ({kare_no} kare, {sure:.1f} sn) ===")
if not tum_obj_id:
    print("Hicbir referans ne SIFT ne DINO tarafindan bulunamadi.")
else:
    print(f"{'obj':>4} {'SIFT':>6} {'DINO':>6} {'CIFTE-ONAY':>11}")
    for oid in tum_obj_id:
        s = sift_bulunma.get(oid, 0)
        d = dino_bulunma.get(oid, 0)
        c = cifte_onay.get(oid, 0)
        print(f"{oid:>4} {s:>6} {d:>6} {c:>11}")

toplam_cifte = sum(cifte_onay.values())
print(f"\nToplam cifte-onayli tespit: {toplam_cifte} / {kare_no} kare")
print(f"Kanit karesi kaydedildi: {kanit_kaydedilen} -> {CIKTI_DIR}/")
if toplam_cifte == 0:
    print("SONUC: Cifte dogrulama bu 300 karede HIC tetiklenmedi.")
