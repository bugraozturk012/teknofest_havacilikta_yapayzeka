"""
Gerçek bir test videonuz yoksa, uçtan uca pipeline testi (sahte_sunucu.py +
main.py) için bu betik kısa, sentetik bir video üretir.

DİKKAT: Bu video gerçek hava görüntüsü değildir — YOLO modeli üzerinde
gerçekçi nesne tespiti beklemeyin. Amaç sadece sunucu<->istemci iletişiminin
ve JSON formatının uçtan uca çalıştığını doğrulamaktır.

Çalıştır: python test/sentetik_video_uret.py
"""

import cv2
import numpy as np
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIKIS_YOLU = os.path.join(BASE_DIR, "video.mp4")
KARE_SAYISI = 150
FPS = 30          # sahte_sunucu.py bunu 7.5 fps'e indirgeyecek (her 4. kare)
GENISLIK, YUKSEKLIK = 1280, 720


def sentetik_kare_uret(kare_no):
    """Hareket eden kutular/şekillerle havadan görüş hissi veren bir kare üretir."""
    kare = np.zeros((YUKSEKLIK, GENISLIK, 3), dtype=np.uint8)
    kare[:] = (30, 60, 30)  # koyu yeşil zemin

    kayma = kare_no * 3
    for x in range(0, GENISLIK + 100, 100):
        cv2.line(kare, ((x - kayma) % (GENISLIK + 100) - 50, 0),
                 ((x - kayma) % (GENISLIK + 100) - 50, YUKSEKLIK), (40, 80, 40), 1)

    # Hareketli "araç" (mavi kutu)
    arx = (150 + kare_no * 4) % (GENISLIK - 100)
    cv2.rectangle(kare, (arx, 300), (arx + 80, 340), (200, 80, 30), -1)

    # Sabit "araç" (kırmızı kutu)
    cv2.rectangle(kare, (600, 400), (680, 440), (30, 30, 200), -1)

    # Sabit "insan" (küçük beyaz daire)
    cv2.circle(kare, (900, 250), 12, (220, 220, 220), -1)

    return kare


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(CIKIS_YOLU, fourcc, FPS, (GENISLIK, YUKSEKLIK))

    for i in range(KARE_SAYISI):
        writer.write(sentetik_kare_uret(i))

    writer.release()
    boyut = os.path.getsize(CIKIS_YOLU) / 1024
    print(f"[OK] {CIKIS_YOLU} oluşturuldu ({KARE_SAYISI} kare, {boyut:.0f} KB)")


if __name__ == "__main__":
    main()
