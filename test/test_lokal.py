"""
Lokal test betiği — yarışma sunucusu olmadan tüm modülleri dener.
Webcam varsa kullanır, yoksa sentetik kare üretir.

Çalıştır: python test_lokal.py
Çıkış   : q tuşu
"""

import cv2
import numpy as np
import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Bu dosya test/ klasöründe olduğu için proje kökü bir üst dizindir.
PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJE_KOKU)

import config
from kaynak_kodlar.nesne_tespiti import NesneTespiti
from kaynak_kodlar.pozisyon_kestirimi import PozisyonKestirimi
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme


# ============================================================
# AYARLAR
# ============================================================
# Kendi test videonuzu bu dosyanın yanına "video.mp4" olarak koyun,
# ya da burayı kendi yolunuzla değiştirin.
VIDEO_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video.mp4")
FPS_HEDEF      = 7.5
ISLEM_GENISLIK = 640   # YOLO bu çözünürlükte işler (daha geniş = yavaş)


def sentetik_kare_uret(kare_no, genislik=1280, yukseklik=720):
    """Gerçek hava görüntüsü yerine test amacıyla hareket eden şekiller."""
    kare = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
    kare[:] = (30, 60, 30)   # koyu yeşil zemin

    # Kaydırılan grid (uçuş hissi)
    kayma = kare_no * 3
    for x in range(0, genislik + 100, 100):
        cv2.line(kare, ((x - kayma) % (genislik + 100) - 50, 0),
                 ((x - kayma) % (genislik + 100) - 50, yukseklik), (40, 80, 40), 1)
    for y in range(0, yukseklik + 100, 100):
        cv2.line(kare, (0, (y - kayma // 2) % (yukseklik + 100) - 50),
                 (genislik, (y - kayma // 2) % (yukseklik + 100) - 50), (40, 80, 40), 1)

    # Hareketli "araç" (mavi kutu)
    arx = (150 + kare_no * 4) % (genislik - 100)
    cv2.rectangle(kare, (arx, 300), (arx + 80, 340), (200, 80, 30), -1)

    # Sabit "araç" (kırmızı kutu)
    cv2.rectangle(kare, (600, 400), (680, 440), (30, 30, 200), -1)

    # Sabit "insan" (küçük beyaz daire)
    cv2.circle(kare, (900, 250), 12, (220, 220, 220), -1)

    return kare


def bilgi_ekrani_ciz(kare, tespit_sonuc, poz_sonuc, eslestirme_sonuc, fps, health):
    """Tespitleri ve metrikleri görüntü üzerine yazar."""
    h, w = kare.shape[:2]
    overlay = kare.copy()

    # Nesne tespiti kutuları
    for obj in tespit_sonuc:
        x1 = int(obj["top_left_x"])
        y1 = int(obj["top_left_y"])
        x2 = int(obj["bottom_right_x"])
        y2 = int(obj["bottom_right_y"])
        sinif = int(obj["cls"])
        renkler = {0: (0, 165, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 200)}
        etiketler = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}
        renk = renkler.get(sinif, (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), renk, 2)
        etiket = f"{etiketler.get(sinif, '?')} m:{obj['motion_status']} l:{obj['landing_status']}"
        cv2.putText(overlay, etiket, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)

    # Görüntü eşleştirme kutuları
    for ue in eslestirme_sonuc:
        x1 = int(ue["top_left_x"])
        y1 = int(ue["top_left_y"])
        x2 = int(ue["bottom_right_x"])
        y2 = int(ue["bottom_right_y"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay, f"REF#{ue['object_id']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Bilgi paneli (sol üst)
    panel_h, panel_w = 370, 720
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    model_var = os.path.exists(os.path.join(PROJE_KOKU, config.NESNE_TESPIT_MODEL))
    satirlar = [
        (f"FPS   : {fps:.1f}",                               (200, 200, 200)),
        (f"Model : {'best.pt' if model_var else 'yolov8n (egitim gerekli)'}",
         (0, 220, 0) if model_var else (0, 140, 255)),
        (f"Health: {'SAGLIKLI' if health == 1 else 'SAGSIZ'}",
         (0, 220, 0) if health == 1 else (0, 80, 255)),
        (f"Poz X : {poz_sonuc['translation_x']:.3f} m",     (200, 200, 200)),
        (f"Poz Y : {poz_sonuc['translation_y']:.3f} m",     (200, 200, 200)),
        (f"Poz Z : {poz_sonuc['translation_z']:.3f} m",     (200, 200, 200)),
        (f"Nesne : {len(tespit_sonuc)}   Ref.Obj: {len(eslestirme_sonuc)}",
         (200, 200, 200)),
    ]
    for i, (sat, renk) in enumerate(satirlar):
        cv2.putText(panel, sat, (14, 44 + i * 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, renk, 3)

    overlay[10:10 + panel_h, 10:10 + panel_w] = panel
    cv2.addWeighted(overlay, 0.85, kare, 0.15, 0, kare)
    return kare


def main():
    print("=" * 55)
    print("  TEKNOFEST 2026 — LOKAL TEST")
    print("=" * 55)

    # Modülleri yükle (model yolu proje köküne göre relatif olduğundan
    # cwd'den bağımsız çalışması için PROJE_KOKU ile birleştiriyoruz)
    print("[INIT] Nesne tespiti yükleniyor...")
    model_yolu = os.path.join(PROJE_KOKU, config.NESNE_TESPIT_MODEL)
    nesne = NesneTespiti(model_yolu=model_yolu)

    print("[INIT] Pozisyon kestirimi hazırlanıyor...")
    pozisyon = PozisyonKestirimi()

    print("[INIT] Görüntü eşleştirme yükleniyor...")
    eslestirme = GoruntuEslestirme()

    # Referans klasörü varsa yükle (proje köküne göre)
    referans_klasor = os.path.join(PROJE_KOKU, "referanslar")
    if os.path.exists(referans_klasor):
        ref_dosyalar = sorted([f for f in os.listdir(referans_klasor)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for i, d in enumerate(ref_dosyalar):
            eslestirme.referans_yukle(i + 1, os.path.join(referans_klasor, d))
        if not ref_dosyalar:
            print("[UYARI] referanslar/ klasörü boş — Görev 3 pasif")
    else:
        print("[UYARI] referanslar/ klasörü yok — Görev 3 pasif")

    # Video dosyasını aç
    if not os.path.exists(VIDEO_YOLU):
        print(f"[HATA] Video bulunamadı: {VIDEO_YOLU}")
        return
    cap = cv2.VideoCapture(VIDEO_YOLU)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {VIDEO_YOLU}")
        return
    toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {VIDEO_YOLU}")
    print(f"[VIDEO] Toplam kare: {toplam_kare}")

    print("\n[BASLA] Çalışıyor — çıkmak için 'q' tuşuna bas\n")

    kare_no = 0
    fps_sayac = 0
    fps_sure = time.time()
    fps = 0.0
    # Health durumu: ilk 60 kare sağlıklı, sonra 0, 150'de tekrar 1
    health_degis = [(0, 1), (60, 0), (150, 1)]

    while True:
        t0 = time.time()

        # Kare al
        ret, frame = cap.read()
        if not ret:
            print("[VIDEO] Bitti, başa sarılıyor...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            kare_no = 0
            ret, frame = cap.read()
            if not ret:
                break

        # İşlem için küçült (FPS artışı), gösterim için orijinal sakla
        h, w = frame.shape[:2]
        if w > ISLEM_GENISLIK:
            oran = ISLEM_GENISLIK / w
            kucuk = cv2.resize(frame, (ISLEM_GENISLIK, int(h * oran)))
        else:
            kucuk = frame

        # Simüle pozisyon verisi (gerçekte sunucudan gelir)
        health = 1
        for es, h in health_degis:
            if kare_no >= es:
                health = h
        ref_x = round(kare_no * 0.05, 3)
        ref_y = round(kare_no * 0.03, 3)
        ref_z = 50.0 + np.sin(kare_no * 0.05) * 5

        # ── Görev 1: Nesne tespiti (küçük kare ile)
        tespit_sonuc = nesne.tespit_et(kucuk)

        # Bbox koordinatlarını orijinal boyuta ölçekle
        if w > ISLEM_GENISLIK:
            oran_geri = w / ISLEM_GENISLIK
            for obj in tespit_sonuc:
                obj["top_left_x"]     = round(obj["top_left_x"]     * oran_geri, 2)
                obj["top_left_y"]     = round(obj["top_left_y"]     * oran_geri, 2)
                obj["bottom_right_x"] = round(obj["bottom_right_x"] * oran_geri, 2)
                obj["bottom_right_y"] = round(obj["bottom_right_y"] * oran_geri, 2)

        # ── Görev 2: Pozisyon kestirimi (küçük kare ile)
        poz_sonuc = pozisyon.guncelle(
            kucuk.copy(), ref_x, ref_y, ref_z, health
        )

        # ── Görev 3: Görüntü eşleştirme (küçük kare ile)
        eslestirme_sonuc = eslestirme.eslestir(kucuk.copy())

        # FPS hesapla
        fps_sayac += 1
        if time.time() - fps_sure >= 1.0:
            fps = fps_sayac / (time.time() - fps_sure)
            fps_sayac = 0
            fps_sure = time.time()

        # Ekranda göster — pencereyi ekrana sığdır
        gosterim = bilgi_ekrani_ciz(
            frame.copy(), tespit_sonuc, poz_sonuc, eslestirme_sonuc, fps, health
        )
        gosterim = cv2.resize(gosterim, (1280, 720))
        cv2.namedWindow("TEKNOFEST 2026 - Lokal Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TEKNOFEST 2026 - Lokal Test", 1280, 720)
        cv2.imshow("TEKNOFEST 2026 - Lokal Test", gosterim)

        # Konsol çıktısı (her 30 karede bir)
        if kare_no % 30 == 0:
            print(f"[{kare_no:4d}] Health:{health}  "
                  f"Nesne:{len(tespit_sonuc)}  "
                  f"Poz:({poz_sonuc['translation_x']:.2f}, "
                  f"{poz_sonuc['translation_y']:.2f}, "
                  f"{poz_sonuc['translation_z']:.2f})  "
                  f"FPS:{fps:.1f}")

        kare_no += 1

        # Hız sınırla
        gecen = time.time() - t0
        bekle = max(1, int((1.0 / FPS_HEDEF - gecen) * 1000))
        if cv2.waitKey(bekle) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[BITTI] Test tamamlandı.")


if __name__ == "__main__":
    main()
