"""
TEKNOFEST 2026 - Sahte Yarışma Sunucusu

Lokal test için gerçek sunucu API'sini taklit eder. v2.1.0 resmi bağlantı
arayüzü (26.06.2026) ile birebir uyumlu:
- GET /progress/     → {frame_index, total_frames, completed, session_name}
- GET /frames/       → Sıradaki TEK karenin listesi ([] ise oturum bitti)
- GET /translation/  → Sıradaki TEK karenin translation'ı (frames/ ile aynı index)
- GET /reference/    → Görev 3 referans nesneleri + aktif kare pencereleri
- GET /media/<dosya> → Kare görüntüsü
- GET /media/referanslar/<dosya> → Referans nesne görüntüsü
- POST /prediction/  → Sonuç alır; SADECE sıradaki kare için kabul eder,
                       kabul edilince sıradaki index'e ilerler (sunucu
                       tahmin gönderilmeden bir sonraki kareyi vermez).

Kullanım: python sahte_sunucu.py
"""

from flask import Flask, request, jsonify, send_file
import cv2
import os
import sys
import glob
import time
import math
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)

# ============================================================
# AYARLAR
# ============================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_YOLU = os.path.join(_BASE_DIR, "video.mp4")
KARELER_KLASORU = os.path.join(_BASE_DIR, "kareler")
REFERANSLAR_KLASORU = os.path.join(_BASE_DIR, "..", "referanslar")
FPS = 7.5  # Şartname: 7.5 FPS
SAGLIKLI_KARE = 450  # İlk 1 dakika sağlıklı
SESSION_NAME = "test_video_V1"


def sahte_pozisyon(kare_no):
    """Sahte ama gerçekçi uçuş rotası üretir."""
    t = kare_no / FPS
    x = 50 * math.sin(0.02 * t) + 0.5 * t
    y = 30 * math.cos(0.015 * t) + 0.3 * t
    z = 15 + 5 * math.sin(0.01 * t)
    return x, y, z


def videodan_kare_cikar():
    """Videoyu karelere ayırır (varsa atlanır)."""
    os.makedirs(KARELER_KLASORU, exist_ok=True)
    mevcut = sorted(glob.glob(os.path.join(KARELER_KLASORU, "*.jpg")))
    if len(mevcut) > 100:
        print(f"[SUNUCU] {len(mevcut)} kare zaten mevcut, atlanıyor.")
        return [os.path.basename(p) for p in mevcut]

    if not os.path.exists(VIDEO_YOLU):
        print(f"[HATA] Video bulunamadı: {VIDEO_YOLU}")
        return []

    cap = cv2.VideoCapture(VIDEO_YOLU)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    atlama = max(1, int(video_fps / FPS))

    dosya_adlari = []
    kare_sayaci = 0
    toplam = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        toplam += 1
        if toplam % atlama == 0:
            dosya = f"frame_{kare_sayaci:06d}.jpg"
            cv2.imwrite(os.path.join(KARELER_KLASORU, dosya), frame)
            dosya_adlari.append(dosya)
            kare_sayaci += 1
            if kare_sayaci >= 2250:  # Şartname: max 2250 kare
                break
    cap.release()
    print(f"[SUNUCU] {kare_sayaci} kare çıkarıldı ({VIDEO_YOLU} -> {KARELER_KLASORU})")
    return dosya_adlari


# ============================================================
# OTURUM DURUMU (tek seferlik hesaplanır, index ile ilerler)
# ============================================================
_kare_dosyalari = videodan_kare_cikar()
TOPLAM_KARE = len(_kare_dosyalari)

_frame_listesi = []
_translation_listesi = []
random.seed(42)  # Sağlık durumu tekrar sorgulamalarda DEĞİŞMESİN (deterministik)
for i, dosya_adi in enumerate(_kare_dosyalari):
    x, y, z = sahte_pozisyon(i)
    if i < SAGLIKLI_KARE:
        health = 1
        tx, ty, tz = x, y, z
    else:
        if random.random() < 0.6:
            health = 0
            tx, ty, tz = "NaN", "NaN", "NaN"
        else:
            health = 1
            tx, ty, tz = x, y, z

    _frame_listesi.append({
        "url": f"http://localhost:5000/frames/{i}/",
        "image_url": f"/{dosya_adi}",
        "video_name": "test_video_V1",
        "session": "http://localhost:5000/session/1/",
    })
    _translation_listesi.append({
        "translation_x": tx, "translation_y": ty, "translation_z": tz,
        "health_status": health
    })

# Sunucu tarafı ilerleme durumu (tek "kullanıcı" varsayımıyla basitleştirildi)
_mevcut_index = 0


def _referanslari_hazirla():
    """
    referanslar/ klasöründeki görselleri Görev 3 referans nesnelerine
    dönüştürür; her birine 2250 kareyi eşit ve ÇAKIŞMAYAN aralıklara bölerek
    sıralı bir [frame_start, frame_end] penceresi atar (Q&A kuralı: aynı
    anda sadece TEK referans aktif, aralıklar çakışmaz/sıralıdır).
    """
    if not os.path.exists(REFERANSLAR_KLASORU) or TOPLAM_KARE == 0:
        return []
    dosyalar = sorted([f for f in os.listdir(REFERANSLAR_KLASORU)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not dosyalar:
        return []

    n = len(dosyalar)
    pencere_boyu = max(1, TOPLAM_KARE // n)
    referanslar = []
    for i, dosya in enumerate(dosyalar):
        baslangic_idx = i * pencere_boyu
        bitis_idx = min(TOPLAM_KARE - 1, baslangic_idx + pencere_boyu - 1) if i < n - 1 else TOPLAM_KARE - 1
        if baslangic_idx > TOPLAM_KARE - 1:
            break
        referanslar.append({
            "url": f"http://localhost:5000/reference/{i + 1}/",
            "session": "http://localhost:5000/session/1/",
            "image_url": f"/referanslar/{dosya}",
            "frame_start_image_url": _frame_listesi[baslangic_idx]["image_url"],
            "frame_end_image_url": _frame_listesi[bitis_idx]["image_url"],
            "order": i + 1,
        })
    return referanslar


_referans_listesi = _referanslari_hazirla()


@app.route('/progress/', methods=['GET'])
def get_progress():
    return jsonify({
        "frame_index": _mevcut_index,
        "total_frames": TOPLAM_KARE,
        "completed": _mevcut_index >= TOPLAM_KARE,
        "session_name": SESSION_NAME,
    })


@app.route('/frames/', methods=['GET'])
def get_frames():
    """Sunucu artık sadece sıradaki TEK kareyi döner (tahmin gönderilmeden ilerlemez)."""
    if _mevcut_index >= TOPLAM_KARE:
        return jsonify([])
    return jsonify([_frame_listesi[_mevcut_index]])


@app.route('/translation/', methods=['GET'])
def get_translation():
    if _mevcut_index >= TOPLAM_KARE:
        return jsonify([])
    return jsonify([_translation_listesi[_mevcut_index]])


@app.route('/reference/', methods=['GET'])
def get_reference():
    return jsonify(_referans_listesi)


@app.route('/media/referanslar/<dosya_adi>', methods=['GET'])
def referans_goruntu_gonder(dosya_adi):
    yol = os.path.join(REFERANSLAR_KLASORU, dosya_adi)
    if os.path.exists(yol):
        return send_file(yol)
    return jsonify({"hata": "Dosya bulunamadı"}), 404


@app.route('/media/<dosya_adi>', methods=['GET'])
def goruntu_gonder(dosya_adi):
    yol = os.path.join(KARELER_KLASORU, dosya_adi)
    if os.path.exists(yol):
        return send_file(yol, mimetype='image/jpeg')
    return jsonify({"hata": "Dosya bulunamadı"}), 404


@app.route('/prediction/', methods=['POST'])
def sonuc_al():
    """
    v2.1.0 formatını doğrular: {frame, detected_objects, detected_translations,
    reference_predictions} — üst seviyede id/user YOK. Sadece sıradaki kare
    için kabul eder ve index'i ilerletir; eski/tekrar gönderimler 406 döner.
    """
    global _mevcut_index
    data = request.json

    gerekli_alanlar = ["frame", "detected_objects", "detected_translations",
                      "reference_predictions"]
    eksik = [alan for alan in gerekli_alanlar if alan not in data]
    if eksik:
        print(f"[UYARI] Eksik alanlar: {eksik}")
        return jsonify({"detail": f"Eksik: {eksik}"}), 400

    for obj in data.get("detected_objects", []):
        nesne_alanlari = ["cls", "landing_status", "moving_status",
                          "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"]
        obj_eksik = [a for a in nesne_alanlari if a not in obj]
        if obj_eksik:
            print(f"[UYARI] Nesne eksik alanlar: {obj_eksik}")

    if _mevcut_index >= TOPLAM_KARE:
        return jsonify({"detail": "Already sent."}), 406

    beklenen_url = _frame_listesi[_mevcut_index]["url"]
    if data.get("frame") != beklenen_url:
        print(f"[UYARI] Beklenmeyen frame: geldi={data.get('frame')} beklenen={beklenen_url}")
        return jsonify({"detail": "Already sent."}), 406

    nesne_sayisi = len(data.get("detected_objects", []))
    poz = data.get("detected_translations", [{}])[0] if data.get("detected_translations") else {}
    ref_sayisi = len(data.get("reference_predictions", []))

    print(f"[SONUC] frame={_mevcut_index} | "
          f"Nesne: {nesne_sayisi} | "
          f"Poz: ({poz.get('translation_x', '?')}, {poz.get('translation_y', '?')}, {poz.get('translation_z', '?')}) | "
          f"Ref.Obj: {ref_sayisi}")

    _mevcut_index += 1

    return jsonify({
        "durum": "basarili",
        "mesaj": "Sonuç alındı",
        "sunucu_saati": time.time()
    }), 201


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  SAHTE YARIŞMA SUNUCUSU (v2.1.0 uyumlu)")
    print(f"  Kareler: {KARELER_KLASORU}")
    print(f"  Toplam: {TOPLAM_KARE} kare")
    print(f"  Referanslar: {len(_referans_listesi)}")
    print("  Adres: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
