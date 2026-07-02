"""
main.py dongusundeki 3 gorevin (Nesne Tespiti, Pozisyon Kestirimi, Goruntu
Eslestirme) gercek per-frame suresini olcer.

sahte_sunucu her karede tam 1 aktif referans penceresi veriyor (12 referans
2250 kareyi bosluksuz bolusturuyor), yani Gorev 3 her karede calisiyor - bu
profil o senaryoyu (1 aktif referans her karede) taklit ediyor. 200 ardisik
karede (tracking/optical-flow zamana bagli oldugundan sirali) her asama
ayri zamanlanir.
"""

import sys
import os
import time
import statistics

sys.path.append('..')
import config
import cv2

from kaynak_kodlar.nesne_tespiti import NesneTespiti
from kaynak_kodlar.pozisyon_kestirimi import PozisyonKestirimi
from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

KARELER_DIR = "kareler"
ORNEK_KARE_SAYISI = 200

print("[PROFIL] Moduller yukleniyor...")
nesne_tespit = NesneTespiti(model_yolu="../" + config.NESNE_TESPIT_MODEL)
pozisyon = PozisyonKestirimi()
eslestirme = GoruntuEslestirme(
    dinov2_model_yolu="../modeller/dinov2_vits14.pth",
    dinov2_repo_yolu="../modeller/dinov2_repo",
)

# Gercek senaryo: her karede TEK bir referans aktif (bkz. docstring)
ref_klasor = "../referanslar"
ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
if ref_dosyalar:
    ilk_ref = ref_dosyalar[0]
    eslestirme.referans_yukle("test_ref", os.path.join(ref_klasor, ilk_ref))
    aktif_ref_anahtarlari = ["test_ref"]
    print(f"[PROFIL] Referans yuklendi: {ilk_ref} (her karede aktif kabul edilecek)")
else:
    aktif_ref_anahtarlari = None
    print("[PROFIL] UYARI: referans bulunamadi, Gorev 3 atlanacak")

kare_dosyalari = sorted(os.listdir(KARELER_DIR))[:ORNEK_KARE_SAYISI]
print(f"[PROFIL] {len(kare_dosyalari)} kare uzerinde profil cikariliyor...\n")

sureler = {"nesne_tespiti": [], "pozisyon": [], "eslestirme": [], "toplam": []}

for i, dosya in enumerate(kare_dosyalari):
    yol = os.path.join(KARELER_DIR, dosya)
    frame = cv2.imread(yol)
    if frame is None:
        continue
    temiz_kare = frame.copy()

    t0 = time.perf_counter()
    nesne_tespit.tespit_et(frame)
    t1 = time.perf_counter()

    pozisyon.guncelle(temiz_kare, None, None, None, 0)  # health=0 -> kendi kestirimi (yariscmada cogu zaman bu durum)
    t2 = time.perf_counter()

    if aktif_ref_anahtarlari:
        eslestirme.eslestir(temiz_kare, aktif_ref_anahtarlari)
    t3 = time.perf_counter()

    sureler["nesne_tespiti"].append(t1 - t0)
    sureler["pozisyon"].append(t2 - t1)
    sureler["eslestirme"].append(t3 - t2)
    sureler["toplam"].append(t3 - t0)

    if (i + 1) % 50 == 0:
        print(f"  ... {i+1}/{len(kare_dosyalari)}")

print("\n=== SONUC (kare basina ortalama, ms) ===")
toplam_ort = statistics.mean(sureler["toplam"]) * 1000
for asama in ["nesne_tespiti", "pozisyon", "eslestirme"]:
    ort = statistics.mean(sureler[asama]) * 1000
    medyan = statistics.median(sureler[asama]) * 1000
    yuzde = 100 * ort / toplam_ort
    print(f"  {asama:15s}: ortalama {ort:6.1f} ms | medyan {medyan:6.1f} ms | toplamin %{yuzde:.1f}'i")

print(f"\n  {'TOPLAM (3 gorev)':15s}: ortalama {toplam_ort:6.1f} ms/kare -> "
      f"{1000/toplam_ort:.2f} FPS -> 2250 kare icin ~{2250*toplam_ort/1000/60:.1f} dakika")
print("\n  NOT: Bu sure agdan/sunucu iletisiminden BAGIMSIZ, sadece 3 gorevin")
print("  hesaplama suresi. Gercek oturumda buna sunucu GET/POST round-trip'i")
print("  ve MIN_KARE_ARALIGI=0.25s tabani da eklenir.")
