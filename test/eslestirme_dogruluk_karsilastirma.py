"""
SIFT kucultme degisikliginin (config.ESLESTIRME_SIFT_OLCEK) eslestirme
dogrulugunu bozup bozmadigini kontrol eder - hiz kazanci zaten olculmustu,
burada olcek=1.0 (eski) ile olcek=0.33 (yeni) arasinda onaylanan eslesme
sayisi/konumu karsilastirilir (12 referans, pencere kisitlamasi olmadan).
"""
import sys, os, json
sys.path.append('..')
import config
import cv2

from kaynak_kodlar.goruntu_eslestirme import GoruntuEslestirme

KARELER_DIR = "kareler"
ORNEK_KARE_SAYISI = 120  # ilk 400'de surec takildi/asiri yavasladi, kapsam kucultuldu


def tam_gecis(olcek):
    config.ESLESTIRME_SIFT_OLCEK = olcek
    eslestirme = GoruntuEslestirme(
        dinov2_model_yolu="../modeller/dinov2_vits14.pth",
        dinov2_repo_yolu="../modeller/dinov2_repo",
    )
    ref_klasor = "../referanslar"
    ref_dosyalar = sorted([f for f in os.listdir(ref_klasor) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    for i, dosya in enumerate(ref_dosyalar):
        eslestirme.referans_yukle(i + 1, os.path.join(ref_klasor, dosya))

    kare_dosyalari = sorted(os.listdir(KARELER_DIR))[:ORNEK_KARE_SAYISI]
    onaylanan_toplam = 0
    onaylanan_detay = []
    for i, dosya in enumerate(kare_dosyalari):
        frame = cv2.imread(os.path.join(KARELER_DIR, dosya))
        if frame is None:
            continue
        sonuc = eslestirme.eslestir(frame, aktif_ref_anahtarlari=None)
        if sonuc:
            onaylanan_toplam += len(sonuc)
            for s in sonuc:
                onaylanan_detay.append((dosya, s["reference"], s["top_left_x"], s["top_left_y"],
                                         s["bottom_right_x"], s["bottom_right_y"]))
        if (i + 1) % 20 == 0:
            print(f"  [olcek={olcek}] ... {i+1}/{len(kare_dosyalari)}", flush=True)

    return onaylanan_toplam, onaylanan_detay


print("=== ESKI DAVRANIS (olcek=1.0) ===")
n_eski, detay_eski = tam_gecis(1.0)
print(f"Onaylanan eslesme: {n_eski}")
for d in detay_eski[:15]:
    print(" ", d)

print("\n=== YENI DAVRANIS (olcek=0.33) ===")
n_yeni, detay_yeni = tam_gecis(0.33)
print(f"Onaylanan eslesme: {n_yeni}")
for d in detay_yeni[:15]:
    print(" ", d)

print(f"\n=== KARSILASTIRMA === eski={n_eski}, yeni={n_yeni}")

with open("degerlendirme_ciktilari/eslestirme_dogruluk_karsilastirma.json", "w", encoding="utf-8") as f:
    json.dump({"eski_olcek_1.0": {"n": n_eski, "detay": detay_eski},
               "yeni_olcek_0.33": {"n": n_yeni, "detay": detay_yeni}}, f, ensure_ascii=False, indent=2)
