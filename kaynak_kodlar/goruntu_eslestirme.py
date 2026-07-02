"""
TEKNOFEST 2026 - Görüntü Eşleştirme Modülü (Görev 3 - %25 Puan)
DINOv2 + SIFT Hibrit Yaklaşım

Şartname Bölüm 2.3'e tam uyumlu:
- Birden fazla referans nesne (oturum başında dinamik yüklenir)
- Cross-modal eşleştirme (termal→RGB, uydu→drone, farklı açı/irtifa)
- Çıktı: reference (referansın kendi URL'i/anahtarı), top_left_x/y, bottom_right_x/y
  (v2.1.0 resmi örnek: "object_id" değil "reference" alanı gönderiliyor,
  ReferencePrediction.reference_url olarak sunucuya iletiliyor)

Görev 3 gerçek davranışı (Q&A toplantısı + v2.1.0 /reference/ endpoint'i): aynı anda
sadece tek bir referans aktif pencerede aranır (frame_start_image_url/frame_end_image_url
aralığı, sunucu tarafından verilir). eslestir()'e aktif_ref_anahtarlari verilirse sadece
o referanslar aranır - bu hem gerçek davranışla örtüşür hem de 12 referansın hepsini her
karede aramanın getirdiği FLANN darboğazını (bkz. test/sift_profil.py) ortadan kaldırır.

Gerçek hibrit davranış: DINOv2 aktifse DINOv2 ve SIFT taramaları AYNI karede
birlikte çalıştırılıp sonuçlar NMS ile birleştirilir (biri kaçırırsa diğeri
yakalayabilir). DINOv2 yoksa SIFT tek başına devreye girer.

Renk kanalı notu: cv2 ile okunan/indirilen görüntüler BGR sıralıdır.
DINOv2/ImageNet normalizasyonu RGB sıralama bekler — bu yüzden DINOv2'ye
giriş öncesi mutlaka BGR→RGB dönüşümü yapılır (_rgb_donustur). SIFT gri
tonlamalı çalıştığı için bu dönüşüme ihtiyaç duymaz.

Offline çalışır: modeller/dinov2_repo + modeller/dinov2_vits14.pth
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import warnings
import sys
import os

sys.path.append('..')
import config

# xFormers uyarılarını sustur (opsiyonel kütüphane, olmasa da çalışır)
warnings.filterwarnings("ignore", message="xFormers is not available")

# Bu dosya kaynak_kodlar/ icinde oldugu icin proje koku bir ust dizindir.
# cwd'ye bagli relatif yol kullanmamak icin (calistirildigi dizin degisirse
# kirilmasin) modeller/ yolu burada mutlak olarak sabitlenir.
_PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GoruntuEslestirme:

    def __init__(self, dinov2_model_yolu=None, dinov2_repo_yolu=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESLESTIRME] Cihaz: {self.device}")

        model_yolu = dinov2_model_yolu or os.path.join(_PROJE_KOKU, "modeller", "dinov2_vits14.pth")
        repo_yolu = dinov2_repo_yolu or os.path.join(_PROJE_KOKU, "modeller", "dinov2_repo")

        # ============================================================
        # DINOv2 YÜKLEME (Tamamen Offline)
        # ============================================================
        try:
            sys.path.insert(0, repo_yolu)
            from dinov2.models.vision_transformer import vit_small

            self.model = vit_small(
                patch_size=14,
                init_values=1.0,
                block_chunks=0,
                num_register_tokens=4,
                img_size=518
            )
            state_dict = torch.load(
                model_yolu, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.dinov2_aktif = True
            print(f"[ESLESTIRME] DINOv2 yüklendi (GPU: {self.device})")
        except Exception as e:
            print(f"[ESLESTIRME] DINOv2 yüklenemedi: {e}")
            print("[ESLESTIRME] SIFT-only moduna geçiliyor")
            self.dinov2_aktif = False

        # ============================================================
        # GÖRÜNTÜ ÖN İŞLEME
        # ============================================================
        self.global_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # ============================================================
        # SIFT (Yedek + hassas bbox)
        # ============================================================
        self.sift = cv2.SIFT_create(nfeatures=500)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # ============================================================
        # REFERANSLAR VE EŞİKLER
        # ============================================================
        self.referanslar = {}
        # Gercek video+referans verisiyle olculen dagilima gore secildi
        # (test/dagilim_tani_v2.py, pencere-ortalama template metrigi):
        # gozlenen max=0.472, p99=0.445 - net bir sinyal/gurultu ayrimi yok,
        # bu yuzden esik bilerek ust kuyrukta (~%0.7 gecis orani) tutuluyor.
        self.DINO_ESIK = 0.45
        self.NMS_ESIK = 0.3
        self.SIFT_MIN_ESLESME = config.ESLESTIRME_MIN_ESLESME
        self.SIFT_RATIO_TEST = config.ESLESTIRME_RATIO_TEST
        self.SIFT_INLIER_ORANI_ESIK = 0.5

        # Ardisik kare dogrulamasi: bir nesne en az 2 ardisik karede
        # benzer konumda gorulmeden ekrana cizilmez (tek-kare gurultu
        # tespitlerini eler), gosterilen kutu da onceki konumla yumusatilir.
        self._kare_sayac = -1
        self._son_tespitler = {}  # object_id -> (x1, y1, x2, y2, kare_no)

    # ================================================================
    # FEATURE ÇIKARMA
    # ================================================================
    def _bgr3_yap(self, resim):
        """Termal, gri veya tek kanallı görüntüyü 3 kanallı BGR'ye çevirir."""
        if resim is None:
            return None
        if len(resim.shape) == 2:
            return cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
        if resim.shape[2] == 1:
            return cv2.cvtColor(resim[:, :, 0], cv2.COLOR_GRAY2BGR)
        return resim

    def _rgb_donustur(self, resim_bgr):
        """3 kanallı BGR görüntüyü DINOv2/ImageNet normalizasyonunun beklediği RGB sıralamasına çevirir."""
        return cv2.cvtColor(resim_bgr, cv2.COLOR_BGR2RGB)

    def _dino_patch_features(self, resim):
        """
        Tek forward pass ile tüm görüntünün patch-level feature haritasını çıkarır.
        518x518 giriş → 37x37 patch grid → her patch 384 boyutlu vektör.
        """
        if not self.dinov2_aktif:
            return None, None, None
        resim_bgr = self._bgr3_yap(resim)
        kare_h, kare_w = resim_bgr.shape[:2]
        resim_rgb = self._rgb_donustur(resim_bgr)

        img_tensor = self.global_transform(resim_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # [1, 1369, 384]

        grid_boyut = int(patch_tokens.shape[1] ** 0.5)  # 37
        feat_map = patch_tokens.reshape(1, grid_boyut, grid_boyut, -1)
        feat_map = feat_map.permute(0, 3, 1, 2)  # [1, 384, 37, 37]

        return feat_map, grid_boyut, (kare_w, kare_h)

    def _dino_template_cikar(self, resim, grid_k=6):
        """
        Referans gorseli icin patch-seviyeli sablon (template) cikarir.

        Onceki yaklasimda referans tek bir 384-boyutlu global vektore
        sikistirilip karenin HER patch'iyle tek tek karsilastiriliyordu -
        bu, nesnenin ic uzamsal yapisini (koselerin/kenarlarin birbirine
        gore konumu) tamamen kaybediyordu. Gercek videoda bu yuzden DINOv2
        "bu bir cati kosesi" gibi cok genel bir oruntuyu yakalayip sahnedeki
        BIRBIRINE BENZER her catida ayri ayri "eslesme" buluyordu
        (test/obj10_kanit/ ile gorsel olarak dogrulandi).

        Bu fonksiyon yerine referansi KxK'lik bir patch gridi (sablon)
        olarak tutar; eslestirme bu sablonu karenin patch gridi uzerinde
        kaydirip (cross-correlation) en iyi hizalanan konumu arar -
        boylece nesnenin ic geometrisi de karsilastirmaya girer.
        """
        if not self.dinov2_aktif:
            return None
        feat_map, grid_boyut, _ = self._dino_patch_features(resim)
        if feat_map is None:
            return None
        k = min(grid_k, grid_boyut)
        template = F.adaptive_avg_pool2d(feat_map, (k, k))
        return F.normalize(template, dim=1)  # [1, 384, k, k]

    # ================================================================
    # REFERANS YÜKLEME
    # ================================================================
    def referans_yukle(self, ref_anahtari, referans_resim):
        """
        Tek referans nesne ekler (dosya yolu veya numpy array).

        ref_anahtari: sunucudan geliyorsa referansın kendi URL'i (ref['url'],
        eslestir() çıktısında "reference" alanına aynen geri gönderilecek),
        lokal fallback'te (referanslar/ klasörü) keyfi bir tamsayı/string olabilir.
        """
        if isinstance(referans_resim, str):
            if not os.path.exists(referans_resim):
                print(f"[ESLESTIRME] Dosya bulunamadı: {referans_resim}")
                return False
            img = cv2.imread(referans_resim)
        else:
            img = referans_resim

        if img is None:
            return False

        ref_data = {"img": img, "boyut": img.shape[:2]}

        # DINOv2 patch-template (bkz. _dino_template_cikar docstring)
        ref_data["dino_template"] = self._dino_template_cikar(img)

        # SIFT feature
        img_gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kp, des = self.sift.detectAndCompute(img_gri, None)
        ref_data["sift_kp"] = kp
        ref_data["sift_des"] = des

        self.referanslar[ref_anahtari] = ref_data
        dino_str = "DINO aktif" if ref_data["dino_template"] is not None else "DINO yok"
        sift_str = f"SIFT {len(kp) if kp else 0}"
        print(f"[ESLESTIRME] Referans {ref_anahtari} yüklendi ({dino_str}, {sift_str})")
        return True

    def referanslari_toplu_yukle(self, referans_listesi):
        """Oturum başında birden fazla referans yükler."""
        yuklenen = sum(1 for r in referans_listesi
                      if self.referans_yukle(r["reference"], r["image"]))
        print(f"[ESLESTIRME] {yuklenen}/{len(referans_listesi)} referans yüklendi.")

    # ================================================================
    # DINOv2 İLE ARAMA
    # ================================================================
    def _dino_ile_ara(self, kare, referans_havuzu):
        """
        DINOv2 patch-template cross-correlation ile referansları arar.

        Referansın KxK'lik iç patch yapısı, karenin tüm patch gridi
        üzerinde kaydırılır (conv2d ile); her konumda referans
        patch'lerinin karşılık gelen kare patch'leriyle ortalama cosine
        benzerliği hesaplanır. Bu, sadece "genel görünüm" değil nesnenin
        iç geometrisini de eşleştirmeye dahil eder (bkz. _dino_template_cikar).
        """
        bulunanlar = []
        feat_map, grid_boyut, kare_boyut = self._dino_patch_features(kare)
        if feat_map is None:
            return bulunanlar
        kare_w, kare_h = kare_boyut
        feat_norm = F.normalize(feat_map, dim=1)  # [1, 384, 37, 37]

        for ref_anahtari, ref_data in referans_havuzu.items():
            template = ref_data.get("dino_template")
            if template is None:
                continue
            k = template.shape[-1]
            if grid_boyut < k:
                continue

            with torch.no_grad():
                skor_haritasi = F.conv2d(feat_norm, weight=template) / (k * k)
            skor_np = skor_haritasi.squeeze().cpu().numpy()
            if skor_np.ndim == 0:
                skor_np = skor_np.reshape(1, 1)

            r0, c0 = np.unravel_index(np.argmax(skor_np), skor_np.shape)
            en_iyi_skor = float(skor_np[r0, c0])

            if en_iyi_skor < self.DINO_ESIK:
                continue

            olcek_x = kare_w / grid_boyut
            olcek_y = kare_h / grid_boyut
            x1 = float(c0 * olcek_x)
            y1 = float(r0 * olcek_y)
            x2 = float((c0 + k) * olcek_x)
            y2 = float((r0 + k) * olcek_y)

            bulunanlar.append({
                "reference": ref_anahtari,
                "top_left_x": max(0, x1),
                "top_left_y": max(0, y1),
                "bottom_right_x": min(kare_w, x2),
                "bottom_right_y": min(kare_h, y2),
                "skor": en_iyi_skor
            })
        return bulunanlar

    # ================================================================
    # SIFT İLE ARAMA (Fallback)
    # ================================================================
    def _sift_ile_ara(self, kare, referans_havuzu):
        """SIFT tabanlı eşleştirme (DINOv2 yoksa)."""
        bulunanlar = []
        kare_gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY) if len(kare.shape) == 3 else kare
        kare_kp, kare_des = self.sift.detectAndCompute(kare_gri, None)
        if kare_des is None or len(kare_kp) < 10:
            return bulunanlar

        for ref_anahtari, ref_data in referans_havuzu.items():
            ref_des = ref_data.get("sift_des")
            ref_kp = ref_data.get("sift_kp")
            if ref_des is None or len(ref_kp) < 5:
                continue
            try:
                matches = self.flann.knnMatch(ref_des, kare_des, k=2)
            except:
                continue

            iyi = []
            for pair in matches:
                if len(pair) == 2 and pair[0].distance < self.SIFT_RATIO_TEST * pair[1].distance:
                    iyi.append(pair[0])
            if len(iyi) < self.SIFT_MIN_ESLESME:
                continue

            src = np.float32([ref_kp[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
            dst = np.float32([kare_kp[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)
            try:
                M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            except:
                continue
            if M is None or (mask is not None and np.sum(mask) / len(mask) < self.SIFT_INLIER_ORANI_ESIK):
                continue

            ref_h, ref_w = ref_data["boyut"]
            pts = np.float32([[0, 0], [0, ref_h-1], [ref_w-1, ref_h-1], [ref_w-1, 0]]).reshape(-1, 1, 2)
            try:
                dst_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            except:
                continue

            x1 = float(np.min(dst_pts[:, 0]))
            y1 = float(np.min(dst_pts[:, 1]))
            x2 = float(np.max(dst_pts[:, 0]))
            y2 = float(np.max(dst_pts[:, 1]))
            h_px, w_px = kare.shape[:2]
            if x1 < -50 or y1 < -50 or x2 > w_px + 50 or y2 > h_px + 50:
                continue
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            bulunanlar.append({
                "reference": ref_anahtari,
                "top_left_x": max(0, x1), "top_left_y": max(0, y1),
                "bottom_right_x": min(w_px, x2), "bottom_right_y": min(h_px, y2),
                "skor": len(iyi) / max(len(matches), 1)
            })
        return bulunanlar

    # ================================================================
    # NMS VE ANA FONKSİYON
    # ================================================================
    def _iou_hesapla(self, a, b):
        x1 = max(a["top_left_x"], b["top_left_x"])
        y1 = max(a["top_left_y"], b["top_left_y"])
        x2 = min(a["bottom_right_x"], b["bottom_right_x"])
        y2 = min(a["bottom_right_y"], b["bottom_right_y"])
        if x1 >= x2 or y1 >= y2:
            return 0.0
        kesisim = (x2 - x1) * (y2 - y1)
        a_alan = (a["bottom_right_x"] - a["top_left_x"]) * (a["bottom_right_y"] - a["top_left_y"])
        b_alan = (b["bottom_right_x"] - b["top_left_x"]) * (b["bottom_right_y"] - b["top_left_y"])
        birlesim = a_alan + b_alan - kesisim
        return kesisim / birlesim if birlesim > 0 else 0.0

    def _nms(self, tespitler):
        if len(tespitler) <= 1:
            return tespitler
        gruplar = {}
        for t in tespitler:
            gruplar.setdefault(t["reference"], []).append(t)
        sonuclar = []
        for oid, grup in gruplar.items():
            grup.sort(key=lambda x: x["skor"], reverse=True)
            secilen = []
            for t in grup:
                if not any(self._iou_hesapla(t, s) > self.NMS_ESIK for s in secilen):
                    secilen.append(t)
            if secilen:
                sonuclar.append(secilen[0])
        return sonuclar

    def _ardisik_kare_dogrula(self, tespitler):
        """
        Bir nesne en az 2 ardisik karede benzer konumda gorulmeden ekrana
        cizilmez (test/eslestirme_tani.py ile olculdu: tek-kareye dayali
        SIFT eslesmeleri ardisik karede %67-90 oranla hic ortusmuyordu -
        bu adim o gurultu tespitlerini eler). Onaylanan kutu onceki
        konumla yumusatilir (EMA), boylece zıplama azalir.
        """
        self._kare_sayac += 1
        onayli = []
        yeni_son_tespitler = {}

        for t in tespitler:
            oid = t["reference"]
            bbox = (t["top_left_x"], t["top_left_y"], t["bottom_right_x"], t["bottom_right_y"])
            onceki = self._son_tespitler.get(oid)

            if onceki is not None and onceki[4] == self._kare_sayac - 1:
                onceki_bbox = onceki[:4]
                ortusme = self._iou_hesapla(
                    {"top_left_x": onceki_bbox[0], "top_left_y": onceki_bbox[1],
                     "bottom_right_x": onceki_bbox[2], "bottom_right_y": onceki_bbox[3]},
                    {"top_left_x": bbox[0], "top_left_y": bbox[1],
                     "bottom_right_x": bbox[2], "bottom_right_y": bbox[3]}
                )
                if ortusme > 0.15:
                    yumusak = tuple(0.5 * o + 0.5 * n for o, n in zip(onceki_bbox, bbox))
                    t["top_left_x"], t["top_left_y"], t["bottom_right_x"], t["bottom_right_y"] = yumusak
                    onayli.append(t)
                    yeni_son_tespitler[oid] = yumusak + (self._kare_sayac,)
                    continue

            # Henuz dogrulanmadi - bu karenin ham konumu bir sonraki kare icin saklanir
            yeni_son_tespitler[oid] = bbox + (self._kare_sayac,)

        self._son_tespitler = yeni_son_tespitler
        return onayli

    def _cifte_dogrulama(self, dino_sonuc, sift_sonuc, iou_esik=0.3):
        """
        DINOv2 tek başına yanlış pozitif üretebiliyor (test/template_kanit/
        ile görsel olarak doğrulandı: futbol sahası referansı çıplak toprak
        yamaca eşleşmişti, ama ardışık kare doğrulamasını geçecek kadar
        "stabil" görünüyordu). Bu yüzden bir nesne sadece DINO VE SIFT aynı
        bölgeyi BAĞIMSIZ olarak işaret ettiğinde güvenilir sayılır - yanlış
        pozitif riski artık tek kaynağa değil, ikisinin AYNI ANDA ve AYNI
        YANLIŞ yerde hata yapmasına bağlı (çok daha düşük olasılık).
        """
        onaylanan = []
        for d in dino_sonuc:
            for s in sift_sonuc:
                if d["reference"] != s["reference"]:
                    continue
                if self._iou_hesapla(d, s) > iou_esik:
                    # SIFT homografi tabanli oldugu icin genelde daha hassas/dar kutu cizer
                    onaylanan.append(s)
                    break
        return onaylanan

    def eslestir(self, kare, aktif_ref_anahtarlari=None):
        """
        Ana eşleştirme fonksiyonu.
        DINOv2 aktifse → DINO ve SIFT bağımsız olarak aynı bölgeyi
        bulduğunda (çifte doğrulama) güvenilir sayılır.
        DINOv2 yoksa → sadece SIFT (fallback).

        aktif_ref_anahtarlari: None ise yüklü TÜM referanslar aranır (lokal
        fallback/test amaçlı). Bir liste verilirse SADECE o anahtarlardaki
        referanslar aranır - gerçek yarışma davranışı budur (aynı anda tek
        referans/dar bir pencere aktif olur, bkz. modül docstring'i) ve
        12 referansın hepsini taramanın FLANN darboğazını önler.

        Returns: [{"reference": <anahtar>, "top_left_x": .., ...}, ...]
        """
        if not self.referanslar or kare is None:
            return []

        if aktif_ref_anahtarlari is not None:
            referans_havuzu = {k: v for k, v in self.referanslar.items()
                              if k in aktif_ref_anahtarlari}
            if not referans_havuzu:
                return []
        else:
            referans_havuzu = self.referanslar

        kare = self._bgr3_yap(kare)  # termal/gri giriş garantisi (BGR, SIFT için doğru format)

        sift_sonuc = self._sift_ile_ara(kare, referans_havuzu)

        if self.dinov2_aktif:
            dino_sonuc = self._dino_ile_ara(kare, referans_havuzu)
            tespitler = self._cifte_dogrulama(dino_sonuc, sift_sonuc)
        else:
            tespitler = sift_sonuc

        tespitler = self._nms(tespitler)
        tespitler = self._ardisik_kare_dogrula(tespitler)

        for t in tespitler:
            t.pop("skor", None)
            for k in ["top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"]:
                t[k] = round(t[k], 2)
        return tespitler