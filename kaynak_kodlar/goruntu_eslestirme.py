"""
TEKNOFEST 2026 - Görüntü Eşleştirme Modülü (Görev 3 - %25 Puan)
DINOv2 + SIFT hibrit yaklaşım.

Verilen referans nesneyi karede bulur. Çıktı alanı "reference" (referansın
kendi URL'i), "object_id" değil - v2.1.0 böyle istiyor.

Aynı anda tek referans aktif pencerede aranır (frame_start_image_url/
frame_end_image_url aralığı sunucudan gelir). eslestir()'e aktif_ref_anahtarlari
verilirse sadece o referanslar aranır - gerçek davranış bu, ayrıca 12
referansın hepsini taramanın FLANN darboğazını da önler.

DINOv2 aktifse DINO ve SIFT aynı karede birlikte çalışıp sonuçlar birleştirilir
(biri kaçırırsa diğeri yakalar). DINOv2 yoksa SIFT tek başına.

cv2 BGR okuyor, DINOv2/ImageNet normalizasyonu RGB bekliyor - bu yüzden
DINOv2 girişi öncesi BGR->RGB dönüşümü şart (_rgb_donustur). SIFT gri
çalıştığı için buna gerek yok.

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

warnings.filterwarnings("ignore", message="xFormers is not available")

# bu dosya kaynak_kodlar/ icinde, proje koku bir ust dizin - cwd degisirse kirilmasin diye mutlak yol
_PROJE_KOKU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GoruntuEslestirme:

    def __init__(self, dinov2_model_yolu=None, dinov2_repo_yolu=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESLESTIRME] Cihaz: {self.device}")

        model_yolu = dinov2_model_yolu or os.path.join(_PROJE_KOKU, "modeller", "dinov2_vits14.pth")
        repo_yolu = dinov2_repo_yolu or os.path.join(_PROJE_KOKU, "modeller", "dinov2_repo")

        # DINOv2 yükleme (tamamen offline)
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

        self.global_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.sift = cv2.SIFT_create(nfeatures=500)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.referanslar = {}

        # eşikler gerçek video verisiyle ölçüldü (test/dagilim_tani_v2.py):
        # max=0.472, p99=0.445 - net bir ayrım yok, eşik bilerek üst kuyrukta tutuldu
        self.DINO_ESIK = 0.45
        self.NMS_ESIK = 0.3
        self.SIFT_MIN_ESLESME = config.ESLESTIRME_MIN_ESLESME
        self.SIFT_RATIO_TEST = config.ESLESTIRME_RATIO_TEST
        self.SIFT_INLIER_ORANI_ESIK = 0.5
        self.SIFT_KARE_OLCEK = config.ESLESTIRME_SIFT_OLCEK

        # bir nesne en az 2 ardışık karede görülmeden çizilmez, tek-kare gürültüyü eler
        self._kare_sayac = -1
        self._son_tespitler = {}  # object_id -> (x1, y1, x2, y2, kare_no)

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
        """BGR -> RGB (DINOv2/ImageNet normalizasyonu için)."""
        return cv2.cvtColor(resim_bgr, cv2.COLOR_BGR2RGB)

    def _dino_patch_features(self, resim):
        """Tek forward pass ile tüm görüntünün patch-level feature haritasını çıkarır (518x518 -> 37x37 grid, 384 boyut)."""
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
        Referansı KxK patch gridi (şablon) olarak tutar; eşleştirme bu şablonu
        karenin patch gridi üzerinde kaydırıp en iyi hizalanan konumu arar.

        Eski yaklaşım referansı tek bir global vektöre sıkıştırıyordu, bu da
        nesnenin iç geometrisini kaybediyordu ve "bu bir çatı köşesi" gibi
        genel örüntüleri her benzer çatıda eşleşme sanıyordu (görsel doğrulandı).
        """
        if not self.dinov2_aktif:
            return None
        feat_map, grid_boyut, _ = self._dino_patch_features(resim)
        if feat_map is None:
            return None
        k = min(grid_k, grid_boyut)
        template = F.adaptive_avg_pool2d(feat_map, (k, k))
        return F.normalize(template, dim=1)  # [1, 384, k, k]

    def referans_yukle(self, ref_anahtari, referans_resim):
        """
        Tek referans nesne ekler (dosya yolu veya numpy array).
        ref_anahtari sunucudan geliyorsa referansın kendi URL'i - eslestir()
        çıktısında "reference" alanına aynen geri gider.
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

        ref_data["dino_template"] = self._dino_template_cikar(img)

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

    def _dino_ile_ara(self, kare, referans_havuzu):
        """
        Referansın KxK patch şablonunu karenin tüm patch gridinde kaydırıp
        (conv2d) en iyi cosine benzerliğini bulur - sadece genel görünüm değil,
        nesnenin iç geometrisi de eşleşmeye dahil olur.
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

    def _sift_ile_ara(self, kare, referans_havuzu):
        """SIFT tabanlı eşleştirme (DINOv2 yoksa fallback)."""
        bulunanlar = []
        kare_gri_tam = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY) if len(kare.shape) == 3 else kare

        # tam çözünürlükte SIFT laptopta çok yavaşladı (bkz. config.ESLESTIRME_SIFT_OLCEK),
        # küçültülmüş karede çalışıp keypoint'leri aşağıda orijinal koordinata geri ölçekliyoruz
        olcek = self.SIFT_KARE_OLCEK
        if olcek < 1.0:
            kare_gri = cv2.resize(kare_gri_tam, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
        else:
            kare_gri = kare_gri_tam

        kare_kp, kare_des = self.sift.detectAndCompute(kare_gri, None)
        if kare_des is None or len(kare_kp) < 10:
            return bulunanlar

        if olcek < 1.0:
            ters_olcek = 1.0 / olcek
            for kp in kare_kp:
                kp.pt = (kp.pt[0] * ters_olcek, kp.pt[1] * ters_olcek)

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
        Bir nesne en az 2 ardışık karede benzer konumda görülmeden çizilmez -
        tek-kareye dayalı SIFT eşleşmeleri komşu karede %67-90 oranla hiç
        örtüşmüyordu (test/eslestirme_tani.py). Onaylanan kutu önceki
        konumla yumuşatılır (EMA), zıplama azalır.
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

            yeni_son_tespitler[oid] = bbox + (self._kare_sayac,)  # henuz dogrulanmadi

        self._son_tespitler = yeni_son_tespitler
        return onayli

    def _cifte_dogrulama(self, dino_sonuc, sift_sonuc, iou_esik=0.3):
        """
        DINOv2 tek başına yanlış pozitif üretebiliyor (futbol sahası referansı
        çıplak toprağa eşlemişti, ama ardışık kare doğrulamasını geçecek kadar
        stabil görünüyordu). Bu yüzden bir nesne
        sadece DINO VE SIFT aynı bölgeyi BAĞIMSIZ işaret ettiğinde güvenilir sayılır.
        """
        onaylanan = []
        for d in dino_sonuc:
            for s in sift_sonuc:
                if d["reference"] != s["reference"]:
                    continue
                if self._iou_hesapla(d, s) > iou_esik:
                    onaylanan.append(s)  # SIFT homografi tabanlı, genelde daha dar/hassas kutu
                    break
        return onaylanan

    def eslestir(self, kare, aktif_ref_anahtarlari=None):
        """
        Ana eşleştirme fonksiyonu. DINOv2 aktifse çifte doğrulama, değilse SIFT tek başına.

        aktif_ref_anahtarlari: None -> yüklü tüm referanslar aranır (lokal
        fallback/test). Liste verilirse sadece o anahtarlar aranır - gerçek
        yarışma davranışı bu.

        Döner: [{"reference": <anahtar>, "top_left_x": .., ...}, ...]
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

        kare = self._bgr3_yap(kare)

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
