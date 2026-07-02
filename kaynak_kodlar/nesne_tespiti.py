"""
TEKNOFEST 2026 - Nesne Tespiti Modülü (Görev 1 - %25 Puan)

4 sınıf: Taşıt(0), İnsan(1), UAP(2), UAİ(3). Taşıt için hareket durumu
(ego-motion kompanzasyonlu), UAP/UAİ için iniş uygunluğu.
Çıktı alanı "moving_status" - v2.1.0'da "motion_status" değil bu.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

sys.path.append('..')
import config


def _gri_yap(frame):
    """BGR, tek kanallı veya gri (termal) görüntüyü gri formatına getirir."""
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _bgr_yap(frame):
    """Termal/gri görüntüyü YOLO'nun beklediği 3 kanallı BGR'ye çevirir."""
    if frame is None:
        return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame


class NesneTespiti:
    """Hava görüntülerinden nesne tespit, hareket analizi ve iniş uygunluğu."""

    def __init__(self, model_yolu=None):
        model_yolu = model_yolu or config.NESNE_TESPIT_MODEL

        if os.path.exists(model_yolu):
            print(f"[NESNE] Model yükleniyor: {model_yolu}")
            self.model = YOLO(model_yolu)
        else:
            print(f"[NESNE] {model_yolu} bulunamadı, {config.NESNE_TESPIT_FALLBACK} kullanılıyor")
            self.model = YOLO(config.NESNE_TESPIT_FALLBACK)

        self.gecmis_konumlar = {}  # {takip_id: son_merkez} - hareket tespiti için
        self.takip_gorulme_sayisi = {}  # {takip_id: kac_karedir_gorunuyor}

        self.onceki_kare_gri = None
        self.homography_matrix = None

    def _kamera_hareketi_hesapla(self, kare_gri):
        """Homography ile kamera ego-motion'ını çıkarır, sabit nesnelerin piksel kaymasını kompanze eder."""
        self.homography_matrix = None

        if self.onceki_kare_gri is None:
            self.onceki_kare_gri = kare_gri.copy()
            return

        if kare_gri.shape != self.onceki_kare_gri.shape:  # termal/RGB karışmış olabilir
            self.onceki_kare_gri = kare_gri.copy()
            return

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(self.onceki_kare_gri, None)
        kp2, des2 = orb.detectAndCompute(kare_gri, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.onceki_kare_gri = kare_gri.copy()
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            self.onceki_kare_gri = kare_gri.copy()
            return

        matches = sorted(matches, key=lambda x: x.distance)[:100]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.homography_matrix = H

        self.onceki_kare_gri = kare_gri.copy()

    def _gercek_hareket_mi(self, eski_merkez, yeni_merkez):
        """Kamera hareketi çıkarıldıktan sonra nesne gerçekten yer değiştirmiş mi."""
        if self.homography_matrix is None:
            mesafe = np.sqrt((yeni_merkez[0] - eski_merkez[0])**2 +
                           (yeni_merkez[1] - eski_merkez[1])**2)
            return mesafe > config.HAREKET_PIKSEL_ESIGI

        # kamera hareketi olmasaydı bu nokta nerede olurdu
        eski_pt = np.float32([[eski_merkez]]).reshape(-1, 1, 2)
        try:
            beklenen_konum = cv2.perspectiveTransform(eski_pt, self.homography_matrix)
            beklenen_x = beklenen_konum[0][0][0]
            beklenen_y = beklenen_konum[0][0][1]

            gercek_kayma = np.sqrt((yeni_merkez[0] - beklenen_x)**2 +
                                  (yeni_merkez[1] - beklenen_y)**2)
            return gercek_kayma > config.HAREKET_PIKSEL_ESIGI
        except:
            return False

    def _inis_uygunlugu_kontrol(self, alan_bbox, tum_tespitler, kare_boyut):
        """
        UAP/UAİ alanı boşsa uygun, üzerinde/yakınında nesne varsa veya kare
        dışına taşıyorsa uygun değil. Perspektif payı (Şekil 11) için alan
        kontrol amaçlı genişletiliyor, raporlanan bbox değişmiyor.
        """
        ax1, ay1, ax2, ay2 = alan_bbox
        kare_h, kare_w = kare_boyut

        if ax1 <= 5 or ay1 <= 5 or ax2 >= kare_w - 5 or ay2 >= kare_h - 5:
            return config.INIS_UYGUN_DEGIL

        pay_x = (ax2 - ax1) * config.INIS_KONTROL_GENISLETME_ORANI
        pay_y = (ay2 - ay1) * config.INIS_KONTROL_GENISLETME_ORANI
        gax1, gay1 = ax1 - pay_x, ay1 - pay_y
        gax2, gay2 = ax2 + pay_x, ay2 + pay_y

        for tespit in tum_tespitler:
            t_sinif = tespit["cls_id"]
            if t_sinif in [config.SINIF_UAP, config.SINIF_UAI]:
                continue

            tx1, ty1, tx2, ty2 = tespit["bbox_xyxy"]

            overlap_x1 = max(gax1, tx1)
            overlap_y1 = max(gay1, ty1)
            overlap_x2 = min(gax2, tx2)
            overlap_y2 = min(gay2, ty2)

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                return config.INIS_UYGUN_DEGIL

        return config.INIS_UYGUN

    def tespit_et(self, frame):
        """
        Karedeki tüm nesneleri döner, şartname formatında:
        [{"cls": "0", "landing_status": "-1", "moving_status": "1",
          "top_left_x": .., "top_left_y": .., "bottom_right_x": .., "bottom_right_y": ..}, ...]
        """
        if frame is None:
            return []

        frame = _bgr_yap(frame)
        kare_gri = _gri_yap(frame)
        kare_h, kare_w = frame.shape[:2]

        self._kamera_hareketi_hesapla(kare_gri)

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=config.NESNE_TESPIT_CONF,
            iou=config.NESNE_TESPIT_IOU
        )

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        ham_tespitler = []

        xyxys = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.int().cpu().tolist()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(xyxys)

        for bbox, sinif_id, takip_id in zip(xyxys, clss, ids):
            x1, y1, x2, y2 = bbox
            merkez_x = (x1 + x2) / 2
            merkez_y = (y1 + y2) / 2

            ham_tespitler.append({
                "cls_id": sinif_id,
                "takip_id": takip_id,
                "bbox_xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "merkez": (merkez_x, merkez_y)
            })

        for tespit in ham_tespitler:
            takip_id = tespit["takip_id"]
            if takip_id != -1:
                self.takip_gorulme_sayisi[takip_id] = self.takip_gorulme_sayisi.get(takip_id, 0) + 1

        # iniş kontrolü aşağıda ham_tespitler (filtresiz) üzerinden çalışıyor -
        # henüz onaylanmamış bir nesne bile alan üzerinde fiziksel engel sayılmalı
        sonuclar = []

        for tespit in ham_tespitler:
            sinif_id = tespit["cls_id"]
            takip_id = tespit["takip_id"]
            x1, y1, x2, y2 = tespit["bbox_xyxy"]
            merkez = tespit["merkez"]

            # yeterince ardışık karede görülmeyen takip henüz raporlanmıyor
            if takip_id != -1 and self.takip_gorulme_sayisi.get(takip_id, 0) < config.NESNE_TESPIT_TAKIP_ONAY_ESIGI:
                continue

            landing_status = str(config.INIS_YOK)
            moving_status = str(config.HAREKET_YOK)

            if sinif_id == config.SINIF_TASIT:
                if takip_id != -1:
                    if takip_id in self.gecmis_konumlar:
                        eski_merkez = self.gecmis_konumlar[takip_id]
                        hareketli = self._gercek_hareket_mi(eski_merkez, merkez)
                        moving_status = str(config.HAREKET_HAREKETLI if hareketli
                                          else config.HAREKET_HAREKETSIZ)
                    else:
                        moving_status = str(config.HAREKET_HAREKETSIZ)  # ilk görülme
                    self.gecmis_konumlar[takip_id] = merkez
                else:
                    moving_status = str(config.HAREKET_HAREKETSIZ)

            elif sinif_id in [config.SINIF_UAP, config.SINIF_UAI]:
                inis = self._inis_uygunlugu_kontrol(
                    (x1, y1, x2, y2), ham_tespitler, (kare_h, kare_w)
                )
                landing_status = str(inis)

            sonuclar.append({
                "cls": str(sinif_id),
                "landing_status": landing_status,
                "moving_status": moving_status,
                "top_left_x": round(x1, 2),
                "top_left_y": round(y1, 2),
                "bottom_right_x": round(x2, 2),
                "bottom_right_y": round(y2, 2)
            })

        return sonuclar
