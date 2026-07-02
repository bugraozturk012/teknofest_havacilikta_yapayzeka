"""
TEKNOFEST 2026 - Nesne Tespiti Modülü (Görev 1 - %25 Puan)

Şartname Bölüm 2.1'e tam uyumlu:
- 4 sınıf: Taşıt(0), İnsan(1), UAP(2), UAİ(3)
- Taşıt için hareket durumu (kamera ego-motion kompanzasyonlu)
- UAP/UAİ için iniş uygunluğu (alan üzerinde cisim kontrolü)
- Çıktı: Şartname JSON formatında (cls, landing_status, moving_status, bbox)
  (v2.1.0 resmi örnek: alan adı "moving_status", "motion_status" değil)
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
    """
    Hava görüntülerinden nesne tespit, hareket analizi ve iniş uygunluğu sistemi.
    """

    def __init__(self, model_yolu=None):
        model_yolu = model_yolu or config.NESNE_TESPIT_MODEL
        
        if os.path.exists(model_yolu):
            print(f"[NESNE] Model yükleniyor: {model_yolu}")
            self.model = YOLO(model_yolu)
        else:
            print(f"[NESNE] {model_yolu} bulunamadı, {config.NESNE_TESPIT_FALLBACK} kullanılıyor")
            self.model = YOLO(config.NESNE_TESPIT_FALLBACK)

        # Hareket tespiti için geçmiş konumlar {tracker_id: [(x, y), ...]}
        self.gecmis_konumlar = {}
        
        # Kamera ego-motion kompanzasyonu için önceki kare
        self.onceki_kare_gri = None
        self.homography_matrix = None

    def _kamera_hareketi_hesapla(self, kare_gri):
        """
        Kamera ego-motion'ını hesaplar (Homography matrisi).
        Bu matris ile sabit nesnelerin piksel kayması kompanze edilir.
        
        Şartname uyarısı: "kamera sürekli hareket halinde, sabit nesneler de
        görüntü üzerinde hareketliymiş gibi algılanabilir"
        """
        self.homography_matrix = None
        
        if self.onceki_kare_gri is None:
            self.onceki_kare_gri = kare_gri.copy()
            return

        # Kanal uyumunu garantile (termal vs RGB karışımı olabilir)
        if kare_gri.shape != self.onceki_kare_gri.shape:
            self.onceki_kare_gri = kare_gri.copy()
            return

        # ORB ile özellik noktaları bul
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(self.onceki_kare_gri, None)
        kp2, des2 = orb.detectAndCompute(kare_gri, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.onceki_kare_gri = kare_gri.copy()
            return

        # BFMatcher ile eşleştir
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            self.onceki_kare_gri = kare_gri.copy()
            return

        # En iyi eşleşmeleri seç
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Homography hesapla (kamera hareketini temsil eder)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.homography_matrix = H

        self.onceki_kare_gri = kare_gri.copy()

    def _gercek_hareket_mi(self, eski_merkez, yeni_merkez):
        """
        Kamera ego-motion'ı çıkarıldıktan sonra nesnenin gerçekten
        hareket edip etmediğini kontrol eder.
        """
        if self.homography_matrix is None:
            # Homography hesaplanamadıysa, basit mesafe kontrolü yap
            mesafe = np.sqrt((yeni_merkez[0] - eski_merkez[0])**2 + 
                           (yeni_merkez[1] - eski_merkez[1])**2)
            return mesafe > config.HAREKET_PIKSEL_ESIGI

        # Eski merkezi homography ile transform et
        # (Kamera hareketi olmasaydı bu nokta nerede olurdu?)
        eski_pt = np.float32([[eski_merkez]]).reshape(-1, 1, 2)
        try:
            beklenen_konum = cv2.perspectiveTransform(eski_pt, self.homography_matrix)
            beklenen_x = beklenen_konum[0][0][0]
            beklenen_y = beklenen_konum[0][0][1]

            # Beklenen konum ile gerçek konum arasındaki fark = gerçek hareket
            gercek_kayma = np.sqrt((yeni_merkez[0] - beklenen_x)**2 + 
                                  (yeni_merkez[1] - beklenen_y)**2)
            return gercek_kayma > config.HAREKET_PIKSEL_ESIGI
        except:
            return False

    def _inis_uygunlugu_kontrol(self, alan_bbox, tum_tespitler, kare_boyut):
        """
        UAP/UAİ alanının iniş için uygun olup olmadığını kontrol eder.

        Şartname kuralları:
        - Alan üzerinde herhangi bir nesne varsa: uygun değil (0)
        - Alan tamamen boşsa: uygun (1)
        - Alanın tamamı kare içinde değilse: uygun değil (0)
        - Perspektif nedeniyle yakın cisimler alan üstündeymiş gibi görünüyorsa: uygun değil (0)
          (Şekil 11) — bu senaryo gerçek derinlik bilgisi olmadan tam çözülemez;
          burada alan bbox'u kontrol amaçlı bir pay ile genişletilerek yaklaşık
          olarak ele alınıyor (raporlanan alan bbox'u değişmez).
        """
        ax1, ay1, ax2, ay2 = alan_bbox
        kare_h, kare_w = kare_boyut

        # 1. Alan tamamı kare içinde mi?
        if ax1 <= 5 or ay1 <= 5 or ax2 >= kare_w - 5 or ay2 >= kare_h - 5:
            return config.INIS_UYGUN_DEGIL

        # 2. Kontrol için alanı bir pay ile genişlet (yakın cisim yaklaşımı)
        pay_x = (ax2 - ax1) * config.INIS_KONTROL_GENISLETME_ORANI
        pay_y = (ay2 - ay1) * config.INIS_KONTROL_GENISLETME_ORANI
        gax1, gay1 = ax1 - pay_x, ay1 - pay_y
        gax2, gay2 = ax2 + pay_x, ay2 + pay_y

        # 3. Genişletilmiş alan üzerinde/yakınında başka nesne var mı?
        for tespit in tum_tespitler:
            t_sinif = tespit["cls_id"]
            # Kendisi hariç diğer nesnelere bak
            if t_sinif in [config.SINIF_UAP, config.SINIF_UAI]:
                continue

            tx1, ty1, tx2, ty2 = tespit["bbox_xyxy"]

            # Çakışma kontrolü (IoU yerine basit overlap)
            overlap_x1 = max(gax1, tx1)
            overlap_y1 = max(gay1, ty1)
            overlap_x2 = min(gax2, tx2)
            overlap_y2 = min(gay2, ty2)

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # Alan üzerinde/yakınında cisim var
                return config.INIS_UYGUN_DEGIL

        return config.INIS_UYGUN

    def tespit_et(self, frame):
        """
        Ana tespit fonksiyonu. Görüntüyü alır, tüm görevleri yapar.
        
        Returns:
            list[dict]: Şartname formatında nesne listesi
            [
                {
                    "cls": "0",
                    "landing_status": "-1",
                    "moving_status": "1",
                    "top_left_x": 100.5,
                    "top_left_y": 200.3,
                    "bottom_right_x": 300.7,
                    "bottom_right_y": 450.2
                },
                ...
            ]
        """
        if frame is None:
            return []

        frame = _bgr_yap(frame)      # termal/gri → 3 kanallı BGR
        kare_gri = _gri_yap(frame)
        kare_h, kare_w = frame.shape[:2]

        # Kamera ego-motion hesapla (hareket tespiti için)
        self._kamera_hareketi_hesapla(kare_gri)

        # YOLO ile tespit + tracking
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

        # Önce tüm tespitleri topla (iniş kontrolü için)
        ham_tespitler = []
        
        xyxys = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
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

        # Şartname formatına dönüştür
        sonuclar = []

        for tespit in ham_tespitler:
            sinif_id = tespit["cls_id"]
            takip_id = tespit["takip_id"]
            x1, y1, x2, y2 = tespit["bbox_xyxy"]
            merkez = tespit["merkez"]

            # Varsayılan değerler
            landing_status = str(config.INIS_YOK)     # -1
            moving_status = str(config.HAREKET_YOK)    # -1

            # === TAŞIT İÇİN HAREKET DURUMU ===
            if sinif_id == config.SINIF_TASIT:
                if takip_id != -1:
                    if takip_id in self.gecmis_konumlar:
                        eski_merkez = self.gecmis_konumlar[takip_id]
                        hareketli = self._gercek_hareket_mi(eski_merkez, merkez)
                        moving_status = str(config.HAREKET_HAREKETLI if hareketli
                                          else config.HAREKET_HAREKETSIZ)
                    else:
                        # İlk görülme, hareketsiz varsay
                        moving_status = str(config.HAREKET_HAREKETSIZ)
                    self.gecmis_konumlar[takip_id] = merkez
                else:
                    moving_status = str(config.HAREKET_HAREKETSIZ)

            # === UAP/UAİ İÇİN İNİŞ DURUMU ===
            elif sinif_id in [config.SINIF_UAP, config.SINIF_UAI]:
                inis = self._inis_uygunlugu_kontrol(
                    (x1, y1, x2, y2), ham_tespitler, (kare_h, kare_w)
                )
                landing_status = str(inis)

            # Şartname JSON formatına dönüştür
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
