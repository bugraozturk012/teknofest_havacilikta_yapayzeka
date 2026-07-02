"""
TEKNOFEST 2026 - Pozisyon Kestirimi Modülü (Görev 2 - %40 Puan)
Kalman Filter + Optical Flow + Keyframe Kalibrasyon

3 eksenli kestirim (x,y,z metre), health_status yönetimi (1=referans, 0=kendi
kestirim), ilk 450 kare kalibrasyon dönemi.

Hata formülü: E = (1/N) * Σ √((x̂ᵢ-xᵢ)² + (ŷᵢ-yᵢ)² + (ẑᵢ-zᵢ)²)
"""

import cv2
import numpy as np
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


class KalmanPozisyonFiltresi:
    """
    6 durumlu Kalman filtresi: [x, y, z, vx, vy, vz].
    Optical flow'un kare başına ürettiği küçük hataları yumuşatır, drift'i azaltır.
    """

    def __init__(self, dt=1.0/7.5):
        """dt: kareler arası süre - şartname 7.5 FPS -> ~0.133s"""
        self.dt = dt

        self.x = np.zeros(6)

        # sabit hız modeli: x_yeni = x_eski + vx*dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.eye(3, 6)  # sadece pozisyon ölçülüyor, hız dolaylı çıkarılıyor

        self.P = np.eye(6) * 10.0  # belirsizlik kovaryansı

        self.Q = np.eye(6) * 0.1  # süreç gürültüsü
        self.Q[3, 3] = 0.5
        self.Q[4, 4] = 0.5
        self.Q[5, 5] = 0.3

        self.R = np.eye(3) * 2.0  # ölçüm gürültüsü (optical flow ne kadar güvenilir)
        self.R[2, 2] = 1.0

    def tahmin(self):
        """Predict adımı."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def guncelle(self, olcum):
        """Update adımı: ölçümle tahmini düzeltir. olcum: [x, y, z]"""
        z = np.array(olcum)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].copy()

    def durumu_ayarla(self, x, y, z):
        """Durumu doğrudan ayarlar (health=1'den referans geldiğinde)."""
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z
        self.P[0, 0] = 0.1
        self.P[1, 1] = 0.1
        self.P[2, 2] = 0.1

    def hiz_guncelle(self, vx, vy, vz):
        """Hız tahminini günceller (kalibrasyon döneminde)."""
        self.x[3] = vx
        self.x[4] = vy
        self.x[5] = vz


class PozisyonKestirimi:
    """Görsel odometri + Kalman filtre tabanlı 3 eksenli pozisyon kestirimi."""

    def __init__(self):
        self.kalman = KalmanPozisyonFiltresi(dt=1.0/7.5)

        self.kestirim_x = config.BASLANGIC_X
        self.kestirim_y = config.BASLANGIC_Y
        self.kestirim_z = config.BASLANGIC_Z

        self.onceki_kare_gri = None

        self.kalibrasyon_tamamlandi = False
        self.olcek_faktoru_x = 1.0
        self.olcek_faktoru_y = 1.0
        self.kalibrasyon_verileri = []  # [(ref_dx, ref_dy, of_dx, of_dy), ...]
        self.onceki_ref_x = None
        self.onceki_ref_y = None
        self.onceki_ref_z = None

        self.z_gecmisi = []  # sağlıklı dönemdeki z değerleri
        self.son_saglıklı_z = 0.0
        self.onceki_ozellik_sayisi = None

        self.spread_z_verileri = []  # [(spread, z), ...] sağlıklı dönemde
        self.spread_z_katsayilari = None  # np.polyfit sonucu

        self.son_saglıklı_x = 0.0
        self.son_saglıklı_y = 0.0
        self.son_saglıklı_kare = 0
        self.saglıksız_baslangic_x = 0.0
        self.saglıksız_baslangic_y = 0.0

        self.lk_params = dict(
            winSize=config.OF_WIN_SIZE,
            maxLevel=config.OF_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.kare_sayaci = 0
        self.onceki_health = 1

    def _optik_akis_hesapla(self, kare_gri):
        """İki ardışık kare arası piksel kaymasını bulur. RANSAC + medyan ile hareketli nesneler filtrelenir."""
        if self.onceki_kare_gri is None:
            self.onceki_kare_gri = kare_gri.copy()
            return 0.0, 0.0

        p0 = cv2.goodFeaturesToTrack(
            self.onceki_kare_gri, mask=None,
            maxCorners=config.OF_MAX_CORNERS,
            qualityLevel=config.OF_QUALITY_LEVEL,
            minDistance=config.OF_MIN_DISTANCE,
            blockSize=7
        )

        if p0 is None or len(p0) < 10:
            self.onceki_kare_gri = kare_gri.copy()
            return 0.0, 0.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.onceki_kare_gri, kare_gri, p0, None, **self.lk_params
        )

        if p1 is None:
            self.onceki_kare_gri = kare_gri.copy()
            return 0.0, 0.0

        iyi_yeni = p1[st == 1]
        iyi_eski = p0[st == 1]

        if len(iyi_yeni) < 5:
            self.onceki_kare_gri = kare_gri.copy()
            return 0.0, 0.0

        # RANSAC ile outlier temizleme (hareketli nesnelerin noktaları yanlış kayma verir)
        try:
            H, mask = cv2.findHomography(iyi_eski, iyi_yeni, cv2.RANSAC, 3.0)
            if mask is not None:
                inlier_mask = mask.ravel() == 1
                if np.sum(inlier_mask) >= 5:
                    iyi_yeni = iyi_yeni[inlier_mask]
                    iyi_eski = iyi_eski[inlier_mask]
        except:
            pass

        hareket = iyi_yeni - iyi_eski
        dx_piksel = float(np.median(hareket[:, 0]))
        dy_piksel = float(np.median(hareket[:, 1]))

        self.onceki_ozellik_sayisi = len(iyi_yeni)

        self.onceki_kare_gri = kare_gri.copy()
        return dx_piksel, dy_piksel

    def _feature_spread_hesapla(self, kare_gri):
        """
        Özellik noktalarının merkeze medyan uzaklığı. Drone yükselince noktalar
        merkeze yaklaşır, alçalınca uzaklaşır - Z kestirimi bu ilişkiyi kullanır.
        """
        p = cv2.goodFeaturesToTrack(
            kare_gri, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        if p is None or len(p) < 10:
            return None
        pts = p.reshape(-1, 2)
        h, w = kare_gri.shape[:2]
        merkez = np.array([w / 2.0, h / 2.0])
        uzakliklar = np.linalg.norm(pts - merkez, axis=1)
        return float(np.median(uzakliklar))

    def _z_kestirim_goruntu_olcegi(self, kare_gri):
        """Feature spread modeli yoksa, son sağlıklı Z'den lineer trend ile tahmin."""
        if len(self.z_gecmisi) < 5:
            return self.son_saglıklı_z

        son_n = min(100, len(self.z_gecmisi))
        z_son = np.array(self.z_gecmisi[-son_n:])
        x_vals = np.arange(son_n)

        try:
            katsayilar = np.polyfit(x_vals, z_son, 1)
            egim = katsayilar[0]

            kare_fark = self.kare_sayaci - self.son_saglıklı_kare

            tahmin_z = self.son_saglıklı_z + egim * kare_fark
            tahmin_z = np.clip(tahmin_z, 0, 200)  # makul aralıkta tut
            return float(tahmin_z)
        except:
            return self.son_saglıklı_z

    def _kalibrasyonu_guncelle(self, ref_x, ref_y, dx_piksel, dy_piksel):
        """Sağlıklı dönemde piksel->metre ölçek faktörünü hesaplar."""
        if self.onceki_ref_x is not None:
            ref_dx = ref_x - self.onceki_ref_x
            ref_dy = ref_y - self.onceki_ref_y

            self.kalibrasyon_verileri.append((ref_dx, ref_dy, dx_piksel, dy_piksel))

        self.onceki_ref_x = ref_x
        self.onceki_ref_y = ref_y

        if len(self.kalibrasyon_verileri) >= 30:
            toplam_dx_m = sum(abs(d[0]) for d in self.kalibrasyon_verileri)
            toplam_dy_m = sum(abs(d[1]) for d in self.kalibrasyon_verileri)
            toplam_dx_px = sum(abs(d[2]) for d in self.kalibrasyon_verileri)
            toplam_dy_px = sum(abs(d[3]) for d in self.kalibrasyon_verileri)

            if toplam_dx_px > 50:
                self.olcek_faktoru_x = toplam_dx_m / toplam_dx_px
            if toplam_dy_px > 50:
                self.olcek_faktoru_y = toplam_dy_m / toplam_dy_px

            if not self.kalibrasyon_tamamlandi:
                self.kalibrasyon_tamamlandi = True
                print(f"[POZISYON] Kalibrasyon tamamlandi: "
                      f"olcek_x={self.olcek_faktoru_x:.6f} m/px, "
                      f"olcek_y={self.olcek_faktoru_y:.6f} m/px")

            if len(self.kalibrasyon_verileri) > 200:
                self.kalibrasyon_verileri = self.kalibrasyon_verileri[-200:]

    def guncelle(self, frame, ref_x, ref_y, ref_z, health_status):
        """
        Her kare için çağrılır.
        ref_x/y/z: sunucudan gelen referans (health=0 ise None)
        Döner: {"translation_x": .., "translation_y": .., "translation_z": ..}
        """
        self.kare_sayaci += 1

        if frame is None:
            if health_status == 1 and ref_x is not None:
                self.kestirim_x = ref_x
                self.kestirim_y = ref_y
                self.kestirim_z = ref_z
            return {
                "translation_x": round(self.kestirim_x, 4),
                "translation_y": round(self.kestirim_y, 4),
                "translation_z": round(self.kestirim_z, 4)
            }

        kare_gri = _gri_yap(frame)

        dx_piksel, dy_piksel = self._optik_akis_hesapla(kare_gri)

        if health_status == 1 and ref_x is not None and ref_y is not None and ref_z is not None:
            # sağlıklı dönem: referans pozisyonu kullan + kalibre et
            self.kalman.durumu_ayarla(ref_x, ref_y, ref_z)

            if self.onceki_ref_x is not None:
                vx = (ref_x - self.onceki_ref_x) * 7.5  # m/s
                vy = (ref_y - self.onceki_ref_y) * 7.5
                vz = (ref_z - self.onceki_ref_z) * 7.5 if self.onceki_ref_z is not None else 0
                self.kalman.hiz_guncelle(vx, vy, vz)

            self._kalibrasyonu_guncelle(ref_x, ref_y, dx_piksel, dy_piksel)

            self.z_gecmisi.append(ref_z)
            self.son_saglıklı_z = ref_z

            spread = self._feature_spread_hesapla(kare_gri)
            if spread is not None:
                self.spread_z_verileri.append((spread, ref_z))
                if len(self.spread_z_verileri) > 200:
                    self.spread_z_verileri = self.spread_z_verileri[-200:]
                if len(self.spread_z_verileri) >= 30:
                    sp_arr = np.array([d[0] for d in self.spread_z_verileri])
                    z_arr = np.array([d[1] for d in self.spread_z_verileri])
                    if np.ptp(z_arr) > 0.5:  # yeterli z varyasyonu varsa modeli güncelle
                        self.spread_z_katsayilari = np.polyfit(sp_arr, z_arr, 1)

            self.son_saglıklı_x = ref_x
            self.son_saglıklı_y = ref_y
            self.son_saglıklı_kare = self.kare_sayaci

            self.kestirim_x = ref_x
            self.kestirim_y = ref_y
            self.kestirim_z = ref_z

            self.onceki_ref_z = ref_z

            if self.onceki_health == 0:
                print(f"[POZISYON] Health 0->1 gecisi: drift sifirlandi "
                      f"(kare {self.kare_sayaci})")

        else:
            # sağlıksız dönem: kendi kestirimimiz
            if self.onceki_health == 1:
                self.saglıksız_baslangic_x = self.kestirim_x
                self.saglıksız_baslangic_y = self.kestirim_y
                print(f"[POZISYON] Health 1->0 gecisi: kendi kestirim basladi "
                      f"(kare {self.kare_sayaci})")

            dx_metre = dx_piksel * self.olcek_faktoru_x
            dy_metre = dy_piksel * self.olcek_faktoru_y

            ham_x = self.kestirim_x - dx_metre
            ham_y = self.kestirim_y + dy_metre

            spread = self._feature_spread_hesapla(kare_gri)
            if spread is not None and self.spread_z_katsayilari is not None:
                ham_z = float(np.polyval(self.spread_z_katsayilari, spread))
                ham_z = float(np.clip(ham_z, 0, 300))
            else:
                ham_z = self._z_kestirim_goruntu_olcegi(kare_gri)

            self.kalman.tahmin()
            filtreli = self.kalman.guncelle([ham_x, ham_y, ham_z])

            self.kestirim_x = filtreli[0]
            self.kestirim_y = filtreli[1]
            self.kestirim_z = filtreli[2]

        self.onceki_health = health_status

        return {
            "translation_x": round(float(self.kestirim_x), 4),
            "translation_y": round(float(self.kestirim_y), 4),
            "translation_z": round(float(self.kestirim_z), 4)
        }
