"""
TEKNOFEST 2026 - Pozisyon Kestirimi Modülü (Görev 2 - %40 Puan)
Kalman Filter + Optical Flow + Keyframe Kalibrasyon

Şartname Bölüm 2.2'ye tam uyumlu:
- 3 eksenli kestirim (x, y, z metre cinsinden)
- health_status yönetimi (health=1 referans, health=0 kendi kestirim)
- İlk 450 kare kesinlikle sağlıklı → kalibrasyon dönemi

İyileştirmeler (eski sürüme göre):
1. Kalman Filter: Ham optical flow çıktısını yumuşatır, ani hataları bastırır
2. RANSAC + Median: Hareketli nesnelerin etkisini filtreler
3. Keyframe kalibrasyon: Sağlıklı dönemde ölçek faktörünü sürekli günceller
4. Z kestirimi: Görüntü ölçeği değişiminden yükseklik tahmini
5. Drift düzeltme: Health=1'e geri dönüldüğünde birikmiş hatayı sıfırlar

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
    6 durumlu Kalman Filtresi: [x, y, z, vx, vy, vz]
    
    Optical flow her karede küçük hatalar üretir.
    2250 kare boyunca bu hatalar birikir → drift.
    Kalman filtre bu hataları yumuşatarak gerçek hareketi izole eder.
    """

    def __init__(self, dt=1.0/7.5):
        """
        Args:
            dt: Kareler arası süre (saniye). Şartname: 7.5 FPS → dt ≈ 0.133s
        """
        self.dt = dt

        # Durum: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)

        # Durum geçiş matrisi (sabit hız modeli)
        # x_yeni = x_eski + vx * dt
        self.F = np.eye(6)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

        # Ölçüm matrisi (sadece pozisyon ölçüyoruz, hızı dolaylı çıkarıyoruz)
        self.H = np.eye(3, 6)  # [x, y, z] ölçümü

        # Kovaryans matrisi (belirsizlik)
        self.P = np.eye(6) * 10.0

        # Süreç gürültüsü (modelin ne kadar sapabileceği)
        self.Q = np.eye(6) * 0.1
        self.Q[3, 3] = 0.5  # Hız değişimi daha belirsiz
        self.Q[4, 4] = 0.5
        self.Q[5, 5] = 0.3  # Z hızı daha stabil

        # Ölçüm gürültüsü (optical flow'un ne kadar hatalı olduğu)
        self.R = np.eye(3) * 2.0
        self.R[2, 2] = 1.0  # Z ölçümü biraz daha güvenilir

    def tahmin(self):
        """Predict adımı: bir sonraki durumu tahmin et."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def guncelle(self, olcum):
        """
        Update adımı: ölçümle tahminini düzelt.
        
        Args:
            olcum: [x, y, z] numpy array
        """
        z = np.array(olcum)
        y = z - self.H @ self.x  # Yenilik (ölçüm - tahmin farkı)
        S = self.H @ self.P @ self.H.T + self.R  # Yenilik kovaryansı
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman kazancı

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].copy()

    def durumu_ayarla(self, x, y, z):
        """Durumu doğrudan ayarla (health=1'den referans geldiğinde)."""
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z
        # Belirsizliği düşür (referans güvenilir)
        self.P[0, 0] = 0.1
        self.P[1, 1] = 0.1
        self.P[2, 2] = 0.1

    def hiz_guncelle(self, vx, vy, vz):
        """Hız tahminini güncelle (kalibrasyon döneminde)."""
        self.x[3] = vx
        self.x[4] = vy
        self.x[5] = vz


class PozisyonKestirimi:
    """
    Görsel odometri + Kalman Filter tabanlı 3 eksenli pozisyon kestirim sistemi.
    """

    def __init__(self):
        # Kalman filtresi
        self.kalman = KalmanPozisyonFiltresi(dt=1.0/7.5)

        # Mevcut kestirim pozisyonu
        self.kestirim_x = config.BASLANGIC_X
        self.kestirim_y = config.BASLANGIC_Y
        self.kestirim_z = config.BASLANGIC_Z

        # Önceki kare (optical flow için)
        self.onceki_kare_gri = None

        # Kalibrasyon
        self.kalibrasyon_tamamlandi = False
        self.olcek_faktoru_x = 1.0
        self.olcek_faktoru_y = 1.0
        self.kalibrasyon_verileri = []  # [(ref_dx, ref_dy, of_dx, of_dy), ...]
        self.onceki_ref_x = None
        self.onceki_ref_y = None
        self.onceki_ref_z = None

        # Z kestirimi için
        self.z_gecmisi = []          # Sağlıklı dönemdeki z değerleri
        self.son_saglıklı_z = 0.0
        self.onceki_ozellik_sayisi = None  # Görüntü ölçeği değişimi için

        # Feature spread tabanlı Z modeli
        self.spread_z_verileri = []       # [(spread, z), ...] sağlıklı dönemde
        self.spread_z_katsayilari = None  # np.polyfit sonucu [a, b]

        # Drift düzeltme
        self.son_saglıklı_x = 0.0
        self.son_saglıklı_y = 0.0
        self.son_saglıklı_kare = 0
        self.saglıksız_baslangic_x = 0.0
        self.saglıksız_baslangic_y = 0.0

        # Optical Flow parametreleri
        self.lk_params = dict(
            winSize=config.OF_WIN_SIZE,
            maxLevel=config.OF_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.kare_sayaci = 0
        self.onceki_health = 1

    def _optik_akis_hesapla(self, kare_gri):
        """
        İki ardışık kare arasındaki piksel kaymasını hesaplar.
        RANSAC ile outlier temizleme + Median (ortalamadan daha robust).
        """
        if self.onceki_kare_gri is None:
            self.onceki_kare_gri = kare_gri.copy()
            return 0.0, 0.0

        # Takip noktaları bul
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

        # Lucas-Kanade Optical Flow
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

        # RANSAC ile outlier temizleme
        # (Hareketli nesnelerin noktaları yanlış kayma verir)
        try:
            H, mask = cv2.findHomography(iyi_eski, iyi_yeni, cv2.RANSAC, 3.0)
            if mask is not None:
                inlier_mask = mask.ravel() == 1
                if np.sum(inlier_mask) >= 5:
                    iyi_yeni = iyi_yeni[inlier_mask]
                    iyi_eski = iyi_eski[inlier_mask]
        except:
            pass

        # Median hareket (ortalamadan daha robust)
        hareket = iyi_yeni - iyi_eski
        dx_piksel = float(np.median(hareket[:, 0]))
        dy_piksel = float(np.median(hareket[:, 1]))

        # Görüntü ölçeği değişimi (Z kestirimi için)
        self.onceki_ozellik_sayisi = len(iyi_yeni)

        self.onceki_kare_gri = kare_gri.copy()
        return dx_piksel, dy_piksel

    def _feature_spread_hesapla(self, kare_gri):
        """
        Görüntüdeki özellik noktalarının merkeze medyan uzaklığını döner.

        Drone yükselince aynı yüzey daha küçük görünür → noktalar merkeze yaklaşır.
        Drone alçalınca yüzey büyür → noktalar merkeze uzaklaşır.
        Bu değer ile Z arasındaki lineer ilişki sağlıklı dönemde kalibrasyon edilir.
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
        """
        Z ekseni kestirimi: Görüntüdeki özellik noktalarının yayılımından
        yükseklik değişimini tahmin eder.
        
        Drone yükselince → nesneler küçülür → özellik noktaları yakınlaşır
        Drone alçalınca → nesneler büyür → özellik noktaları uzaklaşır
        """
        if len(self.z_gecmisi) < 5:
            return self.son_saglıklı_z

        # Son bilinen z'yi temel al
        # Basit yaklaşım: Z genelde stabil, ani değişimler yok
        # Lineer trend ile tahmin
        son_n = min(100, len(self.z_gecmisi))
        z_son = np.array(self.z_gecmisi[-son_n:])
        x_vals = np.arange(son_n)

        try:
            # Lineer fit: z = a*t + b
            katsayilar = np.polyfit(x_vals, z_son, 1)
            egim = katsayilar[0]
            
            # Kaç kare geçti sağlıklı dönemden beri?
            kare_fark = self.kare_sayaci - self.son_saglıklı_kare
            
            # Trend ile tahmin (ama çok uzaklaşmasın)
            tahmin_z = self.son_saglıklı_z + egim * kare_fark
            
            # Z değerinin makul aralıkta kalmasını sağla (0-200m)
            tahmin_z = np.clip(tahmin_z, 0, 200)
            return float(tahmin_z)
        except:
            return self.son_saglıklı_z

    def _kalibrasyonu_guncelle(self, ref_x, ref_y, dx_piksel, dy_piksel):
        """
        Sağlıklı dönemde piksel→metre ölçek faktörünü hesaplar.
        Her yeni referans geldiğinde kalibrasyon verisine ekler.
        """
        if self.onceki_ref_x is not None:
            ref_dx = ref_x - self.onceki_ref_x
            ref_dy = ref_y - self.onceki_ref_y
            
            self.kalibrasyon_verileri.append((ref_dx, ref_dy, dx_piksel, dy_piksel))

        self.onceki_ref_x = ref_x
        self.onceki_ref_y = ref_y

        # En az 30 veri birikince kalibrasyon yap
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

            # Kalibrasyon verisini kırp (son 200 veri yeterli)
            if len(self.kalibrasyon_verileri) > 200:
                self.kalibrasyon_verileri = self.kalibrasyon_verileri[-200:]

    def guncelle(self, frame, ref_x, ref_y, ref_z, health_status):
        """
        Her kare için çağrılan ana fonksiyon.
        
        Args:
            frame: numpy array görüntü
            ref_x, ref_y, ref_z: Sunucudan gelen referans (None ise sağlıksız)
            health_status: 1=sağlıklı, 0=sağlıksız
        
        Returns:
            dict: {"translation_x": float, "translation_y": float, "translation_z": float}
        """
        self.kare_sayaci += 1

        if frame is None:
            # Görüntü yoksa: health=1 ise referans pozisyonu, değilse son bilinen
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

        # Optical flow ile piksel kaymasını her durumda hesapla
        dx_piksel, dy_piksel = self._optik_akis_hesapla(kare_gri)

        if health_status == 1 and ref_x is not None and ref_y is not None and ref_z is not None:
            # ============================================================
            # SAĞLIKLI DÖNEM: Referans pozisyonu kullan + kalibre et
            # ============================================================

            # Kalman filtreyi referansla güncelle
            self.kalman.durumu_ayarla(ref_x, ref_y, ref_z)

            # Hız tahmini (ardışık referanslardan)
            if self.onceki_ref_x is not None:
                vx = (ref_x - self.onceki_ref_x) * 7.5  # m/s
                vy = (ref_y - self.onceki_ref_y) * 7.5
                vz = (ref_z - self.onceki_ref_z) * 7.5 if self.onceki_ref_z is not None else 0
                self.kalman.hiz_guncelle(vx, vy, vz)

            # Kalibrasyon verisi topla
            self._kalibrasyonu_guncelle(ref_x, ref_y, dx_piksel, dy_piksel)

            # Z geçmişi kaydet
            self.z_gecmisi.append(ref_z)
            self.son_saglıklı_z = ref_z

            # Feature spread ile Z modeli kalibre et
            spread = self._feature_spread_hesapla(kare_gri)
            if spread is not None:
                self.spread_z_verileri.append((spread, ref_z))
                if len(self.spread_z_verileri) > 200:
                    self.spread_z_verileri = self.spread_z_verileri[-200:]
                if len(self.spread_z_verileri) >= 30:
                    sp_arr = np.array([d[0] for d in self.spread_z_verileri])
                    z_arr = np.array([d[1] for d in self.spread_z_verileri])
                    # Yeterli z varyasyonu varsa modeli güncelle
                    if np.ptp(z_arr) > 0.5:
                        self.spread_z_katsayilari = np.polyfit(sp_arr, z_arr, 1)

            # Son sağlıklı pozisyonu kaydet
            self.son_saglıklı_x = ref_x
            self.son_saglıklı_y = ref_y
            self.son_saglıklı_kare = self.kare_sayaci

            # Referans pozisyonunu kullan
            self.kestirim_x = ref_x
            self.kestirim_y = ref_y
            self.kestirim_z = ref_z

            self.onceki_ref_z = ref_z

            # Sağlıksız dönemden sağlıklıya geçiş → drift'i sıfırla
            if self.onceki_health == 0:
                print(f"[POZISYON] Health 0->1 gecisi: drift sifirlandi "
                      f"(kare {self.kare_sayaci})")

        else:
            # ============================================================
            # SAĞLIKSIZ DÖNEM: Kendi algoritmamızla kestirim yap
            # ============================================================

            # Sağlıklıdan sağlıksıza geçiş anı
            if self.onceki_health == 1:
                self.saglıksız_baslangic_x = self.kestirim_x
                self.saglıksız_baslangic_y = self.kestirim_y
                print(f"[POZISYON] Health 1->0 gecisi: kendi kestirim basladi "
                      f"(kare {self.kare_sayaci})")

            # Piksel kaymasını metreye çevir
            dx_metre = dx_piksel * self.olcek_faktoru_x
            dy_metre = dy_piksel * self.olcek_faktoru_y

            # Ham optical flow kestirimi
            ham_x = self.kestirim_x - dx_metre
            ham_y = self.kestirim_y + dy_metre

            # Z kestirimi: spread modeli varsa kullan, yoksa lineer trend
            spread = self._feature_spread_hesapla(kare_gri)
            if spread is not None and self.spread_z_katsayilari is not None:
                ham_z = float(np.polyval(self.spread_z_katsayilari, spread))
                ham_z = float(np.clip(ham_z, 0, 300))
            else:
                ham_z = self._z_kestirim_goruntu_olcegi(kare_gri)

            # Kalman filtre ile yumuşat
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