"""
TEKNOFEST 2026 - Sunucu İstemci Modülü

v2.1.0 resmi bağlantı arayüzüne (26.06.2026) göre yeniden yazıldı:
- Kimlik doğrulama (Token tabanlı)
- Sunucu artık kareleri TEK TEK verir (tahmin gönderilmeden ilerlemez) -
  eski "başta tüm listeyi çek" mimarisi kaldırıldı.
- GET /progress/ ile kaldığı yerden devam (resume) desteği.
- GET /reference/ ile Görev 3 referans nesneleri + aktif kare pencereleri.
- POST /prediction/ - detected_objects, detected_translations, reference_predictions
  (üst seviyede artık id/user alanı yok).
"""

import requests
import cv2
import numpy as np
import time
import sys

sys.path.append('..')
import config


class SunucuIstemci:
    """
    Yarışma sunucusuyla iletişimi yöneten sınıf.

    v2.1.0 akışı:
    0. (Gerekirse) Kullanıcı adı/şifre ile giriş yap, token al.
    1. GET /progress/ ile oturumu tespit et (aktif oturum var mı, kaldığı yer neresi).
    2. GET /reference/ ile Görev 3 referanslarını (varsa) tek seferde çek.
    3. Döngü: GET /frames/ + /translation/ (sıradaki TEK kare) → işle → POST /prediction/.
       Tahmin gönderilmeden sunucu bir sonraki kareyi vermez.
    """

    def __init__(self, sunucu_url=None, kullanici_url=None):
        self.sunucu_url = (sunucu_url or config.SUNUCU_URL).rstrip("/")
        self.kullanici_url = kullanici_url or config.KULLANICI_URL
        self.oturum = requests.Session()  # Bağlantı havuzu + auth header için
        self.token = None

        # oturumu_baslat() tarafından doldurulur
        self.oturum_adi = None
        self.toplam_kare = 0
        self.baslangic_index = 0
        self.tamamlandi = False

        if config.OTURUM_ACMA_AKTIF:
            self.giris_yap()

    def giris_yap(self):
        """
        Resmi bağlantı arayüzündeki auth/ akışıyla uyumlu giriş.
        Başarılı olursa alınan token, oturumun tüm isteklerine (görüntü
        indirme dahil) otomatik eklenir.
        """
        try:
            yanit = self.oturum.post(
                f"{self.sunucu_url}/{config.API_AUTH_PATH}",
                data={"username": config.TAKIM_ADI, "password": config.TAKIM_SIFRE},
                timeout=config.ISTEK_TIMEOUT
            )
            if yanit.status_code == 200:
                self.token = yanit.json().get("token")
                if self.token:
                    self.oturum.headers.update({"Authorization": f"Token {self.token}"})
                    print("[SUNUCU] Giriş başarılı, token alındı.")
                    return True
                print("[SUNUCU] Giriş yanıtında token bulunamadı.")
            else:
                print(f"[SUNUCU] Giriş başarısız: HTTP {yanit.status_code} - {yanit.text}")
        except requests.exceptions.RequestException as e:
            print(f"[SUNUCU] Giriş isteği hatası: {e}")
        return False

    def _get_ile_liste_al(self, api_path):
        """GET isteğiyle bir JSON listesi çeker, retry'lı. Boş liste [] dönebilir (normal)."""
        for deneme in range(config.ISTEK_MAX_RETRY):
            try:
                response = self.oturum.get(
                    f"{self.sunucu_url}/{api_path}",
                    timeout=config.ISTEK_TIMEOUT
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"[SUNUCU] HTTP {response.status_code} ({api_path})")
            except requests.exceptions.RequestException as e:
                print(f"[SUNUCU] Bağlantı hatası (deneme {deneme+1}, {api_path}): {e}")
            time.sleep(config.ISTEK_RETRY_BEKLEME)
        return None

    def oturumu_baslat(self):
        """
        GET /progress/ ile aktif oturumu tespit eder ve kaldığı yeri öğrenir.

        Returns:
            dict {"oturum_adi", "toplam_kare", "baslangic_index", "tamamlandi"} → başarılı
            None → sunucuya ulaşılamadı (bağlantı hatası, tekrar denenmeli)

        NOT: session_name=None dönmesi "aktif oturum yok" demektir, bu bağlantı
        hatasından FARKLIDIR (bu durumda None yerine tamamlandi=False, oturum_adi=None
        içeren bir dict döner, çağıran taraf bunu kontrol etmeli).
        """
        for deneme in range(config.ISTEK_MAX_RETRY):
            try:
                response = self.oturum.get(
                    f"{self.sunucu_url}/{config.API_PROGRESS_PATH}",
                    timeout=config.ISTEK_TIMEOUT
                )
                if response.status_code == 200:
                    veri = response.json()
                    self.oturum_adi = veri.get("session_name")
                    self.toplam_kare = veri.get("total_frames", 0)
                    self.baslangic_index = veri.get("frame_index", 0)
                    self.tamamlandi = bool(veri.get("completed", False))
                    print(f"[SUNUCU] İlerleme: {self.baslangic_index}/{self.toplam_kare} "
                          f"(tamamlandı={self.tamamlandi}, oturum={self.oturum_adi})")
                    return {
                        "oturum_adi": self.oturum_adi,
                        "toplam_kare": self.toplam_kare,
                        "baslangic_index": self.baslangic_index,
                        "tamamlandi": self.tamamlandi,
                    }
                else:
                    print(f"[SUNUCU] HTTP {response.status_code} (progress/)")
            except requests.exceptions.RequestException as e:
                print(f"[SUNUCU] İlerleme isteği hatası (deneme {deneme+1}): {e}")
            time.sleep(config.ISTEK_RETRY_BEKLEME)
        return None

    def siradaki_kareyi_al(self):
        """
        Sunucunun o an beklediği (henüz tahmin gönderilmemiş) TEK kareyi alır
        ve görüntüyü indirir. Tahmin gönderilmeden tekrar çağrılırsa AYNI kare
        döner (sunucu ilerlemez) - main.py'deki döngü bu yüzden her karede
        tam olarak bir kez sonuc_gonder() çağırmalı.

        Returns:
            None  → oturum bitti veya aktif oturum yok (döngüden çık)
            dict  → her zaman döner; frame=None ise görüntü indirilemedi demektir,
                    yine de pozisyon ve sonuç sunucuya gönderilmeli (boş/varsayılan)
        """
        kareler = self._get_ile_liste_al(config.API_FRAME_PATH)
        if not kareler:
            print("[SUNUCU] Tüm kareler işlendi (veya aktif oturum yok).")
            return None
        kare_bilgi = kareler[0]

        translationlar = self._get_ile_liste_al(config.API_TRANSLATION_PATH)
        tr_bilgi = translationlar[0] if translationlar else {}
        if not translationlar:
            print("[SUNUCU] UYARI: translation alınamadı, kare yine de işlenecek.")

        tx = tr_bilgi.get("translation_x", "NaN")
        ty = tr_bilgi.get("translation_y", "NaN")
        tz = tr_bilgi.get("translation_z", "NaN")
        health = tr_bilgi.get("health_status")

        tx = None if tx == "NaN" or tx is None else float(tx)
        ty = None if ty == "NaN" or ty is None else float(ty)
        tz = None if tz == "NaN" or tz is None else float(tz)

        image_url = kare_bilgi.get("image_url", "")
        goruntu = self._goruntu_indir(image_url)
        if goruntu is None:
            print(f"[SUNUCU] Kare indirilemedi ({image_url}), boş sonuç gönderilecek.")

        return {
            "frame": goruntu,
            "frame_url": kare_bilgi.get("url", ""),
            "image_url": image_url,
            "translation_x": tx,
            "translation_y": ty,
            "translation_z": tz,
            "health_status": None if health is None else int(health),
        }

    def _goruntu_indir(self, image_url):
        """
        Görüntüyü URL'den indirir ve numpy array olarak döner.

        Resmi_arayuz/TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py:
        görüntüleri "media" öneki ekleyerek indiriyor
        (evaluation_server_url + "media" + image_url). Auth token gerekiyorsa
        self.oturum zaten Authorization header'ını taşıyor (giris_yap()'ta eklendi).
        """
        if not image_url:
            return None
        try:
            if image_url.startswith("http"):
                tam_url = image_url
            elif image_url.startswith("/"):
                tam_url = f"{self.sunucu_url}/media{image_url}"
            else:
                tam_url = f"{self.sunucu_url}/media/{image_url}"

            response = self.oturum.get(tam_url, timeout=config.ISTEK_TIMEOUT)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            print(f"[SUNUCU] Görüntü indirme hatası: {e}")
        return None

    def _cls_alanlarini_donustur(self, detected_objects):
        """
        Şartname Şekil 17 örneğinde ve resmi bağlantı arayüzünde "cls" alanı
        düz bir sayı değil, bir URL'dir (örn: "http://.../classes/1/").
        config.SINIF_URL_FORMATINDA_GONDER True ise dönüşümü burada yaparız;
        nesne tespiti modülü dahili olarak hep düz sayı ("0".."3") üretir.
        """
        if not config.SINIF_URL_FORMATINDA_GONDER:
            return detected_objects

        donusturulmus = []
        for nesne in detected_objects:
            yeni_nesne = dict(nesne)
            try:
                sinif_id = int(yeni_nesne["cls"])
                yeni_nesne["cls"] = (
                    f"{self.sunucu_url}/classes/{sinif_id + config.SINIF_URL_OFSET}/"
                )
            except (KeyError, ValueError, TypeError):
                pass
            donusturulmus.append(yeni_nesne)
        return donusturulmus

    def _reference_predictions_olustur(self, frame_url, reference_predictions):
        """
        goruntu_eslestirme.eslestir() çıktısını ({"reference": .., "top_left_x": ..})
        v2.1.0 ReferencePrediction şemasına ({"reference": .., "frame": .., bbox..}) çevirir.
        """
        return [
            {
                "reference": r["reference"],
                "frame": frame_url,
                "top_left_x": r["top_left_x"],
                "top_left_y": r["top_left_y"],
                "bottom_right_x": r["bottom_right_x"],
                "bottom_right_y": r["bottom_right_y"],
            }
            for r in reference_predictions
        ]

    def sonuc_gonder(self, frame_url, detected_objects, detected_translations,
                     reference_predictions):
        """
        v2.1.0 formatında sonuç gönderir (üst seviyede id/user YOK):
        {
            "frame": "http://.../frames/4000/",
            "detected_objects": [
                {"cls": "http://.../classes/1/", "landing_status": "-1",
                 "moving_status": "-1", "top_left_x": .., ...}
            ],
            "detected_translations": [
                {"translation_x": 0.02, "translation_y": 0.01, "translation_z": 0.03}
            ],
            "reference_predictions": [
                {"reference": "http://.../reference/1/", "frame": "http://.../frames/4000/",
                 "top_left_x": .., ...}
            ]
        }
        """
        payload = {
            "frame": frame_url,
            "detected_objects": self._cls_alanlarini_donustur(detected_objects),
            "detected_translations": detected_translations,
            "reference_predictions": self._reference_predictions_olustur(
                frame_url, reference_predictions
            ),
        }

        for deneme in range(config.ISTEK_MAX_RETRY):
            try:
                response = self.oturum.post(
                    f"{self.sunucu_url}/{config.API_RESULT_PATH}",
                    json=payload,
                    timeout=config.ISTEK_TIMEOUT
                )
                if response.status_code in (200, 201):
                    return True
                if response.status_code == 406:
                    # Bu kare için sonuç zaten gönderilmiş, tekrar denemek anlamsız.
                    print(f"[SUNUCU] Kare için sonuç zaten gönderilmiş (406): {frame_url}")
                    return False
                if response.status_code == 403 and "exceeded" in response.text.lower():
                    # Yeni hız limiti mesajı: {"detail":"...exceeded <rate> limit."}
                    bekleme = 2.0 * (deneme + 1)
                    print(f"[SUNUCU] Hız limiti aşıldı, {bekleme:.1f}s bekleyip tekrar denenecek.")
                    time.sleep(bekleme)
                    continue
                print(f"[SUNUCU] Sonuç gönderme HTTP {response.status_code}: {response.text[:200]}")
            except requests.exceptions.RequestException as e:
                print(f"[SUNUCU] Sonuç gönderme hatası (deneme {deneme+1}): {e}")
            time.sleep(config.ISTEK_RETRY_BEKLEME)
        return False

    def referans_goruntu_indir(self, image_url):
        """Görev 3 referans görüntüsünü indirir (main.py'nin dışarıdan çağırabilmesi için)."""
        return self._goruntu_indir(image_url)

    def referanslari_al(self):
        """
        Oturum başında sunucudan Görev 3 referans nesnelerini çeker (v2.1.0 /reference/).

        Returns:
            list[dict]: [{"url", "session", "image_url", "frame_start_image_url",
                          "frame_end_image_url", "order"}, ...]
            Boş liste [] → referans yok veya istek başarısız (main.py lokal
            'referanslar/' klasörüne fallback yapar).
        """
        veri = self._get_ile_liste_al(config.API_REFERANS_PATH)
        if veri:
            print(f"[SUNUCU] {len(veri)} referans nesne alındı.")
            return veri
        print("[SUNUCU] Referans alınamadı — lokal klasör kullanılacak.")
        return []
