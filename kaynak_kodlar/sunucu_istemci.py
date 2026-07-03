"""
TEKNOFEST 2026 - Sunucu İstemci Modülü

v2.1.0 API: kareler tek tek verilir (tahmin gönderilmeden ilerlemez, eski
"başta tüm listeyi çek" mimarisi yok), /progress/ ile resume, /reference/
ile Görev 3 referansları + aktif pencereler. POST /prediction/ üst
seviyede artık id/user içermiyor.
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
    Yarışma sunucusuyla iletişim.

    Akış: (varsa) giriş yap -> /progress/ ile oturumu ve kaldığı yeri bul ->
    /reference/ ile Görev 3 referanslarını çek -> döngüde /frames/+/translation/
    al, işle, /prediction/ ile gönder. Tahmin gitmeden sunucu sonraki kareyi vermiyor.
    """

    def __init__(self, sunucu_url=None, kullanici_url=None):
        self.sunucu_url = (sunucu_url or config.SUNUCU_URL).rstrip("/")
        self.kullanici_url = kullanici_url or config.KULLANICI_URL
        self.oturum = requests.Session()  # bağlantı havuzu + auth header için
        self.token = None

        # oturumu_baslat() doldurur
        self.oturum_adi = None
        self.toplam_kare = 0
        self.baslangic_index = 0
        self.tamamlandi = False

        if config.OTURUM_ACMA_AKTIF:
            self.giris_yap()

    def giris_yap(self):
        """Token alır, oturumun tüm isteklerine (görüntü indirme dahil) otomatik eklenir."""
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
        """GET ile JSON listesi çeker, retry'lı. Boş liste [] normal bir sonuç olabilir."""
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
            time.sleep(config.ISTEK_RETRY_BEKLEME * (2 ** deneme))
        return None

    def oturumu_baslat(self):
        """
        /progress/ ile aktif oturumu ve kaldığı yeri bulur.
        None dönerse sunucuya ulaşılamadı demektir - bu, session_name=None
        (aktif oturum yok) durumundan farklı, çağıran taraf ikisini ayırt etmeli.
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
            time.sleep(config.ISTEK_RETRY_BEKLEME * (2 ** deneme))
        return None

    def siradaki_kareyi_al(self):
        """
        Sunucunun beklediği tek kareyi alır ve indirir. Tahmin gönderilmeden
        tekrar çağrılırsa aynı kare döner - main.py her karede tam bir kez
        sonuc_gonder() çağırmalı.

        None -> oturum bitti/yok. dict -> her zaman döner, frame=None ise
        indirme başarısız demektir ama yine de sonuç gönderilmeli.
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
        """Görüntüyü indirir, numpy array döner. Resmi kod "media" önekiyle indiriyor, auth token varsa header'da zaten var."""
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

    _SAYISAL_ALANLAR = ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")

    def _detected_objects_donustur(self, detected_objects):
        """
        "cls" düz sayı değil URL olarak gidiyor (classes/1/ gibi). Sayısal alanlar
        da string'e çevrilir - resmi connection_handler'daki create_payload()
        tüm sayısal alanları str() ile gönderiyor, biz aynı formata uyuyoruz.
        """
        donusturulmus = []
        for nesne in detected_objects:
            yeni_nesne = dict(nesne)
            if config.SINIF_URL_FORMATINDA_GONDER:
                try:
                    sinif_id = int(yeni_nesne["cls"])
                    yeni_nesne["cls"] = (
                        f"{self.sunucu_url}/classes/{sinif_id + config.SINIF_URL_OFSET}/"
                    )
                except (KeyError, ValueError, TypeError):
                    pass
            for alan in self._SAYISAL_ALANLAR:
                if alan in yeni_nesne:
                    yeni_nesne[alan] = str(yeni_nesne[alan])
            donusturulmus.append(yeni_nesne)
        return donusturulmus

    def _translations_donustur(self, detected_translations):
        """translation_x/y/z alanlarını string'e çevirir (resmi Translation.create_payload ile aynı format)."""
        return [{k: str(v) for k, v in t.items()} for t in detected_translations]

    def _reference_predictions_olustur(self, frame_url, reference_predictions):
        """eslestir() çıktısını v2.1.0 ReferencePrediction şemasına çevirir (frame alanı eklenir, sayısal alanlar string)."""
        return [
            {
                "reference": r["reference"],
                "frame": frame_url,
                "top_left_x": str(r["top_left_x"]),
                "top_left_y": str(r["top_left_y"]),
                "bottom_right_x": str(r["bottom_right_x"]),
                "bottom_right_y": str(r["bottom_right_y"]),
            }
            for r in reference_predictions
        ]

    def sonuc_gonder(self, frame_url, detected_objects, detected_translations,
                     reference_predictions):
        """v2.1.0 formatında sonucu POST eder (üst seviyede id/user yok)."""
        payload = {
            "frame": frame_url,
            "detected_objects": self._detected_objects_donustur(detected_objects),
            "detected_translations": self._translations_donustur(detected_translations),
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
                    print(f"[SUNUCU] Kare için sonuç zaten gönderilmiş (406): {frame_url}")
                    return False
                if response.status_code == 403 and "exceeded" in response.text.lower():
                    bekleme = 2.0 * (deneme + 1)
                    print(f"[SUNUCU] Hız limiti aşıldı, {bekleme:.1f}s bekleyip tekrar denenecek.")
                    time.sleep(bekleme)
                    continue
                print(f"[SUNUCU] Sonuç gönderme HTTP {response.status_code}: {response.text[:200]}")
            except requests.exceptions.RequestException as e:
                print(f"[SUNUCU] Sonuç gönderme hatası (deneme {deneme+1}): {e}")
            time.sleep(config.ISTEK_RETRY_BEKLEME * (2 ** deneme))
        return False

    def referans_goruntu_indir(self, image_url):
        """Görev 3 referans görüntüsünü indirir (main.py dışarıdan çağırır)."""
        return self._goruntu_indir(image_url)

    def referanslari_al(self):
        """
        /reference/ ile Görev 3 referanslarını çeker.
        Boş liste -> referans yok/istek başarısız, main.py lokal referanslar/ klasörüne düşer.
        """
        veri = self._get_ile_liste_al(config.API_REFERANS_PATH)
        if veri:
            print(f"[SUNUCU] {len(veri)} referans nesne alındı.")
            return veri
        print("[SUNUCU] Referans alınamadı — lokal klasör kullanılacak.")
        return []
