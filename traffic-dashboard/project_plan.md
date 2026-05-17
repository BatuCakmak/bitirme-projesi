# UrbanEye - Şehir Trafik İzleme Sistemi

## 1. Proje Tanımı
Şehir mobese kameralarından gelen görüntüleri kullanarak araç sayımı, trafik analizi ve alternatif rota önerileri sunan profesyonel bir web tabanlı trafik izleme dashboard'u. Hedef kullanıcılar: trafik yönetim merkezi operatörleri ve şehir planlama yetkilileri.

## 2. Sayfa Yapısı
- `/` - Live Map (Canlı Harita)
- `/vehicle-count` - Vehicle Count (Araç Sayımı)
- `/vehicle-types` - Vehicle Types (Araç Türleri)
- `/traffic-analysis` - Traffic Analysis (Trafik Analizi)
- `/alternative-routes` - Alternative Routes (Alternatif Rotalar)
- `/settings` - Settings (Ayarlar)

## 3. Temel Özellikler
- [x] Canlı harita üzerinde kamera konumları ve trafik yoğunluğu
- [x] Mobese kamera canlı izleme grid'i
- [x] Araç sayımı (toplam, saatlik, günlük)
- [x] Araç türü sınıflandırması (otomobil, kamyon, otobüs, motosiklet)
- [x] Trafik yoğunluk analizi ve grafikler
- [x] Alternatif rota önerileri
- [x] Sistem ayarları ve kamera yönetimi

## 4. Veri Modeli
Mock data ile çalışır (Supabase bağlantısı gerekmez):
- Kamera listesi (id, konum, durum, koordinatlar)
- Araç sayım verileri (zaman, kamera, araç türü, sayı)
- Trafik yoğunluk verileri (yol, yoğunluk, hız)
- Rota verileri (başlangıç, bitiş, süre, mesafe, durum)

## 5. Entegrasyon Planı
- Supabase: Gerekmez (mock data yeterli)
- Shopify: Gerekmez
- Stripe: Gerekmez

## 6. Geliştirme Fazları

### Faz 1: Ana Layout + Live Map Sayfası
- Hedef: Sol navigasyon, üst bar ve canlı harita sayfası
- Çıktı: Çalışan dashboard iskelet + Live Map

### Faz 2: Vehicle Count + Vehicle Types Sayfaları
- Hedef: Araç sayım ve tür analizi sayfaları
- Çıktı: Grafikler ve istatistikler

### Faz 3: Traffic Analysis + Alternative Routes
- Hedef: Analiz grafikleri ve rota önerileri
- Çıktı: Detaylı analiz ve rota sistemi

### Faz 4: Settings Sayfası
- Hedef: Sistem ayarları ve kamera yönetimi
- Çıktı: Tam işlevsel ayarlar paneli
