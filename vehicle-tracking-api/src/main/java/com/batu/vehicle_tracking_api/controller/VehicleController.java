package com.batu.vehicle_tracking_api.controller;

import com.batu.vehicle_tracking_api.model.VehicleLog;
import com.batu.vehicle_tracking_api.repository.VehicleLogRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
@RestController
@RequestMapping("/api/vehicles")
@CrossOrigin(origins = "*") // React'in bu API'ye erişebilmesi için gerekli (CORS)
public class VehicleController {

    @Autowired
    private VehicleLogRepository repository;

    @PostMapping("/log")
    public ResponseEntity<String> saveVehicleLog(@RequestBody VehicleLog vehicleLog) {
        if (vehicleLog.getTimestamp() == null) {
            vehicleLog.setTimestamp(LocalDateTime.now());
        }

        repository.save(vehicleLog);

        // BURASI GÜNCELLENDİ: getDensity() yerine getCurrentDensity() oldu
        return ResponseEntity.ok("Araç başarıyla kaydedildi: " + vehicleLog.getType() + " - ID: " + vehicleLog.getVehicleId() + " | Ekrandaki Yoğunluk: " + vehicleLog.getCurrentDensity());
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Long>> getVehicleStats() {
        List<Object[]> rawStats = repository.countVehiclesByType();
        Map<String, Long> statsMap = new HashMap<>();

        long total = 0;
        // React'in hata vermemesi için varsayılan değerleri 0 olarak atıyoruz
        statsMap.put("araba", 0L);
        statsMap.put("motor", 0L);
        statsMap.put("kamyonTir", 0L);
        statsMap.put("otobus", 0L);

        // Veritabanından gelen listeyi okuyup Map'e yerleştiriyoruz
        for (Object[] row : rawStats) {
            String type = (String) row[0];
            Long count = ((Number) row[1]).longValue();
            total += count;

            if ("Araba".equals(type)) statsMap.put("araba", count);
            else if ("Motor".equals(type)) statsMap.put("motor", count);
            else if ("Kamyon/Tir".equals(type)) statsMap.put("kamyonTir", count);
            else if ("Otobus".equals(type)) statsMap.put("otobus", count);
        }

        // Toplam araç sayısını da ekleyip React'e yolluyoruz
        statsMap.put("toplamArac", total);
        return ResponseEntity.ok(statsMap);
    }

    @GetMapping("/flow-rate")
    public ResponseEntity<Integer> getFlowRate() {
        LocalDateTime oneMinuteAgo = LocalDateTime.now().minusMinutes(1);
        int vehiclesPerMinute = repository.countByTimestampAfter(oneMinuteAgo);
        return ResponseEntity.ok(vehiclesPerMinute);
    }

    @GetMapping("/current-density")
    public ResponseEntity<Integer> getCurrentDensity() {
        return repository.findAll().stream()
                .reduce((first, second) -> second)
                // BURASI GÜNCELLENDİ: getDensity() yerine getCurrentDensity() oldu
                .map(log -> ResponseEntity.ok(log.getCurrentDensity()))
                .orElse(ResponseEntity.ok(0));
    }
    @GetMapping("/hourly")
    public ResponseEntity<List<Map<String, Object>>> getHourlyStats() {
        // Bugünün başlangıcı ve şu anki zamanı alıyoruz
        LocalDateTime startOfDay = java.time.LocalDate.now().atStartOfDay();
        LocalDateTime now = LocalDateTime.now();

        // Bugün geçen tüm araçları veritabanından çekiyoruz
        List<VehicleLog> todayLogs = repository.findByTimestampBetween(startOfDay, now);

        List<Map<String, Object>> result = new java.util.ArrayList<>();

        for (int i = 0; i < 24; i++) {
            String hourStr = String.format("%02d:00", i);
            int currentHour = i;

            // Gelecek saatler için veritabanında kayıt olmayacağından otomatik olarak 0 dönecek
            List<VehicleLog> logsInHour = todayLogs.stream()
                    .filter(log -> log.getTimestamp() != null && log.getTimestamp().getHour() == currentHour)
                    .toList();

            long e5Count = logsInHour.stream().filter(l -> "CAM-001".equals(l.getCameraId())).count();
            long temCount = logsInHour.stream().filter(l -> "CAM-002".equals(l.getCameraId())).count();
            long d100Count = logsInHour.stream().filter(l -> "CAM-003".equals(l.getCameraId())).count();
            long vatanCount = logsInHour.stream().filter(l -> "CAM-004".equals(l.getCameraId())).count();

            Map<String, Object> hourData = new java.util.HashMap<>();
            hourData.put("hour", hourStr);
            hourData.put("e5", e5Count);
            hourData.put("tem", temCount);
            hourData.put("d100", d100Count);
            hourData.put("vatan", vatanCount);

            result.add(hourData);
        }

        return ResponseEntity.ok(result);
    }
    @GetMapping("/road-density")
    public ResponseEntity<List<Map<String, Object>>> getRoadDensity() {
        LocalDateTime fiveMinutesAgo = LocalDateTime.now().minusMinutes(5);
        // Akış hızı (dakikada geçen araç) için 1 dakika öncesini alıyoruz
        LocalDateTime oneMinuteAgo = LocalDateTime.now().minusMinutes(1);

        List<VehicleLog> recentLogs = repository.findByTimestampBetween(fiveMinutesAgo, LocalDateTime.now());
        List<Map<String, Object>> result = new java.util.ArrayList<>();

        // [Kamera ID, Yol Adı, Uzunluk, Kapasite]
        String[][] roads = {
                {"CAM-001", "E-5 Karayolu", "42 km", "25"},
                {"CAM-002", "TEM Otoyolu", "38 km", "60"},
                {"CAM-003", "D-100", "56 km", "30"},
                {"CAM-004", "Vatan Caddesi", "8 km", "20"}
        };

        for (String[] road : roads) {
            String camId = road[0];
            String roadName = road[1];
            String length = road[2];
            double maxCarsOnScreen = Double.parseDouble(road[3]);

            List<VehicleLog> camLogs = recentLogs.stream()
                    .filter(l -> camId.equals(l.getCameraId()))
                    .toList();

            int densityPct = 0;
            String status = "low";

            if (!camLogs.isEmpty()) {
                double avgScreenDensity = camLogs.stream()
                        .mapToInt(VehicleLog::getCurrentDensity)
                        .average()
                        .orElse(0.0);
                densityPct = (int) Math.min(100, (avgScreenDensity / maxCarsOnScreen) * 100);
            }

            if (densityPct >= 75) status = "high";
            else if (densityPct >= 40) status = "medium";

            // --- YENİ EKLENEN: KAMERAYA ÖZEL VERİLER ---
            long totalCount = repository.countByCameraId(camId); // O kameradan geçen toplam araç
            long flowRate = repository.countByCameraIdAndTimestampAfter(camId, oneMinuteAgo); // O kameranın anlık akış hızı

            Map<String, Object> roadData = new java.util.HashMap<>();
            roadData.put("cameraId", camId); // React tarafında kolay eşleştirmek için ID'yi de yolluyoruz
            roadData.put("road", roadName);
            roadData.put("length", length);
            roadData.put("density", densityPct);
            roadData.put("status", status);
            roadData.put("totalCount", totalCount);
            roadData.put("flowRate", flowRate);

            result.add(roadData);
        }

        return ResponseEntity.ok(result);
    }
    @GetMapping("/hourly-types")
    public ResponseEntity<List<Map<String, Object>>> getHourlyTypes() {
        // Bugünün başlangıcı ve şu anki zaman
        LocalDateTime startOfDay = java.time.LocalDate.now().atStartOfDay();
        LocalDateTime now = LocalDateTime.now();

        // Bugün geçen tüm araçları çekiyoruz
        List<VehicleLog> todayLogs = repository.findByTimestampBetween(startOfDay, now);

        List<Map<String, Object>> result = new java.util.ArrayList<>();

        // 24 saat için döngü (Grafiğin 24 çubuğu için)
        for (int i = 0; i < 24; i++) {
            String hourStr = String.format("%02d:00", i);
            int currentHour = i;

            // İlgili saat dilimindeki logları filtreliyoruz
            List<VehicleLog> logsInHour = todayLogs.stream()
                    .filter(log -> log.getTimestamp() != null && log.getTimestamp().getHour() == currentHour)
                    .toList();

            // Sınıflara göre sayım (Python YOLO'dan gelen 'type' isimlerine birebir uymalı)
            long carCount = logsInHour.stream().filter(l -> "Araba".equals(l.getType())).count();
            long truckCount = logsInHour.stream().filter(l -> "Kamyon/Tir".equals(l.getType())).count();
            long busCount = logsInHour.stream().filter(l -> "Otobus".equals(l.getType())).count();
            long motoCount = logsInHour.stream().filter(l -> "Motor".equals(l.getType())).count();

            // JSON haritasını oluşturuyoruz (React grafiği bu anahtarları bekliyor)
            Map<String, Object> hourData = new java.util.HashMap<>();
            hourData.put("hour", hourStr);
            hourData.put("car", carCount);
            hourData.put("truck", truckCount);
            hourData.put("bus", busCount);
            hourData.put("minibus", 0); // Python YOLO sınıflandırmanda şu an yok, o yüzden 0 atıyoruz
            hourData.put("moto", motoCount);

            result.add(hourData);
        }

        return ResponseEntity.ok(result);
    }

    @GetMapping("/daily-comparison")
    public ResponseEntity<List<Map<String, Object>>> getDailyComparison() {
        List<Map<String, Object>> result = new java.util.ArrayList<>();
        java.time.LocalDate today = java.time.LocalDate.now();

        // Son 7 günü geriye dönük (örneğin: Pzt, Sal, Çar) hesaplıyoruz
        for (int i = 6; i >= 0; i--) {
            java.time.LocalDate date = today.minusDays(i);
            LocalDateTime start = date.atStartOfDay();
            LocalDateTime end = date.atTime(java.time.LocalTime.MAX);

            // O güne ait toplam araç sayısını çekiyoruz
            int count = repository.countByTimestampBetween(start, end);

            // Günü Türkçe kısa formatta alıyoruz (Pzt, Sal vb.)
            String dayName = date.getDayOfWeek().getDisplayName(
                    java.time.format.TextStyle.SHORT,
                    new java.util.Locale("tr", "TR")
            );

            Map<String, Object> dayData = new java.util.HashMap<>();
            dayData.put("day", dayName);
            dayData.put("count", count);

            result.add(dayData);
        }

        return ResponseEntity.ok(result);
    }
}