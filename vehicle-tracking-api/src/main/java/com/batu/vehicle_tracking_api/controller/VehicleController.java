package com.batu.vehicle_tracking_api.controller;

import com.batu.vehicle_tracking_api.model.VehicleLog;
import com.batu.vehicle_tracking_api.repository.VehicleLogRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

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
    public ResponseEntity<Object> getVehicleStats() {
        List<Object[]> stats = repository.countVehiclesByType();
        return ResponseEntity.ok(stats);
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
}