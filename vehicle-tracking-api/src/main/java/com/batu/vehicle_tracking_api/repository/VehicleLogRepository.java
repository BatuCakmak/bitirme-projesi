package com.batu.vehicle_tracking_api.repository;

import com.batu.vehicle_tracking_api.model.VehicleLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface VehicleLogRepository extends JpaRepository<VehicleLog, Long> {
    // Tüm CRUD (Kaydet, Sil, Güncelle, Listele) işlemleri otomatik olarak burada var.
    // Araç türlerine göre toplam sayıları getiren özel sorgu
    @Query("SELECT v.type, COUNT(v) FROM VehicleLog v GROUP BY v.type")
    List<Object[]> countVehiclesByType();
    // YENİ EKLENEN: Belirli bir zamandan sonra geçen toplam araç sayısını bulur (Canlı Akış Hızı)
    int countByTimestampAfter(LocalDateTime time);
}