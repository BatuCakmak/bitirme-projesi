package com.batu.vehicle_tracking_api.model;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Table(name = "vehicle_logs")
@Data // Lombok sayesinde getter/setter yazmamıza gerek kalmıyor
public class VehicleLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id; // Veritabanındaki otomatik artan anahtarımız

    @Column(name = "camera_id")
    private String cameraId;

    private Long vehicleId; // ByteTrack'in atadığı ID
    private String type;    // Araç türü (Araba, Kamyon vs.)
    private String status;  // Durum (Geçti)
    private LocalDateTime timestamp; // Geçiş zamanı

    // İŞTE DÜZELTİLEN KISIM: Python'daki ismin birebir aynısı olmak ZORUNDA
    @Column(name = "current_density")
    private int currentDensity;
}