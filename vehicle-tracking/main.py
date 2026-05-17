import cv2
import sys
import requests 
from ultralytics import YOLO
from flask import Flask, Response
from flask_cors import CORS

# --- FLASK WEB SUNUCUSU AYARLARI ---
app = Flask(__name__)
CORS(app)

API_URL = "http://localhost:8080/api/vehicles/log"
API_TIMEOUT_SECONDS = 2

# Global 'model' değişkenini kaldırdık. Modelleri aşağıda kameralara özel yükleyeceğiz.
print("Yapay Zeka Modelleri Yükleniyor, Lütfen Bekleyin...")

def send_vehicle_log(payload):
    try:
        response = requests.post(API_URL, json=payload, timeout=API_TIMEOUT_SECONDS)
    except Exception as exc:
        print(f"API baglanti hatasi ({payload.get('cameraId')}): {exc}")

ceviriler = {
    "car": "Araba",
    "motorcycle": "Motor",
    "bus": "Otobus",
    "truck": "Kamyon/Tir",
    "van": "Kamyon/Tir" 
}

# Senin çizgi mantığın aynen korundu
LINE_Y = 500 

# ÇÖZÜM: Her kameranın kendi YAPAY ZEKASI (model) ve kendi hafızası var!
camera_states = {
    "CAM-001": {
        "model": YOLO('motorcycle_finetune_best.pt'), # CAM-001 için ayrı model
        "counted_ids": set(), 
        "track_history": {}, 
        "class_counts": {"Araba": 0, "Motor": 0, "Otobus": 0, "Kamyon/Tir": 0}
    },
    "CAM-002": {
        "model": YOLO('motorcycle_finetune_best.pt'), # CAM-002 için ayrı model
        "counted_ids": set(), 
        "track_history": {}, 
        "class_counts": {"Araba": 0, "Motor": 0, "Otobus": 0, "Kamyon/Tir": 0}
    }
}

def generate_frames(video_path, camera_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: {video_path} dosyası okunamadı.")
        return
    
    # F5 atıldığında bizim sayım hafızamızı temizle
    camera_states[camera_id]["counted_ids"].clear()
    camera_states[camera_id]["track_history"].clear()

    # Bu döngünün çalışacağı kameranın state'ini ve KENDİ MODELİNİ çekiyoruz
    state = camera_states[camera_id]
    cam_model = state["model"] 

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # DİKKAT: Artık global 'model' değil, kameraya özel 'cam_model' kullanılıyor
        results = cam_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, iou=0.45, classes=[0, 1, 2, 3, 4], device=0)
        current_density = len(results[0].boxes)
        annotated_frame = results[0].plot()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (255, 0, 0), 2)

        boxes = results[0].boxes
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for id, box, cls_id in zip(ids, coords, cls_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

                # State (Hafıza) üzerinden kontrol yapıyoruz
                if id not in state["track_history"]:
                    state["track_history"][id] = cy
                
                prev_cy = state["track_history"][id]

                # Senin LINE_Y çizgi geçiş mantığın
                if (prev_cy < LINE_Y and cy >= LINE_Y) or (prev_cy > LINE_Y and cy <= LINE_Y):
                    if id not in state["counted_ids"]:
                        state["counted_ids"].add(id)
                        
                        ingilizce_isim = cam_model.names[int(cls_id)]
                        turkce_isim = ceviriler.get(ingilizce_isim, None)
                        
                        if turkce_isim is not None:
                            state["class_counts"][turkce_isim] += 1
                            
                            data = {
                                "cameraId": camera_id, 
                                "type": turkce_isim,
                                "vehicleId": int(id),
                                "status": "Gecti",
                                "currentDensity": int(current_density)
                            }
                            send_vehicle_log(data)
                
                state["track_history"][id] = cy

        y_offset = 50
        for isim, count in state["class_counts"].items():
            text = f"{isim}: {count}"
            cv2.putText(annotated_frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Sistem kitlenmesin diye ufak bekleme
        cv2.waitKey(1)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- İKİ FARKLI UÇ (ENDPOINT) ---
@app.route('/video_feed/cam1')
def video_feed_cam1():
    return Response(generate_frames("test_video.mp4", "CAM-001"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/cam2')
def video_feed_cam2():
    return Response(generate_frames("test_video2.mp4", "CAM-002"), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Python AI Sunucusu başlatıldı! Canlı yayın 5000 portunda...")
    # Çoklu videonun donmaması için threaded=True özelliği aktifleştirildi
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)