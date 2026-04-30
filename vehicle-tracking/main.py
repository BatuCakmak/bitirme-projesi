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
model = YOLO('best.pt')

# --- SAYIM DEĞİŞKENLERİ ---
LINE_Y = 500 
counted_ids = set()
class_counts = {0: 0, 1: 0, 2: 0, 3: 0} 
class_names = {0: "Araba", 1: "Motor", 2: "Otobus", 3: "Kamyon"}

# Jeneratör Fonksiyonu: Videoyu okur, işler ve web'e uygun kareler (frame) üretir
def generate_frames():
    cap = cv2.VideoCapture("test_video.mp4")
    #kayseri_canli_yayin_url = "https://canliyayin.kayseri.bel.tr/canli/2b9d5580-d7cf-42b3-be63-c977f3900c3e.stream/playlist.m3u8?bltokenstarttime=1775463505&bltokenendtime=1775463805&bltokenCustomParameter=secured&bltokenhash=Y_Bt-hy_SlOK0XMRUzEVIbljuC8l9UQbh5GxZRrLyBo="
    #cap = cv2.VideoCapture(kayseri_canli_yayin_url)
    if not cap.isOpened():
        print("Hata: Video dosyası okunamadı.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Video biterse başa sar (Sürekli yayın için)
            continue

        #results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1,device=0)
        # YENİ HALİ (BoT-SORT ve Optimize Güven Skoru):
        results = model.track(frame, persist=True, tracker="botsort.yaml", conf=0.3, device=0)
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

                margin = 15 
                if (LINE_Y - margin) < cy < (LINE_Y + margin):
                    if id not in counted_ids:
                        counted_ids.add(id)
                        class_counts[cls_id] += 1
                        
                        data = {
                            "type": class_names[int(cls_id)],
                            "vehicleId": int(id),
                            "status": "Gecti",
                            "currentDensity": int(current_density)
                        }
                        try:
                            requests.post(API_URL, json=data, timeout=0.1)
                        except:
                            pass

        # Ekran Yazıları
        y_offset = 50
        for cls_id, count in class_counts.items():
            text = f"{class_names[cls_id]}: {count}"
            cv2.putText(annotated_frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40
        cv2.putText(annotated_frame, f"Anlik Yogunluk: {current_density}", (width - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Görüntüyü Web formatına (JPEG formatında Byte dizisine) çevirme
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Flask'a bu kareyi gönder
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# React'in videoyu çekeceği uç (Endpoint)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Python AI Sunucusu başlatıldı! Canlı yayın 5000 portunda...")
    # Flask sunucusunu başlat (Port: 5000)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)