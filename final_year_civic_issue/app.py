import cv2
import os
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- SETTINGS ---
MODEL_FILE = 'water_model.h5'
REPORT_DIR = 'static/reports'
DATABASE_FILE = 'civic_reports.csv'

# Calibration (Tweak these tomorrow if needed!)
THRESHOLD = 0.40        # Set this BELOW your current 0.47
REQUIRED_FRAMES = 10     # Make it very fast

os.makedirs(REPORT_DIR, exist_ok=True)
MODEL = load_model(MODEL_FILE)

current_location = "Not Set"
monitoring_active = False
detection_counter = 0 

def gen_frames():
    global current_location, monitoring_active, detection_counter
    camera = cv2.VideoCapture(0)
    last_save_time = datetime.datetime.now()

    while True:
        success, frame = camera.read()
        if not success: break

        if monitoring_active:
            img = cv2.resize(frame, (128, 128))
            img = np.expand_dims(img, axis=0) / 255.0
            
            prediction = MODEL.predict(img, verbose=0)[0][0]
            score = round(float(prediction), 2)

            # Logic for the Stability Counter
            if score > THRESHOLD:
                detection_counter += 1
            else:
                detection_counter = max(0, detection_counter - 1) # Slowly decrease if unsure

            # Visual Overlays
            color = (0, 0, 255) if score > THRESHOLD else (0, 255, 0)
            cv2.putText(frame, f"Water Score: {score}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Progress Bar
            bar_len = int((detection_counter / REQUIRED_FRAMES) * 200)
            cv2.rectangle(frame, (10, 80), (10 + min(bar_len, 200), 95), (0, 255, 255), -1)

            # Save Logic
            if detection_counter >= REQUIRED_FRAMES:
                now = datetime.datetime.now()
                if (now - last_save_time).seconds > 10:
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    file_path = os.path.join(REPORT_DIR, f"flood_{timestamp}.jpg")
                    cv2.imwrite(file_path, frame)
                    
                    # Log to CSV
                    log = [[now.strftime("%Y-%m-%d %H:%M:%S"), current_location, file_path]]
                    pd.DataFrame(log).to_csv(DATABASE_FILE, mode='a', header=not os.path.exists(DATABASE_FILE), index=False)
                    
                    last_save_time = now
                    detection_counter = 0 
                    print(f"SAVED: {file_path}")

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/start_system', methods=['POST'])
def start_system():
    global current_location, monitoring_active
    current_location = request.json.get('location', 'Unknown')
    monitoring_active = True
    return jsonify({"status": "active"})

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)