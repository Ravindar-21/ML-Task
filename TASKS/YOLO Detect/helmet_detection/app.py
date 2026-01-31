from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

app = Flask(__name__)

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
violation_count = 0
beep_playing = False


def play_beep():
    global beep_playing
    if not beep_playing:
        beep_playing = True
        playsound("alert.wav")
        beep_playing = False


def generate_frames():
    global violation_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5)

        helmet_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # CLASS 0 = Helmet
                if cls == 0:
                    helmet_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Helmet {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

        # ❌ NO HELMET CASE
        if not helmet_detected:
            violation_count += 1

            cv2.putText(
                frame,
                "NO HELMET!",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4
            )

            threading.Thread(target=play_beep).start()

        cv2.putText(
            frame,
            f"Violations: {violation_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
