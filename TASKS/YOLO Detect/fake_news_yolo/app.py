import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    verdict = None
    verdict_type = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            # Save uploaded image
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            # Run YOLO inference
            results = model(input_path, conf=0.5)

            # ===== DRAW BOUNDING BOXES =====
            annotated_img = results[0].plot()

            # Save annotated output image
            output_filename = "output_" + file.filename
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_img)

            # ===== FINAL VERDICT LOGIC =====
            best_conf = 0
            best_label = None

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    if conf > best_conf:
                        best_conf = conf
                        best_label = model.names[cls]

            if best_label == "fake":
                verdict = "❌ FAKE NEWS"
                verdict_type = "fake"
            elif best_label == "real":
                verdict = "✅ REAL NEWS"
                verdict_type = "real"
            elif best_label == "misleading":
                verdict = "⚠️ MISLEADING NEWS"
                verdict_type = "misleading"
            else:
                verdict = "⚠️ UNABLE TO DETECT"
                verdict_type = "unknown"

            confidence = round(best_conf, 2)
            image_path = output_path

    return render_template(
        "index.html",
        verdict=verdict,
        verdict_type=verdict_type,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)