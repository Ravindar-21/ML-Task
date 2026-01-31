from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Test image path
IMAGE_PATH = "test_images/sample.png"

# Run inference
results = model(IMAGE_PATH, conf=0.5)

# Final decision logic
label = "Unknown"
confidence = 0.0

for box in results[0].boxes:
    cls = int(box.cls)
    conf = float(box.conf)

    if conf > confidence:
        confidence = conf
        label = model.names[cls]

print("====== Fake News Detection Result ======")
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}")