import cv2
from cvzone.ClassificationModule import Classifier

# ================== LOAD MODEL ==================
classifier = Classifier("keras_model.h5", "labels.txt")

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ================== CONTROL ==================
last_spoken = ""

# ================== MAIN LOOP ==================
while True:
    success, img = cap.read()

    # Safety check
    if not success or img is None:
        continue

    # ================== PREDICTION ==================
    prediction, index = classifier.getPrediction(img, draw=False)

    class_name = classifier.list_labels[index].strip()
    confidence = float(prediction[index])

    # ================== DISPLAY ==================
    cv2.putText(img, f"{class_name} ({confidence*100:.2f}%)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Optional: highlight valid detections
    if confidence > 0.9 and class_name.lower() != "nothing":
        cv2.putText(img, f"DETECTED: {class_name}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    # ================== SHOW ==================
    cv2.imshow("Teachable Machine Detection", img)

    # ================== EXIT ==================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()