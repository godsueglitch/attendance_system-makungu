from insightface.app import FaceAnalysis
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FaceAnalysis
app = FaceAnalysis()
app.prepare(ctx_id=0)

print("Starting webcam face detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Webcam - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
