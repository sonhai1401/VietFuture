import cv2
import os
from ultralytics import YOLO
import time
from threading import Thread
model = YOLO("models/yolov8n-face.pt")
#object
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)
def detect_faces(frame, frame_count, bounding_boxes, resize_factor):# hàm phát hiện khuôn mặt
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    results = model(small_frame)
    bounding_boxes.clear()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * (1 / resize_factor)).astype(int)
            conf = box.conf[0].item()
            if conf >= 0.8:# độ tin cậy 0.8 thì nhận
                bounding_boxes.append((x1, y1, x2, y2, conf))
                face = frame[y1:y2, x1:x2]
                face_filename = os.path.join(output_folder, f"face_{frame_count}_{len(bounding_boxes)}.jpg")
                cv2.imwrite(face_filename, face)
def run_face_detection(desired_fps=60, resize_factor=0.25, detection_interval=10):
    cap = cv2.VideoCapture(0)# tăng này lên để sử dụng nhiều cam
    #tôi còn đa luồng chưa xử lý
    frame_duration = 1 / desired_fps
    frame_count = 0
    bounding_boxes = [] #bouding box khuôn mặt
    while cap.isOpened():# mở cam thực lên
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
         # Tạo một luồng mới để phát hiện khuôn mặt
        if frame_count % detection_interval == 0:
            detection_thread = Thread(target=detect_faces, args=(frame, frame_count, bounding_boxes, resize_factor))
            detection_thread.start()
        for (x1, y1, x2, y2, conf) in bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("face_cam", frame)
        elapsed_time = time.time() - start_time
        wait_time = max(int((frame_duration - elapsed_time) * 1000), 1)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    #bổ sung đa luồng chổ này.
    
    run_face_detection()
