import cv2
import os
from ultralytics import YOLO
import time
from threading import Thread, Lock
from deepface import DeepFace
import numpy as np
from db.conn import MongoConnection
from scipy.spatial.distance import cosine

# Kết nối MongoDB
mongoConn = MongoConnection()

# Tải model YOLO
face_model = YOLO("realtime2/models/yolov8n-face.pt")  # Model nhận diện khuôn mặt
object_model = YOLO("realtime2/models/yolov8n.pt")    # Model nhận diện vật thể

# Thư mục lưu khuôn mặt được phát hiện
output_folder = r"D:\VietFuture\VietFuture\detected_faces_output"
os.makedirs(output_folder, exist_ok=True)

# Khóa đồng bộ (Lock) cho việc cập nhật dữ liệu giữa các luồng
bounding_boxes_lock = Lock()

# Hàm tìm kiếm người tương ứng trong cơ sở dữ liệu MongoDB
def find_matching_user(face_embedding):
    # Truy vấn tất cả người dùng trong MongoDB
    users = mongoConn.collection.find()  
    min_distance = float("inf")
    matched_user = None

    for user in users:
        db_embedding = np.array(user['embedding'])  # Truy xuất embedding của người dùng từ MongoDB
        distance = cosine(face_embedding, db_embedding)  # Tính cosine similarity

        if distance < min_distance:
            min_distance = distance
            matched_user = user

    if min_distance < 0.4:  # Ngưỡng cho việc nhận diện khuôn mặt, điều chỉnh nếu cần
        return matched_user["user_name"]
    else:
        return "Unknown"  # Nếu không tìm thấy khớp

# Hàm phát hiện khuôn mặt
def detect_faces(frame, frame_count, bounding_boxes, resize_factor, mongo_connection):
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    results = face_model(small_frame)
    bounding_boxes.clear()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * (1 / resize_factor)).astype(int)
            conf = box.conf[0].item()

            if conf >= 0.6:  # Ngưỡng độ tin cậy cho khuôn mặt
                face = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                # Lưu ảnh khuôn mặt vào thư mục mới
                face_path = f"{output_folder}/face_{frame_count}.jpg"  # Chỉnh sửa đường dẫn lưu ảnh
                cv2.imwrite(face_path, face)

                # Sử dụng DeepFace để nhận diện khuôn mặt và tìm kiếm người trong cơ sở dữ liệu MongoDB
                try:
                    # Tính toán embedding cho khuôn mặt
                    representations = DeepFace.represent(face_path, model_name="Facenet", enforce_detection=False)

                    if representations and isinstance(representations, list):
                        embedding_data = representations[0]
                        if "embedding" in embedding_data:
                            face_embedding = embedding_data["embedding"]
                            # Tìm người trong MongoDB
                            user_name = find_matching_user(face_embedding)
                        else:
                            print("Không có embedding trong representations.")
                            user_name = "Unknown"
                    else:
                        print("Không có dữ liệu representations.")
                        user_name = "Unknown"
                except Exception as e:
                    print(f"Error in face recognition: {e}")
                    user_name = "Unknown"
                    
            with bounding_boxes_lock:
                # Thêm bounding box và thông tin
                bounding_boxes.append((x1, y1, x2, y2, conf, user_name))

# Hàm phát hiện vật thể
def detect_objects(frame, resize_factor, object_bounding_boxes):
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    results = object_model(small_frame)
    object_bounding_boxes.clear()
    target_labels = ["cup", "fork", "spoon", "banana", "apple", "orange", 
                 "sandwich", "broccoli", "carrot", "sausage", "laptop"]
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * (1 / resize_factor)).astype(int)
            conf = box.conf[0].item()
            label = int(box.cls[0].item())  # Nhãn của vật thể
            class_name = object_model.names[label]  # Lấy tên class từ nhãn
            if conf >= 0.5 and class_name in target_labels:  # Lọc theo ngưỡng tin cậy và danh sách nhãn
                object_bounding_boxes.append((x1, y1, x2, y2, conf, label))

# Hàm chạy phát hiện khuôn mặt và vật thể cho một camera
def run_face_and_object_detection(camera_id, desired_fps=30, resize_factor=0.5, detection_interval=5, mongo_connection=None):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_duration = 1 / desired_fps
    frame_count = 0
    face_bounding_boxes = []
    object_bounding_boxes = []

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện khuôn mặt và vật thể mỗi `detection_interval` frame
        if frame_count % detection_interval == 0:
            face_thread = Thread(target=detect_faces, args=(frame, frame_count, face_bounding_boxes, resize_factor, mongo_connection))
            object_thread = Thread(target=detect_objects, args=(frame, resize_factor, object_bounding_boxes))
            face_thread.start()
            object_thread.start()
            face_thread.join()
            object_thread.join()

        # Vẽ bounding boxes cho khuôn mặt
        for (x1, y1, x2, y2, conf, user_name) in face_bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = user_name  # Hiển thị tên người dùng hoặc "Unknown"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow(f"Camera {camera_id}", frame)

        # Tính toán FPS và thời gian chờ
        elapsed_time = time.time() - start_time
        wait_time = max(int((frame_duration - elapsed_time) * 1000), 1)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Chạy chương trình chính
if __name__ == "__main__":
    camera_ids = [0]  # Thêm nhiều ID nếu có nhiều camera
    threads = []

    mongo_connection = MongoConnection()  # Kết nối MongoDB

    for camera_id in camera_ids:
        thread = Thread(target=run_face_and_object_detection, args=(camera_id, 30, 0.5, 5, mongo_connection))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
