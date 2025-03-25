import numpy as np
from deepface import DeepFace
from pymongo import MongoClient
import hnswlib

class MongoConnection:
    def __init__(self):
        self.client = MongoClient('mongodb+srv://sonhai:LIhIaamZwU6Duxak@cluster0.wp705.mongodb.net/dbuser?retryWrites=true&w=majority')
        self.db = self.client["defaultdb"]
        self.collection = self.db["faces"]
    
    def insert(self, user, email, accountnumber, face_encoding):
        # Lưu thông tin người dùng và mã hóa khuôn mặt vào MongoDB
        self.collection.insert_one({"user": user, "email": email, "accountnumber": accountnumber, "encoding": face_encoding})
    
    def insertBalance(self, userName, balance):
        self.collection.insert_one({"balance_" + userName: balance})

    def insertInvoice(self, userName, invoice):
        self.collection.insert_one({"invoice_" + userName: invoice})

    def getBalance(self, userName):
        userName = "balance_" + userName
        query = self.collection.find({userName: {"$exists": True}})
        for q in query:
            return q[userName]
        return None
    
    def updateBalance(self, userName, balance):
        self.collection.update_one({"balance_" + userName: {"$exists": True}}, {"$set": {"balance_" + userName: balance}})

    def get_all(self):
        dimension = 128  # Kích thước vector mã hóa khuôn mặt
        index = hnswlib.Index(space="l2", dim=dimension)  # Thay đổi sang khoảng cách Euclidean
        index.init_index(max_elements=10000, ef_construction=300, M=32)

        ids = 0
        users = []
        for q in self.collection.find():
            try:
                user = q["user"]
                encoding_list = q["encoding"]  # Danh sách mã hóa khuôn mặt
                for encoding in encoding_list:
                    encoding = np.array(encoding, dtype=np.float32)
                    encoding = self.normalize_vector(encoding)  # Chuẩn hóa vector
                    index.add_items(encoding, ids)
                    users.append(user)
                    ids += 1
            except Exception as e:
                print(f"Error while processing user {q}: {e}")
        index.set_ef(100)  # Tăng hiệu quả tìm kiếm
        return users, index

    @staticmethod
    def normalize_vector(vector):
        """Chuẩn hóa vector sử dụng L2 Norm."""
        return vector / np.linalg.norm(vector)
    
    def get_user_data(self, username):
        """Lấy thông tin người dùng cụ thể"""
        try:
            query = self.collection.find_one({"user": username})
            if query:
                return {
                    "user": query.get("user"),
                    "email": query.get("email"),
                    "accountnumber": query.get("accountnumber"),
                    "encoding": query.get("encoding"),
                }
            return None
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return None
        
    def find_user_by_encoding(self, face_encoding, threshold=0.3):
        try:
            query = self.collection.find()
            for record in query:
                user_encoding = record.get("encoding")
                if user_encoding:
                    user_encoding = np.array(user_encoding)  # Chuyển đổi về dạng numpy array
                    # Tính toán khoảng cách Euclidean
                    distance = np.linalg.norm(face_encoding - user_encoding)
                    if distance <= threshold:
                        return record.get("user")  # Trả về tên người dùng
            return None
        except Exception as e:
            print(f"Error finding user by encoding: {e}")
            return None

    def delete_all(self):
        self.collection.delete_many({})

    # Hàm tính toán mã hóa khuôn mặt và lưu vào MongoDB
    def encode_face_and_insert(self, face_image, user, email, accountnumber):
        try:
            # Sử dụng DeepFace để tính toán mã hóa khuôn mặt
            face_encoding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=False)
            if face_encoding:
                encoding = face_encoding[0]["embedding"]  # Lấy embedding của khuôn mặt

                # Lưu thông tin vào MongoDB
                self.insert(user, email, accountnumber, encoding)
            else:
                print(f"Error: No face detected in image for user {user}")
        except Exception as e:
            print(f"Error encoding face for user {user}: {e}")

