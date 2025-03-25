import numpy as np
import dlib # type: ignore
from ultralytics import YOLO # type: ignore
import cv2 # type: ignore   

class DlibModel:
    def __init__(self):
        yoloModel = "yolov8n-face.pt"

        self.pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.yoloModel = YOLO(yoloModel, verbose=False)

    def getFace(self, img):
        try:
            results = self.yoloModel(img, device="cpu", verbose=False)
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
            return [], []

        if len(results) == 0:
            return [], []

        rects = []
        new_boxes = []
        for result in results:
            try:
                bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
            except Exception as e:
                print(f"Error processing bounding boxes: {e}")
                continue

            if len(bboxes) == 0:
                continue

            for bbox in bboxes:
                rects.append(dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3]))
                new_boxes.append(bbox)
        return rects, new_boxes

    def encodeFace(self, face_location, image):
        face_landmarks = self.pose_predictor(image, face_location)
        face = dlib.get_face_chip(image, face_landmarks)
        encoding = np.array(self.face_encoder.compute_face_descriptor(face))

        return encoding

    def face_landmarks(self, image, face_location):
        face_landmarks = self.pose_predictor(image, face_location)
        return face_landmarks

    def normalization(self, face_landmarks, new_img):
        landmarks_tuple = []

        for i in range(0, 68):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            
            landmarks_tuple.append((x, y))
            
        routes = [i for i in range(16, -1, -1)] + [i for i in range(17, 19)] + [i for i in range(24, 26)] + [16]
        routes_coordinates = []
        for i in range(0, len(routes) - 1):
            source_point = routes[i]
            
            source_coordinate = landmarks_tuple[source_point]
            
            routes_coordinates.append(source_coordinate)
        
        mask = np.zeros((new_img.shape[0], new_img.shape[1]))

        mask = cv2.fillConvexPoly(mask, np.array(routes_coordinates), 1)

        mask = mask.astype(bool)

        out = np.zeros_like(new_img)

        out[mask] = new_img[mask]

        return out
    
    def getSimilarity(self, face1_embeddings, face2_embeddings):
        return np.linalg.norm(face1_embeddings-face2_embeddings)