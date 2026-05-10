import cv2
from ultralytics import YOLO

class PlayerRacketDetector:
    def __init__(self, model_path='models/yolo26m.pt'):
        # YOLO will automatically download yolo26m.pt to this path if it doesn't exist.
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        # In COCO dataset used by YOLO:
        # Class 0: person
        # Class 38: tennis racket (can be used for padel racket detection context)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id == 0:
                detections.append({"bbox": [x1, y1, x2, y2], "class": "player", "conf": conf})
            elif cls_id == 38:
                detections.append({"bbox": [x1, y1, x2, y2], "class": "racket", "conf": conf})
                
        return detections