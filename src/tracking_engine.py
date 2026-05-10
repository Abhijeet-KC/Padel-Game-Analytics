import cv2
import numpy as np
import torch
from ultralytics import YOLO

class UnifiedObjectTracker:
    def __init__(self, model_path='models/yolo26m.pt'):
        # Check CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO directly on {self.device} using {model_path}...")
        self.model = YOLO(model_path)
        
        # 0: person, 32: sports ball, 38: tennis racket
        self.target_classes = [0, 32, 38]
        
    def _distance_to_bbox(self, p_bbox, r_bbox):
        px, py = (p_bbox[0]+p_bbox[2])/2, (p_bbox[1]+p_bbox[3])/2
        rx, ry = (r_bbox[0]+r_bbox[2])/2, (r_bbox[1]+r_bbox[3])/2
        return np.sqrt((px-rx)**2 + (py-ry)**2)
        
    def process_frame(self, frame):
        """
        Runs YOLO with bytetrack to keep consistent IDs.
        Filters players based on racket proximity and assigns Player A/B names.
        """
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                   classes=self.target_classes, device=self.device, verbose=False)[0]
        raw_players = []
        rackets = []
        balls = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else np.zeros_like(clss)
            
            for box, cls, conf, track_id in zip(boxes, clss, confs, ids):
                cls = int(cls)
                x1, y1, x2, y2 = map(int, box)
                
                det = {"bbox": [x1, y1, x2, y2], "conf": float(conf), "id": int(track_id)}
                
                if cls == 0:
                    raw_players.append(det)
                elif cls == 38:
                    rackets.append(det)
                elif cls == 32:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    det["x"] = cx
                    det["y"] = cy
                    balls.append(det)
        
        ball = sorted(balls, key=lambda x: x["conf"], reverse=True)[0] if balls else None
        
        # Filter: keep only players holding rackets
        filtered_players = []
        for p in raw_players:
            has_racket = False
            p_w, p_h = p["bbox"][2]-p["bbox"][0], p["bbox"][3]-p["bbox"][1]
            max_dist = max(p_w, p_h) * 1.5 
            
            for r in rackets:
                if self._distance_to_bbox(p["bbox"], r["bbox"]) < max_dist:
                    has_racket = True
                    break
            
            if has_racket:
                filtered_players.append(p)
                
        # Assign Player A / Player B based on Y position
        for p in filtered_players:
            py = (p["bbox"][1] + p["bbox"][3]) / 2
            if py > frame.shape[0] / 2:
                p["name"] = "Player A"
            else:
                p["name"] = "Player B"
                
        return filtered_players, rackets, ball
