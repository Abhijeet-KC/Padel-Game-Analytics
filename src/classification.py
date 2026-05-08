import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ShotClassifier:
    def __init__(self, model_path='models/pose_landmarker_lite.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def classify_shot(self, frame, hit_event):
        """
        Takes the full frame and the hit_event details to classify the shot type.
        """
        x1, y1, x2, y2 = hit_event["player_bbox"]
        
        # Ensure bounding box is within frame limits
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        # Crop player
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return "unknown"
            
        player_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=player_rgb)
        
        results = self.detector.detect(mp_image)
        
        if not results.pose_landmarks:
            return "unknown"
            
        # Landmarks for first detected person
        landmarks = results.pose_landmarks[0]
        
        # Extract required landmarks (by index)
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        
        # Check Serve/Smash: Wrist higher than shoulder (In images, smaller Y is higher up)
        threshold = 0.1 # roughly 10% of crop height
        highest_wrist_y = min(r_wrist.y, l_wrist.y)
        highest_shoulder_y = min(r_shoulder.y, l_shoulder.y)
        
        if highest_wrist_y < (highest_shoulder_y - threshold):
            # According to guide: if ball y < mid-frame it's a serve, else smash
            if hit_event["ball_y"] < (h / 2):
                return "serve"
            else:
                return "smash"
                
        # Forehand vs Backhand
        player_center_x = (x1 + x2) / 2
        ball_x = hit_event["ball_x"]
        
        # Determine handedness based on which wrist is closer to the ball
        # Map normalized wrist coords to absolute frame coords
        r_wrist_abs_x = x1 + r_wrist.x * (x2 - x1)
        r_wrist_abs_y = y1 + r_wrist.y * (y2 - y1)
        l_wrist_abs_x = x1 + l_wrist.x * (x2 - x1)
        l_wrist_abs_y = y1 + l_wrist.y * (y2 - y1)
        
        dist_r = (r_wrist_abs_x - ball_x)**2 + (r_wrist_abs_y - hit_event["ball_y"])**2
        dist_l = (l_wrist_abs_x - ball_x)**2 + (l_wrist_abs_y - hit_event["ball_y"])**2
        
        is_right_handed = dist_r < dist_l
        
        if is_right_handed:
            if ball_x > player_center_x: # Ball to the right of player center
                return "forehand"
            else:
                return "backhand"
        else:
            if ball_x < player_center_x: # Ball to the left of player center
                return "forehand"
            else:
                return "backhand"
