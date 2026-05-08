import cv2
import numpy as np
from collections import deque
import csv

class BallTracker:
    def __init__(self, mode='fallback'):
        self.mode = mode
        # Temporal smoothing buffer
        self.positions = deque(maxlen=5) 
        
        # Color thresholds for fallback padel/tennis ball tracking (HSV)
        self.lower_color = np.array([25, 40, 40])
        self.upper_color = np.array([65, 255, 255])
        self.last_frame_gray = None

    def track(self, frame):
        if self.mode == 'tracknet':
            return self._tracknet_predict(frame)
        else:
            return self._fallback_predict(frame)

    def _fallback_predict(self, frame):
        # 1. Color Thresholding (Yellow/Green)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # 2. Motion Detection (Frame differencing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        
        mask_motion = np.zeros_like(gray)
        if self.last_frame_gray is not None:
            frame_diff = cv2.absdiff(self.last_frame_gray, blur)
            _, mask_motion = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
        self.last_frame_gray = blur

        # 3. Combine Masks (Motion + Color)
        combined_mask = cv2.bitwise_and(mask_color, mask_motion)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter roughly based on ball size limits
            if 10 < area < 600:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_center = (cx, cy)
                    break # Take the first matching blob for fallback
        
        # 5. Temporal Smoothing & Occlusion Handling
        if best_center:
            self.positions.append(best_center)
            x, y = best_center
            visible = True
        else:
            if len(self.positions) > 0:
                # Interpolate from last known
                x, y = self.positions[-1]
            else:
                x, y = 0, 0
            visible = False

        # Apply simple temporal smoothing over last 3 frames if visible
        if visible and len(self.positions) >= 3:
            pts = list(self.positions)[-3:]
            x = int(sum([p[0] for p in pts]) / len(pts))
            y = int(sum([p[1] for p in pts]) / len(pts))

        return {"x": x, "y": y, "visible": visible}

    def _tracknet_predict(self, frame):
        # Placeholder for actual TrackNetV4 logic if weights become available
        return {"x": 0, "y": 0, "visible": False}
