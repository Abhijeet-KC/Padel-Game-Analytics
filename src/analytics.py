import csv
import math

class DirectionResolver:
    def __init__(self, frames_to_wait=15, dx_threshold=40):
        self.frames_to_wait = frames_to_wait
        self.dx_threshold = dx_threshold
        self.pending_hits = []

    def add_hit(self, hit_event):
        """Register a new hit event to trace direction later."""
        self.pending_hits.append(hit_event)

    def process_frame(self, frame_num, ball_pos):
        """Check if any pending hits have reached frame+15 to resolve their direction."""
        resolved_hits = []
        for hit in self.pending_hits[:]:
            if frame_num >= hit["frame"] + self.frames_to_wait:
                if ball_pos["visible"]:
                    dx = ball_pos["x"] - hit["ball_x"]
                    if dx > self.dx_threshold:
                        direction = "right"
                    elif dx < -self.dx_threshold:
                        direction = "left"
                    else:
                        direction = "center"
                else:
                    direction = "unknown"
                
                hit["direction"] = direction
                self.pending_hits.remove(hit)
                resolved_hits.append(hit)
                
                # Also store the end position for visualization
                hit["direction_end_pos"] = (ball_pos["x"], ball_pos["y"])
        return resolved_hits

class BounceDetector:
    def __init__(self, floor_threshold_ratio=0.8):
        self.floor_threshold_ratio = floor_threshold_ratio
        self.history = []
        self.bounces = []

    def process_frame(self, frame_num, ball_pos, frame_height):
        if not ball_pos["visible"]:
            return None
            
        y = ball_pos["y"]
        self.history.append((frame_num, y))
        
        # Keep last 3 points to determine V change (t-2, t-1, t)
        if len(self.history) > 3:
            self.history.pop(0)
            
        if len(self.history) == 3:
            y_t2 = self.history[0][1]
            y_t1 = self.history[1][1]
            y_t0 = self.history[2][1]
            
            # v1 is velocity from t-2 to t-1
            # v2 is velocity from t-1 to t0
            v1 = y_t1 - y_t2
            v2 = y_t0 - y_t1
            
            # Bounce condition:
            # 1. v1 > 0 (moving down, since y increases downwards)
            # 2. v2 < 0 (moving up)
            # 3. y_t1 > floor_threshold (it's in the lower portion of the screen)
            
            floor_threshold = frame_height * self.floor_threshold_ratio
            if v1 > 0 and v2 < 0 and y_t1 > floor_threshold:
                bounce_event = {
                    "frame": self.history[1][0],
                    "x": ball_pos["x"],
                    "y": y_t1
                }
                self.bounces.append(bounce_event)
                return bounce_event
                
        return None

class HitDetector:
    def __init__(self, proximity_threshold=100, angle_change_threshold=45, speed_ratio_threshold=1.5, cooldown_frames=15):
        # Configuration based on Rule-based Logic in Guide.MD
        self.proximity_threshold = proximity_threshold
        self.angle_change_threshold = angle_change_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.cooldown_frames = cooldown_frames
        
        self.ball_history = []
        self.hit_events = []
        self.last_hit_frame = -999

    def process_frame(self, frame_num, ball_pos, players_dets):
        if not ball_pos["visible"]:
            return None

        # Keep a rolling history of the last 6 ball positions for velocity calculation over N=5 frames
        self.ball_history.append((frame_num, ball_pos["x"], ball_pos["y"]))
        if len(self.ball_history) > 6:
            self.ball_history.pop(0)

        if len(self.ball_history) == 6:
            t = self.ball_history[-1]
            t_minus_5 = self.ball_history[0]
            
            vx = t[1] - t_minus_5[1]
            vy = t[2] - t_minus_5[2]
            
            # Simple speed calculation
            speed = math.sqrt(vx**2 + vy**2)
            
            # Angle calculation (-180 to 180 degrees)
            angle = math.degrees(math.atan2(vy, vx))
            
            # To detect sudden changes, we need the vectors from a slightly older window
            # t-5 to t-10 (For simplicity, we'll approximate with frame-by-frame shifts here but
            # tracking N=5 velocity changes requires maintaining velocity history)
            # 
            # We'll refine the logic here for step 4: Find sudden directional or speed shift inside hitting proximity
            
            nearest_player = self._get_nearest_player_bbox(t[1], t[2], players_dets)
            if nearest_player:
                dist = self._distance_to_bbox(t[1], t[2], nearest_player["bbox"])
                
                # We'll need a velocity history to calculate angle_change and speed_ratio
                if dist < self.proximity_threshold:
                    if frame_num - self.last_hit_frame > self.cooldown_frames:
                        # For now, flag as potential hit event space when ball is very near player.
                        # We will implement full velocity-ratio checks when velocity history is added
                        hit = {
                            "frame": frame_num,
                            "ball_x": t[1],
                            "ball_y": t[2],
                            "player_bbox": nearest_player["bbox"],
                            "timestamp_sec": round(frame_num / 30.0, 2)  # Assuming 30fps default
                        }
                        self.last_hit_frame = frame_num
                        return hit
        return None
        
    def _get_nearest_player_bbox(self, bx, by, detections):
        min_dist = float('inf')
        nearest = None
        for det in detections:
            if det['class'] == 'player':
                dist = self._distance_to_bbox(bx, by, det['bbox'])
                if dist < min_dist:
                    min_dist = dist
                    nearest = det
        return nearest

    def _distance_to_bbox(self, x, y, bbox):
        x1, y1, x2, y2 = bbox
        
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return math.sqrt(dx*dx + dy*dy)
