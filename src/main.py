import cv2
import json
import os
import csv
from detection import PlayerRacketDetector
from tracking import BallTracker
from analytics import HitDetector, DirectionResolver, BounceDetector
from classification import ShotClassifier

def main():
    # Setup input/output paths
    video_path = "videos/inference_sample_video.mp4"
    output_video_path = "outputs/inference_annotated_video.mp4"
    output_json_path = "outputs/inference_detections.json"
    output_csv_path = "outputs/inference_ball_trajectory.csv"
    
    if not os.path.exists(video_path):
        print(f"Error: Could not find input video at {video_path}")
        return

    print("Initializing Detectors, Trackers and Classifiers...")
    detector = PlayerRacketDetector(model_path="models/yolo26m.pt")
    ball_tracker = BallTracker(mode='fallback')
    hit_detector = HitDetector()
    classifier = ShotClassifier()
    direction_resolver = DirectionResolver()
    bounce_detector = BounceDetector(floor_threshold_ratio=0.8)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_detections = {}
    ball_positions = {}
    classified_shots = []
    bounces = []
    csv_data = [["frame", "x", "y", "visible"]]
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing Video ({total_frames} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect Players & Racket
        dets = detector.detect(frame)
        frame_detections[frame_count] = dets
        
        # 2. Track Ball
        ball_pos = ball_tracker.track(frame)
        ball_positions[frame_count] = ball_pos
        csv_data.append([frame_count, ball_pos["x"], ball_pos["y"], ball_pos["visible"]])
        
        # 2.5 Detect Bounces
        bounce = bounce_detector.process_frame(frame_count, ball_pos, height)
        if bounce:
            bounces.append(bounce)
            print(f"Frame {frame_count}: BOUNCE Detected at ({bounce['x']}, {bounce['y']})")
        
        # 3. Detect Hits
        hit = hit_detector.process_frame(frame_count, ball_pos, dets)
        
        # 4. Classify Shot if hit occurred
        if hit:
            shot_type = classifier.classify_shot(frame, hit)
            hit["shot_type"] = shot_type
            hit["direction"] = "pending"
            classified_shots.append(hit)
            direction_resolver.add_hit(hit)
            print(f"Frame {frame_count}: Player - {shot_type.upper()}")
            cv2.putText(frame, f"HIT: {shot_type.upper()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # 4.5 Resolve Shot Direction
        resolved_hits = direction_resolver.process_frame(frame_count, ball_pos)
        for rh in resolved_hits:
            print(f"Frame {rh['frame']} hit direction resolved: {rh['direction'].upper()}")
            
            # Draw arrow on frame from hit position toward end position
            if rh["direction"] != "unknown" and "direction_end_pos" in rh:
                start_point = (int(rh["ball_x"]), int(rh["ball_y"]))
                end_point = (int(rh["direction_end_pos"][0]), int(rh["direction_end_pos"][1]))
                
                # Determine arrow color based on shot type Guide
                color = (0, 255, 0) # default green
                if rh["shot_type"] == "forehand":
                    color = (0, 255, 0)
                elif rh["shot_type"] == "backhand":
                    color = (0, 0, 255)
                elif rh["shot_type"] in ["serve", "smash"]:
                    color = (0, 255, 255)
                    
                cv2.arrowedLine(frame, start_point, end_point, color, 3, tipLength=0.2)
            
        # 5. Annotate bounding boxes
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if det['class'] == 'player' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        # 6. Annotate Ball
        if ball_pos["visible"]:
            cv2.circle(frame, (ball_pos["x"], ball_pos["y"]), 6, (0, 0, 255), -1)
            
        # 7. Annotate Bounces
        for b in bounces:
            # only show bounce for a few frames after it happens
            if 0 <= (frame_count - b["frame"]) < 30:
                cv2.circle(frame, (b["x"], int(b["y"])), 15, (255, 0, 0), 3)
                cv2.putText(frame, "BOUNCE", (b["x"] - 30, int(b["y"]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        # Write frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
            
    cap.release()
    out.release()
    
    # Save structured detection data
    with open(output_json_path, 'w') as f:
        json.dump({
            "players_rackets": frame_detections, 
            "ball_positions": ball_positions, 
            "classified_shots": classified_shots,
            "bounces": bounces
        }, f, indent=4)
        
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    print(f"\nStep 5 Finished successfully!")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"Detections JSON saved to: {output_json_path}")

    # Generate analytical dashboard
    print("Generating Analytics Dashboard...")
    try:
        from visualize import generate_dashboard
        generate_dashboard(output_json_path, output_csv_path, "outputs/dashboard.png")
    except Exception as e:
        print(f"Could not generate dashboard: {e}")

if __name__ == "__main__":
    main()