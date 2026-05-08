import cv2
import json
import os
import csv
from detection import PlayerRacketDetector
from tracking import BallTracker

def main():
    # Setup input/output paths
    video_path = "videos/input_sample_video.mp4"
    output_video_path = "data/output/step3_annotated_video.mp4"
    output_json_path = "data/output/step3_detections.json"
    output_csv_path = "data/output/ball_trajectory.csv"
    
    if not os.path.exists(video_path):
        print(f"Error: Could not find input video at {video_path}")
        return

    print("Initializing Detectors and Trackers...")
    detector = PlayerRacketDetector(model_path="models/yolov8n.pt")
    
    # We use mode='fallback' so it works natively via CV2 contour/motion masking
    # without needing heavy TrackNetV4 weights that are hard to obtain automatically
    ball_tracker = BallTracker(mode='fallback')
    
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
        
        # 3. Annotate bounding boxes for players/rackets
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"
            color = (0, 255, 0) if det['class'] == 'player' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # 4. Annotate Ball
        if ball_pos["visible"]:
            cv2.circle(frame, (ball_pos["x"], ball_pos["y"]), 6, (0, 0, 255), -1)
            
        # 5. Write frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
            
    cap.release()
    out.release()
    
    # Save structured detection data
    with open(output_json_path, 'w') as f:
        json.dump({"players_rackets": frame_detections, "ball_positions": ball_positions}, f, indent=4)
        
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    print(f"\nStep 3 Finished successfully!")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"Detections JSON saved to: {output_json_path}")
    print(f"Ball Trajectory CSV saved to: {output_csv_path}")

if __name__ == "__main__":
    main()