import cv2
import json
import os
from detection import PlayerRacketDetector

def main():
    # Setup input/output paths
    video_path = "videos/input_sample_video.mp4"
    output_video_path = "data/output/step2_annotated_video.mp4"
    output_json_path = "data/output/step2_detections.json"
    
    if not os.path.exists(video_path):
        print(f"Error: Could not find input video at {video_path}")
        return

    print("Initializing YOLOv8 Detector...")
    detector = PlayerRacketDetector(model_path="models/yolov8n.pt")
    
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
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing Video ({total_frames} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect
        dets = detector.detect(frame)
        frame_detections[frame_count] = dets
        
        # 2. Annotate bounding boxes
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"
            color = (0, 255, 0) if det['class'] == 'player' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # 3. Write frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
            
    cap.release()
    out.release()
    
    # Save structured detection data
    with open(output_json_path, 'w') as f:
        json.dump(frame_detections, f, indent=4)
        
    print(f"\nStep 2 Finished successfully!")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"Detections JSON saved to: {output_json_path}")

if __name__ == "__main__":
    main()