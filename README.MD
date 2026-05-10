# Padel Game Analytics

An automated pipeline aimed at detecting players, rackets, and balls in padel games, tracking trajectories, identifying bounce/hit events, and classifying shot types (Forehand, Backhand, Serve, Smash).

## 🚀 Demo & Notebooks
To ensure this repository is primarily classified as a Python project, the Heavy Jupyter Notebooks (used for model fine-tuning) and the high-resolution demo videos have been moved externally. 

🔗 **[View Demo Video & Development Notebooks (External Drive Link)](#)**
*(Update the link above with the actual drive folder containing `inference_annotated_video.mp4` and `yolo-finetune-padel.ipynb`)*

## 📊 Output & Results
The pipeline successfully generates analytics files detailing the game. Below is the output structure and visual representation.

### Analytical Dashboard
![Analytical Dashboard](data/output/dashboard.png)
*The dashboard summarizes shot distributions (Forehand vs Backhand vs Serve/Smash) and maps bounce geometries across the court space.*

### Generated Output Files
All executed results are placed in `data/output/`:
- **`inference_annotated_video.mp4`**: Fully annotated inference video showing player bounding boxes (green), racket bounds (red), tracked ball path (red/orange interpolated), hitting events, and directional trajectories.
- **`inference_ball_trajectory.csv`**: Frame-by-frame ball coordinate tracing. Contains fields `[frame, x, y, visible, interpolated]` for precise trajectory analysis.
- **`inference_detections.json`**: Structured metadata covering frame detections, bounce occurrences `{"frame", "x", "y"}`, and detailed shot classifications.
- **`dashboard.png`**: The rendered matplotlib visualizations shown above.

## Experimental Journey & Architecture

During development, multiple neural network architectures and models were evaluated to find the most accurate analytical extraction for padel matches:

**1. Baseline YOLOv8n**
- **Process:** Base object detection without modification.
- **Result:** Struggled heavily with tracking consistency for players. It completely failed to recognize padel rackets and struggled substantially with accurate ball tracking.

**2. Fine-tuning YOLOv11m**
- **Process:** Fine-tuned `yolov11m.pt` on the open-source `padeltracker100` dataset for 30 epochs following ultralytics methodologies. 
- **Result:** Player tracking and ID persistence became highly consistent. However, shot classification and continuous ball tracking still failed. We identified a need to integrate the `racketvision` dataset (from tennis) to fine-tune rackets, as a pristine padel-exclusive 5-point dataset wasn't readily available. Due to time constraints, this fine-tuned architecture could not be finalized.

**3. Specialized Open-Source Tracking Models**
- **Process:** Attempted combinations of `players_keypoint`, `TrackNet`, `InpaintNet`, and `court_keypoint` models.
- **Result:** While promising, these models presented significant pipeline integration and architectural alignment issues, making them untenable for our unified tracking timeframe.

**4. YOLO26m (Selected Architecture)**
- **Process:** Adopted `yolo26m.pt` alongside Mediapipe `pose_landmarker_lite.task` for the final prototype. 
- **Result:** Provided highly solid outputs for player (green), racket (red), and ball (red) tracking. 

## Known Limitations & Observations
- **Ball Flickering:** Ball tracking suffers from a flickering effect across frames, likely from heat-mapping/HSV detection issues parsing highly blurred and small moving objects.
- **Shot Classification Quirks:** Shot classification captures most strokes decently; however, some "Serves" are occasionally misclassified as "Forehands". Hit detection performs reasonably well given the complexity.
- **Background Noise:** In specific full-match inference videos, the existence of up to 4 simultaneous active courts (and occasionally multiple balls bouncing in background frames) added immense levels of noise, diminishing overall recall compared to the isolated, clean input sample video.

---
**Branch Structure Refactoring:** 
- The experimental repository branch `exp2` has been elevated to the new `main` production branch, transitioning the earliest legacy prototype logic to branch `exp0`.
- All unused modeling weights were pruned to optimize repository space, retaining only `yolo26m.pt` and `pose_landmarker_lite.task`.