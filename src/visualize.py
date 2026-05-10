import json
import csv
import matplotlib.pyplot as plt
import os
from collections import Counter

def generate_dashboard(json_path, csv_path, output_png):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    classified_shots = data.get("classified_shots", [])
    
    # Calculate statistics
    shot_types = [shot["shot_type"] for shot in classified_shots if "shot_type" in shot]
    shot_counts = Counter(shot_types)
    
    directions = [shot["direction"] for shot in classified_shots if "direction" in shot and shot["direction"] not in ("unknown", "pending")]
    direction_counts = Counter(directions)
    
    # For player distribution we don't have proper player ID tracking yet
    # We will simulate player sides by X coordinate (Left vs Right player)
    player_sides = []
    # Assume 1280x720 normal shape
    for shot in classified_shots:
        bbox = shot.get("player_bbox", [0, 0, 0, 0])
        x_center = (bbox[0] + bbox[2]) / 2
        if x_center < 640:
            player_sides.append("Left Player")
        else:
            player_sides.append("Right Player")
            
    player_counts = Counter(player_sides)
    
    # Read CSV for tracking statistics
    y_coords = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['visible'] == 'True':
                    y_coords.append(float(row['y']))
                    

    # Dashboard Generation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Shot type counts
    types = ['forehand', 'backhand', 'serve', 'smash']
    counts_t = [shot_counts.get(t, 0) for t in types]
    axes[0, 0].bar(types, counts_t, color=['green', 'red', 'gold', 'orange'])
    axes[0, 0].set_title('Shot Type Distribution')
    axes[0, 0].set_ylabel('Count')

    # Subplot 2: Player comparison pie
    labels = list(player_counts.keys())
    sizes = list(player_counts.values())
    if sizes:
        axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        axes[0, 1].set_title('Shots by Court Side (Player Approximation)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Shot Data', ha='center', va='center')
        axes[0, 1].set_axis_off()

    # Subplot 3: Direction counts
    dirs = ['left', 'center', 'right']
    counts_d = [direction_counts.get(d, 0) for d in dirs]
    axes[1, 0].bar(dirs, counts_d, color=['blue', 'gray', 'purple'])
    axes[1, 0].set_title('Shot Direction Distribution')
    axes[1, 0].set_ylabel('Count')
    
    # Subplot 4: Y-Coordinate Plot (Ball trajectory heatmap / height)
    if y_coords:
        axes[1, 1].hist(y_coords, bins=30, orientation='vertical', color='c', edgecolor='black')
        axes[1, 1].set_title('Ball Height Distribution')
        axes[1, 1].set_xlabel('Y Coordinate (Pixels - Lower is higher up)')
        axes[1, 1].set_ylabel('Frequency')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Trajectory Data', ha='center', va='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    print(f"Dashboard saved to: {output_png}")

if __name__ == "__main__":
    generate_dashboard(
        json_path="outputs/inference_detections.json",
        csv_path="outputs/inference_ball_trajectory.csv",
        output_png="outputs/dashboard.png"
    )