import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ultralytics import YOLO
from PIL import Image
import csv, os
import math

def calculate_direction_angle(drone_lat, drone_lon):
    lat1 = math.radians(14.305029)
    lat2 = math.radians(drone_lat)
    lon1 = math.radians(101.173010)
    lon2 = math.radians(drone_lon)
    
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    bearing = math.atan2(x, y)
    bearing_deg = math.degrees(bearing)
    
    # Normalize to 0-360
    if bearing_deg < 0:
        bearing_deg += 360
        
    return bearing_deg

# --- Load training data ---
data = pd.read_csv(r"C:\Users\khom2\Desktop\tesa\compare\combined_output.csv")

# Use "Altitude" column as per the preferred method
X = data[["x_center", "y_center", "width", "height"]]
y = data[["Latitude", "Longitude", "Altitude"]] # CHANGE: Using "Altitude"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained | MAE: {mae:.6f}")

# --- Load YOLO model ---
obj_model = YOLO(r"C:\Users\khom2\Desktop\tesa\main_best.pt")

# --- Load real coordinates ---
# Use "filename" and "Latitude", "Longitude", "Altitude" as per the preferred method
ref_dict = {row["filename"]: (row["Latitude"], row["Longitude"], row["Altitude"]) for _, row in data.iterrows()} # CHANGE: Using "filename", "Latitude", "Longitude", "Altitude"

def get_real_coordinates(name):
    return ref_dict.get(name, (None, None, None))

# --- Predict loop ---
output_csv = "test_pred.csv"
folder = r"C:\Users\khom2\Desktop\tesa\TRAIN_PO"
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
print(files)
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "image_file", "center_x", "center_y", "width", "height",
        "x_center_norm", "y_center_norm", "area_norm",
        "pred_latitude", "pred_longitude", "pred_altitude",
        "real_latitude", "real_longitude", "real_altitude"
    ])
    
    angle_errors = []
    height_errors = []
    results_list = []
    
    for idx, fname in enumerate(files, 1):
        image_path = os.path.join(folder, fname)
        # Assuming obj_model is the YOLO model instance, use the call method
        results = obj_model(image_path, conf=0.45) 
        boxes = results[0].boxes
        
        if boxes is None or len(boxes) == 0:
            print(f"[{idx}/{len(files)}] No detections in {fname}")
            continue
        
        img = Image.open(image_path)
        img_width, img_height = img.size
        real_lat, real_lon, real_alt = get_real_coordinates(fname[:-4])
        
        if real_lat is None:
            print(f"[{idx}/{len(files)}] No reference data for {fname}")
            continue

        for box in boxes:
            # box.xyxy[0] returns a tensor; tolist() converts it to a list of floats
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            area_norm = w_norm * h_norm

            bbox = np.array([[x_center_norm, y_center_norm, w_norm, h_norm]])
            pred_lat, pred_lon, pred_alt = model.predict(bbox)[0]

            writer.writerow([
                fname,
                f"{x_center:.2f}", f"{y_center:.2f}", f"{w:.2f}", f"{h:.2f}",
                f"{x_center_norm:.6f}", f"{y_center_norm:.6f}", f"{area_norm:.8f}",
                f"{pred_lat:.8f}", f"{pred_lon:.8f}", f"{pred_alt:.2f}",
                f"{real_lat:.8f}", f"{real_lon:.8f}", f"{real_alt:.2f}"
            ])
            
        angle_pred = calculate_direction_angle(pred_lat, pred_lon)
        angle_true = calculate_direction_angle(real_lat, real_lon)
        angle_error = abs(angle_pred - angle_true)
        # Handle wraparound for angles (e.g., 359° and 1° should be 2° apart, not 358°)
        if angle_error > 180:
            angle_error = 360 - angle_error
        
        height_error = abs(pred_alt - real_alt)
        
        # Store errors
        angle_errors.append(angle_error)
        height_errors.append(height_error)

print(f"\n✅ Results saved to {output_csv}")

if angle_errors and height_errors:
    print("\n" + "="*80)
    print("ERROR STATISTICS")
    print("="*80)
    
    mean_angle_error = np.mean(angle_errors)
    std_angle_error = np.std(angle_errors)
    min_angle_error = np.min(angle_errors)
    max_angle_error = np.max(angle_errors)
    
    mean_height_error = np.mean(height_errors)
    std_height_error = np.std(height_errors)
    min_height_error = np.min(height_errors)
    max_height_error = np.max(height_errors)
    
    print(f"\n📐 DIRECTION ANGLE ERROR:")
    print(f"   Mean:    {mean_angle_error:.2f}°")
    print(f"   Std:     {std_angle_error:.2f}°")
    print(f"   Min:     {min_angle_error:.2f}°")
    print(f"   Max:     {max_angle_error:.2f}°")
    
    print(f"\n📏 HEIGHT ERROR:")
    print(f"   Mean:    {mean_height_error:.2f}m")
    print(f"   Std:     {std_height_error:.2f}m")
    print(f"   Min:     {min_height_error:.2f}m")
    print(f"   Max:     {max_height_error:.2f}m")
    
    # Calculate total error
    total_error = 0.7 * mean_angle_error + 0.3 * mean_height_error
    
    print(f"\n📊 COMBINED ERROR:")
    print(f"   Total Error = 0.7 × {mean_angle_error:.2f}° + 0.3 × {mean_height_error:.2f}m")
    print(f"   Total Error = {total_error:.2f}")
    
    # Convert to 9-point scale
    # Normalize errors: assume max acceptable error
    # Angle: max 30° = worst score
    # Height: max 20m = worst score
    
    max_angle_acceptable = 30.0  # degrees
    max_height_acceptable = 20.0  # meters
    
    # Score formula: 9 - (error / max) * 9
    # If error = 0, score = 9 (perfect)
    # If error >= max, score = 0 (worst)
    
    normalized_angle_error = min(mean_angle_error / max_angle_acceptable, 1.0)
    normalized_height_error = min(mean_height_error / max_height_acceptable, 1.0)
    
    normalized_total_error = (0.7 * normalized_angle_error) + (0.3 * normalized_height_error)
    score_9_point = 9.0 * (1.0 - normalized_total_error)
    
    print(f"\n⭐ PERFORMANCE SCORE (9-POINT SCALE):")
    print(f"   Max Acceptable Angle Error:   {max_angle_acceptable}°")
    print(f"   Max Acceptable Height Error:  {max_height_acceptable}m")
    print(f"   Normalized Angle Error:       {normalized_angle_error:.4f}")
    print(f"   Normalized Height Error:      {normalized_height_error:.4f}")
    print(f"   Normalized Total Error:       {normalized_total_error:.4f}")
    print(f"   ⭐ FINAL SCORE: {score_9_point:.2f} / 9.0")
    
    # Add interpretation
    if score_9_point >= 8.0:
        rating = "🌟 Excellent (ยอดเยี่ยม)"
    elif score_9_point >= 7.0:
        rating = "⭐ Very Good (ดีมาก)"
    elif score_9_point >= 6.0:
        rating = "✅ Good (ดี)"
    elif score_9_point >= 5.0:
        rating = "👍 Acceptable (ยอมรับได้)"
    elif score_9_point >= 3.0:
        rating = "⚠️  Fair (พอใจ)"
    else:
        rating = "❌ Poor (ต่ำ)"
    
    print(f"   Rating: {rating}")
    
    print("="*80 + "\n")
    
    # Save summary to file
    with open("calibration_summary.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("DRONE DETECTION & LOCALIZATION PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DIRECTION ANGLE ERROR:\n")
        f.write(f"  Mean:  {mean_angle_error:.2f}°\n")
        f.write(f"  Std:   {std_angle_error:.2f}°\n")
        f.write(f"  Range: {min_angle_error:.2f}° - {max_angle_error:.2f}°\n\n")
        
        f.write("HEIGHT ERROR:\n")
        f.write(f"  Mean:  {mean_height_error:.2f}m\n")
        f.write(f"  Std:   {std_height_error:.2f}m\n")
        f.write(f"  Range: {min_height_error:.2f}m - {max_height_error:.2f}m\n\n")
        
        f.write("COMBINED ERROR:\n")
        f.write(f"  Total Error: {total_error:.2f}\n\n")
        
        f.write("PERFORMANCE SCORE (9-POINT SCALE):\n")
        f.write(f"  Score: {score_9_point:.2f} / 9.0\n")
        f.write(f"  Rating: {rating}\n")
    
    print("✅ Summary saved to 'calibration_summary.txt'")
else:
    print("\n⚠️  No data available for error calculation")