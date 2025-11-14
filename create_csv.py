import os
import csv
import math
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
import joblib

folder = "TEST_PO"
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
output_csv = "output_with_angle.csv"

model = YOLO(r"C:\Users\khom2\Desktop\tesa\sec_best.pt")

# OPTIMIZED MODEL 2 COEFFICIENTS
# Error: 6.3201

CAMERA_LAT = 14.305029
CAMERA_LON = 101.173010
CAMERA_ALT = 37.2

# Model 2: Area + Position
LAT_COEF = -0.00175430
LAT_INTERCEPT = 14.30069233
LON_COEF = -0.01264205
LON_INTERCEPT = 101.17242594

# Load LSQ-fitted altitude coefficients from separate module
try:
    from optimized_lsq_coefficients import (
        ALT_LSQ_area,
        ALT_LSQ_x,
        ALT_LSQ_y,
        ALT_LSQ_intercept,
    )
except Exception:
    # Fallback values (hard-coded) if the import fails
    ALT_LSQ_area = -564.5536210608279
    ALT_LSQ_x = 7.679077257619582
    ALT_LSQ_y = -25.619625435709786
    ALT_LSQ_intercept = 60.45526366578215

# Try to load a saved ML model (if present). This model may expect normalized features
# or pixel features depending on how it was saved. We'll keep it as a fallback.
ALT_MODEL = None
for path in (os.path.join(os.path.dirname(__file__), 'best_altitude_model.pkl'), 'best_altitude_model.pkl'):
    try:
        if os.path.exists(path):
            loaded = joblib.load(path)
            ALT_MODEL = loaded['model'] if isinstance(loaded, dict) and 'model' in loaded else loaded
            print(f"Loaded altitude model from {path}")
            break
    except Exception:
        ALT_MODEL = None


# Load reference data
try:
    ref_data = pd.read_csv(r'C:\Users\khom2\Desktop\tesa\compare\combined_output.csv')
    ref_data.columns = ref_data.columns.str.strip()
except Exception as e:
    print(f"Error loading reference data: {e}")
    ref_data = None

def calculate_drone_position(x_center_norm, y_center_norm, object_area_norm):
    """Calculate drone position from normalized image coordinates"""
    latitude = LAT_COEF * y_center_norm + LAT_INTERCEPT
    longitude = LON_COEF * x_center_norm + LON_INTERCEPT
    # Use fitted least-squares model: intercept + c_area*area + c_x*x + c_y*y
    altitude = (ALT_LSQ_intercept
                + ALT_LSQ_area * object_area_norm
                + ALT_LSQ_x * x_center_norm
                + ALT_LSQ_y * y_center_norm)
    
    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude
    }

def calculate_direction_angle(camera_lat, camera_lon, drone_lat, drone_lon):
    """
    Calculate direction angle (bearing) from camera to drone in degrees
    0° = North, 90° = East, 180° = South, 270° = West
    """
    lat1 = math.radians(camera_lat)
    lat2 = math.radians(drone_lat)
    lon1 = math.radians(camera_lon)
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

def get_real_coordinates(fname):
    """Get real coordinates from reference data by filename"""
    if ref_data is None:
        return None, None, None
    
    matching = ref_data[ref_data['filename'] == fname]
    if len(matching) > 0:
        row = matching.iloc[0]
        return row['Latitude'], row['Longitude'], row['Altitude']
    return None, None, None

# Lists to store errors
angle_errors = []
height_errors = []
results_list = []

print("="*80)
print("DIRECTION ANGLE & HEIGHT ERROR CALCULATION")
print("="*80)

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "image_file",
        "center_x", "center_y", "width", "height",
        "x_center_norm", "y_center_norm", "area_norm",
        "pred_latitude", "pred_longitude", "pred_altitude",
        "real_latitude", "real_longitude", "real_altitude",
        "angle_pred_deg", "angle_true_deg", "angle_error_deg",
        "height_pred_m", "height_true_m", "height_error_m"
    ])

    total_files = len(files)
    
    for idx, fname in enumerate(files, 1):
        image_path = os.path.join(folder, fname)
        results = model(image_path, conf=0.5)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print(f"[{idx}/{total_files}] No detections in {fname}")
            continue

        # Get image dimensions
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            print(f"[{idx}/{total_files}] Error reading image dimensions for {fname}")
            continue

        # Get real coordinates
        real_lat, real_lon, real_alt = get_real_coordinates(fname[:len(fname)-4])
        
        if real_lat is None:
            print(f"[{idx}/{total_files}] No reference data for {fname}")
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Normalize coordinates (0-1)
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            area_norm = w_norm * h_norm
            
            # Calculate predicted drone position
            drone_pos = calculate_drone_position(x_center_norm, y_center_norm, area_norm)
            
            # Calculate direction angles
            angle_pred = calculate_direction_angle(CAMERA_LAT, CAMERA_LON, 
                                                   drone_pos['latitude'], drone_pos['longitude'])
            angle_true = calculate_direction_angle(CAMERA_LAT, CAMERA_LON, 
                                                   real_lat, real_lon)
            
            # Calculate errors
            angle_error = abs(angle_pred - angle_true)
            # Handle wraparound for angles (e.g., 359° and 1° should be 2° apart, not 358°)
            if angle_error > 180:
                angle_error = 360 - angle_error
            
            height_error = abs(drone_pos['altitude'] - real_alt)
            
            # Store errors
            angle_errors.append(angle_error)
            height_errors.append(height_error)
            
            results_list.append({
                'fname': fname,
                'angle_pred': angle_pred,
                'angle_true': angle_true,
                'angle_error': angle_error,
                'height_pred': drone_pos['altitude'],
                'height_true': real_alt,
                'height_error': height_error
            })
            
            writer.writerow([
                fname,
                f"{x_center:.2f}", f"{y_center:.2f}", f"{w:.2f}", f"{h:.2f}",
                f"{x_center_norm:.6f}", f"{y_center_norm:.6f}", f"{area_norm:.8f}",
                f"{drone_pos['latitude']:.8f}", f"{drone_pos['longitude']:.8f}", f"{drone_pos['altitude']:.2f}",
                f"{real_lat:.8f}", f"{real_lon:.8f}", f"{real_alt:.2f}",
                f"{angle_pred:.2f}", f"{angle_true:.2f}", f"{angle_error:.2f}",
                f"{drone_pos['altitude']:.2f}", f"{real_alt:.2f}", f"{height_error:.2f}"
            ])
            
            print(f"[{idx}/{total_files}] {fname}: Angle Pred={angle_pred:.2f}° (True={angle_true:.2f}°) | Height Pred={drone_pos['altitude']:.2f}m (True={real_alt:.2f}m) | Error: Angle={angle_error:.2f}°, Height={height_error:.2f}m")

print(f"\n✅ Results saved to {output_csv}")

# Calculate statistics
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
    
    normalized_total_error = 0.7 * normalized_angle_error + 0.3 * normalized_height_error
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