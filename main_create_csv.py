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
print(f"Model trained | MAE: {mae:.6f}")

# --- Load YOLO model ---
obj_model = YOLO(r"C:\Users\khom2\Desktop\tesa\main_best.pt")

# --- Load real coordinates ---
# Use "filename" and "Latitude", "Longitude", "Altitude" as per the preferred method
ref_dict = {row["filename"]: (row["Latitude"], row["Longitude"], row["Altitude"]) for _, row in data.iterrows()} # CHANGE: Using "filename", "Latitude", "Longitude", "Altitude"

def get_real_coordinates(name):
    return ref_dict.get(name, (None, None, None))

# --- Predict loop ---
output_csv = "re_pred.csv"
folder = r"C:\Users\khom2\Desktop\tesa\P2_DATA_TEST"
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
print(files)
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "ImageName", "Latitude", "Longitude", "Altitude"
    ])
    
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
                fname, f"{pred_lat:.8f}", f"{pred_lon:.8f}", f"{pred_alt:.2f}",
            ])

print(f"\nResults saved to {output_csv}")