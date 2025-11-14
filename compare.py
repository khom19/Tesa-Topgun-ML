import os
import pandas as pd

# === CONFIGURATION ===
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
base_folder = r"C:\Users\khom2\Desktop\tesa\compare"
output_csv = os.path.join(base_folder, "combined_output.csv")
# ======================

combined_data = []

# List all .txt files
for file in os.listdir(base_folder):
    if not file.endswith(".txt"):
        continue

    name = os.path.splitext(file)[0]
    yolo_path = os.path.join(base_folder, file)
    csv_path = os.path.join(base_folder, f"{name}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ No CSV found for {name}, skipping.")
        continue

    # Read GPS CSV (should contain Latitude,Longitude,Altitude)
    try:
        geo_df = pd.read_csv(csv_path)
        if len(geo_df) == 0:
            print(f"⚠️ {csv_path} is empty, skipping.")
            continue

        lat = geo_df.iloc[0]["Latitude"]
        lon = geo_df.iloc[0]["Longitude"]
        alt = geo_df.iloc[0]["Altitude"]
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        continue

    # Read YOLO annotations
    with open(yolo_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, w, h = map(float, parts)

            # Convert normalized YOLO -> pixel values
            x_center_px = x_center * IMAGE_WIDTH
            y_center_px = y_center * IMAGE_HEIGHT
            w_px = w * IMAGE_WIDTH
            h_px = h * IMAGE_HEIGHT

            combined_data.append({
                "filename": name,
                "class": int(cls),
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
                "x_center_px": x_center_px,
                "y_center_px": y_center_px,
                "width_px": w_px,
                "height_px": h_px,
                "Latitude": lat,
                "Longitude": lon,
                "Altitude": alt,
            })

# Save combined output
if combined_data:
    df = pd.DataFrame(combined_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Combined data saved to: {output_csv}")
    print(df.head())
else:
    print("⚠️ No matching data found.")
