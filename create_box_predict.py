import os, csv
from ultralytics import YOLO

folder = r"C:\Users\khom2\Desktop\tesa\TEST_PO"
files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
destination = r"C:\Users\khom2\Desktop\tesa\TEST_PO\output_folder"
output_csv = "output.csv"
model = YOLO(r"C:\Users\khom2\Desktop\tesa\best.pt")
base_path = r"C:\Users\khom2\Desktop\tesa\TEST_PO"
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "center_x", "center_y", "width", "height"])

    for fname in files:
        image_path = os.path.join(folder, fname)
        results = model(image_path,conf = 0.1) 

        img_w, img_h = 1920, 1080

        for box in results[0].boxes:
            cls = int(box.cls[0])  # class ID
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to normalized YOLO format
            x_center = ((x1 + x2) / 2)
            y_center = ((y1 + y2) / 2)
            w = (x2 - x1)
            h = (y2 - y1)

            writer.writerow([fname, x_center, y_center, w, h])

        print(f"Predicting on: {image_path}")
        save_dir = os.path.join(base_path, "drone_predictions")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{fname}.jpg")

        results[0].save(filename=save_path)
        print(f"Prediction saved to {save_path}")