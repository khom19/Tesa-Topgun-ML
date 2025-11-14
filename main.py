import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Paths
base_path = r"C:\Users\khom2\Desktop\TESA"
dataset_dir = os.path.join(base_path, "dataset")
output_dir = os.path.join(base_path, "drone_dataset_split")

# Verify dataset folder
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

# Collect images with labels
image_paths = []
for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(".jpg"):
            label_file = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(root, label_file)
            if os.path.exists(label_path):
                image_paths.append(os.path.join(root, f))
            else:
                print(f"Skipping {f}: label not found")

if len(image_paths) == 0:
    raise FileNotFoundError("No images with labels found in the dataset folder.")

print(f"✅ Found {len(image_paths)} images with labels across all subfolders.")

# Split dataset
train_imgs, temp_imgs = train_test_split(image_paths, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
print(f"📊 Split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

# Create output folders
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Copy images and labels
def move_files(image_list, split):
    for img_path in image_list:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(os.path.dirname(img_path), label_filename)

        shutil.copy(img_path, os.path.join(output_dir, "images", split))
        shutil.copy(label_path, os.path.join(output_dir, "labels", split))

    print(f"✅ Copied {len(image_list)} images and labels to {split}")

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')
move_files(test_imgs, 'test')

# Create YAML for YOLOv8
yaml_path = os.path.join(output_dir, "drone_data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""
train: {os.path.join(output_dir, 'images', 'train')}
val:   {os.path.join(output_dir, 'images', 'val')}
test:  {os.path.join(output_dir, 'images', 'test')}

nc: 1
names: ['Drone']
""")

print(f"✅ Created YAML file: {yaml_path}")

# Train YOLOv8 model
model = YOLO("yolov8n.pt")

results = model.train(
    data=yaml_path,
    epochs=20,
    imgsz=640,
    batch=8,
    project=os.path.join(base_path, "drone_training"),
    name="yolov8_drone",
    patience=5
)

print("🎯 Training complete.")

# Predict on a test image
test_images_dir = os.path.join(output_dir, "images", "test")
test_image_list = [
    os.path.join(test_images_dir, f)
    for f in os.listdir(test_images_dir)
    if f.lower().endswith(".jpg")
]

if test_image_list:
    test_image = test_image_list[0]
    print(f"🧾 Predicting on: {test_image}")
    results = model.predict(source=test_image, conf=0.5, save=True)

    save_dir = os.path.join(base_path, "drone_predictions")
    os.makedirs(save_dir, exist_ok=True)

    # YOLO already saves the predicted image automatically in 'runs/predict'
    print(f"✅ Prediction done. Check runs/predict or {save_dir}")
else:
    print("⚠️ No test images found for prediction.")