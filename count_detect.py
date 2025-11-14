from ultralytics import YOLO

model = YOLO(r"C:\Users\khom2\Desktop\tesa\main_best.pt",
    task='detect')
results = model.predict(source=r"C:\Users\khom2\Desktop\tesa\P2_DATA_TEST",conf=0.5,save_txt=True, save_conf=True, save_crop=True)