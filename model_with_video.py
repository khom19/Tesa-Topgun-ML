import cv2, requests, json, time, os, pytz
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist
from datetime import datetime
from dotenv import load_dotenv
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class SimpleTracker:
    def __init__(self, max_distance=150, max_frames_missing=40):
        self.next_id = 1
        self.tracks = {}  # {tid: {'x': float, 'y': float, 'vx': float, 'vy': float, 'w': float, 'h': float, 'missing': int, 'is_detected': bool}}
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
    
    def update(self, detections, sizes=None):
        assigned_ids = []
        cal_distant = {}
        is_predicted_dict = {}
        
        # Predict positions for all tracks (smooth motion)
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            # Kalman-like prediction with velocity
            track['x'] += track['vx']
            track['y'] += track['vy']
        
        if len(detections) == 0:
            # No detections: increment missing frames for all tracks
            for tid in list(self.tracks.keys()):
                track = self.tracks[tid]
                track['missing'] += 1
                track['is_detected'] = False
                is_predicted_dict[tid] = True
                
                if track['missing'] >= self.max_frames_missing:
                    del self.tracks[tid]
                    self.next_id -= 1
            
            return [], {}, {}
        
        detection_array = np.array([(d[0], d[1]) for d in detections])
        
        if len(self.tracks) == 0:
            # First frame - create new tracks
            for i, det in enumerate(detections):
                tid = self.next_id
                w, h = sizes[i] if sizes and i < len(sizes) else (50, 50)
                
                self.tracks[tid] = {
                    'x': float(det[0]),
                    'y': float(det[1]),
                    'vx': 0.0,
                    'vy': 0.0,
                    'w': float(w),
                    'h': float(h),
                    'missing': 0,
                    'is_detected': True
                }
                assigned_ids.append(tid)
                cal_distant[tid] = 0
                is_predicted_dict[tid] = False
                self.next_id += 1
            
            return assigned_ids, cal_distant, is_predicted_dict
        
        # Hungarian-like matching with predicted positions
        track_ids = list(self.tracks.keys())
        track_positions = np.array([[self.tracks[tid]['x'], self.tracks[tid]['y']] for tid in track_ids])
        
        distances = cdist(detection_array, track_positions)
        used_tracks = set()
        used_dets = set()
        
        # Assign detections to tracks
        for i, det in enumerate(detections):
            best_tid = None
            best_dist = float('inf')
            best_j = None
            
            for j, tid in enumerate(track_ids):
                if j in used_tracks:
                    continue
                if distances[i][j] < best_dist:
                    best_dist = distances[i][j]
                    best_tid = tid
                    best_j = j
            
            if best_tid is not None and best_dist < self.max_distance:
                # Update track with detection
                track = self.tracks[best_tid]
                old_x, old_y = track['x'], track['y']
                
                # Update velocity with exponential moving average
                new_vx = det[0] - old_x
                new_vy = det[1] - old_y
                track['vx'] = 0.7 * track['vx'] + 0.3 * new_vx
                track['vy'] = 0.7 * track['vy'] + 0.3 * new_vy
                
                # Update position
                track['x'] = float(det[0])
                track['y'] = float(det[1])
                track['missing'] = 0
                track['is_detected'] = True
                
                # Update size
                if sizes and i < len(sizes):
                    w, h = sizes[i]
                    track['w'] = 0.8 * track['w'] + 0.2 * float(w)
                    track['h'] = 0.8 * track['h'] + 0.2 * float(h)
                
                distance_moved = int(np.sqrt(new_vx**2 + new_vy**2))
                assigned_ids.append(best_tid)
                cal_distant[best_tid] = distance_moved
                is_predicted_dict[best_tid] = False
                used_tracks.add(best_j)
                used_dets.add(i)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_dets:
                tid = self.next_id
                w, h = sizes[i] if sizes and i < len(sizes) else (50, 50)
                
                self.tracks[tid] = {
                    'x': float(det[0]),
                    'y': float(det[1]),
                    'vx': 0.0,
                    'vy': 0.0,
                    'w': float(w),
                    'h': float(h),
                    'missing': 0,
                    'is_detected': True
                }
                assigned_ids.append(tid)
                cal_distant[tid] = 0
                is_predicted_dict[tid] = False
                self.next_id += 1
        
        # Cleanup old tracks
        self.tracks = {tid: track for tid, track in self.tracks.items() 
                      if track['missing'] < self.max_frames_missing}
        
        return assigned_ids, cal_distant, is_predicted_dict

obj_model = YOLO(r"C:\Users\khom2\Desktop\Tesa-Topgun\main_best.pt")
cap = cv2.VideoCapture(r"C:\Users\khom2\Desktop\Tesa-Topgun\vid\P3_VIDEO.mp4")
img_width, img_height = 1920, 1080

load_dotenv()
token = os.getenv("token")
api_url = os.getenv("url")

colors = [
    (153, 0, 153),   # purple
    (204, 204, 0),   # blue
    (0, 255, 0),     # green
    (0, 0, 255)      # orange
]

track_history = defaultdict(list)
start_y = 30
line_height = 80

tracker = SimpleTracker(max_distance=200, max_frames_missing=240)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# โหลด CSV data
model = joblib.load(r"C:\Users\khom2\Desktop\Tesa-Topgun\model.pkl")

frame_count = 0
send_interval = 30

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = r"C:\Users\khom2\Desktop\Tesa-Topgun\vid\output_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    object_data = []
    
    if not ret:
        break
    
    frame_count += 1
    
    results = obj_model(frame, conf=0.45, verbose=False, iou=0.6)
    result = results[0]
    boxes = result.boxes
    annotated_frame = frame.copy()
    
    # Prepare detections and sizes
    detections = []
    sizes = []
    xyxys_dict = {}  # Store boxes for rendering
    
    if len(boxes) > 0:
        xyxys = boxes.xyxy.cpu().numpy()
        for box in xyxys:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            detections.append((cx, cy))
            sizes.append((w, h))
    
    # Update tracker with detections
    track_ids, distance, is_predicted_dict = tracker.update(detections, sizes)
    
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {len(tracker.tracks)} active tracks - IDs: {sorted(tracker.tracks.keys())}")
    
    # Render ALL active tracks (both detected and predicted)
    for track_id in sorted(tracker.tracks.keys()):
        track = tracker.tracks[track_id]
        x_center = track['x']
        y_center = track['y']
        w = track['w']
        h = track['h']
        is_predicted = is_predicted_dict.get(track_id, True)
        
        # Calculate bounding box
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        # Normalize for prediction
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        bbox = np.array([[x_center_norm, y_center_norm, w_norm, h_norm]])
        pred_lat, pred_lon, pred_alt = model.predict(bbox)[0]
        
        # Color based on track ID
        color = colors[(track_id - 1) % len(colors)]
        
        # Add to history
        track_history[track_id].append((x_center, y_center))
        if len(track_history[track_id]) > 100:
            track_history[track_id].pop(0)
        
        # Draw trajectory
        pts = np.array(track_history[track_id], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)
        
        # Draw bounding box (solid if detected, dashed style if predicted)
        box_thickness = 3 if not is_predicted else 2
        cv2.rectangle(annotated_frame, (int(x1-6), int(y1-4)), (int(x2+8), int(y2+8)), color, box_thickness)
    
        # Draw center point
        cv2.circle(annotated_frame, (int(x_center), int(y_center)), 4, color, -1)
        
        # Draw label on box with text shadow for contrast
        text_x_box = int(x1)
        text_y_box = int(y1) - 15
        if text_y_box < 25:
            text_y_box = int(y2) + 25
        
        label = f"ID: {track_id}"
        
        font_scale = 0.9
        font_thickness = 2
        
        # Draw text shadow (black offset) for better contrast
        cv2.putText(annotated_frame, label, (text_x_box + 1, text_y_box + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        # Draw main text
        cv2.putText(annotated_frame, label, (text_x_box, text_y_box),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        
        # Draw info panel on left - text only with shadow for readability
        text_y = start_y + ((track_id - 1) * line_height)
        info_label = f"ID: {track_id}"
        
        # Draw main ID label
        cv2.putText(annotated_frame, info_label, (12, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Format latitude with shadow
        lat_text = f"Lat: {pred_lat:.5f}"
        cv2.putText(annotated_frame, lat_text, (12, text_y + 24),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Format longitude with shadow
        lon_text = f"Lon: {pred_lon:.5f}"
        cv2.putText(annotated_frame, lon_text, (12, text_y + 44),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Format altitude with shadow
        alt_text = f"Alt: {pred_alt:.1f} m"
        cv2.putText(annotated_frame, alt_text, (12, text_y + 64),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Get speed
        speed = 0
        if track_id in distance:
            speed = int((distance[track_id] * 35) * 5 / 18)
        
        hold_color = list(color)
        object_data.append({
            "obj_id": f"{track_id}",
            "type": "drone",
            "lat": round(float(pred_lat),6),
            "lng": round(float(pred_lon),6),
            "objective": "surveillance",
            "size": "medium",
            "details": {
                "color": f"rgb({hold_color[2]}, {hold_color[1]}, {hold_color[0]})",
                "altitude": round(float(pred_alt),6),
                "speed": speed,
                "status": "predicted" if is_predicted else "detected"
            }
        })
           
    # Send request every 1 sec
    if frame_count % send_interval == 0 and object_data:
        print(f"Frame {frame_count}: {object_data} objects")
        # Uncomment below to send to API
        try:
            timestamp = datetime.now(pytz.timezone("Asia/Bangkok")).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            request_data = {"objects": json.dumps(object_data), "timestamp": timestamp}
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = BytesIO(buffer.tobytes())
            files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
            headers = {"x-camera-token": token}
            response = requests.post(api_url, data=request_data, headers=headers, files=files, timeout=5)
            print(f"API Response: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")

    if not out.isOpened():
        print("Error: Could not create VideoWriter")
        exit()
        
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)   
    cv2.imshow('Frame', annotated_frame)
    out.write(annotated_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to: {output_path}")
print(f"\nTotal frames processed: {frame_count}")
print(f"Total unique tracks: {len(track_history)}")