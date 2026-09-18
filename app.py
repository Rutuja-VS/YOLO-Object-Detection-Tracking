import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort

model = YOLO("yolov8s.pt")
tracker = Sort(max_age=50,min_hits=1,iou_threshold=0.2)

cap = cv2.VideoCapture(0) # WEBCAM
# cap = cv2.VideoCapture("videos/sample.mp4") # VIDEO FILE

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output = cv2.VideoWriter("outputs/output.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30, (640,480))

class_names = model.names
colors = {}
track_classes = {}
prev_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, verbose=False)

    detections = []
    detection_classes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        if confidence > 0.7:
            detections.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                confidence
            ])

            detection_classes.append(class_id)

    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    tracks = tracker.update(detections, detection_classes)

    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        track_id = int(track_id)
        class_id = int(class_id)

        class_name = class_names[class_id]

        color = (
            (track_id * 50) % 255,
            (track_id * 80) % 255,
            (track_id * 120) % 255
        )

        cv2.rectangle(frame,(x1, y1),(x2, y2),color,2)
        label = f"{class_name} | ID: {track_id}"
        cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    current_time = time.time()
    time_diff = current_time - prev_time

    if time_diff > 0:
        fps_display = 1 / time_diff
    else:
        fps_display = 0

    prev_time = current_time

    cv2.rectangle(frame,(10, 10),(150, 60),(0, 0, 0),-1)
    cv2.putText(frame,f"FPS: {int(fps_display)}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)

    cv2.imshow("YOLOv8 Object Detection & Tracking",frame)
    output.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()


        

