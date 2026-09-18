# Real-Time Object Detection and Tracking using YOLOv8 and SORT

A real-time computer vision application that detects objects using **YOLOv8** and tracks individual objects across video frames using the **SORT (Simple Online and Realtime Tracking)** algorithm.

The system displays the detected object's **class name, tracking ID, bounding box, and real-time FPS**.

---

## 🚀 Features

* 🎯 Real-time object detection using **YOLOv8s**
* 🏷️ Identifies detected objects using their class names
* 🔢 Assigns a unique tracking ID to each detected object
* 🔄 Tracks objects across consecutive frames using **SORT**
* 🧠 Class-aware tracking to reduce incorrect associations between different object classes
* 📦 Bounding-box visualization
* 📊 Confidence-based detection filtering
* ⚡ Real-time FPS display
* 🎥 Supports webcam and video-file input
* 💾 Saves processed video output

---

## 🏗️ System Pipeline

```text
             Input Video / Webcam
                     │
                     ▼
              ┌─────────────┐
              │   YOLOv8s   │
              └──────┬──────┘
                     │
          Bounding Box + Confidence
                  + Class ID
                     │
                     ▼
            Confidence Filtering
                 (> 0.7)
                     │
                     ▼
            ┌─────────────────┐
            │  Class-Aware     │
            │      SORT        │
            └────────┬────────┘
                     │
            Bounding Box + ID
                + Class ID
                     │
                     ▼
             Visualization
                     │
                     ▼
       Class Name | Track ID | FPS
```

---

## 🧠 How It Works

### 1. Object Detection

The application uses the **YOLOv8s** pretrained model to detect objects in every video frame.

For each detected object, YOLO provides:

* Bounding box coordinates
* Confidence score
* Class ID

```python
results = model(frame, verbose=False)
```

---

### 2. Confidence Filtering

Only detections with a confidence greater than `0.7` are passed to the tracker.

```python
if confidence > 0.7:
```

This helps remove lower-confidence detections before tracking.

---

### 3. Class-Aware Tracking

The detected bounding boxes and their class IDs are passed to the modified SORT tracker.

```python
tracks = tracker.update(detections, detection_classes)
```

The tracker uses **bounding-box IoU** along with **object class information** when associating detections with existing tracks.

Therefore, a detection belonging to one class is prevented from being matched with a tracker belonging to another class.

For example:

```text
Person detection  → Person track
Bottle detection  → Bottle track

Person detection  ✕ Chair track
Bottle detection  ✕ Person track
```

---

### 4. Tracking IDs

SORT assigns a unique ID to each tracked object.

For example:

```text
person | ID: 1
bottle | ID: 2
laptop | ID: 3
```

The same object can retain its tracking ID across multiple frames while it remains successfully tracked.

---

### 5. Visualization

Each tracked object is displayed using:

* Bounding box
* Object class name
* Tracking ID

Example:

```text
┌─────────────────────┐
│                     │
│       OBJECT        │
│                     │
└─────────────────────┘
laptop | ID: 3
```

The current FPS is also displayed on the video frame.

---

## 🛠️ Technologies Used

| Technology | Purpose                             |
| ---------- | ----------------------------------- |
| Python     | Core programming language           |
| YOLOv8     | Object detection and classification |
| SORT       | Multi-object tracking               |
| OpenCV     | Video processing and visualization  |
| NumPy      | Numerical operations                |
| FilterPy   | Kalman Filter implementation        |
| SciPy      | Hungarian assignment algorithm      |
| Matplotlib | SORT visualization/support          |

---

## 📂 Project Structure

```text
YOLO-Object-Detection-Tracking/
│
├── app.py
├── sort.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── videos/
│   └── sample.mp4
│
└── outputs/
    └── output.mp4
```

> The `videos/` and `outputs/` directories are optional depending on how you use the project.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/YOLO-Object-Detection-Tracking.git
```

### 2. Navigate to the project

```bash
cd YOLO-Object-Detection-Tracking
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run:

```bash
python app.py
```

The application will start the webcam and perform real-time object detection and tracking.

Press:

```text
ESC
```

to exit the application.

---

## 🎥 Using a Video File

To process a video instead of the webcam, change:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture("videos/sample.mp4")
```

Then run:

```bash
python app.py
```

The processed video will be saved to:

```text
outputs/output.mp4
```

---

## 📊 Output

The application produces an annotated video containing:

```text
Object Class
     +
Tracking ID
     +
Bounding Box
     +
Real-Time FPS
```

Example:

```text
person | ID: 1
laptop | ID: 2
bottle | ID: 3
```

---

## 🔧 SORT Configuration

The tracker is initialized using:

```python
tracker = Sort(
    max_age=50,
    min_hits=1,
    iou_threshold=0.2
)
```

### Parameters

| Parameter       | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `max_age`       | Maximum number of frames a track can remain without a matching detection |
| `min_hits`      | Minimum number of successful detections required for a track             |
| `iou_threshold` | Minimum IoU required for associating a detection with an existing track  |

---

## 🧩 Class-Aware SORT

The standard SORT algorithm primarily associates detections using bounding-box overlap.

This implementation additionally stores the detected **class ID** for every tracker.

During association:

```text
Detection Class == Tracker Class
              +
         IoU Threshold
              ↓
       Valid Association
```

Different classes are prevented from being associated with one another.

This helps maintain class consistency during tracking.

---

## ⚠️ Limitations

* The pretrained YOLOv8s model can only recognize the object classes it was trained on.
* Detection accuracy depends on lighting, camera quality, object size, occlusion, and viewpoint.
* SORT does not improve the semantic classification performed by YOLO.
* If YOLO incorrectly classifies an object, SORT cannot correct the object name.
* Tracking IDs can change when objects disappear for too long or tracking is lost.
* Very crowded scenes and heavy occlusion can cause tracking errors.

---

## 🔮 Future Improvements

Potential future improvements include:

* Display detection confidence along with class and ID
* Improve tracking under heavy occlusion
* Compare SORT with ByteTrack and Deep SORT
* Fine-tune YOLO on a custom dataset for domain-specific objects
* Improve detection performance using GPU acceleration
* Add object counting
* Add line-crossing detection
* Add zone-based monitoring
* Add tracking analytics and statistics
* Build a web-based interface for real-time monitoring

---

## 📌 Project Objective

The primary objective of this project is to combine **object detection and multi-object tracking** into a real-time computer vision system.

The system aims to:

```text
Detect Objects
      ↓
Identify Object Classes
      ↓
Track Individual Objects
      ↓
Assign Persistent IDs
      ↓
Visualize Results in Real Time
```

---

## 📜 References

* **YOLOv8** — Ultralytics
* **SORT: Simple Online and Realtime Tracking** — Alex Bewley et al.

---

⭐ If you found this project useful, consider giving the repository a star!
