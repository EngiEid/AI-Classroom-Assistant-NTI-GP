# 🎓 Smart Classroom AI

> **An AI-powered Computer Vision system for student identification, attendance tracking, and classroom activity monitoring.**

Smart Classroom AI is a computer-vision-based classroom monitoring system designed to analyze classroom images and videos, identify registered students using face recognition, detect classroom activities, track students across video frames, and store attendance and event information in a structured database.

The system combines **YOLO-based person detection and tracking**, **InsightFace face recognition**, a custom **YOLO event-detection model**, **ByteTrack**, and a relational SQLite database to build a complete classroom analysis pipeline.

---

## ✨ Key Features

- 👤 **Student Management**
  - Add students to the system.
  - Store student information and associated images.
  - Generate and store face embeddings for recognition.

- 🧠 **Face Recognition**
  - Uses **InsightFace / Buffalo_L**.
  - Extracts face embeddings from student images.
  - Supports multiple images per student.
  - Computes an averaged, L2-normalized embedding for multiple samples.
  - Uses cosine similarity for identity matching.

- 🎯 **Person Detection & Tracking**
  - Uses **YOLOv8n** for full-body person detection.
  - Uses **ByteTrack** to maintain persistent track IDs throughout videos.
  - Associates detected faces with tracked people.

- 🏫 **Classroom Activity Detection**
  - Uses a custom YOLO model (`best.pt`) to detect classroom events/activities.
  - Associates detected activities with the corresponding tracked student.
  - Supports temporal event tracking with start/end timestamps.

- 📝 **Automatic Attendance**
  - When a registered student is successfully recognized, attendance is recorded automatically.
  - Attendance records include the student, class, timestamp, status, and recognition confidence.

- 📊 **Event Logging**
  - Detected classroom activities are stored in the database.
  - Each event can include:
    - Student ID
    - Class ID
    - Event type
    - Start time
    - End time
    - Detection confidence

- 🖼️ **Image Processing**
  - Analyze individual classroom images.
  - Detect students and recognize registered faces.
  - Detect and record classroom activities.

- 🎥 **Video Processing**
  - Process classroom videos frame by frame.
  - Track students across frames.
  - Perform periodic face recognition.
  - Detect and temporally aggregate activities.
  - Generate an annotated processed video.

---

## 🧠 Computer Vision Pipeline

The core video-processing pipeline follows this architecture:

```text
                    Input Video / Image
                           │
                           ▼
              ┌─────────────────────────┐
              │ YOLOv8n Person Detection│
              └────────────┬────────────┘
                           │
                           ▼
                  ByteTrack Tracking
                           │
                           ▼
                 Persistent Track IDs
                           │
                           ▼
                  Face Recognition
                    (InsightFace)
                           │
                           ▼
               Track ID → Student ID
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
   Custom YOLO Event Model       Student Identity
        (best.pt)                       │
             │                          │
             ▼                          ▼
   Object → Person Association     Attendance
             │
             ▼
      Temporal Event Engine
             │
             ▼
        Student Events
             │
             └──────────────┐
                            ▼
                    SQLite Database
                            │
                            ▼
                    Processed Output
```

The implemented `VideoProcessor` follows this sequence by detecting/tracking people, periodically recognizing faces, associating identities with track IDs, detecting events with `best.pt`, associating events with people, and recording attendance/events in the database. 

---

## 🔍 Face Recognition

The face-recognition component is implemented in `core/face_engine.py`.

It uses:

- **InsightFace**
- **Buffalo_L** model
- **CPUExecutionProvider** by default
- **Cosine similarity**
- Configurable similarity threshold

The default threshold is:

```python
sim_threshold = 0.5
```

### Multiple Images per Student

The system is designed to work with multiple images for a student.

For each image:

```text
Student Images
      │
      ▼
Face Detection
      │
      ▼
Face Embedding
      │
      ▼
Valid Embeddings
      │
      ▼
Average Embedding
      │
      ▼
L2 Normalization
      │
      ▼
Stored / Used for Recognition
```

`FaceEngine` provides `process_multiple_images()` and `get_average_embedding()` for this purpose. The average embedding is normalized before being used for recognition.

During recognition, the current face embedding is normalized and compared against the prepared known-face matrix using a vectorized dot product. The highest similarity is selected and accepted only when it reaches the configured threshold.

---

## 🎯 Student Identity Association

Face recognition alone identifies a face, while video tracking provides a persistent person identity across frames.

The system connects the two:

```text
Face
 │
 │ Face Bounding Box
 ▼
Match with Person Bounding Box
 │
 ▼
Track ID
 │
 ▼
Student ID
 │
 ▼
Student Name
```

The `VideoProcessor` maintains a `track_identities` state so that once a track has been associated with a student, that identity can be reused across subsequent frames.

This reduces the need to run face recognition on every single frame.

---

## ⏱️ Periodic Face Recognition

For video processing, face recognition is not necessarily executed on every frame.

The processor uses:

```python
face_recognition_interval = 5
```

By default, recognition is performed periodically, while the tracked identity is maintained between recognition steps.

This helps reduce computational overhead, which is particularly useful when running the system on a CPU.

---

## 🏫 Classroom Event Detection

Classroom activities are detected using the custom YOLO model:

```text
best.pt
```

The event-detection model runs independently from the person detector.

The system then determines which tracked student the detected event belongs to using spatial relationships between bounding boxes.

### Event Association

The association process considers:

- Object center position
- Person bounding box
- Intersection over Union (IoU)
- Normalized distance
- Horizontal overlap constraints
- Small vertical expansion around the person

This helps distinguish an activity belonging to one student from nearby students.

---

## ⏳ Temporal Event Engine

For video input, an activity is not immediately written as a database event every time it appears in a frame.

Instead, the system maintains active events:

```text
Event Detected
      │
      ▼
Create / Update Active Event
      │
      ▼
Keep Updating While Detected
      │
      ▼
Event Disappears
      │
      ▼
Buffer Period
      │
      ▼
Event Ends
      │
      ▼
Record Start Time + End Time
```

The default event buffer is:

```python
event_buffer_frames = 15
```

The system stores the highest confidence observed for the active event.

---

# 🗄️ Database Architecture

The project uses a relational SQLite database.

The main entities are:

```text
Subjects
   │
   ▼
Classes
   │
   ├──────────────► Attendance
   │
   └──────────────► Student Events

Students
   │
   ├──────────────► Student Images
   │
   ├──────────────► Attendance
   │
   └──────────────► Student Events
```

## Database Tables

### `Students`

Stores registered students.

| Column | Description |
|---|---|
| `Stu_ID` | Student primary key |
| `Name` | Student name |

### `Student_Images`

Stores images and face embeddings associated with students.

| Column | Description |
|---|---|
| `Image_ID` | Image primary key |
| `Stu_ID` | Related student |
| `Image_Path` | Path to the student's image |
| `Face_Embedding` | Stored face embedding |

### `Subjects`

Stores available subjects.

| Column | Description |
|---|---|
| `Sub_ID` | Subject primary key |
| `Name` | Subject name |

### `Classes`

Represents classes associated with subjects.

| Column | Description |
|---|---|
| `Class_ID` | Class primary key |
| `Sub_ID` | Related subject |

### `Attendance`

Stores student attendance records.

| Column | Description |
|---|---|
| `Attend_ID` | Attendance primary key |
| `Stu_ID` | Related student |
| `Class_ID` | Related class |
| `Status` | Attendance status |
| `Timestamp` | Time of attendance |
| `Confidence` | Recognition confidence |

### `Student_Events`

Stores detected classroom activities.

| Column | Description |
|---|---|
| `Event_ID` | Event primary key |
| `Stu_ID` | Related student |
| `Class_ID` | Related class |
| `Event_Type` | Detected activity |
| `Start Time` | Event start |
| `End Time` | Event end |
| `Confidence` | Detection confidence |

---

# 📁 Project Structure

```text
smart_classroom_system/
│
├── core/
│   ├── face_engine.py
│   └── video_processor.py
│
├── gui/
│   ├── add_student.py
│   ├── app.py
│   └── dashboard.py
│
├── data/
│   ├── student_images/
│   ├── db.sqlite3
│   └── processed_video...
│
├── CV_INV/
│
├── best.pt
├── yolov8n.pt
│
├── database.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

| Component | Responsibility |
|---|---|
| `core/face_engine.py` | Face detection, embedding extraction, and recognition |
| `core/video_processor.py` | Main computer-vision processing pipeline |
| `database.py` | Database operations |
| `gui/add_student.py` | Student registration workflow |
| `gui/app.py` | Main application interface |
| `gui/dashboard.py` | Dashboard / results interface |
| `best.pt` | Custom classroom event-detection model |
| `yolov8n.pt` | Person detection model |
| `data/student_images/` | Student image data |
| `data/db.sqlite3` | SQLite database |
| `main.py` | Application entry point |

---

# ⚙️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image/video processing |
| Ultralytics YOLO | Object detection and tracking |
| YOLOv8n | Full-body person detection |
| ByteTrack | Multi-object tracking |
| InsightFace | Face detection & recognition |
| NumPy | Numerical operations and embeddings |
| SQLite | Local relational database |
| FFmpeg / imageio-ffmpeg | H.264 video conversion |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd smart_classroom_system
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the main application with:

```bash
streamlit run main.py
```

The GUI provides the interface for interacting with the system, registering students, processing classroom media, and viewing the generated results.

---

# 👤 Student Registration Workflow

A typical student-registration workflow is:

```text
Add Student
    │
    ▼
Enter Student Information
    │
    ▼
Upload / Capture Student Images
    │
    ▼
Face Detection
    │
    ▼
Embedding Extraction
    │
    ▼
Embedding Storage
    │
    ▼
Student Ready for Recognition
```

The face engine can process multiple images for one student and calculate a representative embedding.

---

# 🎥 Video Analysis Workflow

```text
Upload Classroom Video
          │
          ▼
YOLOv8n Person Detection
          │
          ▼
ByteTrack Tracking
          │
          ▼
Periodic Face Recognition
          │
          ▼
Student Identification
          │
          ├──────────────► Attendance
          │
          ▼
Custom Event Detection
          │
          ▼
Event → Student Association
          │
          ▼
Temporal Event Tracking
          │
          ▼
Student Events Database
          │
          ▼
Annotated Processed Video
```

---

# 🖼️ Image Analysis Workflow

For a single image, the system performs:

```text
Input Image
    │
    ├──► Person Detection
    │
    ├──► Face Recognition
    │
    ├──► Student Association
    │
    ├──► Attendance Recording
    │
    ├──► Event Detection
    │
    └──► Event Recording
             │
             ▼
      Annotated Image
```

---

# 💾 Data Flow

```text
                 ┌─────────────────┐
                 │ Student Images  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Face Embeddings │
                 └────────┬────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ Students  │
                    └─────┬─────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       ┌────────────┐          ┌───────────────┐
       │ Attendance │          │ Student Events│
       └────────────┘          └───────────────┘
             │                         │
             └────────────┬────────────┘
                          ▼
                    Classroom Data
```

---

# 🖥️ CPU Execution

The current face-recognition implementation explicitly defaults to:

```python
providers = ['CPUExecutionProvider']
```

Therefore, the system can run without a dedicated NVIDIA GPU.

However, computer-vision processing—especially video processing—can be computationally expensive. The project already reduces some overhead by:

- Performing face recognition periodically instead of every frame.
- Preparing the known embedding matrix once instead of repeatedly rebuilding it.
- Using vectorized embedding comparison.
- Using lightweight `YOLOv8n` for person detection.

For larger classroom videos or real-time processing, a GPU can significantly improve processing speed.

---

# 🔧 Configuration

Important processing parameters are configurable in `VideoProcessor`:

```python
VideoProcessor(
    person_model_path="yolov8n.pt",
    event_model_path="best.pt",
    sim_threshold=0.5,
    face_recognition_interval=5,
    track_conf=0.25,
    event_buffer_frames=15
)
```

### Parameters

| Parameter | Purpose | Default |
|---|---|---:|
| `person_model_path` | Person detection model | `yolov8n.pt` |
| `event_model_path` | Classroom event model | `best.pt` |
| `sim_threshold` | Face similarity threshold | `0.5` |
| `face_recognition_interval` | Frames between recognition operations | `5` |
| `track_conf` | Person detection confidence threshold | `0.25` |
| `event_buffer_frames` | Frames used before closing an event | `15` |

---

# 📈 Output

The system produces two major types of results:

### Visual Output

Annotated media containing:

- Full-body person bounding boxes
- Student names
- Track IDs
- Face bounding boxes
- Detected activities
- Activity labels

### Database Output

Structured records for:

- Attendance
- Student activities/events
- Recognition confidence
- Event confidence
- Start/end timestamps

---

# 🔐 Important Notes

- The system recognizes students based on the face embeddings stored in the database.
- A face is accepted only when its similarity reaches the configured threshold.
- Unknown faces are not assigned to registered student IDs.
- Attendance is recorded when a known student is successfully recognized.
- In video processing, identities are associated with persistent ByteTrack IDs.
- Events are linked to students through spatial association between detected activity objects and tracked people.
- The custom event classes depend on the classes used to train `best.pt`.

---

# 🧪 Current Architecture

The project separates responsibilities into three main layers:

```text
┌───────────────────────────────────────────┐
│                  GUI Layer                │
│                                           │
│  Student Registration │ Processing │ UI   │
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│             Computer Vision Layer         │
│                                           │
│ Person Detection │ Tracking │ Recognition │
│ Event Detection  │ Association │ Temporal│
└───────────────────┬───────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────┐
│               Data Layer                  │
│                                           │
│ SQLite │ Students │ Images │ Attendance   │
│        │ Classes  │ Events                │
└───────────────────────────────────────────┘
```

---

# 🎯 Project Goal

The goal of **Smart Classroom AI** is to transform classroom video data into structured, actionable information by combining:

**Detection → Tracking → Recognition → Activity Understanding → Database Logging**

This enables the system to move beyond simple face recognition and provide a more complete representation of classroom activity.

---

# 👨‍💻 Project Status

The project currently contains the core components required for:

- Student registration
- Face embedding generation
- Face recognition
- Person detection
- Multi-object tracking
- Classroom event detection
- Attendance recording
- Event recording
- Image processing
- Video processing
- Annotated output generation
- Dashboard-oriented data access

---

## 📌 Future Improvements

Potential future improvements include:

- Real-time webcam classroom monitoring
- GPU acceleration
- More advanced student re-identification
- Improved event association
- Additional classroom activity classes
- Performance optimization for large classrooms
- More detailed analytics and reporting
- Advanced attendance statistics
- Model evaluation metrics and benchmarking

---

## 📄 License

Add the project's license here if one is defined.

---

## 👥 Authors

**Smart Classroom AI Project**

Built as a Computer Vision / AI project for intelligent classroom monitoring and analysis.
