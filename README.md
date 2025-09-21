# Real-time-Mediapipe-Demo
To build Python programs that use Mediapipe to detect and track human body features in real time, such as:  Face landmarks (Face Mesh)  Hand landmarks (Hand Tracking)  Eventually full-body pose or gestures
# Mediapipe Real-Time Demos

## Project Overview
This project demonstrates real-time human body feature detection using **Mediapipe**. It includes demos for:
- **Face Mesh**: Detects facial landmarks.
- **Hand Tracking**: Detects hand landmarks.
- Future extensions: Full-body pose detection, gesture recognition.

---

## Environment Setup

### 1. Install Anaconda
Create a virtual environment for the project:
```bash
conda create -n isl python=3.8
conda activate isl
2. Install Required Packages
bash
Copy code
pip install mediapipe opencv-contrib-python numpy matplotlib
3. Verify Installation
bash
Copy code
python -c "import mediapipe as mp; print(mp.__version__)"
Folder Structure
bash
Copy code
mediapipe_project/
│
├─ face_mesh_test.py      # Face landmarks detection demo
├─ hand_tracking_test.py  # Hand landmarks detection demo
├─ other_demo.py          # Optional future demos
Usage
1. Running Face Mesh Demo
bash
Copy code
python face_mesh_test.py
Detects face landmarks in real-time via webcam.

Press q to exit.

2. Running Hand Tracking Demo
bash
Copy code
python hand_tracking_test.py
Detects hand landmarks and draws skeleton.

Press q to exit.

How It Works
Capture video frames using OpenCV.

Process frames with Mediapipe solutions (mp.solutions.face_mesh or mp.solutions.hands).

Draw landmarks and/or connections using Mediapipe drawing utilities.

Display results in a real-time OpenCV window.

Next Steps
Improve Face Mesh demo with connections and screenshots.

Implement additional Hand Tracking features.

Experiment with Pose or Holistic solutions.

Build a custom application such as gesture recognition or AR filters.
