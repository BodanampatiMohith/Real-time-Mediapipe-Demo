🖐️ Sign Language Recognition Using Mediapipe & Machine Learning
📌 Project Overview

This project detects Indian Sign Language (ISL) gestures in real-time using a webcam.
It is designed to recognize emergency signs (Pain, Help, Thief, Accident) and general greetings (Hello, Bye, etc.), making communication easier for hearing-impaired people without requiring an interpreter.

🗂 Project Structure
sign-language-recognition/
│── sign_data_both_hands/         # Folder containing CSV datasets for gestures
│── hand_tracking.py              # Basic Mediapipe hand tracking test
│── multi_sign_data_collection.py # Collect multiple gesture datasets
│── realtime_sign_detection.py    # Real-time ISL sign detection via webcam
│── sign_language.py              # Core script for gesture recognition logic
│── sign_language_model.pkl       # Pre-trained ML model (Random Forest)
│── test_mediapipe.py             # Test Mediapipe setup
│── train_sign_model.py           # Train the classifier on collected CSV data

🛠 Tech Stack
Component	Purpose
Python 3.8+	Main programming language
Mediapipe	Real-time hand landmark detection
OpenCV	Webcam access & visualization
Pandas	CSV dataset handling
Scikit-learn	Model training & evaluation (Random Forest Classifier)
Anaconda / Conda	Dependency management
⚙️ Methodology

Data Collection

Run multi_sign_data_collection.py to capture hand landmarks.

Save CSV datasets for each gesture inside sign_data_both_hands/.

Preprocessing & Training

Run train_sign_model.py to train the ML model.

The trained model is saved as sign_language_model.pkl.

Real-Time Prediction

Run realtime_sign_detection.py to open the webcam.

Mediapipe extracts hand landmarks.

The trained model predicts the gesture.

The gesture name is displayed live on the screen.

▶️ How to Run
1. Setup Environment
conda create -n isl_env python=3.8 -y
conda activate isl_env
pip install opencv-python mediapipe pandas scikit-learn

2. Test Mediapipe
python test_mediapipe.py

3. Collect Gesture Data
python multi_sign_data_collection.py

4. Train Model
python train_sign_model.py

5. Run Real-Time Detection
python realtime_sign_detection.py

➕ Adding New Gestures

Collect new data with:

python multi_sign_data_collection.py


Retrain the model:

python train_sign_model.py


Use updated sign_language_model.pkl for detection.

💡 Tips

Always keep sign_language_model.pkl in the project folder.

If webcam isn’t detected, check OpenCV installation (pip install opencv-python).

For two-hand gestures, ensure you capture both hands during data collection.

🚀 Future Scope

Add more ISL gestures for wider vocabulary.

Use deep learning (CNNs/LSTMs) for better accuracy.

Extend to sentence-level recognition.

Deploy as a mobile or web app for accessibility.

📄 License


