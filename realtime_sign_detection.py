import cv2
import mediapipe as mp
import numpy as np
import joblib

MODEL_PATH = "sign_language_model.pkl"
bundle = joblib.load(MODEL_PATH)
model, scaler, encoder = bundle["model"], bundle["scaler"], bundle["encoder"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords += [lm.x, lm.y, lm.z]

            if handedness.classification[0].label.lower() == "left":
                left_hand = coords
            else:
                right_hand = coords

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
            )
        features = np.array(left_hand + right_hand).reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        sign = encoder.inverse_transform([pred])[0]

        cv2.putText(frame, f"Sign: {sign}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("ISL Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
