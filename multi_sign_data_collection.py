import cv2
import mediapipe as mp
import csv
import os
DATA_DIR = "sign_data_both_hands"
SIGNS = ["hello", "bye", "pain", "thief", "help", "doctor", "accident"]
NUM_SAMPLES_PER_SIGN = 100 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

print("\n➡️ Signs to collect:", SIGNS)
print("Instructions:")
print("- Press 's' to start recording the current sign.")
print("- Press 'n' to move to the NEXT sign after recording is done.")
print("- Press 'q' anytime to quit.\n")

current_sign_index = 0
recording = False
count = 0

while current_sign_index < len(SIGNS):
    sign_name = SIGNS[current_sign_index]
    csv_path = os.path.join(DATA_DIR, f"{sign_name}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = []
            for hand in ["left", "right"]:
                for i in range(21):
                    header += [f"{hand}_x{i}", f"{hand}_y{i}", f"{hand}_z{i}"]
            writer.writerow(header)

    print(f"\n➡️ Ready to record sign: {sign_name}")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

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
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords += [lm.x, lm.y, lm.z]

                    if handedness.classification[0].label.lower() == "left":
                        left_hand = coords
                    else:
                        right_hand = coords
                if recording and count < NUM_SAMPLES_PER_SIGN:
                    writer.writerow(left_hand + right_hand)
                    count += 1
                    print(f"Saved sample {count}/{NUM_SAMPLES_PER_SIGN} for {sign_name}")

                    if count >= NUM_SAMPLES_PER_SIGN:
                        print(f"✅ Finished recording: {sign_name}")
                        recording = False

            cv2.putText(frame,
                        f"Sign: {sign_name} | Samples: {count}/{NUM_SAMPLES_PER_SIGN}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imshow("Data Collection - Both Hands", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                recording = True
                count = 0
                print(f"▶️ Recording started for {sign_name}...")
            elif key == ord("n"):
                recording = False
                count = 0
                current_sign_index += 1
                break
            elif key == ord("q"):
                current_sign_index = len(SIGNS) 
                break

cap.release()
cv2.destroyAllWindows()
print("\n🎯 Data collection completed for all signs!")
