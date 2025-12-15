import cv2
import mediapipe as mp
import csv
import os

OUTPUT_CSV = 'my_data.csv'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

header = []
for i in range(33):
    header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
header.append('label')

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        csv.writer(f).writerow(header)


count_g = 0
count_b = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF


        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])

        if key == ord('b'):
            row.append(0)
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
            count_b += 1
            cv2.putText(frame, f"RECORDING BAD: {count_b}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif key == ord('g'):
            row.append(1)
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
            count_g += 1
            cv2.putText(frame, f"RECORDING GOOD: {count_g}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif key == 27:
            break

    cv2.imshow('Data Recorder', frame)

cap.release()
cv2.destroyAllWindows()