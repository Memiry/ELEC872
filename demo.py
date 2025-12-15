import cv2
import mediapipe as mp
import joblib
import pandas as pd
import numpy as np
import collections

MODEL_PATH = 'hybrid_model.pkl'


SMOOTH_WINDOW = 5


THRES_HIGH = 0.7
THRES_LOW = 0.4


ALERT_FRAMES = 30

model = joblib.load(MODEL_PATH)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


prob_queue = collections.deque(maxlen=SMOOTH_WINDOW)

current_status = "Unknown"
bad_counter = 0
color = (200, 200, 200)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        X = pd.DataFrame([row])


        prob_current = model.predict_proba(X)[0][1]


        prob_queue.append(prob_current)


        prob_avg = sum(prob_queue) / len(prob_queue)


        if prob_avg > THRES_HIGH:
            current_status = "Good"
            bad_counter = 0
        elif prob_avg < THRES_LOW:
            current_status = "Bad"
            bad_counter += 1


        if current_status == "Good":
            display_text = f"GOOD ({prob_avg:.2f})"
            color = (0, 255, 0)
        elif current_status == "Bad":
            display_text = f"BAD ({prob_avg:.2f})"
            color = (0, 165, 255)
        else:
            display_text = "Init..."


        if bad_counter > ALERT_FRAMES:
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 10)
            cv2.putText(frame, "!!! SIT UP !!!", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)

    cv2.putText(frame, f"Status: {current_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Stable Demo', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()