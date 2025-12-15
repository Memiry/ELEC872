import cv2
import mediapipe as mp
import csv
import glob
import os
from pathlib import Path


current_script_dir = Path(__file__).parent
MPI_FOLDER_PATH = current_script_dir / 'SITTING'
OUTPUT_MPI_CSV = 'mpi_converted_data.csv'


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

header = []
for i in range(33):
    header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
header.append('label')

with open(OUTPUT_MPI_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    print(f"正在扫描文件夹: {MPI_FOLDER_PATH}")
    images = glob.glob(os.path.join(MPI_FOLDER_PATH, "*.jpg")) + \
             glob.glob(os.path.join(MPI_FOLDER_PATH, "*.png"))



    count = 0
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None: continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            row = []
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])


            row.append(1)

            writer.writerow(row)
            count += 1

            if count % 50 == 0:
                print(f"已转换 {count} 张...")
