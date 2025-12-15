import cv2
import mediapipe as mp
import os
import shutil
import glob


INPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\Selectedimages'  
OUTPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\Selectedimages2'  


SITTING_THRESHOLD = 0.15


MIN_BODY_SIZE = 0.2

CENTER_TOLERANCE = 0.3


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.6 
)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"正在扫描: {INPUT_FOLDER}")
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

print(f"找到 {len(image_files)} 张图，开始筛选...")

saved_count = 0
rejected_count = 0

for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None: continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

      
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

   
        width = max_x - min_x
        height = max_y - min_y
        area = width * height  


        if area < MIN_BODY_SIZE:
            rejected_count += 1
            continue

 
        center_x = (min_x + max_x) / 2

        if abs(center_x - 0.5) > CENTER_TOLERANCE:
            rejected_count += 1
            continue


        left_hip_y = landmarks[23].y
        left_knee_y = landmarks[25].y
        right_hip_y = landmarks[24].y
        right_knee_y = landmarks[26].y

        diff_leg = (abs(left_hip_y - left_knee_y) + abs(right_hip_y - right_knee_y)) / 2


        shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        torso_lean = abs(shoulder_x - hip_x)


        if diff_leg < SITTING_THRESHOLD and torso_lean < 0.3:

            file_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(OUTPUT_FOLDER, file_name))
            saved_count += 1
        else:
            rejected_count += 1
    else:
        rejected_count += 1

    if (saved_count + rejected_count) % 50 == 0:
        print(f"已处理 {saved_count + rejected_count} ... (保留: {saved_count})")

print(f"\n筛选结束！")
print(f"保留了: {saved_count} 张 (存放在 {OUTPUT_FOLDER})")
print(f"删除了: {rejected_count} 张 (太小/偏离/姿势不对)")