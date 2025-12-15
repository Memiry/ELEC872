import cv2
import mediapipe as mp
import os
import shutil 
import glob

INPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\images'

OUTPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\Selectedimages'


SITTING_THRESHOLD = 0.15

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"已创建新文件夹: {OUTPUT_FOLDER}")

print(f"正在扫描: {INPUT_FOLDER}")
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(INPUT_FOLDER, "*.png"))


valid_count = 0
skipped_count = 0

for img_path in image_files:

    image = cv2.imread(img_path)
    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    is_sitting = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_hip_y = landmarks[23].y
        left_knee_y = landmarks[25].y
        right_hip_y = landmarks[24].y
        right_knee_y = landmarks[26].y

        diff_left = abs(left_hip_y - left_knee_y)
        diff_right = abs(right_hip_y - right_knee_y)
        avg_diff = (diff_left + diff_right) / 2

        if avg_diff < SITTING_THRESHOLD:
            is_sitting = True


    if is_sitting:
        file_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_FOLDER, file_name)

        shutil.copy(img_path, save_path)

        valid_count += 1

    else:
        skipped_count += 1

    total = valid_count + skipped_count
    if total % 20 == 0:
        print(f"已处理 {total} 张... (保留: {valid_count} / 剔除: {skipped_count})")

print(f"原始图片总数: {len(image_files)}")
print(f"成功保留坐姿: {valid_count} 张 -> 存放在 {OUTPUT_FOLDER}")
print(f"剔除站/走/无效: {skipped_count} 张")
