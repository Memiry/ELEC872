import cv2
import mediapipe as mp
import os
import shutil
import glob

# ================== 配置区 ==================
INPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\Selectedimages'  # 原始图片路径
OUTPUT_FOLDER = r'D:\872proj\mpii_human_pose_v1\Selectedimages2'  # 新文件夹

# 1. 坐姿阈值 (0.0 - 1.0)
# 数值越小，要求腿越平，筛选越严格
SITTING_THRESHOLD = 0.15

# 2. 尺寸阈值 (0.0 - 1.0)
# 建议 0.2
MIN_BODY_SIZE = 0.2

# 3. 居中阈值
# 如果人的中心点偏离画面中心太远，扔掉
CENTER_TOLERANCE = 0.3
# ==============================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.6  # 提高置信度
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

        # 1. 计算人尺寸
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 宽和高
        width = max_x - min_x
        height = max_y - min_y
        area = width * height  # 面积 (0.0 - 1.0)

        # 人太小跳过
        if area < MIN_BODY_SIZE:
            rejected_count += 1
            continue

        #  2. 计算是否居中
        center_x = (min_x + max_x) / 2
        # 0.5 是正中心。如果偏离超过 0.3 (即 <0.2 或 >0.8)
        if abs(center_x - 0.5) > CENTER_TOLERANCE:
            rejected_count += 1
            continue

        # 3. 坐姿判断
        # 大腿平不平
        left_hip_y = landmarks[23].y
        left_knee_y = landmarks[25].y
        right_hip_y = landmarks[24].y
        right_knee_y = landmarks[26].y

        diff_leg = (abs(left_hip_y - left_knee_y) + abs(right_hip_y - right_knee_y)) / 2

        # 躯干直不直
        # 比较肩膀和屁股的 X 坐标差
        shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        torso_lean = abs(shoulder_x - hip_x)

        # 综合判断腿 AND 躯干
        if diff_leg < SITTING_THRESHOLD and torso_lean < 0.3:
            # 通过保存
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