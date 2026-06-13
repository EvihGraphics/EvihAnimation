import cv2
import numpy as np

def draw_bbox(img_path, threshold_type='evih'):
    img = cv2.imread(img_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold_type == 'evih':
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # remove ground shadow below the robot manually to get accurate height
        # just find the top and bottom
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sky_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
        robot_mask = cv2.bitwise_not(sky_mask)
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(thresh, thresh, mask=robot_mask)
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        print(f"BBox {threshold_type}: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img

img1 = draw_bbox('frame_0050_evih_new.png', 'evih')
cap = cv2.VideoCapture(r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\motionbricks_evih_replay.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, img2 = cap.read()
cv2.imwrite('frame_0050_mujoco_check.png', img2)
img2 = draw_bbox('frame_0050_mujoco_check.png', 'mujoco')

combined = np.hstack([img1, img2])
cv2.imwrite('bboxes_side_by_side.png', combined)
print("Done")
