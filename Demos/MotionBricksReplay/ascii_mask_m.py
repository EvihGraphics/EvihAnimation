import cv2, numpy as np
img = cv2.imread(r'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\mask_m_test.png')
mask = (img[:,:,0] > 0)
s = ''
for y in range(0, 40, 2):
    for x in range(350, 450, 2):
        s += '#' if mask[y, x] else ' '
    s += '\n'
print(s)
