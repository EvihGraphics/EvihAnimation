import cv2, numpy as np, matplotlib.pyplot as plt
mask_m=cv2.imread('mask_m_test.png',0)>0
mask_e_img = cv2.imread(r'D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay\frames\frame_0051.png')
mask_e = ((mask_e_img[:,:,0]<50) & (mask_e_img[:,:,1]<50) & (mask_e_img[:,:,2]<50))

prof_m = np.sum(mask_m, axis=1)
prof_e = np.sum(mask_e, axis=1)

plt.plot(prof_m, label='MuJoCo')
plt.plot(prof_e, label='Evih')
plt.axvline(240, color='r', linestyle='--', label='Center')
plt.legend()
plt.savefig('y_profile.png')
