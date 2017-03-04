import numpy as np
import cv2
import os
# 0 握手 1 拥抱  2 踢  4 打 5 推
labels = ['handshake', 'hug', 'kick', '##' ,'hit', 'push']
v_list = os.listdir('./video')
print(v_list)
j = 0
for line in v_list:
    fname = './video/' + line
    print(fname)
    cap = cv2.VideoCapture(fname)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    i = 1
    while(1):
        i += 1
        # 保留二人站立影像
        ret, frame = cap.read()
        if frame is None:
            break
        # 去掉全黑
        # 原始图像
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        if i % 5 != 0:
            continue
        # # 镜像翻转
        mirror_frame = cv2.flip(frame, 1)
        mirror_fgmask = fgbg.apply(mirror_frame)
        mirror_fgmask = cv2.morphologyEx(mirror_fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', fgmask)
        # k = cv2.waitKey(30) & 0xff
        label = line.split('_')[2][0]
        imgname = line.split('.')[0]
        cv2.imwrite('./frame2/' + str(j) + '_' + str(i) + '.png', fgmask)
    print(fname + ' PROCESS DONE!')
    cap.release()
    cv2.destroyAllWindows()
    j += 1