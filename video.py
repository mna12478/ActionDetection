# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import Generate_data
import test_multiple
from PIL import Image
from tflearn.data_utils import pil_to_nparray

# 读取参数
filename = str(sys.argv[1])
videoCapture = cv2.VideoCapture("./test/" + filename)
def main():
    '''
    读取视频, 取得帧数，确定步长
    '''
    length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    phase = int(length / 20)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    success, frame = videoCapture.read()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    step = 0
    i = 0
    # 创建一个概率序列
    feature = np.zeros([20])
    while success:
        if step == 20:
            break
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        pil_im = Image.fromarray(fgmask)
        img_array = pil_to_nparray(pil_im)
        img_array.resize([227, 227])
        if i % phase == 0:
            p = Generate_data.model.predict([img_array])[0]
            feature[step] = round(max(p[2], p[4], p[5]), 5)
            step = step + 1
        i += 1
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        success, frame = videoCapture.read()  # 获取下一帧

    # feature = [ 0.04163  0.04163  0.04163  0.08075  0.28016  0.00099  0.0042   0.00196
    #   0.00689  0.01443  0.09599  0.22221  0.73955  0.35479  0.56534  0.62928
    #   0.38863  0.19718  0.00233  0.00092]
    # '''
    result = test_multiple.predict(np.asarray(feature).reshape(1, 20))
    print(feature)
    print("--------------------")
    print(result)

if __name__ == '__main__':
    main()
