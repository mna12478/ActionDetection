import cv2
import os
import numpy as np

import Generate_data
import Test_multiple


'''
先清空一下临时文件夹的文件
'''
old_list = os.listdir("tmp_pics")
for one in old_list:
    filename = "./tmp_pics/" + one
    os.remove(filename)
'''
读取视频，并截图保存在临时文件夹中
'''

filename = "58_10_1.avi"
videoCapture = cv2.VideoCapture("./video/" + filename)
fgbg = cv2.createBackgroundSubtractorMOG2()
success, frame = videoCapture.read()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
step = 0
i = 0
while success:

    fgmask = fgbg.apply(frame)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 腐蚀图像
    # eroded = cv2.erode(fgmask, kernel)
    # 膨胀图像
    # dilated = cv2.dilate(eroded, kernel)

    # fgmask = dilated / 255
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    if step % 5 == 0:
        filename = "./tmp_pics/" + str(i) + ".png"
        cv2.imwrite(filename, fgmask)
        i += 1

    step = step + 1

    # cv2.imshow("Oto Video", frame)  # 显示

    # cv2.waitKey(100)  # 延迟
    success, frame = videoCapture.read()  # 获取下一帧


'''
读出临时文件夹中的图片
'''
test_list = os.listdir("tmp_pics")
# 注意如果视频帧数短于之前的，现在要手动清空tmp_pics文件夹才行
feature = np.zeros((20))
medium = len(test_list) // 2

for j in range(20):
    frame_i = medium + j - 10
    frame_name = str(frame_i) + ".png"
    # print(frame_name)
    if frame_name in test_list:
        file_name = file_name = "tmp_pics/" + frame_name
        image = Generate_data.load_my_image(file_name)
        # print(Train.model.predict([image])[0])
        feature[j] = round(Generate_data.model.predict([image])[0][1], 5)

'''
feature = [ 0.,       0.45467 , 0.0123 ,  0.0123  , 0.0123 ,  0.50625,  0.6196 ,  0.59761,
  0.67474 , 0.76425 , 0.68969 , 0.56172 , 0.55421,  0.76653  ,0.73918 , 0.53565,
  0.49156 , 0.40411 , 0.18931 , 0.     ]
'''
print(feature)

result = Test_multiple.predict(np.asarray(feature).reshape(1, 20))
print("--------------------")
print(result)