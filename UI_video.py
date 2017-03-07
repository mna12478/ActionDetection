
# -*- coding: utf-8 -*-
import cv2
import sys
import time
import numpy as np
import Generate_data
import test_multiple
from PIL import Image
from tflearn.data_utils import pil_to_nparray
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore

# 创建一个概率序列
feature = np.zeros([20])
result = ''
filename = 'output.avi'

class Capture():
    def __init__(self):
        self.capturing = False
        # 读取视频
        self.cap = cv2.VideoCapture("./test/" + filename)
        self.cap2 = cv2.VideoCapture("./test/" + filename)
    def startCapture(self):
        print("pressed Start")
        self.capturing = True
        self.cap = cv2.VideoCapture("./test/" + filename)
        self.cap2 = cv2.VideoCapture("./test/" + filename)
        while(self.capturing):
            # 程序开始时间
            start_clock = time.clock()
            # 读取参数
            videoCapture = self.cap
            videoCapture2 = self.cap2
            # # 预测一个动作用，- 1 防止越界
            # length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            # phase = int((length - 1) / 20)

            # 连续预测，5帧启动一次序列
            phase = 5

            fgbg = cv2.createBackgroundSubtractorMOG2()
            success, frame = videoCapture.read()
            success2, frame2 = videoCapture2.read()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            step = 0
            i = 0
            while success:
                if frame is None:
                    break
                if frame2 is None:
                    break
                if step == 20:
                    result = test_multiple.predict(np.asarray(feature).reshape(1, 20))
                    end_clock = time.clock()
                    window.result_label.setText(str(feature)
                                                + '\n--------------------'
                                                + result
                                                + ' '
                                                + str(end_clock - start_clock))
                    feature[0:15] = feature[5:20]
                    feature[15:20] = 0
                    step -= 5
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

                cv2.imshow('frame2', frame2)

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                success, frame = videoCapture.read()  # 获取下一帧
                success2, frame2 = videoCapture2.read()  # 获取下一帧
            cv2.destroyAllWindows()
            self.cap.release()
            self.cap2.release()

    def endCapture(self):
        print("pressed End")
        cv2.destroyAllWindows()
        self.cap.release()
        self.cap2.release()
        self.capturing = False

    def quitCapture(self):
        print("pressed Quit")
        QtCore.QCoreApplication.quit()

class Window(QWidget):
    def __init__(self):

        QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.capture = Capture()
        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.capture.startCapture)

        self.end_button = QPushButton('End', self)
        self.end_button.clicked.connect(self.capture.endCapture)

        self.quit_button = QPushButton('Quit', self)
        self.quit_button.clicked.connect(self.capture.quitCapture)

        self.result_label = QLabel(self)
        self.result_label.setText('helloworld.')

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)
        vbox.addWidget(self.result_label)

        self.setLayout(vbox)
        self.setGeometry(100, 100 , 200, 200)
        self.show()




if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())