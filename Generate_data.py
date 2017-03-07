# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_image_dataset_from_dir
from PIL import Image
from tflearn.data_utils import pil_to_nparray
import os
import csv

# 输入图像
def load_my_image(fp):
    img = Image.open(fp)
    # img = img.convert('L')
    img = pil_to_nparray(img)
    img.resize([227, 227])
    img /= 255.
    return img


#
#
# Labels = ['handshake',  'hug', 'kick', 'quiet', 'hit', 'push']
#
# # Reading Data
# X, Y = build_image_dataset_from_dir("binary_frame", dataset_file="./file_zip/data.pkl", resize=[227, 227],
#                                     categorical_Y=True, convert_gray=False)
with tf.device('/gpu:0'):
    # Building 'AlexNet'
    network = tf.reshape(input_data(shape=[None, 227, 227]), [-1, 227, 227, 1])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2) # kernal size = [3 x 3]
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 1024, activation='tanh') # origin alexnet is 4096 at this part
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 6, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

model = tflearn.DNN(network, max_checkpoints=1, tensorboard_verbose=3, tensorboard_dir="Board/")
# # model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True,
# #             show_metric=True, batch_size=45, run_id='alexnet_oxflowers17')
#
# # Save model
# model.save('model_alexnet/my_model.tflearn')
#
# # Load model
# model.load('model_alexnet/my_model.tflearn')
#
#
#
# '''
# 取数据
# '''
# # 打开刚才的序号、文件名、标签的csv文件，因为文件中间隔有1行，所以做一些处理
# with open('./csv_file/video_labels.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     your_list = list(reader)
# all_list = []
# for one in your_list:
#     if one[0] != '':
#         all_list.append(one)
#
# # 在一行后边加20个0，预留出来作为一下子填特征值用
# for i in range(100):
#     for j in range(20):
#         all_list[i].append(0)
#     # print(all_list[i])
# # print(all_list)
#
#
# test_list = os.listdir("frame2")
#
# # 用来计算每个视频在这里有多少帧
# initial_value = 0
# list_length = 100
# # sample_list = [ initial_value ]
# sample_list = [initial_value]*list_length
# for frame_name in test_list:
#     sample_list[int(frame_name.split('_')[0])] += 1
# # print(sample_list)
#
#
# # 用于填充特征值
# for i in range(100):
#     medium = sample_list[i] // 2
#
#     for j in range(20):
#         frame_i = (medium + j-9 ) * 5
#         frame_name = str(i) + "_" + str(frame_i) + ".png"
#         # print(frame_name)
#         if frame_name in test_list:
#             file_name = "frame2/" + frame_name
#             image = load_my_image(file_name)
#             # print(model.predict([image]))
#             p = model.predict([image])[0]
#             all_list[i][j+3] = round(max(p[2], p[4], p[5]), 5)
#
# with open("./csv_file/seq.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(all_list)
