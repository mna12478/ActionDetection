# VCheck
***
## 项目描述
```
一个基于神经网络方法检测视频内容是否异常的 toy project
```
***
## 运行方式
```bash
mkdir Board binary_frame csv_file file_zip frame2 model_alexnet model_nn tmp_pics video
python BackSub.py
python Generate_data.py
python Train_multiple.py
Video.py  xx.avi
```
## 运行环境
```
硬件：i7-6700HQ + GTX1060 + 16G DDR4
系统与组件：Win10-64-bit  + Cuda-8.0 + Anaconda3-4.1.1 (Python3.5 .ver)
软件包：Tensorflow-gpu-1.0 + Tflearn-0.3.0 + OpenCV-3.2.0 + Numpy-1.12.0
```
***
## 所使用数据集
```
ICPR 2010 Contest on Semantic Description of Human Activities (SDHA 2010)
```
***
相关链接：[SDHA 2010](http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)
***

## 所使用算法
```
单帧预训练：AlextNet (tflearn, 步长0.001）
概率序列训练：双隐层网络 (tensorflow)
```
***
## 文件夹内容说明
### / borad
```
用于存放Tensorboard文件
```
### / binary_frame
```
用于存放单帧图训练集， 存放格式为:
---- / binary_frame
    --- /0
        -- 1.png
        -- 2.png
            ..
    --- /1
        -- 3.png
        -- 4.png
            ..
    --- /2
        -- 5.png
        -- 6.png
            ..
    ...
    ...
    ...
本项目有六个类别，对应 0~5 ：['handshake',  'hug', 'kick', 'quiet', 'hit', 'push']
```
### / csv_file
```
文件夹下有两个文件：
1， seq.csv, 第一次预训练后，对每个视频隔帧取得的概率序列
2， video_labels.csv, 整段视频原有的标签(正常0, 异常1)，用作第二次训练的标签
```
### / file_zip
```
预训练时Tflearn使用build_image_dataset_from_dir()产生的dataset文件(data.pkl)
```

### / frame2
```
所有视频每隔5帧取一帧，帧上注明该帧所属的视频序号和帧数
文件存放格式:
---- /frame2
    --- 0_5.png
    --- 0_10.png
        ...
    --- 5_15.png
    --- 5_20.png
        ...
```
### / model_alexnet
```
用于存放第一次训练结束后的tflearn model
```
### / moddel_nn
```
用于存放第二次训练结束后的 tensorflow model
```
### / tmp_pics
```
用于存放预测整段视频时隔帧的截图
```
### / video
```
所有视频文件，存放格式为：
---- /video
    --- 0_1_4.avi
    --- 0_11_4.avi
    --- 1_1_2.avi
        ...
```
***
## 脚本
#### BackSub.py
```
1, 读取/video下的视频单帧
2, 执行镜像和背景减除操作
3, 存入对应标签的文件夹 /binary_frame
```
#### Generate_data.py
```
1, 对/binary_frame下的单帧图进行预训练
2, 并将结果模型存入/model_alexnet
3, 并对/frame2下所有视频对应的单帧打上概率标签
4, 将每个视频对应的标签概率序列写入seq.csv
5, 将seq.csv存入/csv_file
```
#### Train_multiple.py
```
1，/video下的视频对应的标签写于/csv_file/video_labels.csv
2，每个视频的概率序列已由Generate_data生成于seq.csv
3，将seq.csv作为训练集，video_labels.csv作为标签，使用双隐层进行训练
4，训练结果模型存入/model_nn
```

#### Test_multiple.py
```
二次训练的预测模块
```

#### Video.py
```
读取视频，返回正常/异常
```

## 可视化
```
cd Vcheck
tensorboard --logdir=Board/
```
***
第一次训练的TensorBoard如下:

Loss/Validation

![](http://p1.bqimg.com/567571/b2bb5be0fbb3d502.jpg)

Accuracy

![](http://p1.bpimg.com/567571/f45bd2cfbd18f8e3.jpg)

Accuracy_validation

![](http://p1.bpimg.com/567571/6818b0757dc42525.jpg)
***

## 运行结果
```
修改Video.py中的文件名，执行后如下图所示
```

![](http://p1.bpimg.com/567571/f691d915d9780601.png)
