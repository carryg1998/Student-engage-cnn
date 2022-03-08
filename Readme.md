## CNN学生行为识别

### 1.下载数据

https://www.kaggle.com/joyee19/studentengagement

放入data文件夹如下

├── data

│ ├── Student-engagement-dataset

│ │ ├──Engaged

│ │ ├──Not engaged

### 2.处理数据

运行crop_face.py生成脸部识别图片，设置测试集比例运行Generate_anno.py生成数据集

### 3.训练

运行train.py进行训练，训练权重保存于logs

### 4.测试

更改test.py的参数看结果