import os
import random

# -----------------------------------------------参数都在这里修改-----------------------------------------------#
data_path = "data/Studenet-crop-face/"
val_percent = 0.2
# -----------------------------------------------参数都在这里修改-----------------------------------------------#

pic_label = [
    "Engaged/confused/",
    "Engaged/engaged/",
    "Engaged/frustrated/",
    "Not engaged/bored/",
    "Not engaged/drowsy/",
    "Not engaged/Looking Away/"
]

label_name = [
    "En-confused",
    "En-engaged",
    "En-frustrated",
    "NE-bored",
    "NE-drowsy",
    "NE-lookingaway"
]

data_anno = []

label_index = 0

for path in pic_label:
    for filename in os.listdir(data_path + path):
        str_lines = os.getcwd() + "/" + data_path + path + filename + "|" + label_name[label_index]
        data_anno.append(str_lines)
    label_index += 1

sum = len(data_anno)
train_data_anno = data_anno
val_data_anno = random.sample(data_anno, int(sum*val_percent))
for testData in val_data_anno: # 将已经选定的测试集数据从数据集中删除
    train_data_anno.remove(testData)
print(len(train_data_anno))
print(len(val_data_anno))

with open("train_lines.txt", "w") as f:
    for i in train_data_anno:
        f.writelines(i.replace("/", "\\") + "\n")

with open("val_lines.txt", "w") as f:
    for i in val_data_anno:
        f.writelines(i.replace("/", "\\") + "\n")
