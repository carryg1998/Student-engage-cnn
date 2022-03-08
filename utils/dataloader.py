from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

import cv2
import dlib
import numpy as np


class StuEnDataset(Dataset):

    def __init__(self, data_lines, input_shape):
        self.data = data_lines
        self.length = len(data_lines)
        self.input_shape = input_shape
        self.label_num = {
            "En-confused": 0,
            "En-engaged": 1,
            "En-frustrated": 2,
            "NE-bored": 3,
            "NE-drowsy": 4,
            "NE-lookingaway": 5
        }
        self.detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pic_path, label_name = self.data[index].split("|")

        image = Image.open(pic_path)

        label = np.array(float(self.label_num[label_name[:-1]]))
        iw, ih = image.size
        h, w = self.input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        return image_data, label

def StuEnDataset_collect_fn(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)
    images = np.array(images)

    return images, labels

if __name__ == "__main__":
    with open(r"C:\Users\CarryG\Desktop\CNN-student-engage\data_lines.txt") as f:
        train_lines = f.readlines()
    data = StuEnDataset(train_lines, [512, 512])
    gen = DataLoader(data, shuffle=True, batch_size=8, num_workers=0, pin_memory=True,
                     drop_last=True, collate_fn=StuEnDataset_collect_fn)
    gen = enumerate(gen)
    for iter, batch in gen:
        print(batch[0].shape)
        print(batch[1])
