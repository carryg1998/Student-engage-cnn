import numpy as np
from PIL import Image

label_num = {
    "En-confused": 0,
    "En-engaged": 1,
    "En-frustrated": 2,
    "NE-bored": 3,
    "NE-drowsy": 4,
    "NE-lookingaway": 5
}

def get_val_data(data_lines, input_shape):

    images = []
    labels = []

    for index in range(len(data_lines)):
        pic_path, label_name = data_lines[index].split("|")

        image = Image.open(pic_path)
        label = label_num[label_name[:-1]]
        iw, ih = image.size
        h, w = input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        images.append(image_data)
        labels.append(label)

    return images, labels
