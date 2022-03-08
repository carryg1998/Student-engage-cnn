import torch
from PIL import Image
import numpy as np
import dlib
import cv2

from model.CNN import CNNnet

label_name = [
    "Engage-confused",
    "Engage-engaged",
    "Engage-frustrated",
    "Not engage-bored",
    "Not engage-drowsy",
    "Not engage-lookingaway"
]

detector = dlib.get_frontal_face_detector()

if __name__ == "__main__":
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#
    test_image = r"data\Studenet-crop-face\Engaged\confused\0020.jpg"
    model_path = "logs/Epoch5.pth"
    # -----------------------------------------------参数都在这里修改-----------------------------------------------#
    input_shape = [256, 256]

    model = CNNnet(6)

    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    im = cv2.imread(test_image)
    image = Image.open(test_image)

    faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)

    if len(faces) != 0:
        face = faces[0]
        img = image.crop((face.left(), face.top(), face.right(), face.bottom()))
    else:
        img = image

    iw, ih = img.size
    h, w = input_shape

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image, np.float32)

    test_img = torch.from_numpy(np.array(image_data)).type(torch.FloatTensor).unsqueeze(0).permute(0, 3, 1, 2)

    y_test = model(test_img)

    label_index = int(torch.argmax(y_test, 1)[0])

    print("Student is " + label_name[label_index])
