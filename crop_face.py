import os
import cv2
import dlib
from PIL import Image

data_path = "data/Student-engagement-dataset/"

pic_label = [
    "Engaged/confused/",
    "Engaged/engaged/",
    "Engaged/frustrated/",
    "Not engaged/bored/",
    "Not engaged/drowsy/",
    "Not engaged/Looking Away/"
]

detector = dlib.get_frontal_face_detector()

if not os.path.exists('data/Studenet-crop-face/'):
    os.mkdir('data/Studenet-crop-face/')

if not os.path.exists('data/Studenet-crop-face/Engaged/'):
    os.mkdir('data/Studenet-crop-face/Engaged/')

if not os.path.exists('data/Studenet-crop-face/Not engaged/'):
    os.mkdir('data/Studenet-crop-face/Not engaged/')

print("Start crop face picture...")
for path in pic_label:
    print("crop " + path + "picture ", end="")
    if not os.path.exists('data/Studenet-crop-face/' + path):
        os.mkdir('data/Studenet-crop-face/' + path)
    for filename in os.listdir(data_path + path):
        file_path = data_path + path + filename
        im = cv2.imread(file_path)
        faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)
        if len(faces) != 0:
            face = faces[0]
            image = Image.open(file_path)
            image = image.crop((face.left(), face.top(), face.right(), face.bottom()))
            image.save("data/Studenet-crop-face/" + path + filename)
        else:
            image = Image.open(file_path)
            image.save("data/Studenet-crop-face/" + path + filename)
    print("Finshied!")
