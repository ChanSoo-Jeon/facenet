from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet

import pickle
import cv2

MyFaceNet = FaceNet()
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'))

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()


cap = cv2.VideoCapture(0)

while True:
    _, gbr1 = cap.read()

    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    for (x1, y1, width, height) in wajah:
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)
        gbr_array = asarray(gbr)

        face = gbr_array[y1:y2, x1:x2]

        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)

        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)

        min_dist = 100
        identity = ' '
        for key, value in database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
        if identity == 'unknown':  # 등록된 얼굴이 아닌 경우 모자이크 처리
            # 해당 얼굴 영역을 모자이크하기 위해 이미지를 가져와 모자이크 처리
            face_censored = gbr1[y1:y2, x1:x2]
            face_censored = cv2.GaussianBlur(face_censored, (99, 99), 30)  # 가우시안 블러를 사용한 모자이크 처리
            gbr1[y1:y2, x1:x2] = face_censored  # 모자이크 처리된 영역을 원본 이미지에 적용

        else:
            cv2.putText(gbr1, identity, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)


    cv2.imshow('res', gbr1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()