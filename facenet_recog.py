from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
import pickle
import cv2
from io import BytesIO


HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

cap = cv2.VideoCapture(0)

while(1):
    _, gbr1 = cap.read()

    kao = HaarCascade.detectMultiScale(gbr1,1.1,4)

    if len(kao)>0:
        x1, y1, width, height = kao[0]
    else:
        x1, y1, width, height = 1, 1, 10, 10

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height


    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)                  # konversi dari OpenCV ke PIL
    gbr_array = asarray(gbr)

    face = gbr_array[y1:y2, x1:x2]

    face = Image.fromarray(face)
    face = face.resize((160,160))
    face = asarray(face)


    face = expand_dims(face, axis=0)
    signature = MyFaceNet.embeddings(face)

    min_dist=100
    identity=' '
    for key, value in database.items() :
        dist = np.linalg.norm(value-signature)
        if dist < min_dist:
            min_dist = dist
            identity = key

    cv2.putText(gbr1,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(gbr1,(x1,y1),(x2,y2), (0,255,0), 2)

    cv2.imshow('res',gbr1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

def take_photo(filename='Img.jpg', quality=0.8):
    # OpenCV로 웹캠에서 이미지 캡처
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # 이미지를 BytesIO에 쓰기
    image_stream = BytesIO()
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(image_stream, format='JPEG', quality=quality*100)
    image_data = image_stream.getvalue()

    # 이미지 저장
    with open(filename, 'wb') as f:
        f.write(image_data)

    # 찾은 얼굴의 좌표
    x, y, w, h = find_faces(frame)

    # 이미지에 얼굴 표시
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 캡처된 이미지 출력
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filename

def find_faces(image):
    # OpenCV를 사용하여 얼굴 찾기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return x, y, w, h
    else:
        return 0, 0, 0, 0

take_photo('captured_photo.jpg')

try:
    filename = take_photo()
    print('Saved to {}'.format(filename))

    cv2.imshow('myfile',filename)

except Exception as err:
    print(str(err))