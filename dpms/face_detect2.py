import os

import cv2
import time
import numpy as np

import sys

from openpyxl.drawing.shapes import Camera

print('Press Esc to exit')
faceCascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
imgWindow = cv2.namedWindow('FaceDetect', cv2.WINDOW_NORMAL)

def detect_face():
    capInput = cv2.VideoCapture(0)
    # 避免处理时间过长造成画面卡顿
    nextCaptureTime = time.time()
    faces = []
    if not capInput.isOpened():
        print('Capture failed because of camera')
    while 1:
        ret, img = capInput.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if nextCaptureTime < time.time():
            nextCaptureTime = time.time() + 0.1
            faces = faceCascade.detectMultiScale(img, 1.3, 5)
        if faces is not None:
            for x, y, w, h in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('FaceDetect', img)
        # 这是简单的读取键盘输入，27即Esc的acsii码
        if cv2.waitKey(1) & 0xFF == 27:
            break
    capInput.release()
    cv2.destroyAllWindows()

def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirname:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == '.directory':
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if sz is not None:
                        im = cv2.resize(im, (200, 200))
                    X.append(np.asarray(im, dtype=np.unit8))
                    y.append(c)
                # except IOError as (errno, strerror):
                #     print('I/O error({0}):{1}'.format(errno, strerror))
                except:
                    print('Unexpected error:', sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]


def face_rec():
    names = ['Joe', 'Jane', 'Jack']
    print(len(sys.argv))
    if len(sys.argv) < 2:
       print('USAGE: facerec_demo.py < /path/to/images')
       sys.exit()
    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(X), np.asarray(y))
    cammer = cv2.VideoCapture(0)
    face_cascale = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while True:
        read, img = Camera.read()
        faces = face_cascale.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x: x+w, y: y+h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print('Label: %s, Confidence: %.2f' %(params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_rec()
