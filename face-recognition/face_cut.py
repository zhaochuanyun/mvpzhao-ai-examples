import socket
import os
import cv2
from imutils import paths
import uuid

'''
https://blog.csdn.net/xingchenbingbuyu/article/details/51105159
https://blog.csdn.net/haohuajie1988/article/details/79163318
'''

img_path = None
save_path = None
if 'captainMBP' in socket.gethostname():
    img_path = '/Users/mvpzhao/Downloads/mj'
    save_path = '/Users/mvpzhao/data/vgg-face/faces_cut/train/Michael_Jordan'
else:
    img_path = ''
    save_path = ''

# 加载分类器
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.3_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.3_2/share/OpenCV/haarcascades/haarcascade_eye.xml')

for _, img_file in enumerate(paths.list_images(img_path)):
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, cv2.CASCADE_SCALE_IMAGE)
    print("发现{0}个人脸!".format(len(faces)))
    if len(faces) > 0:
        for face_x, face_y, face_w, face_h in faces:
            cv2.imwrite(os.path.join(save_path, str(uuid.uuid1()) + '.jpg'), img[face_y: face_y + face_h, face_x:face_x + face_w])
