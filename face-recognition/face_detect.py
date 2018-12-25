import socket
import cv2

'''
https://blog.csdn.net/xingchenbingbuyu/article/details/51105159
https://blog.csdn.net/haohuajie1988/article/details/79163318
'''

img = None
save_path = None
if 'captainMBP' in socket.gethostname():
    img = cv2.imread('/Users/mvpzhao/data/vgg-face/faces/Abbie_Cornish/00000298.jpg')
    save_path = '/Users/mvpzhao/data/vgg-face/faces_cut/1.jpg'
else:
    img = cv2.imread('/Users/mvpzhao/Downloads/vicky8.jpg')
    save_path = '/Users/mvpzhao/data/vgg-face/faces_cut/1.jpg'

# 加载分类器
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.3_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.3_2/share/OpenCV/haarcascades/haarcascade_eye.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸
faces = face_cascade.detectMultiScale(gray, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (50, 50), (100, 100))

print("发现{0}个人脸!".format(len(faces)))

if len(faces) > 0:
    for face_x, face_y, face_w, face_h in faces:
        cv2.imwrite(save_path, img[face_y: face_y + face_h, face_x:face_x + face_w])

        cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2, 8, 0)

        # roi_gray = gray[face_y:face_y + face_h, face_x:face_x + face_w]
        # roi_color = img[face_y:face_y + face_h, face_x:face_x + face_w]
        #
        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (2, 2))
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
