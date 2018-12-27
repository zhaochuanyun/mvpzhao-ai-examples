import os
import socket
import numpy as np
from scipy import misc
from imutils import paths
import tensorflow as tf
import cv2
import mtcnn.mtcnn_detect_face as mtcnn

img_dir = None
if 'captainMBP' in socket.gethostname():
    img_dir = '/Users/mvpzhao/Downloads/data'
    save_dir = '/Users/mvpzhao/data/vgg-face/faces_cut/train/'
else:
    img_dir = ''
    save_dir = ''

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 16
image_size = 224

gpu_memory_fraction = 0.8
model_path = ''


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = list(paths.list_images(facedir))
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


dataset = get_dataset(img_dir)

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

    for _, imageClass in enumerate(dataset):
        out_dir = os.path.join(save_dir, imageClass.name)
        for _, img_path in enumerate(imageClass.image_paths):
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_size = np.asarray(img.shape)[0:2]

            bounding_boxes, points = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            for face_idx in range(len(bounding_boxes)):
                det = np.squeeze(bounding_boxes[face_idx, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]

                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

                idx = len(list(paths.list_images(out_dir))) + 1

                cv2.imwrite(os.path.join(out_dir, imageClass.name + '.' + str(idx) + '.jpg'), aligned)
