import socket
import numpy as np
import matplotlib.pylab as plt
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace

'''
http://www.robots.ox.ac.uk/~vgg/data/
https://github.com/rcmalli/keras-vggface
https://blog.csdn.net/zhuquan945/article/details/53998793
https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799
'''

# Based on VGG16 architecture
vggface_model = VGGFace(
    model='vgg16',
    weights='vggface',
    include_top=True,
    input_shape=(224, 224, 3))

img_path = None

if 'captainMBP' in socket.gethostname():
    img_path = '/Users/mvpzhao/Downloads/vicky8.jpg'
else:
    img_path = '/home/mvpzhao/下载/timg.jpeg'

img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)  # or version=2
preds = vggface_model.predict(x)
print('Predicted:', utils.decode_predictions(preds))
