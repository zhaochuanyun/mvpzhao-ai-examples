import numpy as np
import matplotlib.pylab as plt
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace

'''
https://github.com/rcmalli/keras-vggface
https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799
'''

# Based on VGG16 architecture -> old paper(2015)
vggface_model = VGGFace(
    include_top=True,
    input_shape=(224, 224, 3),
    weights='vggface',
    pooling='avg')

img = image.load_img('/home/mvpzhao/下载/zw.jpeg', target_size=(224, 224))

plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)  # or version=2
preds = vggface_model.predict(x)
print('Predicted:', utils.decode_predictions(preds))
