from keras_vggface.vggface import VGGFace

'''
https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799
https://github.com/rcmalli/keras-vggface
'''

# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16')

# Based on RESNET50 architecture -> new paper(2017)
vggface = VGGFace(model='resnet50')

# Based on SENET50 architecture -> new paper(2017)
vggface = VGGFace(model='senet50')

vggface.summary()
