vggface_model = VGGFace(
    include_top=True,
    input_shape=(224, 224, 3),
    weights='vggface',
    pooling='avg')
