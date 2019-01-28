import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from datasets_generate import load_datasets

MODEL_FILENAME = '~/data/captcha/datasets/captcha_model.hdf5'
MODEL_VIEW = '~/data/captcha/datasets/model_view.png'
TENSOR_BOARD_LOG = '~/data/captcha/datasets/logs'


def captcha_recognition_model(input_shape, model_name='Captcha-Recognition-Model'):
    """
    keras实现的cnn模型
    """
    # Build the neural network!
    model = Sequential(name=model_name)

    # First convolutional layer with max pooling
    model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second convolutional layer with max pooling
    model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))

    # Output layer with 26 nodes (one for each possible letter/number we predict)
    model.add(Dense(26, activation='softmax'))

    return model


def train_cnn(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64, model_view=MODEL_VIEW):
    """
    训练模型
    """
    cnn_model = captcha_recognition_model(X_train[0, :].shape)

    # Ask Keras to build the TensorFlow model behind the scenes
    cnn_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # visualize loss and acc in tensorboard
    tb = TensorBoard(log_dir=os.path.expanduser(TENSOR_BOARD_LOG),
                     histogram_freq=1,
                     batch_size=batch_size,
                     write_graph=True,
                     write_grads=False,
                     write_images=True,
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Train the neural network
    cnn_model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, callbacks=[tb])

    # Save the trained model to disk
    cnn_model.save(os.path.expanduser(MODEL_FILENAME))

    preds = cnn_model.evaluate(x=X_test, y=Y_test)

    print()
    print('Loss = ' + str(preds[0]))
    print('Test Accuracy = ' + str(preds[1]))
    print()

    cnn_model.summary()

    plot_model(cnn_model, to_file=os.path.expanduser(model_view))


if __name__ == '__main__':
    (X_train, Y_train, X_test, Y_test) = load_datasets()
    train_cnn(X_train, Y_train, X_test, Y_test, epochs=10, batch_size=64)
