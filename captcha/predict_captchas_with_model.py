import os
import numpy as np
import cv2
import pickle
from imutils import paths
from keras.models import load_model

from datasets_generate import resize_to_fit, find_letter_image_regions
from datasets_generate import MODEL_LABELS_FILENAME
from train_cnn_model import MODEL_FILENAME

CAPTCHA_PREDICT_FOLDER = '~/data/captcha/datasets/captcha_predict_images'


def predict(model, predict_folder=CAPTCHA_PREDICT_FOLDER, lb_folder=MODEL_LABELS_FILENAME, pad=15):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(lb_folder, 'rb') as f:
        lb = pickle.load(f)

    # Grab some random CAPTCHA images to test against.
    # In the real world, you'd replace this section with code to grab a real
    # CAPTCHA image from a live website.
    captcha_image_files = list(paths.list_images(predict_folder))
    captcha_image_files = np.random.choice(captcha_image_files, size=(20,), replace=False)

    for (idx, captcha_image_file) in enumerate(captcha_image_files):
        image, letter_image_regions = find_letter_image_regions(captcha_image_file, pad=pad)

        # Create an output image and a list to hold our predicted letters
        original = cv2.imread(captcha_image_file)
        output = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        predictions = []

        # loop over the letters
        for letter_bounding_box in letter_image_regions:
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            # Re-size the letter image to 20x20 pixels to match training data
            letter_image = resize_to_fit(letter_image, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

            # draw the prediction on the output image
            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Print the captcha's text
        captcha_text = ''.join(predictions)

        print('correct/predict CAPTCHA text is: {}'.format(captcha_text))

        # Show the annotated image
        cv2.imshow('Output', output)
        cv2.waitKey()


if __name__ == '__main__':
    # Load the trained neural network
    model = load_model(MODEL_FILENAME)
    predict(model)
