import os
import glob
import cv2
import numpy as np
import pickle
import imutils
from imutils import paths

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

"""
数据集到百度网盘下载
"""

CAPTCHA_IMAGE_FOLDER = '~/data/captcha/datasets/captcha_original_images'
LETTER_OUTPUT_FOLDER = '~/data/captcha/datasets/extracted_letter_images'
MODEL_LABELS_FILENAME = '~/data/captcha/datasets/model_labels.dat'


# Load hdf5 datasets
def load_datasets():
    # initialize the data and labels
    data = []
    labels = []

    # loop over the input images
    for image_file in paths.list_images(os.path.expanduser(LETTER_OUTPUT_FOLDER)):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 20x20 pixel box
        image = resize_to_fit(image, 20, 20)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(labels)
    labels = lb.transform(labels)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(os.path.expanduser(MODEL_LABELS_FILENAME), 'wb') as f:
        pickle.dump(lb, f)

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    return X_train, Y_train, X_test, Y_test


def captchas2letters(src_folder=CAPTCHA_IMAGE_FOLDER, out_folder=LETTER_OUTPUT_FOLDER):
    captcha_image_files = glob.glob(os.path.join(os.path.expanduser(src_folder), '*'))

    counts = {}

    for (idx, captcha_image_file) in enumerate(captcha_image_files):
        print('[INFO] processing captcha image {}/{}'.format(idx + 1, len(captcha_image_files)))

        file_name = os.path.basename(captcha_image_file)
        captcha_text = os.path.splitext(file_name)[0]

        image, letter_image_regions = find_letter_image_regions(captcha_image_file)

        # Save out each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_text):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

            # Get the folder to save the image in
            save_path = os.path.join(os.path.expanduser(out_folder), letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(letter_text, 1)
            p = os.path.join(os.path.expanduser(save_path), '{}.png'.format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)

            # increment the count for the current key
            counts[letter_text] = count + 1


def find_letter_image_regions(captcha_image_file, pad=15):
    # read image as gray
    image = cv2.imread(captcha_image_file, 0)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Add some extra padding around the image
    thresh = cv2.copyMakeBorder(thresh, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    if len(letter_image_regions) != 4:
        print('Found more or less than 4 letters!')
        raise Exception('Found more or less than 4 letters!')

    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    return thresh, letter_image_regions


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image


if __name__ == '__main__':
    # load_datasets()
    captchas2letters()
