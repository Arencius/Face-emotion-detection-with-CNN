import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from keras.preprocessing.image import ImageDataGenerator

PICTURE_SIZE = 48
BATCH_SIZE = 64


def string_preprocess(s):
    """
    Converts the pixels in string format into two-dimensional int array.

    :param s: str, text containing 48 pixel values, separated with space
    :return: numpy array with image pixels
    """
    s = np.array(s.split(' ')).astype(int)
    return s.reshape((PICTURE_SIZE, PICTURE_SIZE))


def get_dataset():
    """
    Function loads the fer-2013 dataset, and returns balanced data
    :return: tuple, the dataset with balanced count of each target (emotion)
    """
    data = pd.read_csv('fer2013.csv')
    data.drop('Usage', axis=1, inplace=True)

    # transforming the string pixels to the numpy array
    images = np.array(list(map(string_preprocess, data['pixels'])), dtype=np.float64) / 255
    targets = data['emotion'].values

    # balancing the data
    over_sampler = RandomOverSampler()
    balanced_images, balanced_labels = over_sampler.fit_resample(images.reshape(len(images), PICTURE_SIZE ** 2), targets)
    balanced_images = balanced_images.reshape(len(balanced_images), PICTURE_SIZE, PICTURE_SIZE)

    return balanced_images, balanced_labels


def data_augmentation(images, labels):
    """
    Applies the image augmentation to the dataset.
    :param images: images to perform augmentation on
    :param labels: the labels corresponding to each image
    :return: tuple, augmented dataset
    """
    data_generator = ImageDataGenerator(zoom_range=0.2,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)
    data_generator.fit(images)

    return data_generator.flow(images, labels, batch_size=BATCH_SIZE)

