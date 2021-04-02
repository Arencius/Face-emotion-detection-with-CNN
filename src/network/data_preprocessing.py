import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

PICTURE_SIZE = 48


def string_preprocess(s):
    """
    Converts the pixels in string format into two-dimensional int array.
    """
    s = np.array(s.split(' ')).astype(int)
    return s.reshape((PICTURE_SIZE, PICTURE_SIZE))


def get_dataset():
    data = pd.read_csv('fer2013.csv')
    data.drop('Usage', axis=1, inplace=True)

    images = np.array(list(map(string_preprocess, data['pixels'])), dtype=np.float64) / 255
    targets = data['emotion'].values

    over_sampler = RandomOverSampler()
    balanced_images, balanced_labels = over_sampler.fit_resample(images.reshape(len(images), PICTURE_SIZE ** 2), targets)

    balanced_images = balanced_images.reshape(len(balanced_images), PICTURE_SIZE, PICTURE_SIZE)

    return balanced_images, balanced_labels
