import numpy as np
from PIL import Image
from .data_preprocessing import PICTURE_SIZE

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_emotion(model, pixel_array):
    """
    Loads the image from the given path, preprocesess it and feed it to the network in order to predict the emotion.
    :param model: CNN model used to predict the face emotion on the frame
    :param pixel_array: array of pixels
    """
    image = Image.fromarray(pixel_array).convert('L').resize((PICTURE_SIZE, PICTURE_SIZE))
    img_array = np.expand_dims(np.asarray(image), -1) / 255
    img_array = np.expand_dims(img_array, 0)

    prediction = np.argmax(model.predict(img_array))

    return EMOTIONS[prediction]