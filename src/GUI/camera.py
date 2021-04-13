import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
from src.network.model_testing import predict_emotion
from keras.models import load_model

# hides warnings about tensorflow GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# pretrained CNN model
MODEL = load_model('network/model.h5')


class Camera:
    def __init__(self):
        # setting the window
        self.window = tk.Tk()
        self.window.title('Camera')
        self.window.geometry('600x600')

        # panel to display camera image
        self.main_panel = tk.Label(self.window)
        self.main_panel.pack()

        # label containing predicted emotion
        self.emotion_label = tk.Label(self.window)
        self.emotion_label.pack()

        self.video = cv2.VideoCapture(0)
        self.capture_camera()

        self.window.mainloop()

    def capture_camera(self):
        """
        Captures the camera and displays the image on the screen, along with the emotion predicted by the model
        """
        _, frame = self.video.read()

        # predicts the emotion and displays the output
        emotion = predict_emotion(MODEL, frame)
        self.emotion_label['text'] = emotion

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        # placing the camera image on the screen
        self.main_panel.imgtk = imgtk
        self.main_panel.configure(image=imgtk)
        self.main_panel.after(10, self.capture_camera)