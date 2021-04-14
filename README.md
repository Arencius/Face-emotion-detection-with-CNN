# Face emotion detector with Convolutional Neural Network
## About
A simple application written using keras and opencv, with GUI built in tkinter, that can read and detect one of seven emotions from user's face:
 - anger, 
 - disgust, 
  - fear, 
  - happiness, 
  - sadness, 
  - surprise, 
  - neutral face
  
  The dataset used for training exceeded the 100Mb github files size, thus the file is not included in the repository.
  
  ## Neural network
  Convolutional Neural Network that is responsible for detecting and displaying the emotion was built with four convolutional blocks and was trained on the fer 2013 dataset.
  Training took around 150 epochs, and achieved 79% accuracy on the testing set (the images never seen by model during the training).
  All files related to neural network, i.e. image preprocessing, building the model and the serialized model itself are located in the src/network directory.
  
  ## Launching the app
  For the application to properly work you need to install all the necessary dependecies stored in the requirements.txt file and have access to the webcam.
  Having done that, launch the main.py file in the src folder, andyou should see your face, with botton label displaying the predicted emotion.
