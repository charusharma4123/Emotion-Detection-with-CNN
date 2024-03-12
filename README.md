# Emotion-Detection-with-CNN

![emotion_detection](https://github.com/charusharma4123/Emotion-Detection-with-CNN/blob/main/emoition_detection.png)

### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TranEmotionDetector.py

It will take several hours depends on your processor. (On i5 processor with 16 GB RAM it took me around 5 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model_one.json
emotion_model_one.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
download the sample video from here (https://www.pexels.com/video/three-girls-laughing-5273028/) 

python TestEmotionDetector.py
