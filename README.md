# Emotion detection

## Motivation

This is part of the submission for Hex Cambridge 2021, a Hackathon hosted by the University of Cambridge.

Since the beginning of the COVID-19 pandemic, more and more lectures are delivered online. The aim of this program is to enhance interaction and understanding between teachers and students in online classrooms by creating a Python script that can identify the student's emotion through their webcam using facial recognition AI.

## An emotion detection program

The `train_model.py` script will train a CNN model on the [fer2013](https://www.kaggle.com/msambare/fer2013) dataset using TensorFlow 2. The "fer2013" dataset consists of 48x48 pixel grayscale images of faces, where the emotion of each face is categorized into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). In total, there are 32,298 samples. 

The dataset is first split into 90% train set and 10% train set. An CNN model is then fitted on the train set. Early stopping is specified by monitoring the loss value of the test set.

The accuracy metrics of the trained CNN model evaluated on both the train and test sets:

| Train | Test |
|-------|------|
| 75.3% | 61.2%|

The TensorFlow model is then saved as `model.json` and `model.h5`.

Next, the `emotion.py` script will use the trained CNN model to track the emotion of the student. It will show the webcam image to the student. A rectangular box will be drawn around the student's face along with a text showing the predicted emotion besides it.

The script use `OpenCV` to track human faces captured by the webcam. The models from `dlib` are then used to idenfiy the landmarks of the detected face. This information is used to compute the "drowsiness" of the student by computing the eye aspect ratio (i.e. how closed are the eyes) of the subject. If the ratio is low, it means that the subject is likely closing their eyes. If the ratio is below a predetermined value for some period of time, then it is likely the user is sleeping. Thus, the Python script will print to `stdout` and update the webcam image. Note that the "fer2013" dataset does not have a label for "drowsy" emotion.

The identified face is then preprocessed before feeding it into the trained CNN model to predict the current emotion of the student based on the category with the highest probability. The "angry" emotion was interpreted as "frowning" (i.e. confused/concentrating/unsatisfied). The "surprise" emotion was interpreted as "drowsy", since they are both associated with an open mouth (e.g. yawning). The Python script will also print the emotion to `stdout` and update the webcam image.
