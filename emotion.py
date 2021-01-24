import os
import cv2
import numpy as np
import tensorflow as tf
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 37
# initialize the frame counter
COUNTER = 0
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Load model
model = tf.keras.models.model_from_json(open("model.json", "r").read())

# Load weights
model.load_weights("model.h5")

emotions = ('frowning', 'disgust', 'fear', 'happy',
            'concentrating', 'yawning', 'neutral')

patience = [4] * 6
sleeping = False

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                          "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

time_counter = 0
time_interval = 17
webcam_on = True

while True:
    pressed_key = cv2.waitKey(10)
    if pressed_key == ord('r'):
        # If 'r' key is pressed, then resume the webcam
        cap = cv2.VideoCapture(0)
        webcam_on = True
    elif pressed_key == ord('p'):
        # If "p" key is pressed, then pause the webcam
        cap.release()
        webcam_on = False

    if not webcam_on:
        continue

    ret, test_image = cap.read()

    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray_image, 0)

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray_image, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of
            # then print to stdout
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                sleeping = True
            else:
                sleeping = False
        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter
        else:
            COUNTER = 0

    if (time_counter != time_interval) or not ret:
        time_counter += 1
        continue
    else:
        time_counter = 0

    faces_detected = face_haar_cascade.detectMultiScale(gray_image, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
        roi_gray = gray_image[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255.0

        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        if max_index == 6:
            patience = [3] * 6
        else:
            patience[max_index] -= 1
            if patience[max_index] == 0:
                print(predicted_emotion)

        cv2.putText(test_image, predicted_emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if sleeping:
        print("sleeping")

    resized_image = cv2.resize(test_image, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_image)

    if pressed_key == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
