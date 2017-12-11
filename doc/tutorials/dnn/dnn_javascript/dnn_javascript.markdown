# How to run deep networks in browser {#tutorial_dnn_javascript}

## Introduction
This tutorial will show us how to run deep learning models using OpenCV.js right
in a browser. Tutorial refers a sample of face detection and face recognition
models pipeline.

## Face detection
Face detection network gets BGR image as input and produces set of bounding boxes
that might contain faces. All that we need is just select the boxes with a strong
confidence.

## Face recognition
Network is called OpenFace (project https://github.com/cmusatyalab/openface).
Face recognition model receives RGB face image of size `96x96`. Then it returns
`128`-dimensional unit vector that represents input face as a point on the unit
multidimensional sphere. So difference between two faces is an angle between two
output vectors.

## Sample
All the sample is an HTML page that has JavaScript code to use OpenCV.js functionality.
You may see an insertion of this page below. Press `Start` button to begin a demo.
Press `Add a person` to name a person that is recognized as an unknown one.
Next we'll discuss main parts of the code.

@htmlinclude js_face_recognition.html

-# Run face detection network to detect faces on input image.
@snippet dnn/js_face_recognition.html Run face detection model
You may play with input blob sizes to balance detection quality and efficiency.
The bigger input blob the smaller faces may be detected.

-# Run face recognition network to receive `128`-dimensional unit feature vector by input face image.
@snippet dnn/js_face_recognition.html Get 128 floating points feature vector

-# Perform a recognition.
@snippet dnn/js_face_recognition.html Recognize
Match a new feature vector with registered ones. Return a name of the best matched person.

-# The main loop.
@snippet dnn/js_face_recognition.html Define frames processing
A main loop of our application receives a frames from a camera and makes a recognition
of an every detected face on the frame. We start this function ones when OpenCV.js was
initialized and deep learning models were downloaded.
