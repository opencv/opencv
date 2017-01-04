Cascade Classifier {#tutorial_cascade_classifier}
==================

Goal
----

In this tutorial you will learn how to:

-   Use the @ref cv::CascadeClassifier class to detect objects in a video stream. Particularly, we
    will use the functions:
    -   @ref cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP classifer
    -   @ref cv::CascadeClassifier::detectMultiScale to perform the detection.

Theory
------

Code
----

This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp)
. The second version (using LBP for face detection) can be [found
here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/objectDetection/objectDetection2.cpp)
@include samples/cpp/tutorial_code/objectDetection/objectDetection.cpp

Explanation
-----------

Result
------

-#  Here is the result of running the code above and using as input the video stream of a build-in
    webcam:

    ![](images/Cascade_Classifier_Tutorial_Result_Haar.jpg)

    Remember to copy the files *haarcascade_frontalface_alt.xml* and
    *haarcascade_eye_tree_eyeglasses.xml* in your current directory. They are located in
    *opencv/data/haarcascades*

-#  This is the result of using the file *lbpcascade_frontalface.xml* (LBP trained) for the face
    detection. For the eyes we keep using the file used in the tutorial.

    ![](images/Cascade_Classifier_Tutorial_Result_LBP.jpg)
