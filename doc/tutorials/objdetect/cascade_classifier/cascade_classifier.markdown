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

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp)
@include samples/cpp/tutorial_code/objectDetection/objectDetection.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/3.4/samples/java/tutorial_code/objectDetection/cascade_classifier/ObjectDetectionDemo.java)
@include samples/java/tutorial_code/objectDetection/cascade_classifier/ObjectDetectionDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/3.4/samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py)
@include samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py
@end_toggle

Explanation
-----------

Result
------

-#  Here is the result of running the code above and using as input the video stream of a build-in
    webcam:

    ![](images/Cascade_Classifier_Tutorial_Result_Haar.jpg)

    Be sure the program will find the path of files *haarcascade_frontalface_alt.xml* and
    *haarcascade_eye_tree_eyeglasses.xml*. They are located in
    *opencv/data/haarcascades*

-#  This is the result of using the file *lbpcascade_frontalface.xml* (LBP trained) for the face
    detection. For the eyes we keep using the file used in the tutorial.

    ![](images/Cascade_Classifier_Tutorial_Result_LBP.jpg)

Additional Resources
--------------------

-#  Paul Viola and Michael J. Jones. Robust real-time face detection. International Journal of Computer Vision, 57(2):137–154, 2004. @cite Viola04
-#  Rainer Lienhart and Jochen Maydt. An extended set of haar-like features for rapid object detection. In Image Processing. 2002. Proceedings. 2002 International Conference on, volume 1, pages I–900. IEEE, 2002. @cite Lienhart02
-#  Video Lecture on [Face Detection and Tracking](https://www.youtube.com/watch?v=WfdYYNamHZ8)
-#  An interesting interview regarding Face Detection by [Adam
    Harvey](https://web.archive.org/web/20171204220159/http://www.makematics.com/research/viola-jones/)
-#  [OpenCV Face Detection: Visualized](https://vimeo.com/12774628) on Vimeo by Adam Harvey
