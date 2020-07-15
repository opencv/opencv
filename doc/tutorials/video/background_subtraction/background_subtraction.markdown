How to Use Background Subtraction Methods {#tutorial_background_subtraction}
=========================================

@next_tutorial{tutorial_meanshift}

-   Background subtraction (BS) is a common and widely used technique for generating a foreground
    mask (namely, a binary image containing the pixels belonging to moving objects in the scene) by
    using static cameras.
-   As the name suggests, BS calculates the foreground mask performing a subtraction between the
    current frame and a background model, containing the static part of the scene or, more in
    general, everything that can be considered as background given the characteristics of the
    observed scene.

    ![](images/Background_Subtraction_Tutorial_Scheme.png)

-   Background modeling consists of two main steps:

    -#  Background Initialization;
    -#  Background Update.

    In the first step, an initial model of the background is computed, while in the second step that
    model is updated in order to adapt to possible changes in the scene.

-   In this tutorial we will learn how to perform BS by using OpenCV.

Goals
-----

In this tutorial you will learn how to:

-#  Read data from videos or image sequences by using @ref cv::VideoCapture ;
-#  Create and update the background model by using @ref cv::BackgroundSubtractor class;
-#  Get and show the foreground mask by using @ref cv::imshow ;

Code
----

In the following you can find the source code. We will let the user choose to process either a video
file or a sequence of images.

We will use @ref cv::BackgroundSubtractorMOG2 in this sample, to generate the foreground mask.

The results as well as the input data are shown on the screen.

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/video/bg_sub.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/video/bg_sub.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/video/background_subtraction/bg_sub.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/video/background_subtraction/bg_sub.py
@end_toggle

Explanation
-----------

We discuss the main parts of the code above:

-   A @ref cv::BackgroundSubtractor object will be used to generate the foreground mask. In this
    example, default parameters are used, but it is also possible to declare specific parameters in
    the create function.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/video/bg_sub.cpp create
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java create
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/video/background_subtraction/bg_sub.py create
@end_toggle

-   A @ref cv::VideoCapture object is used to read the input video or input images sequence.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/video/bg_sub.cpp capture
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java capture
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/video/background_subtraction/bg_sub.py capture
@end_toggle

-   Every frame is used both for calculating the foreground mask and for updating the background. If
    you want to change the learning rate used for updating the background model, it is possible to
    set a specific learning rate by passing a parameter to the `apply` method.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/video/bg_sub.cpp apply
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java apply
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/video/background_subtraction/bg_sub.py apply
@end_toggle

-   The current frame number can be extracted from the @ref cv::VideoCapture object and stamped in
    the top left corner of the current frame. A white rectangle is used to highlight the black
    colored frame number.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/video/bg_sub.cpp display_frame_number
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java display_frame_number
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/video/background_subtraction/bg_sub.py display_frame_number
@end_toggle

-   We are ready to show the current input frame and the results.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/video/bg_sub.cpp show
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/video/background_subtraction/BackgroundSubtractionDemo.java show
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/video/background_subtraction/bg_sub.py show
@end_toggle

Results
-------

-   With the `vtest.avi` video, for the following frame:

    ![](images/Background_Subtraction_Tutorial_frame.jpg)

    The output of the program will look as the following for MOG2 method (gray areas are detected shadows):

    ![](images/Background_Subtraction_Tutorial_result_MOG2.jpg)

    The output of the program will look as the following for the KNN method (gray areas are detected shadows):

    ![](images/Background_Subtraction_Tutorial_result_KNN.jpg)

References
----------

-   [Background Models Challenge (BMC) website](https://web.archive.org/web/20140418093037/http://bmc.univ-bpclermont.fr/)
-   A Benchmark Dataset for Foreground/Background Extraction @cite vacavant2013benchmark
