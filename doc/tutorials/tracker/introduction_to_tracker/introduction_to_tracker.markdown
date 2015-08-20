Introduction to OpenCV Tracker {#tutorial_introduction_to_tracker}
===========

Goal
----

In this tutorial you will learn how to

-   Create a tracker object.
-   Using the roiSelector function to select ROI from a given image.
-   Track a specific region from a given image.

Source Code
-----------

@include cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp

Explanation
-----------

-#  **Set up the input video**

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp help

    In this tutorial, you can choose between video or list of images for the program input.
    As written in the help, you should specify the input video as parameter of the program.
    If you want to use image list as input, the image list should have formatted numbering
    as shown in help. In the help, it means that the image fils are numbered with 4 digits
    (e.g. the file naming will be 0001.jpg, 0002.jpg, and so on).

    You can find samples video in Itseez/opencv_extra/testdata/cv/tracking
    <https://github.com/Itseez/opencv_extra/tree/master/testdata/cv/tracking>

-#  **Declares the required variables**

    You need roi to record the bounding box of the tracked object. The value stored in this
    variable will be updated using the tracker object.

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp vars

    The frame variable is used to hold the image data from each frame of the input video or images list.

-#  **Creating a tracker object**

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp create

    There are at least 5 types of tracker algorithm can be used:
    + MIL
    + BOOSTING
    + MEDIANFLOW
    + TLD
    + KCF

    Each tracker algorithm has their own advantages and disadvantages, please refer the documentation of @ref cv::Tracker for more detailed information.

-#  **Select the tracked object**

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp selectroi

    Using this function, you can select the bounding box of the tracked object using a GUI.
    With default parameters, the selection is started from the center of the box and middle cross will be shown.
    See @ref cv::selectROI for more detailed information.

-#  **Initializing the tracker object**

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp init

    Tracker algorithm should be initialized with the provided image data as well as the bounding box of the tracked object.
    Make sure that the bounding box is not valid (size more than zero) to avoid the initialization process failed.

-#  **Update**

    @snippet cpp/tutorial_code/tracker/introduction_to_tracker/introduction_to_tracker.cpp update

    This update function will perform the tracking process and pass the result to the roi variable.
