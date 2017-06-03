Using MultiTracker {#tutorial_multitracker}
==================

Goal
----

In this tutorial you will learn how to

-   Create a MultiTracker object.
-   Track several objects at once using the MultiTracker object.

Source Code
-----------

@include cpp/tutorial_code/tracker/multitracker/tutorial_multitracker.cpp

Explanation
-----------

-#  **Create the MultiTracker object**

    @snippet cpp/tutorial_code/tracker/multitracker/tutorial_multitracker.cpp create

    You can create the MultiTracker object and use same tracking algorithm for all tracked object as shown in the snippet.
    If you should define the tracking algorithm whenever a new object is added to the MultiTracker object.

-#  **Selection of multiple objects**

    @snippet cpp/tutorial_code/tracker/multitracker/tutorial_multitracker.cpp selectmulti

    You can use @ref cv::selectROI to select multiple object with
    the result stored in a vector of @ref cv::Rect2d as shown in the code.
    You can also use another kind of selection scheme, please refer to @ref cv::selectROI for detailed information.

-#  **Adding the tracked object to MultiTracker**

    @snippet cpp/tutorial_code/tracker/multitracker/tutorial_multitracker.cpp init

    You can add all tracked objects at once to the MultiTracker as shown in the code.
    In this case, all object will be tracked using same tracking algorithm as specified in decaration of MultiTracker object.
    If you want to use different tracker algorithm for each tracked object,
    You should add the tracked object one by one and specify their tracking algorithm using the variant of @ref cv::MultiTracker::add.
    @sa cv::MultiTracker::add( const String& trackerType, const Mat& image, const Rect2d& boundingBox )

-#  **Obtaining the result**

    @snippet cpp/tutorial_code/tracker/multitracker/tutorial_multitracker.cpp result

    You can access the result from the public variable @ref cv::MultiTracker::objects provided by the MultiTracker class as shown in the code.
