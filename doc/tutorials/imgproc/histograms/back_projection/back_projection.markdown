Back Projection {#tutorial_back_projection}
===============

@tableofcontents

@prev_tutorial{tutorial_histogram_comparison}
@next_tutorial{tutorial_template_matching}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn:

-   What is Back Projection and why it is useful
-   How to use the OpenCV function @ref cv::calcBackProject to calculate Back Projection
-   How to mix different channels of an image by using the OpenCV function @ref cv::mixChannels

Theory
------

### What is Back Projection?

-   Back Projection is a way of recording how well the pixels of a given image fit the distribution
    of pixels in a histogram model.
-   To make it simpler: For Back Projection, you calculate the histogram model of a feature and then
    use it to find this feature in an image.
-   Application example: If you have a histogram of flesh color (say, a Hue-Saturation histogram ),
    then you can use it to find flesh color areas in an image:

### How does it work?

-   We explain this by using the skin example:
-   Let's say you have gotten a skin histogram (Hue-Saturation) based on the image below. The
    histogram besides is going to be our *model histogram* (which we know represents a sample of
    skin tonality). You applied some mask to capture only the histogram of the skin area:
    ![T0](images/Back_Projection_Theory0.jpg)
    ![T1](images/Back_Projection_Theory1.jpg)

-   Now, let's imagine that you get another hand image (Test Image) like the one below: (with its
    respective histogram):
    ![T2](images/Back_Projection_Theory2.jpg)
    ![T3](images/Back_Projection_Theory3.jpg)


-   What we want to do is to use our *model histogram* (that we know represents a skin tonality) to
    detect skin areas in our Test Image. Here are the steps
    -#  In each pixel of our Test Image (i.e. \f$p(i,j)\f$ ), collect the data and find the
        correspondent bin location for that pixel (i.e. \f$( h_{i,j}, s_{i,j} )\f$ ).
    -#  Lookup the *model histogram* in the correspondent bin - \f$( h_{i,j}, s_{i,j} )\f$ - and read
        the bin value.
    -#  Store this bin value in a new image (*BackProjection*). Also, you may consider to normalize
        the *model histogram* first, so the output for the Test Image can be visible for you.
    -#  Applying the steps above, we get the following BackProjection image for our Test Image:

        ![](images/Back_Projection_Theory4.jpg)

    -#  In terms of statistics, the values stored in *BackProjection* represent the *probability*
        that a pixel in *Test Image* belongs to a skin area, based on the *model histogram* that we
        use. For instance in our Test image, the brighter areas are more probable to be skin area
        (as they actually are), whereas the darker areas have less probability (notice that these
        "dark" areas belong to surfaces that have some shadow on it, which in turns affects the
        detection).

Code
----

-   **What does this program do?**
    -   Loads an image
    -   Convert the original to HSV format and separate only *Hue* channel to be used for the
        Histogram (using the OpenCV function @ref cv::mixChannels )
    -   Let the user to enter the number of bins to be used in the calculation of the histogram.
    -   Calculate the histogram (and update it if the bins change) and the backprojection of the
        same image.
    -   Display the backprojection and the histogram in windows.

@add_toggle_cpp
-   **Downloadable code**:
    -   Click
        [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp)
        for the basic version (explained in this tutorial).
    -   For stuff slightly fancier (using H-S histograms and floodFill to define a mask for the
        skin area) you can check the [improved
        demo](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo2.cpp)
    -   ...or you can always check out the classical
        [camshiftdemo](https://github.com/opencv/opencv/tree/master/samples/cpp/camshiftdemo.cpp)
        in samples.

-   **Code at glance:**
@include samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**:
    -   Click
        [here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java)
        for the basic version (explained in this tutorial).
    -   For stuff slightly fancier (using H-S histograms and floodFill to define a mask for the
        skin area) you can check the [improved
        demo](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo2.java)
    -   ...or you can always check out the classical
        [camshiftdemo](https://github.com/opencv/opencv/tree/master/samples/cpp/camshiftdemo.cpp)
        in samples.

-   **Code at glance:**
@include samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java
@end_toggle

@add_toggle_python
-   **Downloadable code**:
    -   Click
        [here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py)
        for the basic version (explained in this tutorial).
    -   For stuff slightly fancier (using H-S histograms and floodFill to define a mask for the
        skin area) you can check the [improved
        demo](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo2.py)
    -   ...or you can always check out the classical
        [camshiftdemo](https://github.com/opencv/opencv/tree/master/samples/cpp/camshiftdemo.cpp)
        in samples.

-   **Code at glance:**
@include samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py
@end_toggle

Explanation
-----------

-   Read the input image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Read the image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Read the image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Read the image
    @end_toggle

-   Transform it to HSV format:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Transform it to HSV
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Transform it to HSV
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Transform it to HSV
    @end_toggle

-   For this tutorial, we will use only the Hue value for our 1-D histogram (check out the fancier
    code in the links above if you want to use the more standard H-S histogram, which yields better
    results):

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Use only the Hue value
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Use only the Hue value
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Use only the Hue value
    @end_toggle

-   as you see, we use the function @ref cv::mixChannels to get only the channel 0 (Hue) from
    the hsv image. It gets the following parameters:
    -   **&hsv:** The source array from which the channels will be copied
    -   **1:** The number of source arrays
    -   **&hue:** The destination array of the copied channels
    -   **1:** The number of destination arrays
    -   **ch[] = {0,0}:** The array of index pairs indicating how the channels are copied. In this
        case, the Hue(0) channel of &hsv is being copied to the 0 channel of &hue (1-channel)
    -   **1:** Number of index pairs

-   Create a Trackbar for the user to enter the bin values. Any change on the Trackbar means a call
    to the **Hist_and_Backproj** callback function.

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Create Trackbar to enter the number of bins
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Create Trackbar to enter the number of bins
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Create Trackbar to enter the number of bins
    @end_toggle

-   Show the image and wait for the user to exit the program:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Show the image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Show the image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Show the image
    @end_toggle

-   **Hist_and_Backproj function:** Initialize the arguments needed for @ref cv::calcHist . The
    number of bins comes from the Trackbar:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp initialize
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java initialize
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py initialize
    @end_toggle

-   Calculate the Histogram and normalize it to the range \f$[0,255]\f$

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Get the Histogram and normalize it
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Get the Histogram and normalize it
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Get the Histogram and normalize it
    @end_toggle

-   Get the Backprojection of the same image by calling the function @ref cv::calcBackProject

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Get Backprojection
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Get Backprojection
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Get Backprojection
    @end_toggle

-   all the arguments are known (the same as used to calculate the histogram), only we add the
    backproj matrix, which will store the backprojection of the source image (&hue)

-   Display backproj:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Draw the backproj
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Draw the backproj
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Draw the backproj
    @end_toggle

-   Draw the 1-D Hue histogram of the image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp Draw the histogram
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/Histograms_Matching/back_projection/CalcBackProjectDemo1.java Draw the histogram
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/Histograms_Matching/back_projection/calcBackProject_Demo1.py Draw the histogram
    @end_toggle

Results
-------

Here are the output by using a sample image ( guess what? Another hand ). You can play with the
bin values and you will observe how it affects the results:
![R0](images/Back_Projection1_Source_Image.jpg)
![R1](images/Back_Projection1_Histogram.jpg)
![R2](images/Back_Projection1_BackProj.jpg)
