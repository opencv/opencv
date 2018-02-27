Histogram Equalization {#tutorial_histogram_equalization}
======================

Goal
----

In this tutorial you will learn:

-   What an image histogram is and why it is useful
-   To equalize histograms of images by using the OpenCV function @ref cv::equalizeHist

Theory
------

### What is an Image Histogram?

-   It is a graphical representation of the intensity distribution of an image.
-   It quantifies the number of pixels for each intensity value considered.

![](images/Histogram_Equalization_Theory_0.jpg)

### What is Histogram Equalization?

-   It is a method that improves the contrast in an image, in order to stretch out the intensity
    range.
-   To make it clearer, from the image above, you can see that the pixels seem clustered around the
    middle of the available range of intensities. What Histogram Equalization does is to *stretch
    out* this range. Take a look at the figure below: The green circles indicate the
    *underpopulated* intensities. After applying the equalization, we get an histogram like the
    figure in the center. The resulting image is shown in the picture at right.

![](images/Histogram_Equalization_Theory_1.jpg)

### How does it work?

-   Equalization implies *mapping* one distribution (the given histogram) to another distribution (a
    wider and more uniform distribution of intensity values) so the intensity values are spread
    over the whole range.
-   To accomplish the equalization effect, the remapping should be the *cumulative distribution
    function (cdf)* (more details, refer to *Learning OpenCV*). For the histogram \f$H(i)\f$, its
    *cumulative distribution* \f$H^{'}(i)\f$ is:

    \f[H^{'}(i) = \sum_{0 \le j < i} H(j)\f]

    To use this as a remapping function, we have to normalize \f$H^{'}(i)\f$ such that the maximum value
    is 255 ( or the maximum value for the intensity of the image ). From the example above, the
    cumulative function is:

    ![](images/Histogram_Equalization_Theory_2.jpg)

-   Finally, we use a simple remapping procedure to obtain the intensity values of the equalized
    image:

    \f[equalized( x, y ) = H^{'}( src(x,y) )\f]

Code
----

-   **What does this program do?**
    -   Loads an image
    -   Convert the original image to grayscale
    -   Equalize the Histogram by using the OpenCV function @ref cv::equalizeHist
    -   Display the source and equalized images in a window.
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/EqualizeHist_Demo.cpp)
-   **Code at glance:**
    @include samples/cpp/tutorial_code/Histograms_Matching/EqualizeHist_Demo.cpp

Explanation
-----------

-#  Declare the source and destination images as well as the windows names:
    @code{.cpp}
    Mat src, dst;

    char* source_window = "Source image";
    char* equalized_window = "Equalized Image";
    @endcode
-#  Load the source image:
    @code{.cpp}
    src = imread( argv[1], 1 );

    if( !src.data )
      { cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
        return -1;}
    @endcode
-#  Convert it to grayscale:
    @code{.cpp}
    cvtColor( src, src, COLOR_BGR2GRAY );
    @endcode
-#  Apply histogram equalization with the function @ref cv::equalizeHist :
    @code{.cpp}
    equalizeHist( src, dst );
    @endcode
    As it can be easily seen, the only arguments are the original image and the output (equalized)
    image.

-#  Display both images (original and equalized) :
    @code{.cpp}
    namedWindow( source_window, WINDOW_AUTOSIZE );
    namedWindow( equalized_window, WINDOW_AUTOSIZE );

    imshow( source_window, src );
    imshow( equalized_window, dst );
    @endcode
-#  Wait until user exists the program
    @code{.cpp}
    waitKey(0);
    return 0;
    @endcode

Results
-------

-#  To appreciate better the results of equalization, let's introduce an image with not much
    contrast, such as:

    ![](images/Histogram_Equalization_Original_Image.jpg)

    which, by the way, has this histogram:

    ![](images/Histogram_Equalization_Original_Histogram.jpg)

    notice that the pixels are clustered around the center of the histogram.

-#  After applying the equalization with our program, we get this result:

    ![](images/Histogram_Equalization_Equalized_Image.jpg)

    this image has certainly more contrast. Check out its new histogram like this:

    ![](images/Histogram_Equalization_Equalized_Histogram.jpg)

    Notice how the number of pixels is more distributed through the intensity range.

@note
Are you wondering how did we draw the Histogram figures shown above? Check out the following
tutorial!
