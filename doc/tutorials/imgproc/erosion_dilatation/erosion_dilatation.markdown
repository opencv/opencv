Eroding and Dilating {#tutorial_erosion_dilatation}
====================

Goal
----

In this tutorial you will learn how to:

-   Apply two very common morphology operators: Dilation and Erosion. For this purpose, you will use
    the following OpenCV functions:
    -   @ref cv::erode
    -   @ref cv::dilate

Cool Theory
-----------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

Morphological Operations
------------------------

-   In short: A set of operations that process images based on shapes. Morphological operations
    apply a *structuring element* to an input image and generate an output image.
-   The most basic morphological operations are two: Erosion and Dilation. They have a wide array of
    uses, i.e. :
    -   Removing noise
    -   Isolation of individual elements and joining disparate elements in an image.
    -   Finding of intensity bumps or holes in an image
-   We will explain dilation and erosion briefly, using the following image as an example:

    ![](images/Morphology_1_Tutorial_Theory_Original_Image.png)

### Dilation

-   This operations consists of convoluting an image \f$A\f$ with some kernel (\f$B\f$), which can have any
    shape or size, usually a square or circle.
-   The kernel \f$B\f$ has a defined *anchor point*, usually being the center of the kernel.
-   As the kernel \f$B\f$ is scanned over the image, we compute the maximal pixel value overlapped by
    \f$B\f$ and replace the image pixel in the anchor point position with that maximal value. As you can
    deduce, this maximizing operation causes bright regions within an image to "grow" (therefore the
    name *dilation*). Take as an example the image above. Applying dilation we can get:

    ![](images/Morphology_1_Tutorial_Theory_Dilation.png)

The background (bright) dilates around the black regions of the letter.

### Erosion

-   This operation is the sister of dilation. What this does is to compute a local minimum over the
    area of the kernel.
-   As the kernel \f$B\f$ is scanned over the image, we compute the minimal pixel value overlapped by
    \f$B\f$ and replace the image pixel under the anchor point with that minimal value.
-   Analagously to the example for dilation, we can apply the erosion operator to the original image
    (shown above). You can see in the result below that the bright areas of the image (the
    background, apparently), get thinner, whereas the dark zones (the "writing") gets bigger.

    ![](images/Morphology_1_Tutorial_Theory_Erosion.png)

Code
----

This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgProc/Morphology_1.cpp)
@include samples/cpp/tutorial_code/ImgProc/Morphology_1.cpp

Explanation
-----------

-#  Most of the stuff shown is known by you (if you have any doubt, please refer to the tutorials in
    previous sections). Let's check the general structure of the program:

    -   Load an image (can be BGR or grayscale)
    -   Create two windows (one for dilation output, the other for erosion)
    -   Create a set of 02 Trackbars for each operation:
        -   The first trackbar "Element" returns either **erosion_elem** or **dilation_elem**
        -   The second trackbar "Kernel size" return **erosion_size** or **dilation_size** for the
            corresponding operation.
    -   Every time we move any slider, the user's function **Erosion** or **Dilation** will be
        called and it will update the output image based on the current trackbar values.

    Let's analyze these two functions:

-#  **erosion:**
    @code{.cpp}
    /*  @function Erosion  */
    void Erosion( int, void* )
    {
      int erosion_type;
      if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
      else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
      else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

      Mat element = getStructuringElement( erosion_type,
                               Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                           Point( erosion_size, erosion_size ) );
      /// Apply the erosion operation
      erode( src, erosion_dst, element );
      imshow( "Erosion Demo", erosion_dst );
    }
    @endcode
    -   The function that performs the *erosion* operation is @ref cv::erode . As we can see, it
        receives three arguments:
        -   *src*: The source image
        -   *erosion_dst*: The output image
        -   *element*: This is the kernel we will use to perform the operation. If we do not
            specify, the default is a simple `3x3` matrix. Otherwise, we can specify its
            shape. For this, we need to use the function cv::getStructuringElement :
            @code{.cpp}
            Mat element = getStructuringElement( erosion_type,
                                          Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                          Point( erosion_size, erosion_size ) );
            @endcode
            We can choose any of three shapes for our kernel:

            -   Rectangular box: MORPH_RECT
            -   Cross: MORPH_CROSS
            -   Ellipse: MORPH_ELLIPSE

            Then, we just have to specify the size of our kernel and the *anchor point*. If not
            specified, it is assumed to be in the center.

    -   That is all. We are ready to perform the erosion of our image.
@note Additionally, there is another parameter that allows you to perform multiple erosions
(iterations) at once. We are not using it in this simple tutorial, though. You can check out the
Reference for more details.

-#  **dilation:**

    The code is below. As you can see, it is completely similar to the snippet of code for **erosion**.
    Here we also have the option of defining our kernel, its anchor point and the size of the operator
    to be used.
    @code{.cpp}
    /* @function Dilation */
    void Dilation( int, void* )
    {
      int dilation_type;
      if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
      else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
      else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

      Mat element = getStructuringElement( dilation_type,
                                           Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                           Point( dilation_size, dilation_size ) );
      /// Apply the dilation operation
      dilate( src, dilation_dst, element );
      imshow( "Dilation Demo", dilation_dst );
    }
    @endcode

Results
-------

Compile the code above and execute it with an image as argument. For instance, using this image:

![](images/Morphology_1_Tutorial_Original_Image.jpg)

We get the results below. Varying the indices in the Trackbars give different output images,
naturally. Try them out! You can even try to add a third Trackbar to control the number of
iterations.

![](images/Morphology_1_Result.jpg)
