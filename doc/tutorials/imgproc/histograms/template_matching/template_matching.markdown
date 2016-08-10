Template Matching {#tutorial_template_matching}
=================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::matchTemplate to search for matches between an image patch and
    an input image
-   Use the OpenCV function @ref cv::minMaxLoc to find the maximum and minimum values (as well as
    their positions) in a given array.

Theory
------

### What is template matching?

Template matching is a technique for finding areas of an image that match (are similar) to a
template image (patch).

While the patch must be a rectangle it may be that not all of the
rectangle is relevant.  In such a case, a mask can be used to isolate the portion of the patch
that should be used to find the match.

### How does it work?

-   We need two primary components:

    -#  **Source image (I):** The image in which we expect to find a match to the template image
    -#  **Template image (T):** The patch image which will be compared to the template image

    our goal is to detect the highest matching area:

    ![](images/Template_Matching_Template_Theory_Summary.jpg)

-   To identify the matching area, we have to *compare* the template image against the source image
    by sliding it:

    ![](images/Template_Matching_Template_Theory_Sliding.jpg)

-   By **sliding**, we mean moving the patch one pixel at a time (left to right, up to down). At
    each location, a metric is calculated so it represents how "good" or "bad" the match at that
    location is (or how similar the patch is to that particular area of the source image).
-   For each location of **T** over **I**, you *store* the metric in the *result matrix* **(R)**.
    Each location \f$(x,y)\f$ in **R** contains the match metric:

    ![](images/Template_Matching_Template_Theory_Result.jpg)

    the image above is the result **R** of sliding the patch with a metric **TM_CCORR_NORMED**.
    The brightest locations indicate the highest matches. As you can see, the location marked by the
    red circle is probably the one with the highest value, so that location (the rectangle formed by
    that point as a corner and width and height equal to the patch image) is considered the match.

-   In practice, we use the function @ref cv::minMaxLoc to locate the highest value (or lower,
    depending of the type of matching method) in the *R* matrix.

### How does the mask work?
- If masking is needed for the match, three components are required:

    -#  **Source image (I):** The image in which we expect to find a match to the template image
    -#  **Template image (T):** The patch image which will be compared to the template image
    -#  **Mask image (M):** The mask, a grayscale image that masks the template


-   Only two matching methods currently accept a mask: CV_TM_SQDIFF and CV_TM_CCORR_NORMED (see
    below for explanation of all the matching methods available in opencv).


-   The mask must have the same dimensions as the template


-   The mask should have a CV_8U or CV_32F depth and the same number of channels
    as the template image. In CV_8U case, the mask values are treated as binary,
    i.e. zero and non-zero. In CV_32F case, the values should fall into [0..1]
    range and the template pixels will be multiplied by the corresponding mask pixel
    values. Since the input images in the sample have the CV_8UC3 type, the mask
    is also read as color image.

    ![](images/Template_Matching_Mask_Example.jpg)

### Which are the matching methods available in OpenCV?

Good question. OpenCV implements Template matching in the function @ref cv::matchTemplate . The
available methods are 6:

-#  **method=CV_TM_SQDIFF**

    \f[R(x,y)= \sum _{x',y'} (T(x',y')-I(x+x',y+y'))^2\f]

-#  **method=CV_TM_SQDIFF_NORMED**

    \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y')-I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]

-#  **method=CV_TM_CCORR**

    \f[R(x,y)= \sum _{x',y'} (T(x',y')  \cdot I(x+x',y+y'))\f]

-#  **method=CV_TM_CCORR_NORMED**

    \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]

-#  **method=CV_TM_CCOEFF**

    \f[R(x,y)= \sum _{x',y'} (T'(x',y')  \cdot I'(x+x',y+y'))\f]

    where

    \f[\begin{array}{l} T'(x',y')=T(x',y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} T(x'',y'') \\ I'(x+x',y+y')=I(x+x',y+y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} I(x+x'',y+y'') \end{array}\f]

-#  **method=CV_TM_CCOEFF_NORMED**

    \f[R(x,y)= \frac{ \sum_{x',y'} (T'(x',y') \cdot I'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x',y+y')^2} }\f]

Code
----

-   **What does this program do?**
    -   Loads an input image, an image patch (*template*), and optionally a mask
    -   Perform a template matching procedure by using the OpenCV function @ref cv::matchTemplate
        with any of the 6 matching methods described before. The user can choose the method by
        entering its selection in the Trackbar.  If a mask is supplied, it will only be used for
        the methods that support masking
    -   Normalize the output of the matching procedure
    -   Localize the location with higher matching probability
    -   Draw a rectangle around the area corresponding to the highest match
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp)
-   **Code at glance:**
    @include samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp

Explanation
-----------

-#  Declare some global variables, such as the image, template and result matrices, as well as the
    match method and the window names:
    @code{.cpp}
    Mat img; Mat templ; Mat result;
    char* image_window = "Source Image";
    char* result_window = "Result window";

    int match_method;
    int max_Trackbar = 5;
    @endcode
-#  Load the source image, template, and optionally, if supported for the matching method, a mask:
    @code{.cpp}
     bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
  if (use_mask && method_accepts_mask)
    { matchTemplate( img, templ, result, match_method, mask); }
  else
    { matchTemplate( img, templ, result, match_method); }

    @endcode
-#  Create the windows to show the results:
    @code{.cpp}
    namedWindow( image_window, WINDOW_AUTOSIZE );
    namedWindow( result_window, WINDOW_AUTOSIZE );
    @endcode
-#  Create the Trackbar to enter the kind of matching method to be used. When a change is detected
    the callback function **MatchingMethod** is called.
    @code{.cpp}
    char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
    @endcode
-#  Wait until user exits the program.
    @code{.cpp}
    waitKey(0);
    return 0;
    @endcode
-#  Let's check out the callback function. First, it makes a copy of the source image:
    @code{.cpp}
    Mat img_display;
    img.copyTo( img_display );
    @endcode
-#  Next, it creates the result matrix that will store the matching results for each template
    location. Observe in detail the size of the result matrix (which matches all possible locations
    for it)
    @code{.cpp}
    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    result.create( result_rows, result_cols, CV_32FC1 );
    @endcode
-#  Perform the template matching operation:
    @code{.cpp}
    bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
    if (use_mask && method_accepts_mask)
        { matchTemplate( img, templ, result, match_method, mask); }
    else
        { matchTemplate( img, templ, result, match_method); }
    @endcode
    the arguments are naturally the input image **I**, the template **T**, the result **R**, the
    match_method (given by the Trackbar), and optionally the mask image **M**

-#  We normalize the results:
    @code{.cpp}
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
    @endcode
-#  We localize the minimum and maximum values in the result matrix **R** by using @ref
    cv::minMaxLoc .
    @code{.cpp}
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    @endcode
    the function calls as arguments:

    -   **result:** The source array
    -   **&minVal** and **&maxVal:** Variables to save the minimum and maximum values in **result**
    -   **&minLoc** and **&maxLoc:** The Point locations of the minimum and maximum values in the
        array.
    -   **Mat():** Optional mask

-#  For the first two methods ( TM_SQDIFF and MT_SQDIFF_NORMED ) the best match are the lowest
    values. For all the others, higher values represent better matches. So, we save the
    corresponding value in the **matchLoc** variable:
    @code{.cpp}
    if( match_method  == TM_SQDIFF || match_method == TM_SQDIFF_NORMED )
      { matchLoc = minLoc; }
    else
      { matchLoc = maxLoc; }
    @endcode
-#  Display the source image and the result matrix. Draw a rectangle around the highest possible
    matching area:
    @code{.cpp}
    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

    imshow( image_window, img_display );
    imshow( result_window, result );
    @endcode

Results
-------

-#  Testing our program with an input image such as:

    ![](images/Template_Matching_Original_Image.jpg)

    and a template image:

    ![](images/Template_Matching_Template_Image.jpg)

-#  Generate the following result matrices (first row are the standard methods SQDIFF, CCORR and
    CCOEFF, second row are the same methods in its normalized version). In the first column, the
    darkest is the better match, for the other two columns, the brighter a location, the higher the
    match.
    ![Result_0](images/Template_Matching_Correl_Result_0.jpg)
    ![Result_1](images/Template_Matching_Correl_Result_1.jpg)
    ![Result_2](images/Template_Matching_Correl_Result_2.jpg)
    ![Result_3](images/Template_Matching_Correl_Result_3.jpg)
    ![Result_4](images/Template_Matching_Correl_Result_4.jpg)
    ![Result_5](images/Template_Matching_Correl_Result_5.jpg)

-#  The right match is shown below (black rectangle around the face of the guy at the right). Notice
    that CCORR and CCDEFF gave erroneous best matches, however their normalized version did it
    right, this may be due to the fact that we are only considering the "highest match" and not the
    other possible high matches.

    ![](images/Template_Matching_Image_Result.jpg)
