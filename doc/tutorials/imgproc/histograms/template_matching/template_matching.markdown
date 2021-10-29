Template Matching {#tutorial_template_matching}
=================

@tableofcontents

@prev_tutorial{tutorial_back_projection}
@next_tutorial{tutorial_find_contours}

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function **matchTemplate()** to search for matches between an image patch and
    an input image
-   Use the OpenCV function **minMaxLoc()** to find the maximum and minimum values (as well as
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
    -#  **Template image (T):** The patch image which will be compared to the source image

    our goal is to detect the highest matching area:

    ![](images/Template_Matching_Template_Theory_Summary.jpg)

-   To identify the matching area, we have to *compare* the template image against the source image
    by sliding it:

    ![](images/Template_Matching_Template_Theory_Sliding.jpg)

-   By **sliding**, we mean moving the patch one pixel at a time (left to right, up to down). At
    each location, a metric is calculated so it represents how "good" or "bad" the match at that
    location is (or how similar the patch is to that particular area of the source image).
-   For each location of **T** over **I**, you *store* the metric in the *result matrix* **R**.
    Each location \f$(x,y)\f$ in **R** contains the match metric:

    ![](images/Template_Matching_Template_Theory_Result.jpg)

    the image above is the result **R** of sliding the patch with a metric **TM_CCORR_NORMED**.
    The brightest locations indicate the highest matches. As you can see, the location marked by the
    red circle is probably the one with the highest value, so that location (the rectangle formed by
    that point as a corner and width and height equal to the patch image) is considered the match.
-   In practice, we locate the highest value (or lower, depending of the type of matching method) in
    the *R* matrix, using the function **minMaxLoc()**

### How does the mask work?
- If masking is needed for the match, three components are required:

    -#  **Source image (I):** The image in which we expect to find a match to the template image
    -#  **Template image (T):** The patch image which will be compared to the source image
    -#  **Mask image (M):** The mask, a grayscale image that masks the template


-   Only two matching methods currently accept a mask: TM_SQDIFF and TM_CCORR_NORMED (see
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

Good question. OpenCV implements Template matching in the function **matchTemplate()**. The
available methods are 6:

-#  **method=TM_SQDIFF**

    \f[R(x,y)= \sum _{x',y'} (T(x',y')-I(x+x',y+y'))^2\f]

-#  **method=TM_SQDIFF_NORMED**

    \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y')-I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]

-#  **method=TM_CCORR**

    \f[R(x,y)= \sum _{x',y'} (T(x',y')  \cdot I(x+x',y+y'))\f]

-#  **method=TM_CCORR_NORMED**

    \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]

-#  **method=TM_CCOEFF**

    \f[R(x,y)= \sum _{x',y'} (T'(x',y')  \cdot I'(x+x',y+y'))\f]

    where

    \f[\begin{array}{l} T'(x',y')=T(x',y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} T(x'',y'') \\ I'(x+x',y+y')=I(x+x',y+y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} I(x+x'',y+y'') \end{array}\f]

-#  **method=TM_CCOEFF_NORMED**

    \f[R(x,y)= \frac{ \sum_{x',y'} (T'(x',y') \cdot I'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x',y+y')^2} }\f]

Code
----

-   **What does this program do?**
    -   Loads an input image, an image patch (*template*), and optionally a mask
    -   Perform a template matching procedure by using the OpenCV function **matchTemplate()**
        with any of the 6 matching methods described before. The user can choose the method by
        entering its selection in the Trackbar.  If a mask is supplied, it will only be used for
        the methods that support masking
    -   Normalize the output of the matching procedure
    -   Localize the location with higher matching probability
    -   Draw a rectangle around the area corresponding to the highest match

@add_toggle_cpp

-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp)
-   **Code at glance:**
    @include samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp

@end_toggle

@add_toggle_java

-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java)
-   **Code at glance:**
    @include samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java

@end_toggle

@add_toggle_python

-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/imgProc/match_template/match_template.py)
-   **Code at glance:**
    @include samples/python/tutorial_code/imgProc/match_template/match_template.py

@end_toggle

Explanation
-----------

-  Declare some global variables, such as the image, template and result matrices, as well as the
    match method and the window names:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp declare
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java declare
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py global_variables
    @end_toggle

-  Load the source image, template, and optionally, if supported for the matching method, a mask:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp load_image
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java load_image
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py load_image
    @end_toggle

-  Create the Trackbar to enter the kind of matching method to be used. When a change is detected
    the callback function is called.

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp create_trackbar
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java create_trackbar
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py create_trackbar
    @end_toggle

-  Let's check out the callback function. First, it makes a copy of the source image:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp copy_source
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java copy_source
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py copy_source
    @end_toggle

-  Perform the template matching operation. The arguments are naturally the input image **I**,
    the template **T**, the result **R** and the match_method (given by the Trackbar),
    and optionally the mask image **M**.

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp match_template
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java match_template
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py match_template
    @end_toggle

-  We normalize the results:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp normalize
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java normalize
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py normalize
    @end_toggle

-  We localize the minimum and maximum values in the result matrix **R** by using **minMaxLoc()**.

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp best_match
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java best_match
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py best_match
    @end_toggle

-  For the first two methods ( TM_SQDIFF and MT_SQDIFF_NORMED ) the best match are the lowest
    values. For all the others, higher values represent better matches. So, we save the
    corresponding value in the **matchLoc** variable:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp match_loc
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java match_loc
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py match_loc
    @end_toggle

-  Display the source image and the result matrix. Draw a rectangle around the highest possible
    matching area:

    @add_toggle_cpp
        @snippet samples/cpp/tutorial_code/Histograms_Matching/MatchTemplate_Demo.cpp imshow
    @end_toggle

    @add_toggle_java
        @snippet samples/java/tutorial_code/ImgProc/tutorial_template_matching/MatchTemplateDemo.java imshow
    @end_toggle

    @add_toggle_python
        @snippet samples/python/tutorial_code/imgProc/match_template/match_template.py imshow
    @end_toggle

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
