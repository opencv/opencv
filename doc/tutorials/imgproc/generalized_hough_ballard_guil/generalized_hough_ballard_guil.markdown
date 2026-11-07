Object detection with Generalized Ballard and Guil Hough Transform {#tutorial_generalized_hough_ballard_guil}
==================================================================

@tableofcontents

@prev_tutorial{tutorial_hough_circle}
@next_tutorial{tutorial_remap}

|    |    |
| -: | :- |
| Original author | Markus Heck |
| Compatibility | OpenCV >= 3.4 |

Goal
----

In this tutorial you will learn how to:

- Use @ref cv::GeneralizedHoughBallard and @ref cv::GeneralizedHoughGuil to detect an object

Example
-------

### What does this program do?

1. Load the image and template

![image](images/generalized_hough_mini_image.jpg)
![template](images/generalized_hough_mini_template.jpg)

2. Instantiate @ref cv::GeneralizedHoughBallard with the help of `createGeneralizedHoughBallard()`
3. Instantiate @ref cv::GeneralizedHoughGuil with the help of `createGeneralizedHoughGuil()`
4. Set the required parameters for both GeneralizedHough variants
5. Detect and show found results

@note
- Both variants can't be instantiated directly. Using the create methods is required.
- Guil Hough is very slow. Calculating the results for the "mini" files used in this tutorial
  takes only a few seconds. With image and template in a higher resolution, as shown below,
  my notebook requires about 5 minutes to calculate a result.

![image](images/generalized_hough_image.jpg)
![template](images/generalized_hough_template.jpg)

### Code

The complete code for this tutorial is shown below.
@include samples/cpp/tutorial_code/ImgTrans/generalizedHoughTransform.cpp

Explanation
-----------

### Load image, template and setup variables

@snippet samples/cpp/tutorial_code/ImgTrans/generalizedHoughTransform.cpp generalized-hough-transform-load-and-setup

The position vectors will contain the matches the detectors will find.
Every entry contains four floating point values:
position vector

- *[0]*: x coordinate of center point
- *[1]*: y coordinate of center point
- *[2]*: scale of detected object compared to template
- *[3]*: rotation of detected object in degree in relation to template

An example could look as follows: `[200, 100, 0.9, 120]`

### Setup parameters

@snippet samples/cpp/tutorial_code/ImgTrans/generalizedHoughTransform.cpp generalized-hough-transform-setup-parameters

Finding the optimal values can end up in trial and error and depends on many factors, such as the image resolution.

### Run detection

@snippet samples/cpp/tutorial_code/ImgTrans/generalizedHoughTransform.cpp generalized-hough-transform-run

As mentioned above, this step will take some time, especially with larger images and when using Guil.

### Draw results and show image

@snippet samples/cpp/tutorial_code/ImgTrans/generalizedHoughTransform.cpp generalized-hough-transform-draw-results

Result
------

![result image](images/generalized_hough_result_img.jpg)

The blue rectangle shows the result of @ref cv::GeneralizedHoughBallard and the green rectangles the results of @ref
cv::GeneralizedHoughGuil.

Getting perfect results like in this example is unlikely if the parameters are not perfectly adapted to the sample.
An example with less perfect parameters is shown below.
For the Ballard variant, only the center of the result is marked as a black dot on this image. The rectangle would be
the same as on the previous image.

![less perfect result](images/generalized_hough_less_perfect_result_img.jpg)
