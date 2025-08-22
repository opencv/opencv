Chromatic Aberration Correction {#tutorial_py_chromatic_aberration}
================

Goal
----

In this chapter, we will learn how to
    -   Calibrate your camera and get the coefficients to correct lateral chromatic aberration.
    -   Export these coefficients that model the red/blue channel misalignments.
    -   Correct images using functions in OpenCV.

Basics
------

Lateral chromatic aberration appears when different wavelengths focus to slightly different image positions. This results in red/blue fringes at the edges of the photo, and is particularly common in old or low-quality cameras and lenses. It is a property of a lens and appears in every image taken with that camera and lens.

We treat lateral chromatic aberration as a geometric distortion of red and blue channels comapred to the reference green, and aim to estimate a mapping that brings the red and blue channels in alignment with green.

The correction is mainly based on the paper of Rudakova et al. on the lateral chromatic aberration. The misalignment in each channel is modeled as a polynomial model of some degree. The distance between the precise locations of centres in red/blue and green channels is minimized with a warp of these centres.

The paper also proposed to use the calibration pattern of black discs, many more than the polynomial model coefficients count to get a proper fit. The degree 11 is used, but we found that smaller model degrees could also produce similar level of accuracy and perform much better.

Calibration
------

To create a model of the misalignments of the channels, we use the following calibration procedure:

Firstly, print out the calibration photo available in opencv_extra/testdata/cv/cameracalibration/chromatic_aberration/chromatic_aberration_pattern_a3.png. The photo is a grid of black discs on a white background, and as the chromatic aberration fringes appear on the edges of objects in the photo, we will be able to see many different misalignments and model them precisely.

After that, take a photo of the printed out calibration grid using your camera. Make sure that al of the discs are in the photo, and that the grid takes up sa much place as possible, as the chromatic aberration is the biggest at the edges and corners of the photo. You should be able to visually check whether there is visible chromatic aberration.

To calibrate the camera, use the app [ca_calibration.py](opencv/apps/chromatic-aberration-calibration/ca_calibration.py). The app can be used as follows:

ca_calibration.py calibrate [-h] [--degree DEGREE] --coeffs_file YAML image
ca_calibration.py correct   [-h] --coeffs_file YAML [-o OUTPUT] image
ca_calibration.py full      [-h] [--degree DEGREE] --coeffs_file YAML [-o OUTPUT] image
ca_calibration.py scan      [-h] --degree_range k0 k1 image

Calibrate estimates polynomial coefficients and outputs them to a YAML file to be used with correction functions.

What it does (under the hood):
- Splits BGR, finds disk centers per channel at sub-pixel precision.
- Pairs centers to green via KD-tree NN with a 30 px cap.
- Builds monomial terms up to `--degree` and solves least squares, then refines with another optimization algorithm.
- Saves a YAML with:
  - `image_width`, `image_height`
  - `red_channel/blue_channel`: `coeffs_x`, `coeffs_y` (length `M=(d+1)(d+2)/2`), and `rms` residuals.

Scan sweeps polynomial degree range and compares quality. Although higher degrees should almost always model the aberration better, lower degrees can be much faster.

What it does (under the hood):
- Runs calibration for each degree in k0,..,k1 inclusive to fit models for each degree.
- Extracts full disk contours per channel.
- Warps R/B contours toward G using each degree’s polynomials and measures nearest-neighbor distances.
- Prints a table of max / mean / std distances (in pixels) for red and blue.
- The user can then choose what degree works best and calibrate the camera with that specific degree.


Code
----



Additional Resources
--------------------


1.  Rudakova, V., Monasse, P. (2014). Precise Correction of Lateral Chromatic Aberration in Images. In: Klette, R., Rivera, M., Satoh, S. (eds) Image and Video Technology. PSIVT 2013. Lecture Notes in Computer Science, vol 8333. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-53842-1_2
