Motion Deblur Filter {#tutorial_motion_deblur_filter}
==========================

@prev_tutorial{tutorial_out_of_focus_deblur_filter}
@next_tutorial{tutorial_anisotropic_image_segmentation_by_a_gst}

Goal
----

In this tutorial you will learn:

-   what the PSF of a motion blur image is
-   how to restore a motion blur image

Theory
------

For the degradation image model theory and the Wiener filter theory you can refer to the tutorial @ref tutorial_out_of_focus_deblur_filter "Out-of-focus Deblur Filter".
On this page only a linear motion blur distortion is considered. The motion blur image on this page is a real world image. The blur was caused by a moving subject.

### What is the PSF of a motion blur image?

The point spread function (PSF) of a linear motion blur distortion is a line segment. Such a PSF is specified by two parameters: \f$LEN\f$ is the length of the blur and \f$THETA\f$ is the angle of motion.

![Point spread function of a linear motion blur distortion](images/motion_psf.png)

### How to restore a blurred image?

On this page the Wiener filter is used as the restoration filter, for details you can refer to the tutorial @ref tutorial_out_of_focus_deblur_filter "Out-of-focus Deblur Filter".
In order to synthesize the Wiener filter for a motion blur case, it needs to specify the signal-to-noise ratio (\f$SNR\f$), \f$LEN\f$ and \f$THETA\f$ of the PSF.

Source code
-----------

You can find source code in the `samples/cpp/tutorial_code/ImgProc/motion_deblur_filter/motion_deblur_filter.cpp` of the OpenCV source code library.

@include cpp/tutorial_code/ImgProc/motion_deblur_filter/motion_deblur_filter.cpp

Explanation
-----------

A motion blur image recovering algorithm consists of PSF generation, Wiener filter generation and filtering a blurred image in a frequency domain:
@snippet samples/cpp/tutorial_code/ImgProc/motion_deblur_filter/motion_deblur_filter.cpp main

A function calcPSF() forms a PSF according to input parameters \f$LEN\f$ and \f$THETA\f$ (in degrees):
@snippet samples/cpp/tutorial_code/ImgProc/motion_deblur_filter/motion_deblur_filter.cpp calcPSF

A function edgetaper() tapers the input image’s edges in order to reduce the ringing effect in a restored image:
@snippet samples/cpp/tutorial_code/ImgProc/motion_deblur_filter/motion_deblur_filter.cpp edgetaper

The functions calcWnrFilter(), fftshift() and filter2DFreq() realize an image filtration by a specified PSF in the frequency domain.  The functions are copied from the tutorial
@ref tutorial_out_of_focus_deblur_filter "Out-of-focus Deblur Filter".

Result
------

Below you can see the real world image with motion blur distortion. The license plate is not readable on both cars. The red markers show the car’s license plate location.
![Motion blur image. The license plates are not readable](images/motion_original.jpg)


Below you can see the restoration result for the black car license plate. The result has been computed with \f$LEN\f$ = 125, \f$THETA\f$ = 0, \f$SNR\f$ = 700.
![The restored image of the black car license plate](images/black_car.jpg)

Below you can see the restoration result for the white car license plate. The result has been computed with \f$LEN\f$ = 78, \f$THETA\f$ = 15, \f$SNR\f$ = 300.
![The restored image of the white car license plate](images/white_car.jpg)

The values of \f$SNR\f$, \f$LEN\f$ and \f$THETA\f$ were selected manually to give the best possible visual result. The \f$THETA\f$ parameter coincides with the car’s moving direction, and the
\f$LEN\f$ parameter depends on the car’s moving speed.
The result is not perfect, but at least it gives us a hint of the image’s content. With some effort, the car license plate is now readable.

@note The parameters \f$LEN\f$ and \f$THETA\f$ are the most important. You should adjust \f$LEN\f$ and \f$THETA\f$ first, then \f$SNR\f$.

You can also find a quick video demonstration of a license plate recovering method
[YouTube](https://youtu.be/xSrE0hdhb4o).
@youtube{xSrE0hdhb4o}
