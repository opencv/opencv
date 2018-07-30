Out-of-focus Deblur Filter {#tutorial_out_of_focus_deblur_filter}
==========================

Goal
----

In this tutorial you will learn:

-   what is a degradation image model
-   what is PSF of out-of-focus image
-   how to restore a blurred image
-   what is Wiener filter

Theory
------

@note The explanation is based on the books @cite gonzalez and @cite gruzman. Also, you can refer to Matlab's tutorial [Image Deblurring in Matlab] and an article [SmartDeblur].
@note An out-of-focus image on this page is a real world  image. An out-of-focus was done manually by camera optics.

### What is a degradation image model?

A mathematical model of the image degradation in frequency domain representation is:

\f[S = H\cdot U + N\f]

where
\f$S\f$ is a spectrum of blurred (degraded) image,
\f$U\f$ is a spectrum of original true (undegraded) image,
\f$H\f$ is frequency response of point spread function (PSF),
\f$N\f$ is a spectrum of additive noise.

Circular PSF is a good approximation of out-of-focus distortion. Such PSF is specified by only one parameter - radius \f$R\f$. Circular PSF is used in this work.

![Circular point spread function](psf.png)

### How to restore an blurred image?

The objective of restoration (deblurring) is to obtain an estimate of the original image. Restoration formula in frequency domain is:

\f[U' = H_w\cdot S\f]

where
\f$U'\f$ is spectrum of estimation of original image \f$U\f$,
\f$H_w\f$ is restoration filter, for example, Wiener filter.

### What is Wiener filter?

Wiener filter is a way to restore a blurred image. Let's suppose that PSF is a real and symmetric signal, a power spectrum of the original true image and noise are not known,
then simplified Wiener formula is:

\f[H_w = \frac{H}{|H|^2+\frac{1}{SNR}} \f]

where
\f$SNR\f$ is signal-to-noise ratio.

So, in order to recover an out-of-focus image by Wiener filter, it needs to know \f$SNR\f$ and \f$R\f$ of circular PSF.


Source code
-----------

You can find source code in the `samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp` of the OpenCV source code library.

@include cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp

Explanation
-----------

An out-of-focus image recovering algorithm consists of PSF generation, Wiener filter generation and filtering an blurred image in frequency domain:
@snippet samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp main

A function calcPSF() forms an circular PSF according to input parameter radius \f$R\f$:
@snippet samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp calcPSF

A function calcWnrFilter() synthesizes simplified Wiener filter \f$H_w\f$ according to formula described above:
@snippet samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp calcWnrFilter

A function fftshift() rearranges PSF. This code was just copied from tutorial @ref tutorial_discrete_fourier_transform "Discrete Fourier Transform":
@snippet samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp fftshift

A function filter2DFreq() filters an blurred image in frequency domain:
@snippet samples/cpp/tutorial_code/ImgProc/out_of_focus_deblur_filter/out_of_focus_deblur_filter.cpp filter2DFreq

Result
------

Below you can see real out-of-focus image:
![Out-of-focus image](images/original.jpg)


Below result was done by \f$R\f$ = 53 and \f$SNR\f$ = 5200 parameters:
![The restored (deblurred) image](images/recovered.jpg)

The Wiener filter was used, values of \f$R\f$ and \f$SNR\f$ were selected manually to give the best possible visual result.
We can see that the result is not perfect, but it gives us a hint to the image content. With some difficulty, the text is readable.

@note The parameter \f$R\f$ is the most important. So you should adjust \f$R\f$ first, then \f$SNR\f$.
@note Sometimes you can observe the ringing effect in an restored image. This effect can be reduced by several methods. For example, you can taper input image edges.

You can also find a quick video demonstration of this on
[YouTube](https://youtu.be/0bEcE4B0XP4).
@youtube{0bEcE4B0XP4}

References
------
- [Image Deblurring in Matlab] - Image Deblurring in Matlab
- [SmartDeblur] - SmartDeblur site

<!-- invisible references list -->
[Digital Image Processing]: http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/Digital_Image_Processing_2ndEd.pdf
[Image Deblurring in Matlab]: https://www.mathworks.com/help/images/image-deblurring.html
[SmartDeblur]: http://yuzhikov.com/articles/BlurredImagesRestoration1.htm
