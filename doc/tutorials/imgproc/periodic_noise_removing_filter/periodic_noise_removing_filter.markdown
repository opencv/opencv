Periodic Noise Removing Filter {#tutorial_periodic_noise_removing_filter}
==========================

@tableofcontents

@prev_tutorial{tutorial_anisotropic_image_segmentation_by_a_gst}

|    |    |
| -: | :- |
| Original author | Karpushin Vladislav |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn:

-   how to remove periodic noise in the Fourier domain

Theory
------

@note The explanation is based on the book @cite gonzalez. The image on this page is a real world image.

Periodic noise produces spikes in the Fourier domain that can often be detected by visual analysis.

### How to remove periodic noise in the Fourier domain?

Periodic noise can be reduced significantly via frequency domain filtering. On this page we use a notch reject filter with an appropriate radius to completely enclose the noise spikes in the Fourier domain. The notch filter rejects frequencies in predefined neighborhoods around a center frequency. The number of notch filters is arbitrary. The shape of the notch areas can also be arbitrary (e.g. rectangular or circular). On this page we use three circular shape notch reject filters. Power spectrum densify of an image is used for the noise spike’s visual detection.

Source code
-----------

You can find source code in the `samples/cpp/tutorial_code/ImgProc/periodic_noise_removing_filter/periodic_noise_removing_filter.cpp` of the OpenCV source code library.

@include samples/cpp/tutorial_code/ImgProc/periodic_noise_removing_filter/periodic_noise_removing_filter.cpp

Explanation
-----------

Periodic noise reduction by frequency domain filtering consists of power spectrum density calculation (for the noise spikes visual detection), notch reject filter synthesis and frequency filtering:
@snippet samples/cpp/tutorial_code/ImgProc/periodic_noise_removing_filter/periodic_noise_removing_filter.cpp main

A function calcPSD() calculates power spectrum density of an image:
@snippet samples/cpp/tutorial_code/ImgProc/periodic_noise_removing_filter/periodic_noise_removing_filter.cpp calcPSD

A function synthesizeFilterH() forms a transfer function of an ideal circular shape notch reject filter according to a center frequency and a radius:
@snippet samples/cpp/tutorial_code/ImgProc/periodic_noise_removing_filter/periodic_noise_removing_filter.cpp synthesizeFilterH

A function filter2DFreq() filters an image in the frequency domain. The functions fftshift() and filter2DFreq() are copied from the tutorial @ref tutorial_out_of_focus_deblur_filter "Out-of-focus Deblur Filter".

Result
------

The figure below shows an image heavily corrupted by periodical noise of various frequencies.
![Image corrupted by periodic noise](images/period_input.jpg)

The noise components are easily seen as bright dots (spikes) in the Power spectrum density shown in the figure below.
![Power spectrum density showing periodic noise](images/period_psd.jpg)

The figure below shows a notch reject filter with an appropriate radius to completely enclose the noise spikes.
![Notch reject filter](images/period_filter.jpg)

The result of processing the image with the notch reject filter is shown below.
![Result of filtering](images/period_output.jpg)

The improvement is quite evident. This image contains significantly less visible periodic noise than the original image.

You can also find a quick video demonstration of this filtering idea on [YouTube](https://youtu.be/Qne51TcWwAc).
@youtube{Qne51TcWwAc}
