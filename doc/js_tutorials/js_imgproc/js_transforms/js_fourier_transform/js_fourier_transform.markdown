Fourier Transform {#tutorial_js_fourier_transform}
=================

Goal
----

In this section, we will learn
    -   To find the Fourier Transform of images using OpenCV
    -   Some applications of Fourier Transform
    -   We will see following functions : **cv.dft()**, **cv.idft()** etc

Theory
------

Fourier Transform is used to analyze the frequency characteristics of various filters. For images,
**2D Discrete Fourier Transform (DFT)** is used to find the frequency domain. A fast algorithm
called **Fast Fourier Transform (FFT)** is used for calculation of DFT. Details about these can be
found in any image processing or signal processing textbooks. Please see Additional Resources_
section.

For a sinusoidal signal, \f$x(t) = A \sin(2 \pi ft)\f$, we can say \f$f\f$ is the frequency of signal, and
if its frequency domain is taken, we can see a spike at \f$f\f$. If signal is sampled to form a discrete
signal, we get the same frequency domain, but is periodic in the range \f$[- \pi, \pi]\f$ or \f$[0,2\pi]\f$
(or \f$[0,N]\f$ for N-point DFT). You can consider an image as a signal which is sampled in two
directions. So taking fourier transform in both X and Y directions gives you the frequency
representation of image.

More intuitively, for the sinusoidal signal, if the amplitude varies so fast in short time, you can
say it is a high frequency signal. If it varies slowly, it is a low frequency signal. You can extend
the same idea to images. Where does the amplitude varies drastically in images ? At the edge points,
or noises. So we can say, edges and noises are high frequency contents in an image. If there is no
much changes in amplitude, it is a low frequency component. ( Some links are added to
Additional Resources_ which explains frequency transform intuitively with examples).

Performance of DFT calculation is better for some array size. It is fastest when array size is power
of two. The arrays whose size is a product of 2’s, 3’s, and 5’s are also processed quite
efficiently. So if you are worried about the performance of your code, you can modify the size of
the array to any optimal size (by padding zeros) before finding DFT. OpenCV provides a function, **cv.getOptimalDFTSize()** for this. 

Now we will see how to find the Fourier Transform.

Fourier Transform in OpenCV
---------------------------

@note cv.dft() is not in the white list.

We use the functions: **cv.dft (src, dst, flags = 0, nonzeroRows = 0)** 

@param src           input array that could be real or complex.
@param dst           output array whose size and type depends on the flags.
@param flags         transformation flags, representing a combination of the cv.DftFlags
@param nonzeroRows   when the parameter is not zero, the function assumes that only the first nonzeroRows rows of the input array (DFT_INVERSE is not set) or only the first nonzeroRows of the output array (DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the rows more efficiently and save some time; this technique is very useful for calculating array cross-correlation or convolution using DFT.

**cv.idft (src, dst, flags = 0, nonzeroRows = 0)** 

@param src           input floating-point real or complex array.
@param dst           output array whose size and type depends on the flags.
@param flags         operation flags (see cv.dft and cv.DftFlags).
@param nonzeroRows   number of dst rows to process; the rest of the rows have undefined content.

**cv.getOptimalDFTSize (vecsize)**

**cv.copyMakeBorder (src, dst, top, bottom, left, right, borderType, value = Scalar())**

**cv.magnitude (x, y, magnitude)**

cv.merge

cv.split

cv.log

cv.normalize

@note You can also use **cv.cartToPolar()** which returns both magnitude and phase in a single shot


Additional Resources
--------------------

-#  [An Intuitive Explanation of Fourier
    Theory](http://cns-alumni.bu.edu/~slehar/fourier/fourier.html) by Steven Lehar
2.  [Fourier Transform](http://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm) at HIPR
3.  [What does frequency domain denote in case of images?](http://dsp.stackexchange.com/q/1637/818)
