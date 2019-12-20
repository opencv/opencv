Fourier Transform {#tutorial_js_fourier_transform}
=================

Goal
----

-   To find the Fourier Transform of images using OpenCV
-   Some applications of Fourier Transform
-   We will learn following functions : **cv.dft()** etc

Theory
------

Fourier Transform is used to analyze the frequency characteristics of various filters. For images,
**2D Discrete Fourier Transform (DFT)** is used to find the frequency domain. A fast algorithm
called **Fast Fourier Transform (FFT)** is used for calculation of DFT. Details about these can be
found in any image processing or signal processing textbooks.

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
much changes in amplitude, it is a low frequency component.

Performance of DFT calculation is better for some array size. It is fastest when array size is power
of two. The arrays whose size is a product of 2’s, 3’s, and 5’s are also processed quite
efficiently. So if you are worried about the performance of your code, you can modify the size of
the array to any optimal size (by padding zeros) before finding DFT. OpenCV provides a function, **cv.getOptimalDFTSize()** for this.

Now we will see how to find the Fourier Transform.

Fourier Transform in OpenCV
---------------------------

Performance of DFT calculation is better for some array size. It is fastest when array size is power of two. The arrays whose size is a product of 2’s, 3’s, and 5’s are also processed quite efficiently. So if you are worried about the performance of your code, you can modify the size of the array to any optimal size (by padding zeros). So how do we find this optimal size ? OpenCV provides a function, cv.getOptimalDFTSize() for this.

We use the functions: **cv.dft (src, dst, flags = 0, nonzeroRows = 0)**

@param src           input array that could be real or complex.
@param dst           output array whose size and type depends on the flags.
@param flags         transformation flags, representing a combination of the cv.DftFlags
@param nonzeroRows   when the parameter is not zero, the function assumes that only the first nonzeroRows rows of the input array (DFT_INVERSE is not set) or only the first nonzeroRows of the output array (DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the rows more efficiently and save some time; this technique is very useful for calculating array cross-correlation or convolution using DFT.

**cv.getOptimalDFTSize (vecsize)**

@param vecsize   vector size.

**cv.copyMakeBorder (src, dst, top, bottom, left, right, borderType, value = new cv.Scalar())**

@param src           input array that could be real or complex.
@param dst           output array whose size and type depends on the flags.
@param top           parameter specifying how many top pixels in each direction from the source image rectangle to extrapolate.
@param bottom        parameter specifying how many bottom pixels in each direction from the source image rectangle to extrapolate.
@param left          parameter specifying how many left pixels in each direction from the source image rectangle to extrapolate.
@param right         parameter specifying how many right pixels in each direction from the source image rectangle to extrapolate.
@param borderType    border type.
@param value         border value if borderType == cv.BORDER_CONSTANT.

**cv.magnitude (x, y, magnitude)**

@param x          floating-point array of x-coordinates of the vectors.
@param y          floating-point array of y-coordinates of the vectors; it must have the same size as x.
@param magnitude  output array of the same size and type as x.

**cv.split (m, mv)**

@param m     input multi-channel array.
@param mv    output vector of arrays; the arrays themselves are reallocated, if needed.

**cv.merge (mv, dst)**

@param mv      input vector of matrices to be merged; all the matrices in mv must have the same size and the same depth.
@param dst     output array of the same size and the same depth as mv[0]; The number of channels will be the total number of channels in the matrix array.

Try it
------

\htmlonly
<iframe src="../../js_fourier_transform_dft.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly