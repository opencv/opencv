Image Gradients {#tutorial_js_gradients}
===============

Goal
----

-   Find Image gradients, edges etc
-   We will learn following functions : **cv.Sobel()**, **cv.Scharr()**, **cv.Laplacian()** etc

Theory
------

OpenCV provides three types of gradient filters or High-pass filters, Sobel, Scharr and Laplacian.
We will see each one of them.

### 1. Sobel and Scharr Derivatives

Sobel operators is a joint Gausssian smoothing plus differentiation operation, so it is more
resistant to noise. You can specify the direction of derivatives to be taken, vertical or horizontal
(by the arguments, yorder and xorder respectively). You can also specify the size of kernel by the
argument ksize. If ksize = -1, a 3x3 Scharr filter is used which gives better results than 3x3 Sobel
filter. Please see the docs for kernels used.

We use the functions: **cv.Sobel (src, dst, ddepth, dx, dy, ksize = 3, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth(see cv.combinations); in the case of 8-bit input images it will result in truncated derivatives.
@param dx          order of the derivative x.
@param dy          order of the derivative y.
@param ksize       size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
@param scale       optional scale factor for the computed derivative values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

**cv.Scharr (src, dst, ddepth, dx, dy, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth(see cv.combinations).
@param dx          order of the derivative x.
@param dy          order of the derivative y.
@param scale       optional scale factor for the computed derivative values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

Try it
------

\htmlonly
<iframe src="../../js_gradients_Sobel.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

### 2. Laplacian Derivatives

It calculates the Laplacian of the image given by the relation,
\f$\Delta src = \frac{\partial ^2{src}}{\partial x^2} + \frac{\partial ^2{src}}{\partial y^2}\f$ where
each derivative is found using Sobel derivatives. If ksize = 1, then following kernel is used for
filtering:

\f[kernel = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0  \end{bmatrix}\f]

We use the function: **cv.Laplacian (src, dst, ddepth, ksize = 1, scale = 1, delta = 0, borderType = cv.BORDER_DEFAULT)**
@param src         input image.
@param dst         output image of the same size and the same number of channels as src.
@param ddepth      output image depth.
@param ksize       aperture size used to compute the second-derivative filters.
@param scale       optional scale factor for the computed Laplacian values.
@param delta       optional delta value that is added to the results prior to storing them in dst.
@param borderType  pixel extrapolation method(see cv.BorderTypes).

Try it
------

\htmlonly
<iframe src="../../js_gradients_Laplacian.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

One Important Matter!
---------------------

In our last example, output datatype is cv.CV_8U. But there is a slight problem with
that. Black-to-White transition is taken as Positive slope (it has a positive value) while
White-to-Black transition is taken as a Negative slope (It has negative value). So when you convert
data to cv.CV_8U, all negative slopes are made zero. In simple words, you miss that edge.

If you want to detect both edges, better option is to keep the output datatype to some higher forms,
like cv.CV_16S, cv.CV_64F etc, take its absolute value and then convert back to cv.CV_8U.
Below code demonstrates this procedure for a horizontal Sobel filter and difference in results.

Try it
------

\htmlonly
<iframe src="../../js_gradients_absSobel.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly