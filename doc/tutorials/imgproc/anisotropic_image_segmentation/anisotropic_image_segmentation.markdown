Anisotropic image segmentation by a gradient structure tensor {#tutorial_anisotropic_image_segmentation_by_a_gst}
==========================

@prev_tutorial{tutorial_motion_deblur_filter}
@next_tutorial{tutorial_periodic_noise_removing_filter}

Goal
----

In this tutorial you will learn:

-   what the gradient structure tensor is
-   how to estimate orientation and coherency of an anisotropic image by a gradient structure tensor
-   how to segment an anisotropic image with a single local orientation by a gradient structure tensor

Theory
------

@note The explanation is based on the books @cite jahne2000computer, @cite bigun2006vision and @cite van1995estimators. Good physical explanation of a gradient structure tensor is given in @cite yang1996structure. Also, you can refer to a wikipedia page [Structure tensor].
@note A anisotropic image on this page is a real world  image.

### What is the gradient structure tensor?

In mathematics, the gradient structure tensor (also referred to as the second-moment matrix, the second order moment tensor, the inertia tensor, etc.) is a matrix derived from the gradient of a function. It summarizes the predominant directions of the gradient in a specified neighborhood of a point, and the degree to which those directions are coherent (coherency). The gradient structure tensor is widely used in image processing and computer vision for 2D/3D image segmentation, motion detection, adaptive filtration, local image features detection, etc.

Important features of anisotropic images include orientation and coherency of a local anisotropy. In this paper we will show how to estimate orientation and coherency, and how to segment an anisotropic image with a single local orientation by a gradient structure tensor.

The gradient structure tensor of an image is a 2x2 symmetric matrix. Eigenvectors of the gradient structure tensor indicate local orientation, whereas eigenvalues give coherency (a measure of anisotropism).

The gradient structure tensor \f$J\f$ of an image \f$Z\f$ can be written as:

\f[J = \begin{bmatrix}
J_{11} & J_{12}  \\
J_{12} & J_{22}
\end{bmatrix}\f]

where \f$J_{11} = M[Z_{x}^{2}]\f$, \f$J_{22} = M[Z_{y}^{2}]\f$, \f$J_{12} = M[Z_{x}Z_{y}]\f$ - components of the tensor, \f$M[]\f$ is a symbol of mathematical expectation (we can consider this operation as averaging in a window w), \f$Z_{x}\f$ and \f$Z_{y}\f$ are partial derivatives of an image \f$Z\f$ with respect to \f$x\f$ and \f$y\f$.

The eigenvalues of the tensor can be found in the below formula:
\f[\lambda_{1,2} = J_{11} + J_{22} \pm \sqrt{(J_{11} - J_{22})^{2} + 4J_{12}^{2}}\f]
where \f$\lambda_1\f$ - largest eigenvalue, \f$\lambda_2\f$ - smallest eigenvalue.

### How to estimate orientation and coherency of an anisotropic image by gradient structure tensor?

The orientation of an anisotropic image:
\f[\alpha = 0.5arctg\frac{2J_{12}}{J_{22} - J_{11}}\f]

Coherency:
\f[C = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2}\f]

The coherency ranges from 0 to 1. For ideal local orientation (\f$\lambda_2\f$ = 0, \f$\lambda_1\f$ > 0) it is one, for an isotropic gray value structure (\f$\lambda_1\f$ = \f$\lambda_2\f$ \> 0) it is zero.

Source code
-----------

You can find source code in the `samples/cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp` of the OpenCV source code library.

@add_toggle_cpp
    @include cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp
@end_toggle

@add_toggle_python
    @include samples/python/tutorial_code/imgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.py
@end_toggle

Explanation
-----------
An anisotropic image segmentation algorithm consists of a gradient structure tensor calculation, an orientation calculation, a coherency calculation and an orientation and coherency thresholding:

@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp main
@end_toggle

@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.py main
@end_toggle

A function calcGST() calculates orientation and coherency by using a gradient structure tensor. An input parameter w defines a window size:

@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp calcGST
@end_toggle

@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.py calcGST
@end_toggle


The below code applies a thresholds LowThr and HighThr to image orientation and a threshold C_Thr to image coherency calculated by the previous function. LowThr and HighThr define orientation range:

@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp thresholding
@end_toggle

@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.py thresholding
@end_toggle


And finally we combine thresholding results:

@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.cpp combining
@end_toggle

@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/anisotropic_image_segmentation/anisotropic_image_segmentation.py combining
@end_toggle


Result
------

Below you can see the real anisotropic image with single direction:
![Anisotropic image with the single direction](images/gst_input.jpg)

Below you can see the orientation and coherency of the anisotropic image:
![Orientation](images/gst_orientation.jpg)
![Coherency](images/gst_coherency.jpg)

Below you can see the segmentation result:
![Segmentation result](images/gst_result.jpg)

The result has been computed with w = 52, C_Thr = 0.43, LowThr = 35, HighThr = 57. We can see that the algorithm selected only the areas with one single direction.

References
------
- [Structure tensor] - structure tensor description on the wikipedia

<!-- invisible references list -->
[Structure tensor]: https://en.wikipedia.org/wiki/Structure_tensor
