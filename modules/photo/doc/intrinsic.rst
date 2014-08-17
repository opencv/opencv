Intrinsic Image Decomposition
=============================

.. highlight:: cpp

intrinsicDecompose
------------------

Intrinsic Image Decomposition is referred as the separation of illumination (shading) and reflectance components from an input photograph. Many computer vision algorithms, such as segmentation, recognition, and motion estimation are confounded by illumination effects in the image. The performance of these algorithms may beneﬁt substantially from reliable estimation of illumination-invariant material properties for all objects in the scene [JS11]_.

.. ocv:function:: void intrinsicDecompose(InputArray _src, OutputArray _ref, OutputArray _shade, int window = 3, int no_of_iter = 100, float rho = 1.9)

    :param src: Input 8-bit 3-channel image.

    :param shading: Output 8-bit 1-channel image.

    :param reflectance: Output 8-bit 3-channel image.

    :param window: Local window of the source image. Default value is 3.

    :no_of_iter: Number of iterations required to converge. Default value is 100.

    :rho: Default value is 1.9

.. [JS11] Jianbing Shen, Xiaoshan Yang, Yunde Jia, Xuelong Li, "Intrinsic images using optimization", CVPR 2011
