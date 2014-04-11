Decolorization
==============

.. highlight:: cpp

decolor
-------

Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized black-and-white photograph rendering, and in many single channel image processing applications [CL12]_.

.. ocv:function:: void decolor( InputArray src, OutputArray grayscale, OutputArray color_boost )

    :param src: Input 8-bit 3-channel image.

    :param grayscale: Output 8-bit 1-channel image.

    :param color_boost: Output 8-bit 3-channel image.

This function is to be applied on color images.

.. [CL12] Cewu Lu, Li Xu, Jiaya Jia, "Contrast Preserving Decolorization", IEEE International Conference on Computational Photography (ICCP), 2012.
