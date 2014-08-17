ColorConstancy
==============

.. highlight:: cpp

Color constancy is the ability to measure colors of objects independent of the color of the light source. Obtaining color constancy is of importance for many computer vision applications such as image retrieval, image classification, color object recognition and object tracking [JW07]_.

Grey-Edge
---------

Grey Edge color constancy algorithm is based on low-level image features [JW07]_. This framework also includes well known algorithms like Grey-World, max-RGB and Shades of Grey.

.. ocv:function::  void greyEdge(InputArray _src, OutputArray _dst, int diff_order = 1, int mink_norm = 5, int sigma = 2)

    :param src: Input 8-bit 3-channel image.

    :param dst: Output 8-bit 3-channel image.

    :param diff_order: Diff_order = 1 or 2, for 1st-order or 2nd-order derivative.

    :param mink_norm: Minkowski norm can be any number between 1 and infinity.

    :param sigma: Filter size.

    :param Algorithms:

    * **Grey-World** - diff_order = 0, mink_norm = 1, sigma = 0

    * **Max-RGB** - diff_order = 0, mink_norm = -1, sigma = 0

    * **Shades of Grey** - diff_order = 0, mink_norm = 5, sigma = 0

    * **General Grey-World** - diff_order = 0, mink_norm = 5, sigma = 2

    * **Grey-Edge** - diff_order = 1, mink_norm = 5, sigma = 2

    * **2nd Order Grey-Edge** - diff_order = 2, mink_norm = 5, sigma = 2

Weighted Grey-Edge
------------------

Weighted Grey-Edge algorithm improves the performance of edge-based color constancy by computing a weighted average of the edges [AG12]_.

.. ocv:function::  void weightedGreyEdge(InputArray _src, OutputArray _dst, int kappa = 10, int mink_norm = 5, int sigma = 2)

    :param src: Input 8-bit 3-channel image.

    :param dst: Output 8-bit 3-channel image.

    :param kappa: Kappa determines the weight given to the weight-map.

    :param mink_norm: Minkowski norm can be any number between 1 and infinity.

    :param sigma: Filter size.

.. [JW07] J. van de Weijer, T. Gevers and A. Gijsenij, "Edge-Based Color Constancy", IEEE Transactions on Image Processing, 2007.

.. [AG12] A. Gijsenij, T. Gevers and J. van de Weijer, "Improving Color Constancy by Photometric Edge Weighting", IEEE Transactions on Pattern Analysis and Machine Intelligence(PAMI), 2012.
