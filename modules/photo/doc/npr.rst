Non-Photorealistic Rendering
============================

.. highlight:: cpp

edgePreservingFilter
--------------------

Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing filters are used in many different applications.

.. ocv:function:: void edgePreservingFilter(InputArray src, OutputArray dst, int flags = 1, float sigma_s = 60, float sigma_r = 0.4f)

    :param src: Input 8-bit 3-channel image.

    :param dst: Output 8-bit 3-channel image.

    :param flags: Edge preserving filters:

            * **RECURS_FILTER**

            * **NORMCONV_FILTER**

    :param sigma_s: Range between 0 to 200.

    :param sigma_r: Range between 0 to 1.


detailEnhance
-------------
This filter enhances the details of a particular image.

.. ocv:function:: void detailEnhance(InputArray src, OutputArray dst, float sigma_s = 10, float sigma_r = 0.15f)

    :param src: Input 8-bit 3-channel image.

    :param dst: Output image with the same size and type as  ``src``.

    :param sigma_s: Range between 0 to 200.

    :param sigma_r: Range between 0 to 1.


pencilSketch
------------
Pencil-like non-photorealistic line drawing

.. ocv:function:: void pencilSketch(InputArray src, OutputArray dst1, OutputArray dst2, float sigma_s = 60, float sigma_r = 0.07f, float shade_factor = 0.02f)

    :param src: Input 8-bit 3-channel image.

    :param dst1: Output 8-bit 1-channel image.

    :param dst2: Output image with the same size and type as  ``src``.

    :param sigma_s: Range between 0 to 200.

    :param sigma_r: Range between 0 to 1.

    :param shade_factor: Range between 0 to 0.1.


stylization
-----------
Stylization aims to produce digital imagery with a wide variety of effects not focused on photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low contrast while preserving, or enhancing, high-contrast features.

.. ocv:function:: void stylization(InputArray src, OutputArray dst, float sigma_s = 60, float sigma_r = 0.45f)

    :param src: Input 8-bit 3-channel image.

    :param dst: Output image with the same size and type as  ``src``.

    :param sigma_s: Range between 0 to 200.

    :param sigma_r: Range between 0 to 1.
