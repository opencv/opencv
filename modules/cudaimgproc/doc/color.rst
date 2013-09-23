Color space processing
======================

.. highlight:: cpp



cuda::cvtColor
--------------
Converts an image from one color space to another.

.. ocv:function:: void cuda::cvtColor(InputArray src, OutputArray dst, int code, int dcn = 0, Stream& stream = Stream::Null())

    :param src: Source image with  ``CV_8U`` , ``CV_16U`` , or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image.

    :param code: Color space conversion code. For details, see  :ocv:func:`cvtColor` .

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

3-channel color spaces (like ``HSV``, ``XYZ``, and so on) can be stored in a 4-channel image for better performance.

.. seealso:: :ocv:func:`cvtColor`



cuda::demosaicing
-----------------
Converts an image from Bayer pattern to RGB or grayscale.

.. ocv:function:: void cuda::demosaicing(InputArray src, OutputArray dst, int code, int dcn = -1, Stream& stream = Stream::Null())

    :param src: Source image (8-bit or 16-bit single channel).

    :param dst: Destination image.

    :param code: Color space conversion code (see the description below).

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

The function can do the following transformations:

* Demosaicing using bilinear interpolation

    * ``COLOR_BayerBG2GRAY`` , ``COLOR_BayerGB2GRAY`` , ``COLOR_BayerRG2GRAY`` , ``COLOR_BayerGR2GRAY``

    * ``COLOR_BayerBG2BGR`` , ``COLOR_BayerGB2BGR`` , ``COLOR_BayerRG2BGR`` , ``COLOR_BayerGR2BGR``

* Demosaicing using Malvar-He-Cutler algorithm ([MHT2011]_)

    * ``COLOR_BayerBG2GRAY_MHT`` , ``COLOR_BayerGB2GRAY_MHT`` , ``COLOR_BayerRG2GRAY_MHT`` , ``COLOR_BayerGR2GRAY_MHT``

    * ``COLOR_BayerBG2BGR_MHT`` , ``COLOR_BayerGB2BGR_MHT`` , ``COLOR_BayerRG2BGR_MHT`` , ``COLOR_BayerGR2BGR_MHT``

.. seealso:: :ocv:func:`cvtColor`



cuda::swapChannels
------------------
Exchanges the color channels of an image in-place.

.. ocv:function:: void cuda::swapChannels(InputOutputArray image, const int dstOrder[4], Stream& stream = Stream::Null())

    :param image: Source image. Supports only ``CV_8UC4`` type.

    :param dstOrder: Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.

    :param stream: Stream for the asynchronous version.

The methods support arbitrary permutations of the original channels, including replication.



cuda::gammaCorrection
---------------------
Routines for correcting image color gamma.

.. ocv:function:: void cuda::gammaCorrection(InputArray src, OutputArray dst, bool forward = true, Stream& stream = Stream::Null())

    :param src: Source image (3- or 4-channel 8 bit).

    :param dst: Destination image.

    :param forward: ``true`` for forward gamma correction or ``false`` for inverse gamma correction.

    :param stream: Stream for the asynchronous version.



cuda::alphaComp
---------------
Composites two images using alpha opacity values contained in each image.

.. ocv:function:: void cuda::alphaComp(InputArray img1, InputArray img2, OutputArray dst, int alpha_op, Stream& stream = Stream::Null())

    :param img1: First image. Supports ``CV_8UC4`` , ``CV_16UC4`` , ``CV_32SC4`` and ``CV_32FC4`` types.

    :param img2: Second image. Must have the same size and the same type as ``img1`` .

    :param dst: Destination image.

    :param alpha_op: Flag specifying the alpha-blending operation:

            * **ALPHA_OVER**
            * **ALPHA_IN**
            * **ALPHA_OUT**
            * **ALPHA_ATOP**
            * **ALPHA_XOR**
            * **ALPHA_PLUS**
            * **ALPHA_OVER_PREMUL**
            * **ALPHA_IN_PREMUL**
            * **ALPHA_OUT_PREMUL**
            * **ALPHA_ATOP_PREMUL**
            * **ALPHA_XOR_PREMUL**
            * **ALPHA_PLUS_PREMUL**
            * **ALPHA_PREMUL**

    :param stream: Stream for the asynchronous version.

.. note::

   * An example demonstrating the use of alphaComp can be found at opencv_source_code/samples/gpu/alpha_comp.cpp


.. [MHT2011] Pascal Getreuer, Malvar-He-Cutler Linear Image Demosaicking, Image Processing On Line, 2011
