Color space processing
======================

.. highlight:: cpp



gpu::cvtColor
-----------------
Converts an image from one color space to another.

.. ocv:function:: void gpu::cvtColor(const GpuMat& src, GpuMat& dst, int code, int dcn = 0, Stream& stream = Stream::Null())

    :param src: Source image with  ``CV_8U`` , ``CV_16U`` , or  ``CV_32F`` depth and 1, 3, or 4 channels.

    :param dst: Destination image with the same size and depth as  ``src`` .

    :param code: Color space conversion code. For details, see  :ocv:func:`cvtColor` . Conversion to/from Luv and Bayer color spaces is not supported.

    :param dcn: Number of channels in the destination image. If the parameter is 0, the number of the channels is derived automatically from  ``src`` and the  ``code`` .

    :param stream: Stream for the asynchronous version.

3-channel color spaces (like ``HSV``, ``XYZ``, and so on) can be stored in a 4-channel image for better performance.

.. seealso:: :ocv:func:`cvtColor`



gpu::swapChannels
-----------------
Exchanges the color channels of an image in-place.

.. ocv:function:: void gpu::swapChannels(GpuMat& image, const int dstOrder[4], Stream& stream = Stream::Null())

    :param image: Source image. Supports only ``CV_8UC4`` type.

    :param dstOrder: Integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [3,2,1,0] converts this to ABGR channel order.

    :param stream: Stream for the asynchronous version.

The methods support arbitrary permutations of the original channels, including replication.



gpu::alphaComp
-------------------
Composites two images using alpha opacity values contained in each image.

.. ocv:function:: void gpu::alphaComp(const GpuMat& img1, const GpuMat& img2, GpuMat& dst, int alpha_op, Stream& stream = Stream::Null())

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
