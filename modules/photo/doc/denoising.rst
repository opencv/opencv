Denoising
==========

.. highlight:: cpp

fastNlMeansDenoising
--------------------
Perform image denoising using Non-local Means Denoising algorithm http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
with several computational optimizations. Noise expected to be a gaussian white noise

.. ocv:function:: void fastNlMeansDenoising( InputArray src, OutputArray dst, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

.. ocv:pyfunction:: cv2.fastNlMeansDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst

    :param src: Input 8-bit 1-channel, 2-channel or 3-channel image.

    :param dst: Output image with the same size and type as  ``src`` .

    :param templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels

    :param h: Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

This function expected to be applied to grayscale images. For colored images look at ``fastNlMeansDenoisingColored``.
Advanced usage of this functions can be manual denoising of colored image in different colorspaces.
Such approach is used in ``fastNlMeansDenoisingColored`` by converting image to CIELAB colorspace and then separately denoise L and AB components with different h parameter.

fastNlMeansDenoisingColored
---------------------------
Modification of ``fastNlMeansDenoising`` function for colored images

.. ocv:function:: void fastNlMeansDenoisingColored( InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )

.. ocv:pyfunction:: cv2.fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst

    :param src: Input 8-bit 3-channel image.

    :param dst: Output image with the same size and type as  ``src`` .

    :param templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels

    :param h: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

    :param hForColorComponents: The same as h but for color components. For most images value equals 10 will be enought to remove colored noise and do not distort colors

The function converts image to CIELAB colorspace and then separately denoise L and AB components with given h parameters using ``fastNlMeansDenoising`` function.

fastNlMeansDenoisingMulti
-------------------------
Modification of ``fastNlMeansDenoising`` function for images sequence where consequtive images have been captured in small period of time. For example video. This version of the function is for grayscale images or for manual manipulation with colorspaces.
For more details see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394

.. ocv:function:: void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst, int imgToDenoiseIndex, int temporalWindowSize, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

.. ocv:pyfunction:: cv2.fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst

    :param srcImgs: Input 8-bit 1-channel, 2-channel or 3-channel images sequence. All images should have the same type and size.

    :param imgToDenoiseIndex: Target image to denoise index in ``srcImgs`` sequence

    :param temporalWindowSize: Number of surrounding images to use for target image denoising. Should be odd. Images from ``imgToDenoiseIndex - temporalWindowSize / 2`` to ``imgToDenoiseIndex - temporalWindowSize / 2`` from ``srcImgs`` will be used to denoise ``srcImgs[imgToDenoiseIndex]`` image.

    :param dst: Output image with the same size and type as ``srcImgs`` images.

    :param templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels

    :param h: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

fastNlMeansDenoisingColoredMulti
--------------------------------
Modification of ``fastNlMeansDenoisingMulti`` function for colored images sequences

.. ocv:function:: void fastNlMeansDenoisingColoredMulti( InputArrayOfArrays srcImgs, OutputArray dst, int imgToDenoiseIndex, int temporalWindowSize, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )

.. ocv:pyfunction:: cv2.fastNlMeansDenoisingColoredMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst

    :param srcImgs: Input 8-bit 3-channel images sequence. All images should have the same type and size.

    :param imgToDenoiseIndex: Target image to denoise index in ``srcImgs`` sequence

    :param temporalWindowSize: Number of surrounding images to use for target image denoising. Should be odd. Images from ``imgToDenoiseIndex - temporalWindowSize / 2`` to ``imgToDenoiseIndex - temporalWindowSize / 2`` from ``srcImgs`` will be used to denoise ``srcImgs[imgToDenoiseIndex]`` image.

    :param dst: Output image with the same size and type as ``srcImgs`` images.

    :param templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels

    :param h: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise.

    :param hForColorComponents: The same as h but for color components.

The function converts images to CIELAB colorspace and then separately denoise L and AB components with given h parameters using ``fastNlMeansDenoisingMulti`` function.



cuda::nonLocalMeans
-------------------
Performs pure non local means denoising without any simplification, and thus it is not fast.

.. ocv:function:: void cuda::nonLocalMeans(const GpuMat& src, GpuMat& dst, float h, int search_window = 21, int block_size = 7, int borderMode = BORDER_DEFAULT, Stream& s = Stream::Null())

    :param src: Source image. Supports only CV_8UC1, CV_8UC2 and CV_8UC3.

    :param dst: Destination image.

    :param h: Filter sigma regulating filter strength for color.

    :param search_window: Size of search window.

    :param block_size: Size of block used for computing weights.

    :param borderMode:  Border type. See :ocv:func:`borderInterpolate` for details. ``BORDER_REFLECT101`` , ``BORDER_REPLICATE`` , ``BORDER_CONSTANT`` , ``BORDER_REFLECT`` and ``BORDER_WRAP`` are supported for now.

    :param stream: Stream for the asynchronous version.

.. seealso::

    :ocv:func:`fastNlMeansDenoising`



cuda::FastNonLocalMeansDenoising
--------------------------------
.. ocv:class:: cuda::FastNonLocalMeansDenoising

    ::

        class FastNonLocalMeansDenoising
        {
        public:
            //! Simple method, recommended for grayscale images (though it supports multichannel images)
            void simpleMethod(const GpuMat& src, GpuMat& dst, float h, int search_window = 21, int block_size = 7, Stream& s = Stream::Null())
            //! Processes luminance and color components separatelly
            void labMethod(const GpuMat& src, GpuMat& dst, float h_luminance, float h_color, int search_window = 21, int block_size = 7, Stream& s = Stream::Null())
        };

The class implements fast approximate Non Local Means Denoising algorithm.



cuda::FastNonLocalMeansDenoising::simpleMethod()
------------------------------------------------
Perform image denoising using Non-local Means Denoising algorithm http://www.ipol.im/pub/algo/bcm_non_local_means_denoising with several computational optimizations. Noise expected to be a gaussian white noise

.. ocv:function:: void cuda::FastNonLocalMeansDenoising::simpleMethod(const GpuMat& src, GpuMat& dst, float h, int search_window = 21, int block_size = 7, Stream& s = Stream::Null())

    :param src: Input 8-bit 1-channel, 2-channel or 3-channel image.

    :param dst: Output image with the same size and type as  ``src`` .

    :param h: Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

    :param search_window: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater search_window - greater denoising time. Recommended value 21 pixels

    :param block_size: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param stream: Stream for the asynchronous invocations.

This function expected to be applied to grayscale images. For colored images look at ``FastNonLocalMeansDenoising::labMethod``.

.. seealso::

    :ocv:func:`fastNlMeansDenoising`



cuda::FastNonLocalMeansDenoising::labMethod()
---------------------------------------------
Modification of ``FastNonLocalMeansDenoising::simpleMethod`` for color images

.. ocv:function:: void cuda::FastNonLocalMeansDenoising::labMethod(const GpuMat& src, GpuMat& dst, float h_luminance, float h_color, int search_window = 21, int block_size = 7, Stream& s = Stream::Null())

    :param src: Input 8-bit 3-channel image.

    :param dst: Output image with the same size and type as  ``src`` .

    :param h_luminance: Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

    :param float: The same as h but for color components. For most images value equals 10 will be enought to remove colored noise and do not distort colors

    :param search_window: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater search_window - greater denoising time. Recommended value 21 pixels

    :param block_size: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param stream: Stream for the asynchronous invocations.

The function converts image to CIELAB colorspace and then separately denoise L and AB components with given h parameters using ``FastNonLocalMeansDenoising::simpleMethod`` function.

.. seealso::

    :ocv:func:`fastNlMeansDenoisingColored`
