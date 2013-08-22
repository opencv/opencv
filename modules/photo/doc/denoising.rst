Denoising
==========

.. highlight:: cpp

fastNlMeansDenoising
--------------------
Perform image denoising using Non-local Means Denoising algorithm http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/
with several computational optimizations. Noise expected to be a gaussian white noise

.. ocv:function:: void fastNlMeansDenoising( InputArray src, OutputArray dst, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

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

    :param srcImgs: Input 8-bit 3-channel images sequence. All images should have the same type and size.

    :param imgToDenoiseIndex: Target image to denoise index in ``srcImgs`` sequence

    :param temporalWindowSize: Number of surrounding images to use for target image denoising. Should be odd. Images from ``imgToDenoiseIndex - temporalWindowSize / 2`` to ``imgToDenoiseIndex - temporalWindowSize / 2`` from ``srcImgs`` will be used to denoise ``srcImgs[imgToDenoiseIndex]`` image.

    :param dst: Output image with the same size and type as ``srcImgs`` images.

    :param templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels

    :param searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels

    :param h: Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise.

    :param hForColorComponents: The same as h but for color components.

The function converts images to CIELAB colorspace and then separately denoise L and AB components with given h parameters using ``fastNlMeansDenoisingMulti`` function.
