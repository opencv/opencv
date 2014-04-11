Seamless Cloning
================

.. highlight:: cpp

seamlessClone
-------------
Image editing tasks concern either global changes (color/intensity corrections, filters, deformations) or local changes concerned to a selection.
Here we are interested in achieving local changes, ones that are restricted to a region manually selected (ROI), in a seamless and effortless manner.
The extent of the changes ranges from slight distortions to complete replacement by novel content [PM03]_.

.. ocv:function:: void seamlessClone( InputArray src, InputArray dst, InputArray mask, Point p, OutputArray blend, int flags)

    :param src: Input 8-bit 3-channel image.

    :param dst: Input 8-bit 3-channel image.

    :param mask: Input 8-bit 1 or 3-channel image.

    :param p: Point in dst image where object is placed.

    :param result: Output image with the same size and type as ``dst``.

    :param flags: Cloning method that could be one of the following:

            * **NORMAL_CLONE**     The power of the method is fully expressed when inserting objects with complex outlines into a new background

            * **MIXED_CLONE**    The classic method, color-based selection and alpha masking might be time consuming and often leaves an undesirable halo. Seamless cloning, even averaged with the original image, is not effective. Mixed seamless cloning based on a loose selection proves effective.

            * **FEATURE_EXCHANGE**     Feature exchange allows the user to easily replace certain features of one object by alternative features.



colorChange
-----------
Given an original color image, two differently colored versions of this image can be mixed seamlessly.

.. ocv:function:: void colorChange( InputArray src, InputArray mask, OutputArray dst, float red_mul = 1.0f, float green_mul = 1.0f, float blue_mul = 1.0f)

    :param src: Input 8-bit 3-channel image.

    :param mask: Input 8-bit 1 or 3-channel image.

    :param dst: Output image with the same size and type as  ``src`` .

    :param red_mul: R-channel multiply factor.

    :param green_mul: G-channel multiply factor.

    :param blue_mul: B-channel multiply factor.

Multiplication factor is between .5 to 2.5.


illuminationChange
------------------
Applying an appropriate non-linear transformation to the gradient field inside the selection and then integrating back with a Poisson
solver, modifies locally the apparent illumination of an image.

.. ocv:function:: void illuminationChange(InputArray src, InputArray mask, OutputArray dst, float alpha = 0.2f, float beta = 0.4f)

    :param src: Input 8-bit 3-channel image.

    :param mask: Input 8-bit 1 or 3-channel image.

    :param dst: Output image with the same size and type as  ``src``.

    :param alpha: Value ranges between 0-2.

    :param beta: Value ranges between 0-2.

This is useful to highlight under-exposed foreground objects or to reduce specular reflections.

textureFlattening
-----------------
By retaining only the gradients at edge locations, before integrating with the Poisson solver, one washes out the texture of the selected
region, giving its contents a flat aspect. Here Canny Edge Detector is used.

.. ocv:function:: void textureFlattening(InputArray src, InputArray mask, OutputArray dst, double low_threshold=30 , double high_threshold=45, int kernel_size=3)

    :param src: Input 8-bit 3-channel image.

    :param mask: Input 8-bit 1 or 3-channel image.

    :param dst: Output image with the same size and type as  ``src``.

    :param low_threshold: Range from 0 to 100.

    :param high_threshold: Value > 100.

    :param kernel_size: The size of the Sobel kernel to be used.

**NOTE:**

The algorithm assumes that the color of the source image is close to that of the destination. This assumption means that when the colors don't match, the source image color gets tinted toward the color of the destination image.

.. [PM03] Patrick Perez, Michel Gangnet, Andrew Blake, "Poisson image editing", ACM Transactions on Graphics (SIGGRAPH), 2003.
