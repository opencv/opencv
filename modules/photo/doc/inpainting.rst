Inpainting
==========

.. highlight:: cpp

inpaint
-----------
Restores the selected region in an image using the region neighborhood.

.. ocv:function:: void inpaint( InputArray src, InputArray inpaintMask, OutputArray dst, double inpaintRadius, int flags )

.. ocv:pyfunction:: cv2.inpaint(src, inpaintMask, inpaintRange, flags[, dst]) -> dst

.. ocv:cfunction:: void cvInpaint( const CvArr* src, const CvArr* mask, CvArr* dst, double inpaintRadius, int flags)
.. ocv:pyoldfunction:: cv.Inpaint(src, mask, dst, inpaintRadius, flags) -> None

    :param src: Input 8-bit 1-channel or 3-channel image.

    :param inpaintMask: Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that needs to be inpainted.

    :param dst: Output image with the same size and type as  ``src`` .
    
    :param inpaintRadius: Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.

    :param flags: Inpainting method that could be one of the following:

            * **INPAINT_NS**     Navier-Stokes based method.

            * **INPAINT_TELEA**     Method by Alexandru Telea  [Telea04]_.

The function reconstructs the selected image area from the pixel near the area boundary. The function may be used to remove dust and scratches from a scanned photo, or to remove undesirable objects from still images or video. See
http://en.wikipedia.org/wiki/Inpainting
for more details.
