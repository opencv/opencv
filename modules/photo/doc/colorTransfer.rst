ColorTransfer
=============

.. highlight:: cpp

colorTransfer
-------------

Color Transfer is a technique to impose one image’s color characteristics on another. We can achieve color correction by choosing an appropriate source image and apply its characteristic to another image. Applications range from subtle postprocessing on images to improve their appearance to more dramatic alterations, such as converting a daylight image into a night scene [ER01]_.

.. ocv:function::  void colorTransfer(InputArray _src, InputArray _dst, OutputArray _color_transfer, int flag = 0)

    :param src: Input 8-bit 3-channel image.

    :param dst: Input 8-bit 3-channel image.

    :param color_transfer: Output 8-bit 3-channel image.

    :param flag: 0 for computation in Lab colorspace and 1 for RGB colorspace.

The color is transferred from destination to source image.

.. [ER01] Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley, "Color Transfer between Images", IEEE Computer Graphics and Applications, 2001.
