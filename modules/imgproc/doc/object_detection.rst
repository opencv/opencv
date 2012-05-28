Object Detection
================

.. highlight:: cpp

matchTemplate
-----------------
Compares a template against overlapped image regions.

.. ocv:function:: void matchTemplate( InputArray image, InputArray templ, OutputArray result, int method )

.. ocv:pyfunction:: cv2.matchTemplate(image, templ, method[, result]) -> result

.. ocv:cfunction:: void cvMatchTemplate( const CvArr* image, const CvArr* templ, CvArr* result, int method )
.. ocv:pyoldfunction:: cv.MatchTemplate(image, templ, result, method)-> None

    :param image: Image where the search is running. It must be 8-bit or 32-bit floating-point.

    :param templ: Searched template. It must be not greater than the source image and have the same data type.

    :param result: Map of comparison results. It must be single-channel 32-bit floating-point. If  ``image``  is  :math:`W \times H`  and ``templ``  is  :math:`w \times h` , then  ``result``  is  :math:`(W-w+1) \times (H-h+1)` .

    :param method: Parameter specifying the comparison method (see below).

The function slides through ``image`` , compares the
overlapped patches of size
:math:`w \times h` against ``templ`` using the specified method and stores the comparison results in ``result`` . Here are the formulae for the available comparison
methods (
:math:`I` denotes ``image``, :math:`T` ``template``, :math:`R` ``result`` ). The summation is done over template and/or the
image patch:
:math:`x' = 0...w-1, y' = 0...h-1`
* method=CV\_TM\_SQDIFF

    .. math::

        R(x,y)= \sum _{x',y'} (T(x',y')-I(x+x',y+y'))^2

* method=CV\_TM\_SQDIFF\_NORMED

    .. math::

        R(x,y)= \frac{\sum_{x',y'} (T(x',y')-I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}

* method=CV\_TM\_CCORR

    .. math::

        R(x,y)= \sum _{x',y'} (T(x',y')  \cdot I(x+x',y+y'))

* method=CV\_TM\_CCORR\_NORMED

    .. math::

        R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}

* method=CV\_TM\_CCOEFF

    .. math::

        R(x,y)= \sum _{x',y'} (T'(x',y')  \cdot I'(x+x',y+y'))

    where

    .. math::

        \begin{array}{l} T'(x',y')=T(x',y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} T(x'',y'') \\ I'(x+x',y+y')=I(x+x',y+y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} I(x+x'',y+y'') \end{array}

* method=CV\_TM\_CCOEFF\_NORMED

    .. math::

        R(x,y)= \frac{ \sum_{x',y'} (T'(x',y') \cdot I'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x',y+y')^2} }

After the function finishes the comparison, the best matches can be found as global minimums (when ``CV_TM_SQDIFF`` was used) or maximums (when ``CV_TM_CCORR`` or ``CV_TM_CCOEFF`` was used) using the
:ocv:func:`minMaxLoc` function. In case of a color image, template summation in the numerator and each sum in the denominator is done over all of the channels and separate mean values are used for each channel. That is, the function can take a color template and a color image. The result will still be a single-channel image, which is easier to analyze.

