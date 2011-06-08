Object Detection
================

.. highlight:: python



.. index:: MatchTemplate

.. _MatchTemplate:

MatchTemplate
-------------




.. function:: MatchTemplate(image,templ,result,method)-> None

    Compares a template against overlapped image regions.





    
    :param image: Image where the search is running; should be 8-bit or 32-bit floating-point 
    
    :type image: :class:`CvArr`
    
    
    :param templ: Searched template; must be not greater than the source image and the same data type as the image 
    
    :type templ: :class:`CvArr`
    
    
    :param result: A map of comparison results; single-channel 32-bit floating-point.
        If  ``image``  is  :math:`W \times H`  and ``templ``  is  :math:`w \times h`  then  ``result``  must be  :math:`(W-w+1) \times (H-h+1)` 
    
    :type result: :class:`CvArr`
    
    
    :param method: Specifies the way the template must be compared with the image regions (see below) 
    
    :type method: int
    
    
    
The function is similar to
:ref:`CalcBackProjectPatch`
. It slides through 
``image``
, compares the
overlapped patches of size 
:math:`w \times h`
against 
``templ``
using the specified method and stores the comparison results to
``result``
. Here are the formulas for the different comparison
methods one may use (
:math:`I`
denotes 
``image``
, 
:math:`T`
``template``
,
:math:`R`
``result``
). The summation is done over template and/or the
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
    
    
    
    
After the function finishes the comparison, the best matches can be found as global minimums (
``CV_TM_SQDIFF``
) or maximums (
``CV_TM_CCORR``
and 
``CV_TM_CCOEFF``
) using the 
:ref:`MinMaxLoc`
function. In the case of a color image, template summation in the numerator and each sum in the denominator is done over all of the channels (and separate mean values are used for each channel).

