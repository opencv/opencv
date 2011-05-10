Histograms
==========

.. highlight:: python



.. index:: CvHistogram

.. _CvHistogram:

CvHistogram
-----------



.. class:: CvHistogram



Multi-dimensional histogram.

A CvHistogram is a multi-dimensional histogram, created by function 
:ref:`CreateHist`
.  It has an attribute 
``bins``
a 
:ref:`CvMatND`
containing the histogram counts.

.. index:: CalcBackProject

.. _CalcBackProject:

CalcBackProject
---------------




.. function:: CalcBackProject(image,back_project,hist)-> None

    Calculates the back projection.





    
    :param image: Source images (though you may pass CvMat** as well) 
    
    :type image: sequence of :class:`IplImage`
    
    
    :param back_project: Destination back projection image of the same type as the source images 
    
    :type back_project: :class:`CvArr`
    
    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    
The function calculates the back project of the histogram. For each
tuple of pixels at the same position of all input single-channel images
the function puts the value of the histogram bin, corresponding to the
tuple in the destination image. In terms of statistics, the value of
each output image pixel is the probability of the observed tuple given
the distribution (histogram). For example, to find a red object in the
picture, one may do the following:



    

#.
    Calculate a hue histogram for the red object assuming the image contains only this object. The histogram is likely to have a strong maximum, corresponding to red color.
     
    

#.
    Calculate back projection of a hue plane of input image where the object is searched, using the histogram. Threshold the image.
     
    

#.
    Find connected components in the resulting picture and choose the right component using some additional criteria, for example, the largest connected component.
    
    
That is the approximate algorithm of Camshift color object tracker, except for the 3rd step, instead of which CAMSHIFT algorithm is used to locate the object on the back projection given the previous object position.


.. index:: CalcBackProjectPatch

.. _CalcBackProjectPatch:

CalcBackProjectPatch
--------------------




.. function:: CalcBackProjectPatch(images,dst,patch_size,hist,method,factor)-> None

    Locates a template within an image by using a histogram comparison.





    
    :param images: Source images (though, you may pass CvMat** as well) 
    
    :type images: sequence of :class:`IplImage`
    
    
    :param dst: Destination image 
    
    :type dst: :class:`CvArr`
    
    
    :param patch_size: Size of the patch slid though the source image 
    
    :type patch_size: :class:`CvSize`
    
    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param method: Comparison method, passed to  :ref:`CompareHist`  (see description of that function) 
    
    :type method: int
    
    
    :param factor: Normalization factor for histograms, will affect the normalization scale of the destination image, pass 1 if unsure 
    
    :type factor: float
    
    
    
The function calculates the back projection by comparing histograms of the source image patches with the given histogram. Taking measurement results from some image at each location over ROI creates an array 
``image``
. These results might be one or more of hue, 
``x``
derivative, 
``y``
derivative, Laplacian filter, oriented Gabor filter, etc. Each measurement output is collected into its own separate image. The 
``image``
image array is a collection of these measurement images. A multi-dimensional histogram 
``hist``
is constructed by sampling from the 
``image``
image array. The final histogram is normalized. The 
``hist``
histogram has as many dimensions as the number of elements in 
``image``
array.

Each new image is measured and then converted into an 
``image``
image array over a chosen ROI. Histograms are taken from this 
``image``
image in an area covered by a "patch" with an anchor at center as shown in the picture below. The histogram is normalized using the parameter 
``norm_factor``
so that it may be compared with 
``hist``
. The calculated histogram is compared to the model histogram; 
``hist``
uses The function 
``cvCompareHist``
with the comparison method=
``method``
). The resulting output is placed at the location corresponding to the patch anchor in the probability image 
``dst``
. This process is repeated as the patch is slid over the ROI. Iterative histogram update by subtracting trailing pixels covered by the patch and adding newly covered pixels to the histogram can save a lot of operations, though it is not implemented yet.

Back Project Calculation by Patches



.. image:: ../pics/backprojectpatch.png




.. index:: CalcHist

.. _CalcHist:

CalcHist
--------




.. function:: CalcHist(image,hist,accumulate=0,mask=NULL)-> None

    Calculates the histogram of image(s).





    
    :param image: Source images (though you may pass CvMat** as well) 
    
    :type image: sequence of :class:`IplImage`
    
    
    :param hist: Pointer to the histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param accumulate: Accumulation flag. If it is set, the histogram is not cleared in the beginning. This feature allows user to compute a single histogram from several images, or to update the histogram online 
    
    :type accumulate: int
    
    
    :param mask: The operation mask, determines what pixels of the source images are counted 
    
    :type mask: :class:`CvArr`
    
    
    
The function calculates the histogram of one or more
single-channel images. The elements of a tuple that is used to increment
a histogram bin are taken at the same location from the corresponding
input images.

.. include:: /Users/vp/Projects/ocv/opencv/doc/python_fragments/calchist.py
    :literal:



.. index:: CalcProbDensity

.. _CalcProbDensity:

CalcProbDensity
---------------




.. function:: CalcProbDensity(hist1,hist2,dst_hist,scale=255)-> None

    Divides one histogram by another.





    
    :param hist1: first histogram (the divisor) 
    
    :type hist1: :class:`CvHistogram`
    
    
    :param hist2: second histogram 
    
    :type hist2: :class:`CvHistogram`
    
    
    :param dst_hist: destination histogram 
    
    :type dst_hist: :class:`CvHistogram`
    
    
    :param scale: scale factor for the destination histogram 
    
    :type scale: float
    
    
    
The function calculates the object probability density from the two histograms as:



.. math::

    \texttt{dist\_hist} (I)= \forkthree{0}{if $\texttt{hist1}(I)=0$}{\texttt{scale}}{if $\texttt{hist1}(I) \ne 0$ and $\texttt{hist2}(I) > \texttt{hist1}(I)$}{\frac{\texttt{hist2}(I) \cdot \texttt{scale}}{\texttt{hist1}(I)}}{if $\texttt{hist1}(I) \ne 0$ and $\texttt{hist2}(I) \le \texttt{hist1}(I)$} 


So the destination histogram bins are within less than 
``scale``
.


.. index:: ClearHist

.. _ClearHist:

ClearHist
---------




.. function:: ClearHist(hist)-> None

    Clears the histogram.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    
The function sets all of the histogram bins to 0 in the case of a dense histogram and removes all histogram bins in the case of a sparse array.


.. index:: CompareHist

.. _CompareHist:

CompareHist
-----------




.. function:: CompareHist(hist1,hist2,method)->float

    Compares two dense histograms.





    
    :param hist1: The first dense histogram 
    
    :type hist1: :class:`CvHistogram`
    
    
    :param hist2: The second dense histogram 
    
    :type hist2: :class:`CvHistogram`
    
    
    :param method: Comparison method, one of the following: 
        
                
            * **CV_COMP_CORREL** Correlation 
            
               
            * **CV_COMP_CHISQR** Chi-Square 
            
               
            * **CV_COMP_INTERSECT** Intersection 
            
               
            * **CV_COMP_BHATTACHARYYA** Bhattacharyya distance 
            
            
    
    :type method: int
    
    
    
The function compares two dense histograms using the specified method (
:math:`H_1`
denotes the first histogram, 
:math:`H_2`
the second):



    

* Correlation (method=CV\_COMP\_CORREL)
    
    
    .. math::
    
        d(H_1,H_2) =  \frac{\sum_I (H'_1(I) \cdot H'_2(I))}{\sqrt{\sum_I(H'_1(I)^2) \cdot \sum_I(H'_2(I)^2)}} 
    
    
    where
    
    
    .. math::
    
        H'_k(I) =  \frac{H_k(I) - 1}{N \cdot \sum_J H_k(J)} 
    
    
    where N is the number of histogram bins.
    
    

* Chi-Square (method=CV\_COMP\_CHISQR)
    
    
    .. math::
    
        d(H_1,H_2) =  \sum _I  \frac{(H_1(I)-H_2(I))^2}{H_1(I)+H_2(I)} 
    
    
    

* Intersection (method=CV\_COMP\_INTERSECT)
    
    
    .. math::
    
        d(H_1,H_2) =  \sum _I  \min (H_1(I), H_2(I))  
    
    
    

* Bhattacharyya distance (method=CV\_COMP\_BHATTACHARYYA)
    
    
    .. math::
    
        d(H_1,H_2) =  \sqrt{1 - \sum_I \frac{\sqrt{H_1(I) \cdot H_2(I)}}{ \sqrt{ \sum_I H_1(I) \cdot \sum_I H_2(I) }}} 
    
    
    
    
The function returns 
:math:`d(H_1, H_2)`
.

Note: the method 
``CV_COMP_BHATTACHARYYA``
only works with normalized histograms.

To compare a sparse histogram or more general sparse configurations of weighted points, consider using the 
:ref:`CalcEMD2`
function.


.. index:: CreateHist

.. _CreateHist:

CreateHist
----------




.. function:: CreateHist(dims, type, ranges, uniform = 1) -> hist

    Creates a histogram.





    
    :param dims: for an N-dimensional histogram, list of length N giving the size of each dimension 
    
    :type dims: sequence of int
    
    
    :param type: Histogram representation format:  ``CV_HIST_ARRAY``  means that the histogram data is represented as a multi-dimensional dense array CvMatND;  ``CV_HIST_SPARSE``  means that histogram data is represented as a multi-dimensional sparse array CvSparseMat 
    
    :type type: int
    
    
    :param ranges: Array of ranges for the histogram bins. Its meaning depends on the  ``uniform``  parameter value. The ranges are used for when the histogram is calculated or backprojected to determine which histogram bin corresponds to which value/tuple of values from the input image(s) 
    
    :type ranges: list of tuples of ints
    
    
    :param uniform: Uniformity flag; if not 0, the histogram has evenly
        spaced bins and for every  :math:`0<=i<cDims`   ``ranges[i]`` 
        is an array of two numbers: lower and upper boundaries for the i-th
        histogram dimension.
        The whole range [lower,upper] is then split
        into  ``dims[i]``  equal parts to determine the  ``i-th``  input
        tuple value ranges for every histogram bin. And if  ``uniform=0`` ,
        then  ``i-th``  element of  ``ranges``  array contains ``dims[i]+1``  elements: :math:`\texttt{lower}_0, \texttt{upper}_0, 
        \texttt{lower}_1, \texttt{upper}_1 = \texttt{lower}_2,
        ...
        \texttt{upper}_{dims[i]-1}` 
        where :math:`\texttt{lower}_j`  and  :math:`\texttt{upper}_j` 
        are lower and upper
        boundaries of  ``i-th``  input tuple value for  ``j-th`` 
        bin, respectively. In either case, the input values that are beyond
        the specified range for a histogram bin are not counted by :ref:`CalcHist`  and filled with 0 by  :ref:`CalcBackProject` 
    
    :type uniform: int
    
    
    
The function creates a histogram of the specified
size and returns a pointer to the created histogram. If the array
``ranges``
is 0, the histogram bin ranges must be specified later
via the function 
:ref:`SetHistBinRanges`
. Though 
:ref:`CalcHist`
and 
:ref:`CalcBackProject`
may process 8-bit images without setting
bin ranges, they assume thy are equally spaced in 0 to 255 bins.


.. index:: GetMinMaxHistValue

.. _GetMinMaxHistValue:

GetMinMaxHistValue
------------------




.. function:: GetMinMaxHistValue(hist)-> (min_value,max_value,min_idx,max_idx)

    Finds the minimum and maximum histogram bins.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param min_value: Minimum value of the histogram 
    
    :type min_value: :class:`CvScalar`
    
    
    :param max_value: Maximum value of the histogram 
    
    :type max_value: :class:`CvScalar`
    
    
    :param min_idx: Coordinates of the minimum 
    
    :type min_idx: sequence of int
    
    
    :param max_idx: Coordinates of the maximum 
    
    :type max_idx: sequence of int
    
    
    
The function finds the minimum and
maximum histogram bins and their positions. All of output arguments are
optional. Among several extremas with the same value the ones with the
minimum index (in lexicographical order) are returned. In the case of several maximums
or minimums, the earliest in lexicographical order (extrema locations)
is returned.


.. index:: NormalizeHist

.. _NormalizeHist:

NormalizeHist
-------------




.. function:: NormalizeHist(hist,factor)-> None

    Normalizes the histogram.





    
    :param hist: Pointer to the histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param factor: Normalization factor 
    
    :type factor: float
    
    
    
The function normalizes the histogram bins by scaling them, such that the sum of the bins becomes equal to 
``factor``
.


.. index:: QueryHistValue_1D

.. _QueryHistValue_1D:

QueryHistValue_1D
-----------------




.. function:: QueryHistValue_1D(hist, idx0) -> float

    Returns the value from a 1D histogram bin.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param idx0: bin index 0 
    
    :type idx0: int
    
    
    

.. index:: QueryHistValue_2D

.. _QueryHistValue_2D:

QueryHistValue_2D
-----------------




.. function:: QueryHistValue_2D(hist, idx0, idx1) -> float

    Returns the value from a 2D histogram bin.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param idx0: bin index 0 
    
    :type idx0: int
    
    
    :param idx1: bin index 1 
    
    :type idx1: int
    
    
    

.. index:: QueryHistValue_3D

.. _QueryHistValue_3D:

QueryHistValue_3D
-----------------




.. function:: QueryHistValue_3D(hist, idx0, idx1, idx2) -> float

    Returns the value from a 3D histogram bin.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param idx0: bin index 0 
    
    :type idx0: int
    
    
    :param idx1: bin index 1 
    
    :type idx1: int
    
    
    :param idx2: bin index 2 
    
    :type idx2: int
    
    
    

.. index:: QueryHistValue_nD

.. _QueryHistValue_nD:

QueryHistValue_nD
-----------------




.. function:: QueryHistValue_nD(hist, idx) -> float

    Returns the value from a 1D histogram bin.





    
    :param hist: Histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param idx: list of indices, of same length as the dimension of the histogram's bin. 
    
    :type idx: sequence of int
    
    
    

.. index:: ThreshHist

.. _ThreshHist:

ThreshHist
----------




.. function:: ThreshHist(hist,threshold)-> None

    Thresholds the histogram.





    
    :param hist: Pointer to the histogram 
    
    :type hist: :class:`CvHistogram`
    
    
    :param threshold: Threshold level 
    
    :type threshold: float
    
    
    
The function clears histogram bins that are below the specified threshold.

