Histograms
==========

.. highlight:: c



.. index:: CvHistogram

.. _CvHistogram:

CvHistogram
-----------

`id=0.29416496784 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CvHistogram>`__

.. ctype:: CvHistogram



Multi-dimensional histogram.




::


    
    typedef struct CvHistogram
    {
        int     type;
        CvArr*  bins;
        float   thresh[CV_MAX_DIM][2]; /* for uniform histograms */
        float** thresh2; /* for non-uniform histograms */
        CvMatND mat; /* embedded matrix header for array histograms */
    }
    CvHistogram;
    

..


.. index:: CalcBackProject

.. _CalcBackProject:

CalcBackProject
---------------

`id=0.262445080297 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcBackProject>`__




.. cfunction:: void cvCalcBackProject(  IplImage** image, CvArr* back_project, const CvHistogram* hist )

    Calculates the back projection.





    
    :param image: Source images (though you may pass CvMat** as well) 
    
    
    :param back_project: Destination back projection image of the same type as the source images 
    
    
    :param hist: Histogram 
    
    
    
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

`id=0.510320009557 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcBackProjectPatch>`__




.. cfunction:: void cvCalcBackProjectPatch(  IplImage** images, CvArr* dst, CvSize patch_size, CvHistogram* hist, int method, double factor )

    Locates a template within an image by using a histogram comparison.





    
    :param images: Source images (though, you may pass CvMat** as well) 
    
    
    :param dst: Destination image 
    
    
    :param patch_size: Size of the patch slid though the source image 
    
    
    :param hist: Histogram 
    
    
    :param method: Comparison method, passed to  :ref:`CompareHist`  (see description of that function) 
    
    
    :param factor: Normalization factor for histograms, will affect the normalization scale of the destination image, pass 1 if unsure 
    
    
    
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

`id=0.247250829359 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcHist>`__




.. cfunction:: void cvCalcHist(  IplImage** image, CvHistogram* hist, int accumulate=0, const CvArr* mask=NULL )

    Calculates the histogram of image(s).





    
    :param image: Source images (though you may pass CvMat** as well) 
    
    
    :param hist: Pointer to the histogram 
    
    
    :param accumulate: Accumulation flag. If it is set, the histogram is not cleared in the beginning. This feature allows user to compute a single histogram from several images, or to update the histogram online 
    
    
    :param mask: The operation mask, determines what pixels of the source images are counted 
    
    
    
The function calculates the histogram of one or more
single-channel images. The elements of a tuple that is used to increment
a histogram bin are taken at the same location from the corresponding
input images.




::


    
    #include <cv.h>
    #include <highgui.h>
    
    int main( int argc, char** argv )
    {
        IplImage* src;
        if( argc == 2 && (src=cvLoadImage(argv[1], 1))!= 0)
        {
            IplImage* h_plane = cvCreateImage( cvGetSize(src), 8, 1 );
            IplImage* s_plane = cvCreateImage( cvGetSize(src), 8, 1 );
            IplImage* v_plane = cvCreateImage( cvGetSize(src), 8, 1 );
            IplImage* planes[] = { h_plane, s_plane };
            IplImage* hsv = cvCreateImage( cvGetSize(src), 8, 3 );
            int h_bins = 30, s_bins = 32;
            int hist_size[] = {h_bins, s_bins};
            /* hue varies from 0 (~0 deg red) to 180 (~360 deg red again) */
            float h_ranges[] = { 0, 180 };
            /* saturation varies from 0 (black-gray-white) to
               255 (pure spectrum color) */
            float s_ranges[] = { 0, 255 };
            float* ranges[] = { h_ranges, s_ranges };
            int scale = 10;
            IplImage* hist_img =
                cvCreateImage( cvSize(h_bins*scale,s_bins*scale), 8, 3 );
            CvHistogram* hist;
            float max_value = 0;
            int h, s;
    
            cvCvtColor( src, hsv, CV_BGR2HSV );
            cvCvtPixToPlane( hsv, h_plane, s_plane, v_plane, 0 );
            hist = cvCreateHist( 2, hist_size, CV_HIST_ARRAY, ranges, 1 );
            cvCalcHist( planes, hist, 0, 0 );
            cvGetMinMaxHistValue( hist, 0, &max_value, 0, 0 );
            cvZero( hist_img );
    
            for( h = 0; h < h_bins; h++ )
            {
                for( s = 0; s < s_bins; s++ )
                {
                    float bin_val = cvQueryHistValue_2D( hist, h, s );
                    int intensity = cvRound(bin_val*255/max_value);
                    cvRectangle( hist_img, cvPoint( h*scale, s*scale ),
                                 cvPoint( (h+1)*scale - 1, (s+1)*scale - 1),
                                 CV_RGB(intensity,intensity,intensity),
                                 CV_FILLED );
                }
            }
    
            cvNamedWindow( "Source", 1 );
            cvShowImage( "Source", src );
    
            cvNamedWindow( "H-S Histogram", 1 );
            cvShowImage( "H-S Histogram", hist_img );
    
            cvWaitKey(0);
        }
    }
    

..


.. index:: CalcProbDensity

.. _CalcProbDensity:

CalcProbDensity
---------------

`id=0.806356307482 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CalcProbDensity>`__




.. cfunction:: void  cvCalcProbDensity(  const CvHistogram* hist1, const CvHistogram* hist2, CvHistogram* dst_hist, double scale=255 )

    Divides one histogram by another.





    
    :param hist1: first histogram (the divisor) 
    
    
    :param hist2: second histogram 
    
    
    :param dst_hist: destination histogram 
    
    
    :param scale: scale factor for the destination histogram 
    
    
    
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

`id=0.835401602212 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ClearHist>`__




.. cfunction:: void cvClearHist( CvHistogram* hist )

    Clears the histogram.





    
    :param hist: Histogram 
    
    
    
The function sets all of the histogram bins to 0 in the case of a dense histogram and removes all histogram bins in the case of a sparse array.


.. index:: CompareHist

.. _CompareHist:

CompareHist
-----------

`id=0.50848857362 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CompareHist>`__




.. cfunction:: double cvCompareHist(  const CvHistogram* hist1, const CvHistogram* hist2, int method )

    Compares two dense histograms.





    
    :param hist1: The first dense histogram 
    
    
    :param hist2: The second dense histogram 
    
    
    :param method: Comparison method, one of the following: 
        
                
            * **CV_COMP_CORREL** Correlation 
            
               
            * **CV_COMP_CHISQR** Chi-Square 
            
               
            * **CV_COMP_INTERSECT** Intersection 
            
               
            * **CV_COMP_BHATTACHARYYA** Bhattacharyya distance 
            
            
    
    
    
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


.. index:: CopyHist

.. _CopyHist:

CopyHist
--------

`id=0.454990024463 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CopyHist>`__




.. cfunction:: void cvCopyHist( const CvHistogram* src, CvHistogram** dst )

    Copies a histogram.





    
    :param src: Source histogram 
    
    
    :param dst: Pointer to destination histogram 
    
    
    
The function makes a copy of the histogram. If the
second histogram pointer 
``*dst``
is NULL, a new histogram of the
same size as 
``src``
is created. Otherwise, both histograms must
have equal types and sizes. Then the function copies the source histogram's
bin values to the destination histogram and sets the same bin value ranges
as in 
``src``
.


.. index:: CreateHist

.. _CreateHist:

CreateHist
----------

`id=0.761254826094 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CreateHist>`__




.. cfunction:: CvHistogram* cvCreateHist( int dims, int* sizes, int type, float** ranges=NULL, int uniform=1 )

    Creates a histogram.





    
    :param dims: Number of histogram dimensions 
    
    :param sizes: Array of the histogram dimension sizes 
    
    
    :param type: Histogram representation format:  ``CV_HIST_ARRAY``  means that the histogram data is represented as a multi-dimensional dense array CvMatND;  ``CV_HIST_SPARSE``  means that histogram data is represented as a multi-dimensional sparse array CvSparseMat 
    
    
    :param ranges: Array of ranges for the histogram bins. Its meaning depends on the  ``uniform``  parameter value. The ranges are used for when the histogram is calculated or backprojected to determine which histogram bin corresponds to which value/tuple of values from the input image(s) 
    
    
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


.. index:: GetHistValue*D

.. _GetHistValue*D:

GetHistValue*D
--------------

`id=0.909653638644 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetHistValue%2AD>`__




.. cfunction:: float cvGetHistValue_1D(hist, idx0)



.. cfunction:: float cvGetHistValue_2D(hist, idx0, idx1)



.. cfunction:: float cvGetHistValue_3D(hist, idx0, idx1, idx2)



.. cfunction:: float cvGetHistValue_nD(hist, idx)

    Returns a pointer to the histogram bin.





    
    :param hist: Histogram 
    
    
    :param idx0, idx1, idx2, idx3: Indices of the bin 
    
    
    :param idx: Array of indices 
    
    
    



::


    
    #define cvGetHistValue_1D( hist, idx0 ) 
        ((float*)(cvPtr1D( (hist)->bins, (idx0), 0 ))
    #define cvGetHistValue_2D( hist, idx0, idx1 ) 
        ((float*)(cvPtr2D( (hist)->bins, (idx0), (idx1), 0 )))
    #define cvGetHistValue_3D( hist, idx0, idx1, idx2 ) 
        ((float*)(cvPtr3D( (hist)->bins, (idx0), (idx1), (idx2), 0 )))
    #define cvGetHistValue_nD( hist, idx ) 
        ((float*)(cvPtrND( (hist)->bins, (idx), 0 )))
    

..

The macros 
``GetHistValue``
return a pointer to the specified bin of the 1D, 2D, 3D or N-D histogram. In the case of a sparse histogram the function creates a new bin and sets it to 0, unless it exists already.

.. index:: GetMinMaxHistValue

.. _GetMinMaxHistValue:

GetMinMaxHistValue
------------------

`id=0.649289865958 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/GetMinMaxHistValue>`__




.. cfunction:: void cvGetMinMaxHistValue(  const CvHistogram* hist, float* min_value, float* max_value, int* min_idx=NULL, int* max_idx=NULL )

    Finds the minimum and maximum histogram bins.





    
    :param hist: Histogram 
    
    
    :param min_value: Pointer to the minimum value of the histogram 
    
    
    :param max_value: Pointer to the maximum value of the histogram 
    
    
    :param min_idx: Pointer to the array of coordinates for the minimum 
    
    
    :param max_idx: Pointer to the array of coordinates for the maximum 
    
    
    
The function finds the minimum and
maximum histogram bins and their positions. All of output arguments are
optional. Among several extremas with the same value the ones with the
minimum index (in lexicographical order) are returned. In the case of several maximums
or minimums, the earliest in lexicographical order (extrema locations)
is returned.


.. index:: MakeHistHeaderForArray

.. _MakeHistHeaderForArray:

MakeHistHeaderForArray
----------------------

`id=0.153593673347 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/MakeHistHeaderForArray>`__




.. cfunction:: CvHistogram*  cvMakeHistHeaderForArray(  int dims, int* sizes, CvHistogram* hist, float* data, float** ranges=NULL, int uniform=1 )

    Makes a histogram out of an array.





    
    :param dims: Number of histogram dimensions 
    
    
    :param sizes: Array of the histogram dimension sizes 
    
    
    :param hist: The histogram header initialized by the function 
    
    
    :param data: Array that will be used to store histogram bins 
    
    
    :param ranges: Histogram bin ranges, see  :ref:`CreateHist` 
    
    
    :param uniform: Uniformity flag, see  :ref:`CreateHist` 
    
    
    
The function initializes the histogram, whose header and bins are allocated by th user. 
:ref:`ReleaseHist`
does not need to be called afterwards. Only dense histograms can be initialized this way. The function returns 
``hist``
.

.. index:: NormalizeHist

.. _NormalizeHist:

NormalizeHist
-------------

`id=0.494984568711 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/NormalizeHist>`__




.. cfunction:: void cvNormalizeHist( CvHistogram* hist, double factor )

    Normalizes the histogram.





    
    :param hist: Pointer to the histogram 
    
    
    :param factor: Normalization factor 
    
    
    
The function normalizes the histogram bins by scaling them, such that the sum of the bins becomes equal to 
``factor``
.


.. index:: QueryHistValue*D

.. _QueryHistValue*D:

QueryHistValue*D
----------------

`id=0.0495732815752 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/QueryHistValue%2AD>`__




.. cfunction:: float QueryHistValue_1D(CvHistogram hist, int idx0)

    Queries the value of the histogram bin.





    
    :param hist: Histogram 
    
    
    :param idx0, idx1, idx2, idx3: Indices of the bin 
    
    
    :param idx: Array of indices 
    
    
    



::


    
    #define cvQueryHistValue_1D( hist, idx0 ) \
        cvGetReal1D( (hist)->bins, (idx0) )
    #define cvQueryHistValue_2D( hist, idx0, idx1 ) \
        cvGetReal2D( (hist)->bins, (idx0), (idx1) )
    #define cvQueryHistValue_3D( hist, idx0, idx1, idx2 ) \
        cvGetReal3D( (hist)->bins, (idx0), (idx1), (idx2) )
    #define cvQueryHistValue_nD( hist, idx ) \
        cvGetRealND( (hist)->bins, (idx) )
    

..

The macros return the value of the specified bin of the 1D, 2D, 3D or N-D histogram. In the case of a sparse histogram the function returns 0, if the bin is not present in the histogram no new bin is created.

.. index:: ReleaseHist

.. _ReleaseHist:

ReleaseHist
-----------

`id=0.635490375005 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ReleaseHist>`__




.. cfunction:: void cvReleaseHist( CvHistogram** hist )

    Releases the histogram.





    
    :param hist: Double pointer to the released histogram 
    
    
    
The function releases the histogram (header and the data). The pointer to the histogram is cleared by the function. If 
``*hist``
pointer is already 
``NULL``
, the function does nothing.


.. index:: SetHistBinRanges

.. _SetHistBinRanges:

SetHistBinRanges
----------------

`id=0.097775620677 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/SetHistBinRanges>`__




.. cfunction:: void cvSetHistBinRanges(  CvHistogram* hist, float** ranges, int uniform=1 )

    Sets the bounds of the histogram bins.





    
    :param hist: Histogram 
    
    
    :param ranges: Array of bin ranges arrays, see  :ref:`CreateHist` 
    
    
    :param uniform: Uniformity flag, see  :ref:`CreateHist` 
    
    
    
The function is a stand-alone function for setting bin ranges in the histogram. For a more detailed description of the parameters 
``ranges``
and 
``uniform``
see the 
:ref:`CalcHist`
function, that can initialize the ranges as well. Ranges for the histogram bins must be set before the histogram is calculated or the backproject of the histogram is calculated.


.. index:: ThreshHist

.. _ThreshHist:

ThreshHist
----------

`id=0.2471087134 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/imgproc/ThreshHist>`__




.. cfunction:: void cvThreshHist( CvHistogram* hist, double threshold )

    Thresholds the histogram.





    
    :param hist: Pointer to the histogram 
    
    
    :param threshold: Threshold level 
    
    
    
The function clears histogram bins that are below the specified threshold.

