Histograms
==========

.. highlight:: cpp

.. index:: calcHist

calcHist
------------
.. cpp:function:: void calcHist( const Mat* arrays, int narrays, const int* channels, InputArray mask,               OutputArray hist, int dims, const int* histSize, const float** ranges, bool uniform=true,               bool accumulate=false )

.. cpp:function:: void calcHist( const Mat* arrays, int narrays, const int* channels, InputArray mask,               SparseMat& hist, int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )

    Calculates a histogram of a set of arrays.

    :param arrays: Source arrays. They all should have the same depth,  ``CV_8U``  or  ``CV_32F`` , and the same size. Each of them can have an arbitrary number of channels.

    :param narrays: Number of source arrays.

    :param channels: List of the  ``dims``  channels used to compute the histogram. The first array channels are numerated from 0 to  ``arrays[0].channels()-1`` , the second array channels are counted from  ``arrays[0].channels()``  to  ``arrays[0].channels() + arrays[1].channels()-1``  etc.

    :param mask: Optional mask. If the matrix is not empty, it must be an 8-bit array of the same size as  ``arrays[i]`` . The non-zero mask elements mark the array elements counted in the histogram.

    :param hist: Output histogram, which is a dense or sparse  ``dims`` -dimensional array.

    :param dims: Histogram dimensionality that must be positive and not greater than  ``CV_MAX_DIMS`` (=32 in the current OpenCV version).

    :param histSize: Array of histogram sizes in each dimension.

    :param ranges: Array of the ``dims``  arrays of the histogram bin boundaries in each dimension. When the histogram is uniform ( ``uniform`` =true), then for each dimension  ``i``  it is enough to specify the lower (inclusive) boundary  :math:`L_0`  of the 0-th histogram bin and the upper (exclusive) boundary  :math:`U_{\texttt{histSize}[i]-1}`  for the last histogram bin  ``histSize[i]-1`` . That is, in case of a uniform histogram each of  ``ranges[i]``  is an array of 2 elements. When the histogram is not uniform ( ``uniform=false`` ), then each of  ``ranges[i]``  contains  ``histSize[i]+1``  elements:  :math:`L_0, U_0=L_1, U_1=L_2, ..., U_{\texttt{histSize[i]}-2}=L_{\texttt{histSize[i]}-1}, U_{\texttt{histSize[i]}-1}` . The array elements, that are not between  :math:`L_0`  and  :math:`U_{\texttt{histSize[i]}-1}` , are not counted in the histogram.

    :param uniform: Flag indicatinfg whether the histogram is uniform or not (see above).

    :param accumulate: Accumulation flag. If it is set, the histogram is not cleared in the beginning when it is allocated. This feature enables you to compute a single histogram from several sets of arrays, or to update the histogram in time.

The functions ``calcHist`` calculate the histogram of one or more
arrays. The elements of a tuple used to increment
a histogram bin are taken from the corresponding
input arrays at the same location. The sample below shows how to compute a 2D Hue-Saturation histogram for a color image. ::

    #include <cv.h>
    #include <highgui.h>

    using namespace cv;

    int main( int argc, char** argv )
    {
        Mat src, hsv;
        if( argc != 2 || !(src=imread(argv[1], 1)).data )
            return -1;

        cvtColor(src, hsv, CV_BGR2HSV);

        // Quantize the hue to 30 levels
        // and the saturation to 32 levels
        int hbins = 30, sbins = 32;
        int histSize[] = {hbins, sbins};
        // hue varies from 0 to 179, see cvtColor
        float hranges[] = { 0, 180 };
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        float sranges[] = { 0, 256 };
        const float* ranges[] = { hranges, sranges };
        MatND hist;
        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};

        calcHist( &hsv, 1, channels, Mat(), // do not use mask
                 hist, 2, histSize, ranges,
                 true, // the histogram is uniform
                 false );
        double maxVal=0;
        minMaxLoc(hist, 0, &maxVal, 0, 0);

        int scale = 10;
        Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

        for( int h = 0; h < hbins; h++ )
            for( int s = 0; s < sbins; s++ )
            {
                float binVal = hist.at<float>(h, s);
                int intensity = cvRound(binVal*255/maxVal);
                rectangle( histImg, Point(h*scale, s*scale),
                            Point( (h+1)*scale - 1, (s+1)*scale - 1),
                            Scalar::all(intensity),
                            CV_FILLED );
            }

        namedWindow( "Source", 1 );
        imshow( "Source", src );

        namedWindow( "H-S Histogram", 1 );
        imshow( "H-S Histogram", histImg );
        waitKey();
    }


.. index:: calcBackProject

calcBackProject
-------------------
.. cpp:function:: void calcBackProject( const Mat* arrays, int narrays, const int* channels, InputArray hist, OutputArray backProject, const float** ranges, double scale=1, bool uniform=true )

.. cpp:function:: void calcBackProject( const Mat* arrays, int narrays, const int* channels, const SparseMat& hist, OutputArray backProject, const float** ranges, double scale=1, bool uniform=true )

    Calculates the back projection of a histogram.

    :param arrays: Source arrays. They all should have the same depth,  ``CV_8U``  or  ``CV_32F`` , and the same size. Each of them can have an arbitrary number of channels.

    :param narrays: Number of source arrays.

    :param channels: The list of channels that are used to compute the back projection. The number of channels must match the histogram dimensionality. The first array channels are numerated from 0 to  ``arrays[0].channels()-1`` , the second array channels are counted from  ``arrays[0].channels()``  to  ``arrays[0].channels() + arrays[1].channels()-1``  and so on.

    :param hist: Input histogram that can be dense or sparse.

    :param backProject: Destination back projection aray that is a single-channel array of the same size and depth as  ``arrays[0]`` .
	
    :param ranges: Array of arrays of the histogram bin boundaries in each dimension. See  :cpp:func:`calcHist` .
	
    :param scale: Optional scale factor for the output back projection.

    :param uniform: Flag indicating whether the histogram is uniform or not (see above).

The functions ``calcBackProject`` calculate the back project of the histogram. That is, similarly to ``calcHist`` , at each location ``(x, y)`` the function collects the values from the selected channels in the input images and finds the corresponding histogram bin. But instead of incrementing it, the function reads the bin value, scales it by ``scale`` , and stores in ``backProject(x,y)`` . In terms of statistics, the function computes probability of each element value in respect with the empirical probability distribution represented by the histogram. See how, for example, you can find and track a bright-colored object in a scene:

#.
    Before tracking, show the object to the camera so that it covers almost the whole frame. Calculate a hue histogram. The histogram may have strong maximums, corresponding to the dominant colors in the object.

#.
    When tracking, calculate a back projection of a hue plane of each input video frame using that pre-computed histogram. Threshold the back projection to suppress weak colors. It may also make sense to suppress pixels with non-sufficient color saturation and too dark or too bright pixels.

#.
    Find connected components in the resulting picture and choose, for example, the largest component.

This is an approximate algorithm of the
:cpp:func:`CAMShift` color object tracker.

See Also:
:cpp:func:`calcHist`

.. index:: compareHist

compareHist
-----------

.. cpp:function:: double compareHist( InputArray H1, InputArray H2, int method )

.. cpp:function:: double compareHist( const SparseMat& H1,  const SparseMat& H2, int method )

    Compares two histograms.

    :param H1: The first compared histogram.

    :param H2: The second compared histogram of the same size as  ``H1`` .
    :param method: Comparison method that could be one of the following:

            * **CV_COMP_CORREL** 	Correlation

            * **CV_COMP_CHISQR** 	Chi-Square

            * **CV_COMP_INTERSECT** 	Intersection

            * **CV_COMP_BHATTACHARYYA** 	Bhattacharyya distance

The functions ``compareHist`` compare two dense or two sparse histograms using the specified method:

* Correlation (method=CV\_COMP\_CORREL)

    .. math::

        d(H_1,H_2) =  \frac{\sum_I (H_1(I) - \bar{H_1}) (H_2(I) - \bar{H_2})}{\sqrt{\sum_I(H_1(I) - \bar{H_1})^2 \sum_I(H_2(I) - \bar{H_2})^2}}

    where

    .. math::

        \bar{H_k} =  \frac{1}{N} \sum _J H_k(J)

    and
    :math:`N`     is a total number of histogram bins.

* Chi-Square (method=CV\_COMP\_CHISQR)

    .. math::

        d(H_1,H_2) =  \sum _I  \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)+H_2(I)}

* Intersection (method=CV\_COMP\_INTERSECT)

    .. math::

        d(H_1,H_2) =  \sum _I  \min (H_1(I), H_2(I))

* Bhattacharyya distance (method=CV\_COMP\_BHATTACHARYYA)

    .. math::

        d(H_1,H_2) =  \sqrt{1 - \frac{1}{\sqrt{\bar{H_1} \bar{H_2} N^2}} \sum_I \sqrt{H_1(I) \cdot H_2(I)}}

The function returns
:math:`d(H_1, H_2)` .

While the function works well with 1-, 2-, 3-dimensional dense histograms, it may not be suitable for high-dimensional sparse histograms. In such histograms,  because of aliasing and sampling problems, the coordinates of non-zero histogram bins can slightly shift. To compare such histograms or more general sparse configurations of weighted points, consider using the
:cpp:func:`EMD` function.


.. index:: EMD

EMD
------
.. cpp:function:: float EMD( InputArray signature1, InputArray signature2, int distType, InputArray cost=noArray(), float* lowerBound=0, OutputArray flow=noArray() )

    Computes the "minimal work" distance between two weighted point configurations.


    :param signature1: The first signature, a  :math:`\texttt{size1}\times \texttt{dims}+1`  floating-point matrix. Each row stores the point weight followed by the point coordinates. The matrix is allowed to have a single column (weights only) if the user-defined cost matrix is used.

    :param signature2: The second signature of the same format as  ``signature1`` , though the number of rows may be different. The total weights may be different, in this case an extra "dummy" point is added to either  ``signature1``  or  ``signature2`` .

    :param distType: Used metric.  ``CV_DIST_L1, CV_DIST_L2`` , and  ``CV_DIST_C``  stand for one of the standard metrics;  ``CV_DIST_USER``  means that a pre-calculated cost matrix ``cost``  is used.

    :param cost: The user-defined  :math:`\texttt{size1}\times \texttt{size2}`  cost matrix. Also, if a cost matrix is used, lower boundary  ``lowerBound``  can not be calculated, because it needs a metric function.

    :param lowerBound: Optional input/output parameter: lower boundary of distance between the two signatures that is a distance between mass centers. The lower boundary may not be calculated if the user-defined cost matrix is used, the total weights of point configurations are not equal, or if the signatures consist of weights only (i.e. the signature matrices have a single column). The user  **must**  initialize  ``*lowerBound`` . If the calculated distance between mass centers is greater or equal to  ``*lowerBound``  (it means that the signatures are far enough) the function does not calculate EMD. In any case  ``*lowerBound``  is set to the calculated distance between mass centers on return. Thus, if user wants to calculate both distance between mass centers and EMD,  ``*lowerBound``  should be set to 0.

    :param flow: The resultant  :math:`\texttt{size1} \times \texttt{size2}`  flow matrix:  :math:`\texttt{flow}_{i,j}`  is a flow from  :math:`i`  th point of  ``signature1``  to  :math:`j`  th point of  ``signature2``  .

The function computes the earth mover distance and/or a lower boundary of the distance between the two weighted point configurations. One of the applications described in :ref:`RubnerSept98` is multi-dimensional histogram comparison for image retrieval. EMD is a transportation problem that is solved using some modification of a simplex algorithm, thus the complexity is exponential in the worst case, though, on average it is much faster. In the case of a real metric the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used to determine roughly whether the two signatures are far enough so that they cannot relate to the same object.


.. index:: equalizeHist

equalizeHist
----------------
.. cpp:function:: void equalizeHist( InputArray src, OutputArray dst )

    Equalizes the histogram of a grayscale image.

    :param src: Source 8-bit single channel image.

    :param dst: Destination image of the same size and type as  ``src`` .

The function equalizes the histogram of the input image using the following algorithm:

#.
    Calculate the histogram
    :math:`H`     for ``src``  .

#.
    Normalize the histogram so that the sum of histogram bins is 255.

#.
    Compute the integral of the histogram:

    .. math::

        H'_i =  \sum _{0  \le j < i} H(j)

#.
    Transform the image using
    :math:`H'`     as a look-up table:
    :math:`\texttt{dst}(x,y) = H'(\texttt{src}(x,y))`

The algorithm normalizes the brightness and increases the contrast of the image.
