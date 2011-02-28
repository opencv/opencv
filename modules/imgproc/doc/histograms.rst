Histograms
==========

.. highlight:: cpp

.. index:: calcHist

calcHist
------------
.. c:function:: void calcHist( const Mat* arrays, int narrays,               const int* channels, const Mat\& mask,               MatND\& hist, int dims, const int* histSize,               const float** ranges, bool uniform=true,               bool accumulate=false )

.. c:function:: void calcHist( const Mat* arrays, int narrays,               const int* channels, const Mat\& mask,               SparseMat\& hist, int dims, const int* histSize,               const float** ranges, bool uniform=true,               bool accumulate=false )

    Calculates histogram of a set of arrays

    :param arrays: Source arrays. They all should have the same depth,  ``CV_8U``  or  ``CV_32F`` , and the same size. Each of them can have an arbitrary number of channels

    :param narrays: The number of source arrays

    :param channels: The list of  ``dims``  channels that are used to compute the histogram. The first array channels are numerated from 0 to  ``arrays[0].channels()-1`` , the second array channels are counted from  ``arrays[0].channels()``  to  ``arrays[0].channels() + arrays[1].channels()-1``  etc.

    :param mask: The optional mask. If the matrix is not empty, it must be 8-bit array of the same size as  ``arrays[i]`` . The non-zero mask elements mark the array elements that are counted in the histogram

    :param hist: The output histogram, a dense or sparse  ``dims`` -dimensional array

    :param dims: The histogram dimensionality; must be positive and not greater than  ``CV_MAX_DIMS`` (=32 in the current OpenCV version)

    :param histSize: The array of histogram sizes in each dimension

    :param ranges: The array of  ``dims``  arrays of the histogram bin boundaries in each dimension. When the histogram is uniform ( ``uniform`` =true), then for each dimension  ``i``  it's enough to specify the lower (inclusive) boundary  :math:`L_0`  of the 0-th histogram bin and the upper (exclusive) boundary  :math:`U_{\texttt{histSize}[i]-1}`  for the last histogram bin  ``histSize[i]-1`` . That is, in the case of uniform histogram each of  ``ranges[i]``  is an array of 2 elements. When the histogram is not uniform ( ``uniform=false`` ), then each of  ``ranges[i]``  contains  ``histSize[i]+1``  elements:  :math:`L_0, U_0=L_1, U_1=L_2, ..., U_{\texttt{histSize[i]}-2}=L_{\texttt{histSize[i]}-1}, U_{\texttt{histSize[i]}-1}` . The array elements, which are not between  :math:`L_0`  and  :math:`U_{\texttt{histSize[i]}-1}` , are not counted in the histogram

    :param uniform: Indicates whether the histogram is uniform or not, see above

    :param accumulate: Accumulation flag. If it is set, the histogram is not cleared in the beginning (when it is allocated). This feature allows user to compute a single histogram from several sets of arrays, or to update the histogram in time

The functions ``calcHist`` calculate the histogram of one or more
arrays. The elements of a tuple that is used to increment
a histogram bin are taken at the same location from the corresponding
input arrays. The sample below shows how to compute 2D Hue-Saturation histogram for a color imag ::

    #include <cv.h>
    #include <highgui.h>

    using namespace cv;

    int main( int argc, char** argv )
    {
        Mat src, hsv;
        if( argc != 2 || !(src=imread(argv[1], 1)).data )
            return -1;

        cvtColor(src, hsv, CV_BGR2HSV);

        // let's quantize the hue to 30 levels
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
.. c:function:: void calcBackProject( const Mat* arrays, int narrays,                      const int* channels, const MatND\& hist,                      Mat\& backProject, const float** ranges,                      double scale=1, bool uniform=true )

.. c:function:: void calcBackProject( const Mat* arrays, int narrays,                      const int* channels, const SparseMat\& hist,                      Mat\& backProject, const float** ranges,                      double scale=1, bool uniform=true )

    Calculates the back projection of a histogram.

    :param arrays: Source arrays. They all should have the same depth,  ``CV_8U``  or  ``CV_32F`` , and the same size. Each of them can have an arbitrary number of channels

    :param narrays: The number of source arrays

    :param channels: The list of channels that are used to compute the back projection. The number of channels must match the histogram dimensionality. The first array channels are numerated from 0 to  ``arrays[0].channels()-1`` , the second array channels are counted from  ``arrays[0].channels()``  to  ``arrays[0].channels() + arrays[1].channels()-1``  etc.

    :param hist: The input histogram, a dense or sparse

    :param backProject: Destination back projection aray; will be a single-channel array of the same size and the same depth as  ``arrays[0]``
    :param ranges: The array of arrays of the histogram bin boundaries in each dimension. See  :func:`calcHist`
    :param scale: The optional scale factor for the output back projection

    :param uniform: Indicates whether the histogram is uniform or not, see above

The functions ``calcBackProject`` calculate the back project of the histogram. That is, similarly to ``calcHist`` , at each location ``(x, y)`` the function collects the values from the selected channels in the input images and finds the corresponding histogram bin. But instead of incrementing it, the function reads the bin value, scales it by ``scale`` and stores in ``backProject(x,y)`` . In terms of statistics, the function computes probability of each element value in respect with the empirical probability distribution represented by the histogram. Here is how, for example, you can find and track a bright-colored object in a scene:

#.
    Before the tracking, show the object to the camera such that covers almost the whole frame. Calculate a hue histogram. The histogram will likely have a strong maximums, corresponding to the dominant colors in the object.

#.
    During the tracking, calculate back projection of a hue plane of each input video frame using that pre-computed histogram. Threshold the back projection to suppress weak colors. It may also have sense to suppress pixels with non sufficient color saturation and too dark or too bright pixels.

#.
    Find connected components in the resulting picture and choose, for example, the largest component.

That is the approximate algorithm of
:func:`CAMShift` color object tracker.

See also:
:func:`calcHist`
.. index:: compareHist

compareHist
---------------
.. c:function:: double compareHist( const MatND\& H1, const MatND\& H2, int method )

.. c:function:: double compareHist( const SparseMat\& H1,  const SparseMat\& H2, int method )

    Compares two histograms

    :param H1: The first compared histogram

    :param H2: The second compared histogram of the same size as  ``H1``
    :param method: The comparison method, one of the following:

            * **CV_COMP_CORREL** Correlation

            * **CV_COMP_CHISQR** Chi-Square

            * **CV_COMP_INTERSECT** Intersection

            * **CV_COMP_BHATTACHARYYA** Bhattacharyya distance

The functions ``compareHist`` compare two dense or two sparse histograms using the specified method:

* Correlation (method=CV\_COMP\_CORREL)

    .. math::

        d(H_1,H_2) =  \frac{\sum_I (H_1(I) - \bar{H_1}) (H_2(I) - \bar{H_2})}{\sqrt{\sum_I(H_1(I) - \bar{H_1})^2 \sum_I(H_2(I) - \bar{H_2})^2}}

    where

    .. math::

        \bar{H_k} =  \frac{1}{N} \sum _J H_k(J)

    and
    :math:`N`     is the total number of histogram bins.

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

While the function works well with 1-, 2-, 3-dimensional dense histograms, it may not be suitable for high-dimensional sparse histograms, where, because of aliasing and sampling problems the coordinates of non-zero histogram bins can slightly shift. To compare such histograms or more general sparse configurations of weighted points, consider using the
:func:`calcEMD` function.

.. index:: equalizeHist

equalizeHist
----------------
.. c:function:: void equalizeHist( const Mat\& src, Mat\& dst )

    Equalizes the histogram of a grayscale image.

    :param src: The source 8-bit single channel image

    :param dst: The destination image; will have the same size and the same type as  ``src``
The function equalizes the histogram of the input image using the following algorithm:

#.
    calculate the histogram
    :math:`H`     for ``src``     .

#.
    normalize the histogram so that the sum of histogram bins is 255.

#.
    compute the integral of the histogram:

    .. math::

        H'_i =  \sum _{0  \le j < i} H(j)

#.
    transform the image using
    :math:`H'`     as a look-up table:
    :math:`\texttt{dst}(x,y) = H'(\texttt{src}(x,y))`
The algorithm normalizes the brightness and increases the contrast of the image.

