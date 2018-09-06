// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"

CV_IMPL CvScalar cvSum( const CvArr* srcarr )
{
    cv::Scalar sum = cv::sum(cv::cvarrToMat(srcarr, false, true, 1));
    if( CV_IS_IMAGE(srcarr) )
    {
        int coi = cvGetImageCOI((IplImage*)srcarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            sum = cv::Scalar(sum[coi-1]);
        }
    }
    return cvScalar(sum);
}

CV_IMPL int cvCountNonZero( const CvArr* imgarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);
    return countNonZero(img);
}


CV_IMPL  CvScalar
cvAvg( const void* imgarr, const void* maskarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    cv::Scalar mean = !maskarr ? cv::mean(img) : cv::mean(img, cv::cvarrToMat(maskarr));
    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
        }
    }
    return cvScalar(mean);
}


CV_IMPL  void
cvAvgSdv( const CvArr* imgarr, CvScalar* _mean, CvScalar* _sdv, const void* maskarr )
{
    cv::Scalar mean, sdv;

    cv::Mat mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    cv::meanStdDev(cv::cvarrToMat(imgarr, false, true, 1), mean, sdv, mask );

    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
            sdv = cv::Scalar(sdv[coi-1]);
        }
    }

    if( _mean )
        *(cv::Scalar*)_mean = mean;
    if( _sdv )
        *(cv::Scalar*)_sdv = sdv;
}


CV_IMPL void
cvMinMaxLoc( const void* imgarr, double* _minVal, double* _maxVal,
             CvPoint* _minLoc, CvPoint* _maxLoc, const void* maskarr )
{
    cv::Mat mask, img = cv::cvarrToMat(imgarr, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);

    cv::minMaxLoc( img, _minVal, _maxVal,
                   (cv::Point*)_minLoc, (cv::Point*)_maxLoc, mask );
}


CV_IMPL  double
cvNorm( const void* imgA, const void* imgB, int normType, const void* maskarr )
{
    cv::Mat a, mask;
    if( !imgA )
    {
        imgA = imgB;
        imgB = 0;
    }

    a = cv::cvarrToMat(imgA, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    if( a.channels() > 1 && CV_IS_IMAGE(imgA) && cvGetImageCOI((const IplImage*)imgA) > 0 )
        cv::extractImageCOI(imgA, a);

    if( !imgB )
        return !maskarr ? cv::norm(a, normType) : cv::norm(a, normType, mask);

    cv::Mat b = cv::cvarrToMat(imgB, false, true, 1);
    if( b.channels() > 1 && CV_IS_IMAGE(imgB) && cvGetImageCOI((const IplImage*)imgB) > 0 )
        cv::extractImageCOI(imgB, b);

    return !maskarr ? cv::norm(a, b, normType) : cv::norm(a, b, normType, mask);
}
