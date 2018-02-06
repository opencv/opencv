// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"


CV_IMPL void
cvSplit( const void* srcarr, void* dstarr0, void* dstarr1, void* dstarr2, void* dstarr3 )
{
    void* dptrs[] = { dstarr0, dstarr1, dstarr2, dstarr3 };
    cv::Mat src = cv::cvarrToMat(srcarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += dptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> dvec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( dptrs[i] != 0 )
        {
            dvec[j] = cv::cvarrToMat(dptrs[i]);
            CV_Assert( dvec[j].size() == src.size() );
            CV_Assert( dvec[j].depth() == src.depth() );
            CV_Assert( dvec[j].channels() == 1 );
            CV_Assert( i < src.channels() );
            pairs[j*2] = i;
            pairs[j*2+1] = j;
            j++;
        }
    }
    if( nz == src.channels() )
        cv::split( src, dvec );
    else
    {
        cv::mixChannels( &src, 1, &dvec[0], nz, &pairs[0], nz );
    }
}


CV_IMPL void
cvMerge( const void* srcarr0, const void* srcarr1, const void* srcarr2,
         const void* srcarr3, void* dstarr )
{
    const void* sptrs[] = { srcarr0, srcarr1, srcarr2, srcarr3 };
    cv::Mat dst = cv::cvarrToMat(dstarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += sptrs[i] != 0;
    CV_Assert( nz > 0 );
    std::vector<cv::Mat> svec(nz);
    std::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( sptrs[i] != 0 )
        {
            svec[j] = cv::cvarrToMat(sptrs[i]);
            CV_Assert( svec[j].size == dst.size &&
                svec[j].depth() == dst.depth() &&
                svec[j].channels() == 1 && i < dst.channels() );
            pairs[j*2] = j;
            pairs[j*2+1] = i;
            j++;
        }
    }

    if( nz == dst.channels() )
        cv::merge( svec, dst );
    else
    {
        cv::mixChannels( &svec[0], nz, &dst, 1, &pairs[0], nz );
    }
}


CV_IMPL void
cvMixChannels( const CvArr** src, int src_count,
               CvArr** dst, int dst_count,
               const int* from_to, int pair_count )
{
    cv::AutoBuffer<cv::Mat> buf(src_count + dst_count);

    int i;
    for( i = 0; i < src_count; i++ )
        buf[i] = cv::cvarrToMat(src[i]);
    for( i = 0; i < dst_count; i++ )
        buf[i+src_count] = cv::cvarrToMat(dst[i]);
    cv::mixChannels(&buf[0], src_count, &buf[src_count], dst_count, from_to, pair_count);
}


CV_IMPL void
cvConvertScaleAbs( const void* srcarr, void* dstarr,
                   double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && dst.type() == CV_8UC(src.channels()));
    cv::convertScaleAbs( src, dst, scale, shift );
}


CV_IMPL void
cvConvertScale( const void* srcarr, void* dstarr,
                double scale, double shift )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() );
    src.convertTo(dst, dst.type(), scale, shift);
}


CV_IMPL void cvLUT( const void* srcarr, void* dstarr, const void* lutarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), lut = cv::cvarrToMat(lutarr);

    CV_Assert( dst.size() == src.size() && dst.type() == CV_MAKETYPE(lut.depth(), src.channels()) );
    cv::LUT( src, lut, dst );
}


CV_IMPL void cvNormalize( const CvArr* srcarr, CvArr* dstarr,
                          double a, double b, int norm_type, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    CV_Assert( dst.size() == src.size() && src.channels() == dst.channels() );
    cv::normalize( src, dst, a, b, norm_type, dst.type(), mask );
}
