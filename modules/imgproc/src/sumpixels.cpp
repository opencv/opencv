/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{

template<typename QT> inline QT sqr(uchar a) { return a*a; }
template<typename QT> inline QT sqr(float a) { return a*a; }
template<typename QT> inline QT sqr(double a) { return a*a; }
template<> inline double sqr(uchar a) { return CV_8TO32F_SQR(a); }


template<typename T, typename ST, typename QT>
void integral_( const Mat& _src, Mat& _sum, Mat& _sqsum, Mat& _tilted )
{
    int cn = _src.channels();
    Size size = _src.size();
    int x, y, k;

    const T* src = (const T*)_src.data;
    ST* sum = (ST*)_sum.data;
    ST* tilted = (ST*)_tilted.data;
    QT* sqsum = (QT*)_sqsum.data;

    int srcstep = (int)(_src.step/sizeof(T));
    int sumstep = (int)(_sum.step/sizeof(ST));
    int tiltedstep = (int)(_tilted.step/sizeof(ST));
    int sqsumstep = (int)(_sqsum.step/sizeof(QT));

    size.width *= cn;

    memset( sum, 0, (size.width+cn)*sizeof(sum[0]));
    sum += sumstep + cn;

    if( sqsum )
    {
        memset( sqsum, 0, (size.width+cn)*sizeof(sqsum[0]));
        sqsum += sqsumstep + cn;
    }

    if( tilted )
    {
        memset( tilted, 0, (size.width+cn)*sizeof(tilted[0]));
        tilted += tiltedstep + cn;
    }

    if( sqsum == 0 && tilted == 0 )
    {
        for( y = 0; y < size.height; y++, src += srcstep - cn, sum += sumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++ )
            {
                ST s = sum[-cn] = 0;
                for( x = 0; x < size.width; x += cn )
                {
                    s += src[x];
                    sum[x] = sum[x - sumstep] + s;
                }
            }
        }
    }
    else if( tilted == 0 )
    {
        for( y = 0; y < size.height; y++, src += srcstep - cn,
                        sum += sumstep - cn, sqsum += sqsumstep - cn )
        {
            for( k = 0; k < cn; k++, src++, sum++, sqsum++ )
            {
                ST s = sum[-cn] = 0;
                QT sq = sqsum[-cn] = 0;
                for( x = 0; x < size.width; x += cn )
                {
                    T it = src[x];
                    s += it;
                    sq += sqr<QT>(it);
                    ST t = sum[x - sumstep] + s;
                    QT tq = sqsum[x - sqsumstep] + sq;
                    sum[x] = t;
                    sqsum[x] = tq;
                }
            }
        }
    }
    else
    {
        AutoBuffer<ST> _buf(size.width+cn);
        ST* buf = _buf;
        ST s;
        QT sq;
        for( k = 0; k < cn; k++, src++, sum++, tilted++, sqsum++, buf++ )
        {
            sum[-cn] = tilted[-cn] = 0;
            sqsum[-cn] = 0;

            for( x = 0, s = 0, sq = 0; x < size.width; x += cn )
            {
                T it = src[x];
                buf[x] = tilted[x] = it;
                s += it;
                sq += sqr<QT>(it);
                sum[x] = s;
                sqsum[x] = sq;
            }

            if( size.width == cn )
                buf[cn] = 0;
        }

        for( y = 1; y < size.height; y++ )
        {
            src += srcstep - cn;
            sum += sumstep - cn;
            sqsum += sqsumstep - cn;
            tilted += tiltedstep - cn;
            buf += -cn;

            for( k = 0; k < cn; k++, src++, sum++, sqsum++, tilted++, buf++ )
            {
                T it = src[0];
                ST t0 = s = it;
                QT tq0 = sq = sqr<QT>(it);

                sum[-cn] = 0;
                sqsum[-cn] = 0;
                tilted[-cn] = tilted[-tiltedstep];

                sum[0] = sum[-sumstep] + t0;
                sqsum[0] = sqsum[-sqsumstep] + tq0;
                tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];

                for( x = cn; x < size.width - cn; x += cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = sqr<QT>(it);
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    sqsum[x] = sqsum[x - sqsumstep] + sq;
                    t1 += buf[x + cn] + t0 + tilted[x - tiltedstep - cn];
                    tilted[x] = t1;
                }

                if( size.width > cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = sqr<QT>(it);
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    sqsum[x] = sqsum[x - sqsumstep] + sq;
                    tilted[x] = t0 + t1 + tilted[x - tiltedstep - cn];
                    buf[x] = t0;
                }
            }
        }
    }
}

typedef void (*IntegralFunc)(const Mat& _src, Mat& _sum, Mat& _sqsum, Mat& _tilted );

static void
integral( const Mat& src, Mat& sum, Mat* _sqsum, Mat* _tilted, int sdepth )
{
    int depth = src.depth(), cn = src.channels();
    Size isize(src.cols + 1, src.rows+1);
    Mat sqsum, tilted;

    if( sdepth <= 0 )
        sdepth = depth == CV_8U ? CV_32S : CV_64F;
    sdepth = CV_MAT_DEPTH(sdepth);
    sum.create( isize, CV_MAKETYPE(sdepth, cn) );
    
    if( _tilted )
        _tilted->create( isize, CV_MAKETYPE(sdepth, cn) );
    else
        _tilted = &tilted;
    
    if( !_sqsum )
        _sqsum = &sqsum;
    
    if( _sqsum != &sqsum || _tilted->data )
        _sqsum->create( isize, CV_MAKETYPE(CV_64F, cn) );

    IntegralFunc func = 0;

    if( depth == CV_8U && sdepth == CV_32S )
        func = integral_<uchar, int, double>;
    else if( depth == CV_8U && sdepth == CV_32F )
        func = integral_<uchar, float, double>;
    else if( depth == CV_8U && sdepth == CV_64F )
        func = integral_<uchar, double, double>;
    else if( depth == CV_32F && sdepth == CV_64F )
        func = integral_<float, double, double>;
    else if( depth == CV_64F && sdepth == CV_64F )
        func = integral_<double, double, double>;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    func( src, sum, *_sqsum, *_tilted );
}

void integral( const Mat& src, Mat& sum, int sdepth )
{
    integral( src, sum, 0, 0, sdepth );
}

void integral( const Mat& src, Mat& sum, Mat& sqsum, int sdepth )
{
    integral( src, sum, &sqsum, 0, sdepth );
}

void integral( const Mat& src, Mat& sum, Mat& sqsum, Mat& tilted, int sdepth )
{
    integral( src, sum, &sqsum, &tilted, sdepth );
}

}


CV_IMPL void
cvIntegral( const CvArr* image, CvArr* sumImage,
            CvArr* sumSqImage, CvArr* tiltedSumImage )
{
    cv::Mat src = cv::cvarrToMat(image), sum = cv::cvarrToMat(sumImage), sum0 = sum;
    cv::Mat sqsum0, sqsum, tilted0, tilted;
    cv::Mat *psqsum = 0, *ptilted = 0;

    if( sumSqImage )
    {
        sqsum0 = sqsum = cv::cvarrToMat(sumSqImage);
        psqsum = &sqsum;
    }

    if( tiltedSumImage )
    {
        tilted0 = tilted = cv::cvarrToMat(tiltedSumImage);
        ptilted = &tilted;
    }
    cv::integral( src, sum, psqsum, ptilted, sum.depth() );

    CV_Assert( sum.data == sum0.data && sqsum.data == sqsum0.data && tilted.data == tilted0.data );
}

/* End of file. */
