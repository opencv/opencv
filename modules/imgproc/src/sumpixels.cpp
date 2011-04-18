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

template<typename T, typename ST, typename QT>
void integral_( const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
                QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
                Size size, int cn )
{
    int x, y, k;

    int srcstep = (int)(_srcstep/sizeof(T));
    int sumstep = (int)(_sumstep/sizeof(ST));
    int tiltedstep = (int)(_tiltedstep/sizeof(ST));
    int sqsumstep = (int)(_sqsumstep/sizeof(QT));

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
                    sq += (QT)it*it;
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
        for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
        {
            sum[-cn] = tilted[-cn] = 0;

            for( x = 0, s = 0, sq = 0; x < size.width; x += cn )
            {
                T it = src[x];
                buf[x] = tilted[x] = it;
                s += it;
                sq += (QT)it*it;
                sum[x] = s;
                if( sqsum )
                    sqsum[x] = sq;
            }

            if( size.width == cn )
                buf[cn] = 0;
            
            if( sqsum )
            {
                sqsum[-cn] = 0;
                sqsum++;
            }
        }

        for( y = 1; y < size.height; y++ )
        {
            src += srcstep - cn;
            sum += sumstep - cn;
            tilted += tiltedstep - cn;
            buf += -cn;
            
            if( sqsum )
                sqsum += sqsumstep - cn;

            for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
            {
                T it = src[0];
                ST t0 = s = it;
                QT tq0 = sq = (QT)it*it;

                sum[-cn] = 0;
                if( sqsum )
                    sqsum[-cn] = 0;
                tilted[-cn] = tilted[-tiltedstep];

                sum[0] = sum[-sumstep] + t0;
                if( sqsum )
                    sqsum[0] = sqsum[-sqsumstep] + tq0;
                tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];

                for( x = cn; x < size.width - cn; x += cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = (QT)it*it;
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    if( sqsum )
                        sqsum[x] = sqsum[x - sqsumstep] + sq;
                    t1 += buf[x + cn] + t0 + tilted[x - tiltedstep - cn];
                    tilted[x] = t1;
                }

                if( size.width > cn )
                {
                    ST t1 = buf[x];
                    buf[x - cn] = t1 + t0;
                    t0 = it = src[x];
                    tq0 = (QT)it*it;
                    s += t0;
                    sq += tq0;
                    sum[x] = sum[x - sumstep] + s;
                    if( sqsum )
                        sqsum[x] = sqsum[x - sqsumstep] + sq;
                    tilted[x] = t0 + t1 + tilted[x - tiltedstep - cn];
                    buf[x] = t0;
                }
                
                if( sqsum )
                    sqsum++;
            }
        }
    }
}

    
#define DEF_INTEGRAL_FUNC(suffix, T, ST, QT) \
void integral_##suffix( T* src, size_t srcstep, ST* sum, size_t sumstep, QT* sqsum, size_t sqsumstep, \
                        ST* tilted, size_t tiltedstep, Size size, int cn ) \
{ integral_(src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tiltedstep, size, cn); }

DEF_INTEGRAL_FUNC(8u32s, uchar, int, double)
DEF_INTEGRAL_FUNC(8u32f, uchar, float, double)
DEF_INTEGRAL_FUNC(8u64f, uchar, double, double)
DEF_INTEGRAL_FUNC(32f, float, float, float)
DEF_INTEGRAL_FUNC(32f64f, float, double, double)
DEF_INTEGRAL_FUNC(64f, double, double, double)
    
typedef void (*IntegralFunc)(const uchar* src, size_t srcstep, uchar* sum, size_t sumstep,
                             uchar* sqsum, size_t sqsumstep, uchar* tilted, size_t tstep,
                             Size size, int cn );

}


void cv::integral( const InputArray& _src, OutputArray _sum, OutputArray _sqsum, OutputArray _tilted, int sdepth )
{
    Mat src = _src.getMat(), sum, sqsum, tilted;
    int depth = src.depth(), cn = src.channels();
    Size isize(src.cols + 1, src.rows+1);

    if( sdepth <= 0 )
        sdepth = depth == CV_8U ? CV_32S : CV_64F;
    sdepth = CV_MAT_DEPTH(sdepth);
    _sum.create( isize, CV_MAKETYPE(sdepth, cn) );
    sum = _sum.getMat();
    
    if( _tilted.needed() )
    {
        _tilted.create( isize, CV_MAKETYPE(sdepth, cn) );
        tilted = _tilted.getMat();
    }
    
    if( _sqsum.needed() )
    {
        _sqsum.create( isize, CV_MAKETYPE(CV_64F, cn) );
        sqsum = _sqsum.getMat();
    }
    
    IntegralFunc func = 0;

    if( depth == CV_8U && sdepth == CV_32S )
        func = (IntegralFunc)integral_8u32s;
    else if( depth == CV_8U && sdepth == CV_32F )
        func = (IntegralFunc)integral_8u32f;
    else if( depth == CV_8U && sdepth == CV_64F )
        func = (IntegralFunc)integral_8u64f;
    else if( depth == CV_32F && sdepth == CV_32F )
        func = (IntegralFunc)integral_32f;
    else if( depth == CV_32F && sdepth == CV_64F )
        func = (IntegralFunc)integral_32f64f;
    else if( depth == CV_64F && sdepth == CV_64F )
        func = (IntegralFunc)integral_64f;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    func( src.data, src.step, sum.data, sum.step, sqsum.data, sqsum.step,
          tilted.data, tilted.step, src.size(), cn );
}
    
void cv::integral( const InputArray& src, OutputArray sum, int sdepth )
{
    integral( src, sum, OutputArray(), OutputArray(), sdepth );
}

void cv::integral( const InputArray& src, OutputArray sum, OutputArray sqsum, int sdepth )
{
    integral( src, sum, sqsum, OutputArray(), sdepth );
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
    cv::integral( src, sum, psqsum ? cv::OutputArray(*psqsum) : cv::OutputArray(),
                  ptilted ? cv::OutputArray(*ptilted) : cv::OutputArray(), sum.depth() );

    CV_Assert( sum.data == sum0.data && sqsum.data == sqsum0.data && tilted.data == tilted0.data );
}

/* End of file. */
