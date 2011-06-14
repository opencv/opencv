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
/
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

template<typename T, typename AT> void
acc_( const T* src, AT* dst, const uchar* mask, int len, int cn )
{
    int i = 0;
    
    if( !mask )
    {
        len *= cn;
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = src[i] + dst[i];
            t1 = src[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;
            
            t0 = src[i+2] + dst[i+2];
            t1 = src[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        
        for( ; i < len; i++ )
            dst[i] += src[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += src[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = src[0] + dst[0];
                AT t1 = src[1] + dst[1];
                AT t2 = src[2] + dst[2];
                
                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += src[k];
            }
    }
}

    
template<typename T, typename AT> void
accSqr_( const T* src, AT* dst, const uchar* mask, int len, int cn )
{
    int i = 0;
    
    if( !mask )
    {
        len *= cn;
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = (AT)src[i]*src[i] + dst[i];
            t1 = (AT)src[i+1]*src[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;
            
            t0 = (AT)src[i+2]*src[i+2] + dst[i+2];
            t1 = (AT)src[i+3]*src[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        
        for( ; i < len; i++ )
            dst[i] += (AT)src[i]*src[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += (AT)src[i]*src[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = (AT)src[0]*src[0] + dst[0];
                AT t1 = (AT)src[1]*src[1] + dst[1];
                AT t2 = (AT)src[2]*src[2] + dst[2];
                
                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += (AT)src[k]*src[k];
            }
    }
}
   
    
template<typename T, typename AT> void
accProd_( const T* src1, const T* src2, AT* dst, const uchar* mask, int len, int cn )
{
    int i = 0;
    
    if( !mask )
    {
        len *= cn;
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = (AT)src1[i]*src2[i] + dst[i];
            t1 = (AT)src1[i+1]*src2[i+1] + dst[i+1];
            dst[i] = t0; dst[i+1] = t1;
            
            t0 = (AT)src1[i+2]*src2[i+2] + dst[i+2];
            t1 = (AT)src1[i+3]*src2[i+3] + dst[i+3];
            dst[i+2] = t0; dst[i+3] = t1;
        }
        
        for( ; i < len; i++ )
            dst[i] += (AT)src1[i]*src2[i];
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] += (AT)src1[i]*src2[i];
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src1 += 3, src2 += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = (AT)src1[0]*src2[0] + dst[0];
                AT t1 = (AT)src1[1]*src2[1] + dst[1];
                AT t2 = (AT)src1[2]*src2[2] + dst[2];
                
                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src1 += cn, src2 += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += (AT)src1[k]*src2[k];
            }
    }
}

    
template<typename T, typename AT> void
accW_( const T* src, AT* dst, const uchar* mask, int len, int cn, double alpha )
{
    AT a = (AT)alpha, b = 1 - a;
    int i = 0;
    
    if( !mask )
    {
        len *= cn;
        for( ; i <= len - 4; i += 4 )
        {
            AT t0, t1;
            t0 = src[i]*a + dst[i]*b;
            t1 = src[i+1]*a + dst[i+1]*b;
            dst[i] = t0; dst[i+1] = t1;
            
            t0 = src[i+2]*a + dst[i+2]*b;
            t1 = src[i+3]*a + dst[i+3]*b;
            dst[i+2] = t0; dst[i+3] = t1;
        }
        
        for( ; i < len; i++ )
            dst[i] = src[i]*a + dst[i]*b;
    }
    else if( cn == 1 )
    {
        for( ; i < len; i++ )
        {
            if( mask[i] )
                dst[i] = src[i]*a + dst[i]*b;
        }
    }
    else if( cn == 3 )
    {
        for( ; i < len; i++, src += 3, dst += 3 )
        {
            if( mask[i] )
            {
                AT t0 = src[0]*a + dst[0]*b;
                AT t1 = src[1]*a + dst[1]*b;
                AT t2 = src[2]*a + dst[2]*b;
                
                dst[0] = t0; dst[1] = t1; dst[2] = t2;
            }
        }
    }
    else
    {
        for( ; i < len; i++, src += cn, dst += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    dst[k] += src[k]*a + dst[k]*b;
            }
    }
}


#define DEF_ACC_FUNCS(suffix, type, acctype) \
static void acc_##suffix(const type* src, acctype* dst, \
                         const uchar* mask, int len, int cn) \
{ acc_(src, dst, mask, len, cn); } \
\
static void accSqr_##suffix(const type* src, acctype* dst, \
                            const uchar* mask, int len, int cn) \
{ accSqr_(src, dst, mask, len, cn); } \
\
static void accProd_##suffix(const type* src1, const type* src2, \
                             acctype* dst, const uchar* mask, int len, int cn) \
{ accProd_(src1, src2, dst, mask, len, cn); } \
\
static void accW_##suffix(const type* src, acctype* dst, \
                          const uchar* mask, int len, int cn, double alpha) \
{ accW_(src, dst, mask, len, cn, alpha); }


DEF_ACC_FUNCS(8u32f, uchar, float)
DEF_ACC_FUNCS(8u64f, uchar, double)
DEF_ACC_FUNCS(16u32f, ushort, float)
DEF_ACC_FUNCS(16u64f, ushort, double)
DEF_ACC_FUNCS(32f, float, float)
DEF_ACC_FUNCS(32f64f, float, double)
DEF_ACC_FUNCS(64f, double, double)
    
    
typedef void (*AccFunc)(const uchar*, uchar*, const uchar*, int, int);
typedef void (*AccProdFunc)(const uchar*, const uchar*, uchar*, const uchar*, int, int);
typedef void (*AccWFunc)(const uchar*, uchar*, const uchar*, int, int, double);

static AccFunc accTab[] =
{
    (AccFunc)acc_8u32f, (AccFunc)acc_8u64f,
    (AccFunc)acc_16u32f, (AccFunc)acc_16u64f,
    (AccFunc)acc_32f, (AccFunc)acc_32f64f,
    (AccFunc)acc_64f
};

static AccFunc accSqrTab[] =
{
    (AccFunc)accSqr_8u32f, (AccFunc)accSqr_8u64f,
    (AccFunc)accSqr_16u32f, (AccFunc)accSqr_16u64f,
    (AccFunc)accSqr_32f, (AccFunc)accSqr_32f64f,
    (AccFunc)accSqr_64f
};

static AccProdFunc accProdTab[] =
{
    (AccProdFunc)accProd_8u32f, (AccProdFunc)accProd_8u64f,
    (AccProdFunc)accProd_16u32f, (AccProdFunc)accProd_16u64f,
    (AccProdFunc)accProd_32f, (AccProdFunc)accProd_32f64f,
    (AccProdFunc)accProd_64f
};

static AccWFunc accWTab[] =
{
    (AccWFunc)accW_8u32f, (AccWFunc)accW_8u64f,
    (AccWFunc)accW_16u32f, (AccWFunc)accW_16u64f,
    (AccWFunc)accW_32f, (AccWFunc)accW_32f64f,
    (AccWFunc)accW_64f
};

inline int getAccTabIdx(int sdepth, int ddepth)
{
    return sdepth == CV_8U && ddepth == CV_32F ? 0 :
           sdepth == CV_8U && ddepth == CV_64F ? 1 :
           sdepth == CV_16U && ddepth == CV_32F ? 2 :
           sdepth == CV_16U && ddepth == CV_64F ? 3 :
           sdepth == CV_32F && ddepth == CV_32F ? 4 :
           sdepth == CV_32F && ddepth == CV_64F ? 5 :
           sdepth == CV_64F && ddepth == CV_64F ? 6 : -1;
}    
    
}
    
void cv::accumulate( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
    int sdepth = src.depth(), ddepth = dst.depth(), cn = src.channels();
    
    CV_Assert( dst.size == src.size && dst.channels() == cn );
    CV_Assert( mask.empty() || (mask.size == src.size && mask.type() == CV_8U) );
    
    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accTab[fidx] : 0;
    CV_Assert( func != 0 );
    
    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, cn);
}


void cv::accumulateSquare( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
    int sdepth = src.depth(), ddepth = dst.depth(), cn = src.channels();
    
    CV_Assert( dst.size == src.size && dst.channels() == cn );
    CV_Assert( mask.empty() || (mask.size == src.size && mask.type() == CV_8U) );
    
    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accSqrTab[fidx] : 0;
    CV_Assert( func != 0 );
    
    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, cn);
}

void cv::accumulateProduct( InputArray _src1, InputArray _src2,
                            InputOutputArray _dst, InputArray _mask )
{
    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
    int sdepth = src1.depth(), ddepth = dst.depth(), cn = src1.channels();
    
    CV_Assert( src2.size && src1.size && src2.type() == src1.type() );
    CV_Assert( dst.size == src1.size && dst.channels() == cn );
    CV_Assert( mask.empty() || (mask.size == src1.size && mask.type() == CV_8U) );
    
    int fidx = getAccTabIdx(sdepth, ddepth);
    AccProdFunc func = fidx >= 0 ? accProdTab[fidx] : 0;
    CV_Assert( func != 0 );
    
    const Mat* arrays[] = {&src1, &src2, &dst, &mask, 0};
    uchar* ptrs[4];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], ptrs[3], len, cn);
}


void cv::accumulateWeighted( InputArray _src, CV_IN_OUT InputOutputArray _dst,
                             double alpha, InputArray _mask )
{
    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
    int sdepth = src.depth(), ddepth = dst.depth(), cn = src.channels();
    
    CV_Assert( dst.size == src.size && dst.channels() == cn );
    CV_Assert( mask.empty() || (mask.size == src.size && mask.type() == CV_8U) );
    
    int fidx = getAccTabIdx(sdepth, ddepth);
    AccWFunc func = fidx >= 0 ? accWTab[fidx] : 0;
    CV_Assert( func != 0 );
    
    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, cn, alpha);
}


CV_IMPL void
cvAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulate( src, dst, mask );
}

CV_IMPL void
cvSquareAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateSquare( src, dst, mask );
}

CV_IMPL void
cvMultiplyAcc( const void* arr1, const void* arr2,
               void* sumarr, const void* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(arr1), src2 = cv::cvarrToMat(arr2);
    cv::Mat dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateProduct( src1, src2, dst, mask );
}

CV_IMPL void
cvRunningAvg( const void* arr, void* sumarr, double alpha, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateWeighted( src, dst, alpha, mask );
}

/* End of file. */
