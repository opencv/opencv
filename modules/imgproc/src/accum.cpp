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

inline float sqr(uchar a) { return CV_8TO32F_SQR(a); }
inline float sqr(float a) { return a*a; }
    
inline double sqr(double a) { return a*a; }
    
inline Vec3f sqr(const Vec3b& a)
{
    return Vec3f(CV_8TO32F_SQR(a[0]), CV_8TO32F_SQR(a[1]), CV_8TO32F_SQR(a[2]));
}
inline Vec3f sqr(const Vec3f& a)
{
    return Vec3f(a[0]*a[0], a[1]*a[1], a[2]*a[2]);
}
inline Vec3d sqr(const Vec3d& a)
{
    return Vec3d(a[0]*a[0], a[1]*a[1], a[2]*a[2]);
}   
inline float multiply(uchar a, uchar b) { return CV_8TO32F(a)*CV_8TO32F(b); }
inline float multiply(float a, float b) { return a*b; }
inline double multiply(double a, double b) { return a*b; }    
inline Vec3f multiply(const Vec3b& a, const Vec3b& b)
{
    return Vec3f(
        CV_8TO32F(a[0])*CV_8TO32F(b[0]),
        CV_8TO32F(a[1])*CV_8TO32F(b[1]),
        CV_8TO32F(a[2])*CV_8TO32F(b[2]));
}
inline Vec3f multiply(const Vec3f& a, const Vec3f& b)
{
    return Vec3f(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
inline Vec3d multiply(const Vec3d& a, const Vec3d& b)
{
    return Vec3d(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
    
inline float addw(uchar a, float alpha, float b, float beta)
{
    return b*beta + CV_8TO32F(a)*alpha;
}
inline float addw(float a, float alpha, float b, float beta)
{
    return b*beta + a*alpha;
}
inline double addw(uchar a, double alpha, double b, double beta)
{
    return b*beta + CV_8TO32F(a)*alpha;
}
inline double addw(float a, double alpha, double b, double beta)
{
    return b*beta + a*alpha;
}
inline double addw(double a, double alpha, double b, double beta)
{
    return b*beta + a*alpha;
}
    
inline Vec3f addw(const Vec3b& a, float alpha, const Vec3f& b, float beta)
{
    return Vec3f(b[0]*beta + CV_8TO32F(a[0])*alpha,
                 b[1]*beta + CV_8TO32F(a[1])*alpha,
                 b[2]*beta + CV_8TO32F(a[2])*alpha);
}
inline Vec3f addw(const Vec3f& a, float alpha, const Vec3f& b, float beta)
{
    return Vec3f(b[0]*beta + a[0]*alpha, b[1]*beta + a[1]*alpha, b[2]*beta + a[2]*alpha);
}
inline Vec3d addw(const Vec3b& a, double alpha, const Vec3d& b, double beta)
{
    return Vec3d(b[0]*beta + CV_8TO32F(a[0])*alpha,
                 b[1]*beta + CV_8TO32F(a[1])*alpha,
                 b[2]*beta + CV_8TO32F(a[2])*alpha);
}
inline Vec3d addw(const Vec3f& a, double alpha, const Vec3d& b, double beta)
{
    return Vec3d(b[0]*beta + a[0]*alpha, b[1]*beta + a[1]*alpha, b[2]*beta + a[2]*alpha);
}
inline Vec3d addw(const Vec3d& a, double alpha, const Vec3d& b, double beta)
{
    return Vec3d(b[0]*beta + a[0]*alpha, b[1]*beta + a[1]*alpha, b[2]*beta + a[2]*alpha);
}    

template<typename T, typename AT> void
acc_( const Mat& _src, Mat& _dst )
{
    Size size = _src.size();
    size.width *= _src.channels();

    if( _src.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);

        for( j = 0; j <= size.width - 4; j += 4 )
        {
            AT t0 = dst[j] + src[j], t1 = dst[j+1] + src[j+1];
            dst[j] = t0; dst[j+1] = t1;
            t0 = dst[j+2] + src[j+2]; t1 = dst[j+3] + src[j+3];
            dst[j+2] = t0; dst[j+3] = t1;
        }

        for( ; j < size.width; j++ )
            dst[j] += src[j];
    }
}


template<typename T, typename AT> void
accSqr_( const Mat& _src, Mat& _dst )
{
    Size size = _src.size();
    size.width *= _src.channels();

    if( _src.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);

        for( j = 0; j <= size.width - 4; j += 4 )
        {
            AT t0 = dst[j] + sqr(src[j]), t1 = dst[j+1] + sqr(src[j+1]);
            dst[j] = t0; dst[j+1] = t1;
            t0 = dst[j+2] + sqr(src[j+2]); t1 = dst[j+3] + sqr(src[j+3]);
            dst[j+2] = t0; dst[j+3] = t1;
        }

        for( ; j < size.width; j++ )
            dst[j] += sqr(src[j]);
    }
}


template<typename T, typename AT> void
accProd_( const Mat& _src1, const Mat& _src2, Mat& _dst )
{
    Size size = _src1.size();
    size.width *= _src1.channels();

    if( _src1.isContinuous() && _src2.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src1 = (const T*)(_src1.data + _src1.step*i);
        const T* src2 = (const T*)(_src2.data + _src2.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);

        for( j = 0; j <= size.width - 4; j += 4 )
        {
            AT t0, t1;
            t0 = dst[j] + multiply(src1[j], src2[j]);
            t1 = dst[j+1] + multiply(src1[j+1], src2[j+1]);
            dst[j] = t0; dst[j+1] = t1;
            t0 = dst[j+2] + multiply(src1[j+2], src2[j+2]);
            t1 = dst[j+3] + multiply(src1[j+3], src2[j+3]);
            dst[j+2] = t0; dst[j+3] = t1;
        }

        for( ; j < size.width; j++ )
            dst[j] += multiply(src1[j], src2[j]);
    }
}


template<typename T, typename AT> void
accW_( const Mat& _src, Mat& _dst, double _alpha )
{
    AT alpha = (AT)_alpha, beta = (AT)(1 - _alpha);
    Size size = _src.size();
    size.width *= _src.channels();

    if( _src.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);

        for( j = 0; j <= size.width - 4; j += 4 )
        {
            AT t0, t1;
            t0 = addw(src[j], alpha, dst[j], beta);
            t1 = addw(src[j+1], alpha, dst[j+1], beta);
            dst[j] = t0; dst[j+1] = t1;
            t0 = addw(src[j+2], alpha, dst[j+2], beta);
            t1 = addw(src[j+3], alpha, dst[j+3], beta);
            dst[j+2] = t0; dst[j+3] = t1;
        }

        for( ; j < size.width; j++ )
            dst[j] = addw(src[j], alpha, dst[j], beta);
    }
}


template<typename T, typename AT> void
accMask_( const Mat& _src, Mat& _dst, const Mat& _mask )
{
    Size size = _src.size();

    if( _src.isContinuous() && _dst.isContinuous() && _mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);
        const uchar* mask = _mask.data + _mask.step*i;

        for( j = 0; j < size.width; j++ )
            if( mask[j] )
                dst[j] += src[j];
    }
}


template<typename T, typename AT> void
accSqrMask_( const Mat& _src, Mat& _dst, const Mat& _mask )
{
    Size size = _src.size();

    if( _src.isContinuous() && _dst.isContinuous() && _mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);
        const uchar* mask = _mask.data + _mask.step*i;

        for( j = 0; j < size.width; j++ )
            if( mask[j] )
                dst[j] += sqr(src[j]);
    }
}


template<typename T, typename AT> void
accProdMask_( const Mat& _src1, const Mat& _src2, Mat& _dst, const Mat& _mask )
{
    Size size = _src1.size();

    if( _src1.isContinuous() && _src2.isContinuous() &&
        _dst.isContinuous() && _mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src1 = (const T*)(_src1.data + _src1.step*i);
        const T* src2 = (const T*)(_src2.data + _src2.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);
        const uchar* mask = _mask.data + _mask.step*i;

        for( j = 0; j < size.width; j++ )
            if( mask[j] )
                dst[j] += multiply(src1[j], src2[j]);
    }
}


template<typename T, typename AT> void
accWMask_( const Mat& _src, Mat& _dst, double _alpha, const Mat& _mask )
{
    typedef typename DataType<AT>::channel_type AT1;
    AT1 alpha = (AT1)_alpha, beta = (AT1)(1 - _alpha);
    Size size = _src.size();

    if( _src.isContinuous() && _dst.isContinuous() && _mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        const T* src = (const T*)(_src.data + _src.step*i);
        AT* dst = (AT*)(_dst.data + _dst.step*i);
        const uchar* mask = _mask.data + _mask.step*i;

        for( j = 0; j < size.width; j++ )
            if( mask[j] )
                dst[j] = addw(src[j], alpha, dst[j], beta);
    }
}


typedef void (*AccFunc)(const Mat&, Mat&);
typedef void (*AccMaskFunc)(const Mat&, Mat&, const Mat&);
typedef void (*AccProdFunc)(const Mat&, const Mat&, Mat&);
typedef void (*AccProdMaskFunc)(const Mat&, const Mat&, Mat&, const Mat&);
typedef void (*AccWFunc)(const Mat&, Mat&, double);
typedef void (*AccWMaskFunc)(const Mat&, Mat&, double, const Mat&);

void accumulate( const Mat& src, Mat& dst, const Mat& mask )
{
    CV_Assert( dst.size() == src.size() && dst.channels() == src.channels() );
    
    if( !mask.data )
    {
        AccFunc func = 0;
        if( src.depth() == CV_8U && dst.depth() == CV_32F )
            func = acc_<uchar, float>;
        else if( src.depth() == CV_8U && dst.depth() == CV_64F )
            func = acc_<uchar, double>;
        else if( src.depth() == CV_32F && dst.depth() == CV_32F )
            func = acc_<float, float>;
        else if( src.depth() == CV_32F && dst.depth() == CV_64F )
            func = acc_<float, double>;
        else if( src.depth() == CV_64F && dst.depth() == CV_64F )
            func = acc_<double, double>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst );
    }
    else
    {
        CV_Assert( mask.size() == src.size() && mask.type() == CV_8UC1 );

        AccMaskFunc func = 0;
        if( src.type() == CV_8UC1 && dst.type() == CV_32FC1 )
            func = accMask_<uchar, float>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_32FC3 )
            func = accMask_<Vec3b, Vec3f>;
        else if( src.type() == CV_8UC1 && dst.type() == CV_64FC1 )
            func = accMask_<uchar, double>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_64FC3 )
            func = accMask_<Vec3b, Vec3d>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_32FC1 )
            func = accMask_<float, float>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_32FC3 )
            func = accMask_<Vec3f, Vec3f>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_64FC1 )
            func = accMask_<float, double>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_64FC3 )
            func = accMask_<Vec3f, Vec3d>;
        else if( src.type() == CV_64FC1 && dst.type() == CV_64FC1 )
            func = accMask_<double, double>;
        else if( src.type() == CV_64FC3 && dst.type() == CV_64FC3 )
            func = accMask_<Vec3d, Vec3d>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, mask );
    }
}


void accumulateSquare( const Mat& src, Mat& dst, const Mat& mask )
{
    CV_Assert( dst.size() == src.size() && dst.channels() == src.channels() );
    
    if( !mask.data )
    {
        AccFunc func = 0;
        if( src.depth() == CV_8U && dst.depth() == CV_32F )
            func = accSqr_<uchar, float>;
        else if( src.depth() == CV_8U && dst.depth() == CV_64F )
            func = accSqr_<uchar, double>;
        else if( src.depth() == CV_32F && dst.depth() == CV_32F )
            func = accSqr_<float, float>;
        else if( src.depth() == CV_32F && dst.depth() == CV_64F )
            func = accSqr_<float, double>;
        else if( src.depth() == CV_64F && dst.depth() == CV_64F )
            func = accSqr_<double, double>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst );
    }
    else
    {
        CV_Assert( mask.size() == src.size() && mask.type() == CV_8UC1 );

        AccMaskFunc func = 0;
        if( src.type() == CV_8UC1 && dst.type() == CV_32FC1 )
            func = accSqrMask_<uchar, float>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_32FC3 )
            func = accSqrMask_<Vec3b, Vec3f>;
        else if( src.type() == CV_8UC1 && dst.type() == CV_64FC1 )
            func = accSqrMask_<uchar, double>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_64FC3 )
            func = accSqrMask_<Vec3b, Vec3d>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_32FC1 )
            func = accSqrMask_<float, float>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_32FC3 )
            func = accSqrMask_<Vec3f, Vec3f>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_64FC1 )
            func = accSqrMask_<float, double>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_64FC3 )
            func = accSqrMask_<Vec3f, Vec3d>;
        else if( src.type() == CV_64FC1 && dst.type() == CV_64FC1 )
            func = accSqrMask_<double, double>;
        else if( src.type() == CV_64FC3 && dst.type() == CV_64FC3 )
            func = accSqrMask_<Vec3d, Vec3d>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, mask );
    }
}


void accumulateProduct( const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask )
{
    CV_Assert( dst.size() == src1.size() && dst.channels() == src1.channels() &&
               src1.size() == src2.size() && src1.type() == src2.type() );
    
    if( !mask.data )
    {
        AccProdFunc func = 0;
        if( src1.depth() == CV_8U && dst.depth() == CV_32F )
            func = accProd_<uchar, float>;
        else if( src1.depth() == CV_8U && dst.depth() == CV_64F )
            func = accProd_<uchar, double>;
        else if( src1.depth() == CV_32F && dst.depth() == CV_32F )
            func = accProd_<float, float>;
        else if( src1.depth() == CV_32F && dst.depth() == CV_64F )
            func = accProd_<float, double>;
        else if( src1.depth() == CV_64F && dst.depth() == CV_64F )
            func = accProd_<double, double>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src1, src2, dst );
    }
    else
    {
        CV_Assert( mask.size() == src1.size() && mask.type() == CV_8UC1 );

        AccProdMaskFunc func = 0;
        if( src1.type() == CV_8UC1 && dst.type() == CV_32FC1 )
            func = accProdMask_<uchar, float>;
        else if( src1.type() == CV_8UC3 && dst.type() == CV_32FC3 )
            func = accProdMask_<Vec3b, Vec3f>;
        else if( src1.type() == CV_8UC1 && dst.type() == CV_64FC1 )
            func = accProdMask_<uchar, double>;
        else if( src1.type() == CV_8UC3 && dst.type() == CV_64FC3 )
            func = accProdMask_<Vec3b, Vec3d>;
        else if( src1.type() == CV_32FC1 && dst.type() == CV_32FC1 )
            func = accProdMask_<float, float>;
        else if( src1.type() == CV_32FC3 && dst.type() == CV_32FC3 )
            func = accProdMask_<Vec3f, Vec3f>;
        else if( src1.type() == CV_32FC1 && dst.type() == CV_64FC1 )
            func = accProdMask_<float, double>;
        else if( src1.type() == CV_32FC3 && dst.type() == CV_64FC3 )
            func = accProdMask_<Vec3f, Vec3d>;
        else if( src1.type() == CV_64FC1 && dst.type() == CV_64FC1 )
            func = accProdMask_<double, double>;
        else if( src1.type() == CV_64FC3 && dst.type() == CV_64FC3 )
            func = accProdMask_<Vec3d, Vec3d>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src1, src2, dst, mask );
    }
}


void accumulateWeighted( const Mat& src, Mat& dst, double alpha, const Mat& mask )
{
    CV_Assert( dst.size() == src.size() && dst.channels() == src.channels() );
    
    if( !mask.data )
    {
        AccWFunc func = 0;
        if( src.depth() == CV_8U && dst.depth() == CV_32F )
            func = accW_<uchar, float>;
        else if( src.depth() == CV_8U && dst.depth() == CV_64F )
            func = accW_<uchar, double>;
        else if( src.depth() == CV_32F && dst.depth() == CV_32F )
            func = accW_<float, float>;
        else if( src.depth() == CV_32F && dst.depth() == CV_64F )
            func = accW_<float, double>;
        else if( src.depth() == CV_64F && dst.depth() == CV_64F )
            func = accW_<double, double>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, alpha );
    }
    else
    {
        CV_Assert( mask.size() == src.size() && mask.type() == CV_8UC1 );

        AccWMaskFunc func = 0;
        if( src.type() == CV_8UC1 && dst.type() == CV_32FC1 )
            func = accWMask_<uchar, float>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_32FC3 )
            func = accWMask_<Vec3b, Vec3f>;
        else if( src.type() == CV_8UC1 && dst.type() == CV_64FC1 )
            func = accWMask_<uchar, double>;
        else if( src.type() == CV_8UC3 && dst.type() == CV_64FC3 )
            func = accWMask_<Vec3b, Vec3d>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_32FC1 )
            func = accWMask_<float, float>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_32FC3 )
            func = accWMask_<Vec3f, Vec3f>;
        else if( src.type() == CV_32FC1 && dst.type() == CV_64FC1 )
            func = accWMask_<float, double>;
        else if( src.type() == CV_32FC3 && dst.type() == CV_64FC3 )
            func = accWMask_<Vec3f, Vec3d>;
        else if( src.type() == CV_64FC1 && dst.type() == CV_64FC1 )
            func = accWMask_<double, double>;
        else if( src.type() == CV_64FC3 && dst.type() == CV_64FC3 )
            func = accWMask_<Vec3d, Vec3d>;
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, alpha, mask );
    }
}

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
