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

template<typename T> static inline Scalar rawToScalar(const T& v)
{
    Scalar s;
    typedef typename DataType<T>::channel_type T1;
    int i, n = DataType<T>::channels;
    for( i = 0; i < n; i++ )
        s.val[i] = ((T1*)&v)[i];
    return s;
}    

/****************************************************************************************\
*                                        sum                                             *
\****************************************************************************************/

template<typename T, typename WT, typename ST, int BLOCK_SIZE>
static Scalar sumBlock_( const Mat& srcmat )
{
    assert( DataType<T>::type == srcmat.type() );
    Size size = getContinuousSize( srcmat );
    ST s0 = 0;
    WT s = 0;
    int y, remaining = BLOCK_SIZE;

    for( y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x <= limit - 4; x += 4 )
            {
                s += src[x];
                s += src[x+1];
                s += src[x+2];
                s += src[x+3];
            }
            for( ; x < limit; x++ )
                s += src[x];
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 += s;
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return rawToScalar(s0);
}

template<typename T, typename ST>
static Scalar sum_( const Mat& srcmat )
{
    assert( DataType<T>::type == srcmat.type() );
    Size size = getContinuousSize( srcmat );
    ST s = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            s += src[x];
            s += src[x+1];
            s += src[x+2];
            s += src[x+3];
        }
        for( ; x < size.width; x++ )
            s += src[x];
    }
    return rawToScalar(s);
}

typedef Scalar (*SumFunc)(const Mat& src);

Scalar sum( const Mat& m )
{
    static SumFunc tab[]=
    {
        sumBlock_<uchar, unsigned, double, 1<<24>,
        sumBlock_<schar, int, double, 1<<24>,
        sumBlock_<ushort, unsigned, double, 1<<16>,
        sumBlock_<short, int, double, 1<<16>,
        sum_<int, double>,
        sum_<float, double>,
        sum_<double, double>, 0,

        sumBlock_<Vec<uchar, 2>, Vec<unsigned, 2>, Vec<double, 2>, 1<<24>,
        sumBlock_<Vec<schar, 2>, Vec<int, 2>, Vec<double, 2>, 1<<24>,
        sumBlock_<Vec<ushort, 2>, Vec<unsigned, 2>, Vec<double, 2>, 1<<16>,
        sumBlock_<Vec<short, 2>, Vec<int, 2>, Vec<double, 2>, 1<<16>,
        sum_<Vec<int, 2>, Vec<double, 2> >,
        sum_<Vec<float, 2>, Vec<double, 2> >,
        sum_<Vec<double, 2>, Vec<double, 2> >, 0,

        sumBlock_<Vec<uchar, 3>, Vec<unsigned, 3>, Vec<double, 3>, 1<<24>,
        sumBlock_<Vec<schar, 3>, Vec<int, 3>, Vec<double, 3>, 1<<24>,
        sumBlock_<Vec<ushort, 3>, Vec<unsigned, 3>, Vec<double, 3>, 1<<16>,
        sumBlock_<Vec<short, 3>, Vec<int, 3>, Vec<double, 3>, 1<<16>,
        sum_<Vec<int, 3>, Vec<double, 3> >,
        sum_<Vec<float, 3>, Vec<double, 3> >,
        sum_<Vec<double, 3>, Vec<double, 3> >, 0,

        sumBlock_<Vec<uchar, 4>, Vec<unsigned, 4>, Vec<double, 4>, 1<<24>,
        sumBlock_<Vec<schar, 4>, Vec<int, 4>, Vec<double, 4>, 1<<24>,
        sumBlock_<Vec<ushort, 4>, Vec<unsigned, 4>, Vec<double, 4>, 1<<16>,
        sumBlock_<Vec<short, 4>, Vec<int, 4>, Vec<double, 4>, 1<<16>,
        sum_<Vec<int, 4>, Vec<double, 4> >,
        sum_<Vec<float, 4>, Vec<double, 4> >,
        sum_<Vec<double, 4>, Vec<double, 4> >, 0
    };

    CV_Assert( m.channels() <= 4 );

    SumFunc func = tab[m.type()];
    CV_Assert( func != 0 );

    if( m.dims > 2 )
    {
        const Mat* arrays[] = {&m, 0};
        Mat planes[1];
        NAryMatIterator it(arrays, planes);
        Scalar s;
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            s += func(it.planes[0]);
        return s;
    }
    
    return func(m);
}

/****************************************************************************************\
*                                     countNonZero                                       *
\****************************************************************************************/

template<typename T>
static int countNonZero_( const Mat& srcmat )
{
    //assert( DataType<T>::type == srcmat.type() );
    const T* src = (const T*)srcmat.data;
    size_t step = srcmat.step/sizeof(src[0]);
    Size size = getContinuousSize( srcmat );
    int nz = 0;

    for( ; size.height--; src += step )
    {
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
            nz += (src[x] != 0) + (src[x+1] != 0) + (src[x+2] != 0) + (src[x+3] != 0);
        for( ; x < size.width; x++ )
            nz += src[x] != 0;
    }
    return nz;
}

typedef int (*CountNonZeroFunc)(const Mat& src);

int countNonZero( const Mat& m )
{
    static CountNonZeroFunc tab[] =
    {
        countNonZero_<uchar>, countNonZero_<uchar>, countNonZero_<ushort>,
        countNonZero_<ushort>, countNonZero_<int>, countNonZero_<float>,
        countNonZero_<double>, 0
    };
    
    CountNonZeroFunc func = tab[m.depth()];
    CV_Assert( m.channels() == 1 && func != 0 );
    
    if( m.dims > 2 )
    {
        const Mat* arrays[] = {&m, 0};
        Mat planes[1];
        NAryMatIterator it(arrays, planes);
        int nz = 0;
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            nz += func(it.planes[0]);
        return nz;
    }
    
    return func(m);
}


/****************************************************************************************\
*                                         mean                                           *
\****************************************************************************************/

template<typename T, typename WT, typename ST, int BLOCK_SIZE>
static Scalar meanBlock_( const Mat& srcmat, const Mat& maskmat )
{
    assert( DataType<T>::type == srcmat.type() &&
        CV_8U == maskmat.type() && srcmat.size() == maskmat.size() );
    Size size = getContinuousSize( srcmat, maskmat );
    ST s0 = 0;
    WT s = 0;
    int y, remaining = BLOCK_SIZE, pix = 0;

    for( y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x < limit; x++ )
                if( mask[x] )
                    s += src[x], pix++;
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 += s;
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return rawToScalar(s0)*(1./std::max(pix, 1));
}


template<typename T, typename ST>
static Scalar mean_( const Mat& srcmat, const Mat& maskmat )
{
    assert( DataType<T>::type == srcmat.type() &&
        CV_8U == maskmat.type() && srcmat.size() == maskmat.size() );
    Size size = getContinuousSize( srcmat, maskmat );
    ST s = 0;
    int y, pix = 0;

    for( y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        for( int x = 0; x < size.width; x++ )
            if( mask[x] )
                s += src[x], pix++;
    }
    return rawToScalar(s)*(1./std::max(pix, 1));
}

typedef Scalar (*MeanMaskFunc)(const Mat& src, const Mat& mask);

Scalar mean(const Mat& m)
{
    return sum(m)*(1./m.total());
}

Scalar mean( const Mat& m, const Mat& mask )
{
    static MeanMaskFunc tab[]=
    {
        meanBlock_<uchar, unsigned, double, 1<<24>, 0,
        meanBlock_<ushort, unsigned, double, 1<<16>,
        meanBlock_<short, int, double, 1<<16>,
        mean_<int, double>,
        mean_<float, double>,
        mean_<double, double>, 0,

        meanBlock_<Vec<uchar, 2>, Vec<unsigned, 2>, Vec<double, 2>, 1<<24>, 0,
        meanBlock_<Vec<ushort, 2>, Vec<unsigned, 2>, Vec<double, 2>, 1<<16>,
        meanBlock_<Vec<short, 2>, Vec<int, 2>, Vec<double, 2>, 1<<16>,
        mean_<Vec<int, 2>, Vec<double, 2> >,
        mean_<Vec<float, 2>, Vec<double, 2> >,
        mean_<Vec<double, 2>, Vec<double, 2> >, 0,

        meanBlock_<Vec<uchar, 3>, Vec<unsigned, 3>, Vec<double, 3>, 1<<24>, 0,
        meanBlock_<Vec<ushort, 3>, Vec<unsigned, 3>, Vec<double, 3>, 1<<16>,
        meanBlock_<Vec<short, 3>, Vec<int, 3>, Vec<double, 3>, 1<<16>,
        mean_<Vec<int, 3>, Vec<double, 3> >,
        mean_<Vec<float, 3>, Vec<double, 3> >,
        mean_<Vec<double, 3>, Vec<double, 3> >, 0,

        meanBlock_<Vec<uchar, 4>, Vec<unsigned, 4>, Vec<double, 4>, 1<<24>, 0,
        meanBlock_<Vec<ushort, 4>, Vec<unsigned, 4>, Vec<double, 4>, 1<<16>,
        meanBlock_<Vec<short, 4>, Vec<int, 4>, Vec<double, 4>, 1<<16>,
        mean_<Vec<int, 4>, Vec<double, 4> >,
        mean_<Vec<float, 4>, Vec<double, 4> >,
        mean_<Vec<double, 4>, Vec<double, 4> >, 0
    };
    
    if( !mask.data )
        return mean(m);

    CV_Assert( m.channels() <= 4 && mask.type() == CV_8U );

    MeanMaskFunc func = tab[m.type()];
    CV_Assert( func != 0 );
    
    if( m.dims > 2 )
    {
        const Mat* arrays[] = {&m, &mask, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        double total = 0;
        Scalar s;
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            int n = countNonZero(it.planes[1]);
            s += mean(it.planes[0], it.planes[1])*(double)n;
            total += n;
        }
        return (s * 1./std::max(total, 1.));
    }
    
    CV_Assert( m.size() == mask.size() );
    return func( m, mask );
}

/****************************************************************************************\
*                                       meanStdDev                                       *
\****************************************************************************************/

template<typename T, typename SqT> struct SqrC1
{
    typedef T type1;
    typedef SqT rtype;
    rtype operator()(type1 x) const { return (SqT)x*x; }
};

template<typename T, typename SqT> struct SqrC2
{
    typedef Vec<T, 2> type1;
    typedef Vec<SqT, 2> rtype;
    rtype operator()(const type1& x) const { return rtype((SqT)x[0]*x[0], (SqT)x[1]*x[1]); }
};

template<typename T, typename SqT> struct SqrC3
{
    typedef Vec<T, 3> type1;
    typedef Vec<SqT, 3> rtype;
    rtype operator()(const type1& x) const
    { return rtype((SqT)x[0]*x[0], (SqT)x[1]*x[1], (SqT)x[2]*x[2]); }
};

template<typename T, typename SqT> struct SqrC4
{
    typedef Vec<T, 4> type1;
    typedef Vec<SqT, 4> rtype;
    rtype operator()(const type1& x) const
    { return rtype((SqT)x[0]*x[0], (SqT)x[1]*x[1], (SqT)x[2]*x[2], (SqT)x[3]*x[3]); }
};

template<> inline double SqrC1<uchar, double>::operator()(uchar x) const
{ return CV_SQR_8U(x); }

template<> inline Vec<double, 2> SqrC2<uchar, double>::operator()(const Vec<uchar, 2>& x) const
{ return Vec<double, 2>(CV_SQR_8U(x[0]), CV_SQR_8U(x[1])); }

template<> inline Vec<double, 3> SqrC3<uchar, double>::operator() (const Vec<uchar, 3>& x) const
{ return Vec<double, 3>(CV_SQR_8U(x[0]), CV_SQR_8U(x[1]), CV_SQR_8U(x[2])); }

template<> inline Vec<double, 4> SqrC4<uchar, double>::operator() (const Vec<uchar, 4>& x) const
{ return Vec<double, 4>(CV_SQR_8U(x[0]), CV_SQR_8U(x[1]), CV_SQR_8U(x[2]), CV_SQR_8U(x[3])); }


template<class SqrOp> static void
meanStdDev_( const Mat& srcmat, Scalar& _mean, Scalar& _stddev )
{
    SqrOp sqr;
    typedef typename SqrOp::type1 T;
    typedef typename SqrOp::rtype ST;
    typedef typename DataType<ST>::channel_type ST1;
    
    assert( DataType<T>::type == srcmat.type() );
    Size size = getContinuousSize( srcmat );
    ST s = 0, sq = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        for( int x = 0; x < size.width; x++ )
        {
            T v = src[x];
            s += v;
            sq += sqr(v);
        }
    }

    _mean = _stddev = Scalar();
    double scale = 1./std::max(size.width*size.height, 1);
    for( int i = 0; i < DataType<ST>::channels; i++ )
    {
        double t = ((ST1*)&s)[i]*scale;
        _mean.val[i] = t;
        _stddev.val[i] = std::sqrt(std::max(((ST1*)&sq)[i]*scale - t*t, 0.));
    }
}

template<class SqrOp> static void
meanStdDevMask_( const Mat& srcmat, const Mat& maskmat,
                 Scalar& _mean, Scalar& _stddev )
{
    SqrOp sqr;
    typedef typename SqrOp::type1 T;
    typedef typename SqrOp::rtype ST;
    typedef typename DataType<ST>::channel_type ST1;

    assert( DataType<T>::type == srcmat.type() &&
            CV_8U == maskmat.type() &&
            srcmat.size() == maskmat.size() );
    Size size = getContinuousSize( srcmat, maskmat );
    ST s = 0, sq = 0;
    int pix = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        for( int x = 0; x < size.width; x++ )
            if( mask[x] )
            {
                T v = src[x];
                s += v;
                sq += sqr(v);
                pix++;
            }
    }
    _mean = _stddev = Scalar();
    double scale = 1./std::max(pix, 1);
    for( int i = 0; i < DataType<ST>::channels; i++ )
    {
        double t = ((ST1*)&s)[i]*scale;
        _mean.val[i] = t;
        _stddev.val[i] = std::sqrt(std::max(((ST1*)&sq)[i]*scale - t*t, 0.));
    }
}

typedef void (*MeanStdDevFunc)(const Mat& src, Scalar& mean, Scalar& stddev);

typedef void (*MeanStdDevMaskFunc)(const Mat& src, const Mat& mask,
                                   Scalar& mean, Scalar& stddev);

void meanStdDev( const Mat& m, Scalar& mean, Scalar& stddev, const Mat& mask )
{
    static MeanStdDevFunc tab[]=
    {
        meanStdDev_<SqrC1<uchar, double> >, 0,
        meanStdDev_<SqrC1<ushort, double> >,
        meanStdDev_<SqrC1<short, double> >,
        meanStdDev_<SqrC1<int, double> >,
        meanStdDev_<SqrC1<float, double> >,
        meanStdDev_<SqrC1<double, double> >, 0,

        meanStdDev_<SqrC2<uchar, double> >, 0,
        meanStdDev_<SqrC2<ushort, double> >,
        meanStdDev_<SqrC2<short, double> >,
        meanStdDev_<SqrC2<int, double> >,
        meanStdDev_<SqrC2<float, double> >,
        meanStdDev_<SqrC2<double, double> >, 0,

        meanStdDev_<SqrC3<uchar, double> >, 0,
        meanStdDev_<SqrC3<ushort, double> >,
        meanStdDev_<SqrC3<short, double> >,
        meanStdDev_<SqrC3<int, double> >,
        meanStdDev_<SqrC3<float, double> >,
        meanStdDev_<SqrC3<double, double> >, 0,

        meanStdDev_<SqrC4<uchar, double> >, 0,
        meanStdDev_<SqrC4<ushort, double> >,
        meanStdDev_<SqrC4<short, double> >,
        meanStdDev_<SqrC4<int, double> >,
        meanStdDev_<SqrC4<float, double> >,
        meanStdDev_<SqrC4<double, double> >, 0
    };

    static MeanStdDevMaskFunc mtab[]=
    {
        meanStdDevMask_<SqrC1<uchar, double> >, 0,
        meanStdDevMask_<SqrC1<ushort, double> >,
        meanStdDevMask_<SqrC1<short, double> >,
        meanStdDevMask_<SqrC1<int, double> >,
        meanStdDevMask_<SqrC1<float, double> >,
        meanStdDevMask_<SqrC1<double, double> >, 0,

        meanStdDevMask_<SqrC2<uchar, double> >, 0,
        meanStdDevMask_<SqrC2<ushort, double> >,
        meanStdDevMask_<SqrC2<short, double> >,
        meanStdDevMask_<SqrC2<int, double> >,
        meanStdDevMask_<SqrC2<float, double> >,
        meanStdDevMask_<SqrC2<double, double> >, 0,

        meanStdDevMask_<SqrC3<uchar, double> >, 0,
        meanStdDevMask_<SqrC3<ushort, double> >,
        meanStdDevMask_<SqrC3<short, double> >,
        meanStdDevMask_<SqrC3<int, double> >,
        meanStdDevMask_<SqrC3<float, double> >,
        meanStdDevMask_<SqrC3<double, double> >, 0,

        meanStdDevMask_<SqrC4<uchar, double> >, 0,
        meanStdDevMask_<SqrC4<ushort, double> >,
        meanStdDevMask_<SqrC4<short, double> >,
        meanStdDevMask_<SqrC4<int, double> >,
        meanStdDevMask_<SqrC4<float, double> >,
        meanStdDevMask_<SqrC4<double, double> >, 0
    };

    CV_Assert( m.channels() <= 4 && (mask.empty() || mask.type() == CV_8U) );
    
    MeanStdDevFunc func = tab[m.type()];
    MeanStdDevMaskFunc mfunc = mtab[m.type()];
    CV_Assert( func != 0 || mfunc != 0 );
    
    if( m.dims > 2 )
    {
        Scalar s, sq;
        double total = 0;
        
        const Mat* arrays[] = {&m, &mask, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        int k, cn = m.channels();
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            Scalar _mean, _stddev;
            double nz = (double)(mask.data ? countNonZero(it.planes[1]) : it.planes[0].rows*it.planes[0].cols);
            
            if( func )
                func(it.planes[0], _mean, _stddev);
            else
                mfunc(it.planes[0], it.planes[1], _mean, _stddev);
            
            total += nz;
            for( k = 0; k < cn; k++ )
            {
                s[k] += _mean[k]*nz;
                sq[k] += (_stddev[k]*_stddev[k] + _mean[k]*_mean[k])*nz;
            }
        }
        
        mean = stddev = Scalar();
        total = 1./std::max(total, 1.);
        for( k = 0; k < cn; k++ )
        {
            mean[k] = s[k]*total;
            stddev[k] = std::sqrt(std::max(sq[k]*total - mean[k]*mean[k], 0.));
        }
        return;
    }
    
    if( mask.data )
    {
        CV_Assert( mask.size() == m.size() ); 
        mfunc( m, mask, mean, stddev );
    }
    else
        func( m, mean, stddev );
}


/****************************************************************************************\
*                                       minMaxLoc                                        *
\****************************************************************************************/

template<typename T> static void
minMaxIndx_( const Mat& srcmat, double* minVal, double* maxVal, int* minLoc, int* maxLoc )
{
    assert( DataType<T>::type == srcmat.type() );
    const T* src = (const T*)srcmat.data;
    size_t step = srcmat.step/sizeof(src[0]);
    T min_val = src[0], max_val = min_val;
    int min_loc = 0, max_loc = 0;
    int x, loc = 0;
    Size size = getContinuousSize( srcmat );

    for( ; size.height--; src += step, loc += size.width )
    {
        for( x = 0; x < size.width; x++ )
        {
            T val = src[x];
            if( val < min_val )
            {
                min_val = val;
                min_loc = loc + x;
            }
            else if( val > max_val )
            {
                max_val = val;
                max_loc = loc + x;
            }
        }
    }

    *minLoc = min_loc;
    *maxLoc = max_loc;
    *minVal = min_val;
    *maxVal = max_val;
}


template<typename T> static void
minMaxIndxMask_( const Mat& srcmat, const Mat& maskmat,
    double* minVal, double* maxVal, int* minLoc, int* maxLoc )
{
    assert( DataType<T>::type == srcmat.type() &&
        CV_8U == maskmat.type() &&
        srcmat.size() == maskmat.size() );
    const T* src = (const T*)srcmat.data;
    const uchar* mask = maskmat.data;
    size_t step = srcmat.step/sizeof(src[0]);
    size_t maskstep = maskmat.step;
    T min_val = 0, max_val = 0;
    int min_loc = -1, max_loc = -1;
    int x = 0, y, loc = 0;
    Size size = getContinuousSize( srcmat, maskmat );

    for( y = 0; y < size.height; y++, src += step, mask += maskstep, loc += size.width )
    {
        for( x = 0; x < size.width; x++ )
            if( mask[x] != 0 )
            {
                min_loc = max_loc = loc + x;
                min_val = max_val = src[x];
                break;
            }
        if( x < size.width )
            break;
    }

    for( ; y < size.height; x = 0, y++, src += step, mask += maskstep, loc += size.width )
    {
        for( ; x < size.width; x++ )
        {
            T val = src[x];
            int m = mask[x];

            if( val < min_val && m )
            {
                min_val = val;
                min_loc = loc + x;
            }
            else if( val > max_val && m )
            {
                max_val = val;
                max_loc = loc + x;
            }
        }
    }

    *minLoc = min_loc;
    *maxLoc = max_loc;
    *minVal = min_val;
    *maxVal = max_val;
}

typedef void (*MinMaxIndxFunc)(const Mat&, double*, double*, int*, int*);

typedef void (*MinMaxIndxMaskFunc)(const Mat&, const Mat&,
                                    double*, double*, int*, int*);

void minMaxLoc( const Mat& img, double* minVal, double* maxVal,
                Point* minLoc, Point* maxLoc, const Mat& mask )
{
    CV_Assert(img.dims <= 2);
    
    static MinMaxIndxFunc tab[] =
        {minMaxIndx_<uchar>, 0, minMaxIndx_<ushort>, minMaxIndx_<short>,
        minMaxIndx_<int>, minMaxIndx_<float>, minMaxIndx_<double>, 0};
    static MinMaxIndxMaskFunc tabm[] =
        {minMaxIndxMask_<uchar>, 0, minMaxIndxMask_<ushort>, minMaxIndxMask_<short>,
        minMaxIndxMask_<int>, minMaxIndxMask_<float>, minMaxIndxMask_<double>, 0};

    int depth = img.depth();
    double minval=0, maxval=0;
    int minloc=0, maxloc=0;

    CV_Assert( img.channels() == 1 );

    if( !mask.data )
    {
        MinMaxIndxFunc func = tab[depth];
        CV_Assert( func != 0 );
        func( img, &minval, &maxval, &minloc, &maxloc );
    }
    else
    {
        CV_Assert( img.size() == mask.size() && mask.type() == CV_8U );
        MinMaxIndxMaskFunc func = tabm[depth];
        CV_Assert( func != 0 );
        func( img, mask, &minval, &maxval, &minloc, &maxloc );
    }

    if( minVal )
        *minVal = minval;
    if( maxVal )
        *maxVal = maxval;
    if( minLoc )
    {
        if( minloc >= 0 )
        {
            minLoc->y = minloc/img.cols;
            minLoc->x = minloc - minLoc->y*img.cols;
        }
        else
            minLoc->x = minLoc->y = -1;
    }
    if( maxLoc )
    {
        if( maxloc >= 0 )
        {
            maxLoc->y = maxloc/img.cols;
            maxLoc->x = maxloc - maxLoc->y*img.cols;
        }
        else
            maxLoc->x = maxLoc->y = -1;
    }
}

static void ofs2idx(const Mat& a, size_t ofs, int* idx)
{
    int i, d = a.dims;
    for( i = 0; i < d; i++ )
    {
        idx[i] = (int)(ofs / a.step[i]);
        ofs %= a.step[i];
    }
}

void minMaxIndx(const Mat& a, double* minVal,
                double* maxVal, int* minIdx, int* maxIdx,
                const Mat& mask)
{
    if( a.dims <= 2 )
    {
        Point minLoc, maxLoc;
        minMaxLoc(a, minVal, maxVal, &minLoc, &maxLoc, mask);
        if( minIdx )
            minIdx[0] = minLoc.x, minIdx[1] = minLoc.y;
        if( maxIdx )
            maxIdx[0] = maxLoc.x, maxIdx[1] = maxLoc.y;
        return;
    }
    
    const Mat* arrays[] = {&a, &mask, 0};
    Mat planes[2];
    NAryMatIterator it(arrays, planes);
    double minval = DBL_MAX, maxval = -DBL_MAX;
    size_t minofs = 0, maxofs = 0, esz = a.elemSize();
    
    for( int i = 0; i < it.nplanes; i++, ++it )
    {
        double val0 = 0, val1 = 0;
        Point pt0, pt1;
        minMaxLoc( it.planes[0], &val0, &val1, &pt0, &pt1, it.planes[1] );
        if( val0 < minval )
        {
            minval = val0;
            minofs = (it.planes[0].data - a.data) + pt0.x*esz;
        }
        if( val1 > maxval )
        {
            maxval = val1;
            maxofs = (it.planes[0].data - a.data) + pt1.x*esz;
        }
    }
    
    if( minVal )
        *minVal = minval;
    if( maxVal )
        *maxVal = maxval;
    if( minIdx )
        ofs2idx(a, minofs, minIdx);
    if( maxIdx )
        ofs2idx(a, maxofs, maxIdx);
}    
    
/****************************************************************************************\
*                                         norm                                           *
\****************************************************************************************/

template<typename T, typename WT=T> struct OpAbs
{
    typedef T type1;
    typedef WT rtype;
    rtype operator()(type1 x) const { return (WT)std::abs(x); }
};

template<> inline uchar OpAbs<uchar, uchar>::operator()(uchar x) const { return x; }
template<> inline ushort OpAbs<ushort, ushort>::operator()(ushort x) const { return x; }

template<class ElemFunc, class UpdateFunc, class GlobUpdateFunc, int BLOCK_SIZE>
static double normBlock_( const Mat& srcmat )
{
    ElemFunc f;
    UpdateFunc update;
    GlobUpdateFunc globUpdate;
    typedef typename ElemFunc::type1 T;
    typedef typename UpdateFunc::rtype WT;
    typedef typename GlobUpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat.depth() );
    Size size = getContinuousSize( srcmat, srcmat.channels() );
    ST s0 = 0; // luckily, 0 is the correct starting value for both + and max update operations
    WT s = 0;
    int y, remaining = BLOCK_SIZE;

    for( y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x <= limit - 4; x += 4 )
            {
                s = update(s, (WT)f(src[x]));
                s = update(s, (WT)f(src[x+1]));
                s = update(s, (WT)f(src[x+2]));
                s = update(s, (WT)f(src[x+3]));
            }
            for( ; x < limit; x++ )
                s = update(s, (WT)f(src[x]));
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 = globUpdate(s0, (ST)s);
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return s0;
}

template<class ElemFunc, class UpdateFunc>
static double norm_( const Mat& srcmat )
{
    ElemFunc f;
    UpdateFunc update;
    typedef typename ElemFunc::type1 T;
    typedef typename UpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat.depth() );
    Size size = getContinuousSize( srcmat, srcmat.channels() );
    ST s = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            s = update(s, (ST)f(src[x]));
            s = update(s, (ST)f(src[x+1]));
            s = update(s, (ST)f(src[x+2]));
            s = update(s, (ST)f(src[x+3]));
        }
        for( ; x < size.width; x++ )
            s = update(s, (ST)f(src[x]));
    }
    return s;
}

template<class ElemFunc, class UpdateFunc, class GlobUpdateFunc, int BLOCK_SIZE>
static double normMaskBlock_( const Mat& srcmat, const Mat& maskmat )
{
    ElemFunc f;
    UpdateFunc update;
    GlobUpdateFunc globUpdate;
    typedef typename ElemFunc::type1 T;
    typedef typename UpdateFunc::rtype WT;
    typedef typename GlobUpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat.depth() );
    Size size = getContinuousSize( srcmat, maskmat );
    ST s0 = 0;
    WT s = 0;
    int y, remaining = BLOCK_SIZE;

    for( y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x <= limit - 4; x += 4 )
            {
                if( mask[x] )
                    s = update(s, (WT)f(src[x]));
                if( mask[x+1] )
                    s = update(s, (WT)f(src[x+1]));
                if( mask[x+2] )
                    s = update(s, (WT)f(src[x+2]));
                if( mask[x+3] )
                    s = update(s, (WT)f(src[x+3]));
            }
            for( ; x < limit; x++ )
            {
                if( mask[x] )
                    s = update(s, (WT)f(src[x]));
            }
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 = globUpdate(s0, (ST)s);
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return s0;
}

template<class ElemFunc, class UpdateFunc>
static double normMask_( const Mat& srcmat, const Mat& maskmat )
{
    ElemFunc f;
    UpdateFunc update;
    typedef typename ElemFunc::type1 T;
    typedef typename UpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat.depth() );
    Size size = getContinuousSize( srcmat, maskmat );
    ST s = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            if( mask[x] )
                s = update(s, (ST)f(src[x]));
            if( mask[x+1] )
                s = update(s, (ST)f(src[x+1]));
            if( mask[x+2] )
                s = update(s, (ST)f(src[x+2]));
            if( mask[x+3] )
                s = update(s, (ST)f(src[x+3]));
        }
        for( ; x < size.width; x++ )
        {
            if( mask[x] )
                s = update(s, (ST)f(src[x]));
        }
    }
    return s;
}

template<typename T, class ElemFunc, class UpdateFunc, class GlobUpdateFunc, int BLOCK_SIZE>
static double normDiffBlock_( const Mat& srcmat1, const Mat& srcmat2 )
{
    ElemFunc f;
    UpdateFunc update;
    GlobUpdateFunc globUpdate;
    typedef typename UpdateFunc::rtype WT;
    typedef typename GlobUpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat1.depth() );
    Size size = getContinuousSize( srcmat1, srcmat2, srcmat1.channels() );
    ST s0 = 0;
    WT s = 0;
    int y, remaining = BLOCK_SIZE;

    for( y = 0; y < size.height; y++ )
    {
        const T* src1 = (const T*)(srcmat1.data + srcmat1.step*y);
        const T* src2 = (const T*)(srcmat2.data + srcmat2.step*y);
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x <= limit - 4; x += 4 )
            {
                s = update(s, (WT)f(src1[x] - src2[x]));
                s = update(s, (WT)f(src1[x+1] - src2[x+1]));
                s = update(s, (WT)f(src1[x+2] - src2[x+2]));
                s = update(s, (WT)f(src1[x+3] - src2[x+3]));
            }
            for( ; x < limit; x++ )
                s = update(s, (WT)f(src1[x] - src2[x]));
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 = globUpdate(s0, (ST)s);
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return s0;
}

template<typename T, class ElemFunc, class UpdateFunc>
static double normDiff_( const Mat& srcmat1, const Mat& srcmat2 )
{
    ElemFunc f;
    UpdateFunc update;
    typedef typename UpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat1.depth() );
    Size size = getContinuousSize( srcmat1, srcmat2, srcmat1.channels() );
    ST s = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src1 = (const T*)(srcmat1.data + srcmat1.step*y);
        const T* src2 = (const T*)(srcmat2.data + srcmat2.step*y);
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            s = update(s, (ST)f(src1[x] - src2[x]));
            s = update(s, (ST)f(src1[x+1] - src2[x+1]));
            s = update(s, (ST)f(src1[x+2] - src2[x+2]));
            s = update(s, (ST)f(src1[x+3] - src2[x+3]));
        }
        for( ; x < size.width; x++ )
            s = update(s, (ST)f(src1[x] - src2[x]));
    }
    return s;
}

template<typename T, class ElemFunc, class UpdateFunc, class GlobUpdateFunc, int BLOCK_SIZE>
static double normDiffMaskBlock_( const Mat& srcmat1, const Mat& srcmat2, const Mat& maskmat )
{
    ElemFunc f;
    UpdateFunc update;
    GlobUpdateFunc globUpdate;
    typedef typename UpdateFunc::rtype WT;
    typedef typename GlobUpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat1.depth() );
    Size size = getContinuousSize( srcmat1, srcmat2, maskmat );
    ST s0 = 0;
    WT s = 0;
    int y, remaining = BLOCK_SIZE;

    for( y = 0; y < size.height; y++ )
    {
        const T* src1 = (const T*)(srcmat1.data + srcmat1.step*y);
        const T* src2 = (const T*)(srcmat2.data + srcmat2.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        int x = 0;
        while( x < size.width )
        {
            int limit = std::min( remaining, size.width - x );
            remaining -= limit;
            limit += x;
            for( ; x <= limit - 4; x += 4 )
            {
                if( mask[x] )
                    s = update(s, (WT)f(src1[x] - src2[x]));
                if( mask[x+1] )
                    s = update(s, (WT)f(src1[x+1] - src2[x+1]));
                if( mask[x+2] )
                    s = update(s, (WT)f(src1[x+2] - src2[x+2]));
                if( mask[x+3] )
                    s = update(s, (WT)f(src1[x+3] - src2[x+3]));
            }
            for( ; x < limit; x++ )
                if( mask[x] )
                    s = update(s, (WT)f(src1[x] - src2[x]));
            if( remaining == 0 || (x == size.width && y == size.height-1) )
            {
                s0 = globUpdate(s0, (ST)s);
                s = 0;
                remaining = BLOCK_SIZE;
            }
        }
    }
    return s0;
}

template<typename T, class ElemFunc, class UpdateFunc>
static double normDiffMask_( const Mat& srcmat1, const Mat& srcmat2, const Mat& maskmat )
{
    ElemFunc f;
    UpdateFunc update;
    typedef typename UpdateFunc::rtype ST;
    
    assert( DataType<T>::depth == srcmat1.depth() );
    Size size = getContinuousSize( srcmat1, srcmat2, maskmat );
    ST s = 0;

    for( int y = 0; y < size.height; y++ )
    {
        const T* src1 = (const T*)(srcmat1.data + srcmat1.step*y);
        const T* src2 = (const T*)(srcmat2.data + srcmat2.step*y);
        const uchar* mask = maskmat.data + maskmat.step*y;
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            if( mask[x] )
                s = update(s, (ST)f(src1[x] - src2[x]));
            if( mask[x+1] )
                s = update(s, (ST)f(src1[x+1] - src2[x+1]));
            if( mask[x+2] )
                s = update(s, (ST)f(src1[x+2] - src2[x+2]));
            if( mask[x+3] )
                s = update(s, (ST)f(src1[x+3] - src2[x+3]));
        }
        for( ; x < size.width; x++ )
            if( mask[x] )
                s = update(s, (ST)f(src1[x] - src2[x]));

    }
    return s;
}

typedef double (*NormFunc)(const Mat& src);
typedef double (*NormDiffFunc)(const Mat& src1, const Mat& src2);
typedef double (*NormMaskFunc)(const Mat& src1, const Mat& mask);
typedef double (*NormDiffMaskFunc)(const Mat& src1, const Mat& src2, const Mat& mask);

double norm( const Mat& a, int normType )
{
    static NormFunc tab[3][8] =
    {
        {
            norm_<OpAbs<uchar>, OpMax<int> >,
            norm_<OpAbs<schar>, OpMax<int> >,
            norm_<OpAbs<ushort>, OpMax<int> >,
            norm_<OpAbs<short, int>, OpMax<int> >,
            norm_<OpAbs<int>, OpMax<int> >,
            norm_<OpAbs<float>, OpMax<float> >,
            norm_<OpAbs<double>, OpMax<double> >
        },
        
        { 
            normBlock_<OpAbs<uchar>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normBlock_<OpAbs<schar>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normBlock_<OpAbs<ushort>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normBlock_<OpAbs<short, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            norm_<OpAbs<int>, OpAdd<double> >,
            norm_<OpAbs<float>, OpAdd<double> >,
            norm_<OpAbs<double>, OpAdd<double> >
        },

        { 
            normBlock_<SqrC1<uchar, unsigned>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normBlock_<SqrC1<schar, unsigned>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            norm_<SqrC1<ushort, double>, OpAdd<double> >,
            norm_<SqrC1<short, double>, OpAdd<double> >,
            norm_<SqrC1<int, double>, OpAdd<double> >,
            norm_<SqrC1<float, double>, OpAdd<double> >,
            norm_<SqrC1<double, double>, OpAdd<double> >
        }
    };

    normType &= 7;
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);
    NormFunc func = tab[normType >> 1][a.depth()];
    CV_Assert(func != 0);
    
    double r = 0;
    if( a.dims > 2 )
    {
        const Mat* arrays[] = {&a, 0};
        Mat planes[1];
        NAryMatIterator it(arrays, planes);
    
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            double n = func(it.planes[0]);
            if( normType == NORM_INF )
                r = std::max(r, n);
            else
                r += n;
        }
    }
    else
        r = func(a);
    
    return normType == NORM_L2 ? std::sqrt(r) : r;
}


double norm( const Mat& a, int normType, const Mat& mask )
{
    static NormMaskFunc tab[3][8] =
    {
        {
            normMask_<OpAbs<uchar>, OpMax<int> >,
            normMask_<OpAbs<schar>, OpMax<int> >,
            normMask_<OpAbs<ushort>, OpMax<int> >,
            normMask_<OpAbs<short, int>, OpMax<int> >,
            normMask_<OpAbs<int>, OpMax<int> >,
            normMask_<OpAbs<float>, OpMax<float> >,
            normMask_<OpAbs<double>, OpMax<double> >
        },
        
        { 
            normMaskBlock_<OpAbs<uchar>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normMaskBlock_<OpAbs<schar>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normMaskBlock_<OpAbs<ushort>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normMaskBlock_<OpAbs<short, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normMask_<OpAbs<int>, OpAdd<double> >,
            normMask_<OpAbs<float>, OpAdd<double> >,
            normMask_<OpAbs<double>, OpAdd<double> >
        },

        { 
            normMaskBlock_<SqrC1<uchar, unsigned>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normMaskBlock_<SqrC1<schar, unsigned>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normMask_<SqrC1<ushort, double>, OpAdd<double> >,
            normMask_<SqrC1<short, double>, OpAdd<double> >,
            normMask_<SqrC1<int, double>, OpAdd<double> >,
            normMask_<SqrC1<float, double>, OpAdd<double> >,
            normMask_<SqrC1<double, double>, OpAdd<double> >
        }
    };

    if( !mask.data )
        return norm(a, normType);

    normType &= 7;
    CV_Assert((normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2) &&
              mask.type() == CV_8U && a.channels() == 1);
    NormMaskFunc func = tab[normType >> 1][a.depth()];
    CV_Assert(func != 0);
    
    double r = 0;
    if( a.dims > 2 )
    {
        const Mat* arrays[] = {&a, &mask, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            double n = func(it.planes[0], it.planes[1]);
            if( normType == NORM_INF )
                r = std::max(r, n);
            else
                r += n;
        }
    }
    else
    {
        CV_Assert( a.size() == mask.size() );
        r = func(a, mask);
    }
    
    return normType == NORM_L2 ? std::sqrt(r) : r;
}


double norm( const Mat& a, const Mat& b, int normType )
{
    static NormDiffFunc tab[3][8] =
    {
        {
            normDiff_<uchar, OpAbs<int>, OpMax<int> >,
            normDiff_<schar, OpAbs<int>, OpMax<int> >,
            normDiff_<ushort, OpAbs<int>, OpMax<int> >,
            normDiff_<short, OpAbs<int>, OpMax<int> >,
            normDiff_<int, OpAbs<int>, OpMax<int> >,
            normDiff_<float, OpAbs<float>, OpMax<float> >,
            normDiff_<double, OpAbs<double>, OpMax<double> >
        },
        
        { 
            normDiffBlock_<uchar, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normDiffBlock_<schar, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normDiffBlock_<ushort, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffBlock_<short, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiff_<int, OpAbs<int>, OpAdd<double> >,
            normDiff_<float, OpAbs<float>, OpAdd<double> >,
            normDiff_<double, OpAbs<double>, OpAdd<double> >
        },

        { 
            normDiffBlock_<uchar, SqrC1<int, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffBlock_<schar, SqrC1<int, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiff_<ushort, SqrC1<int, double>, OpAdd<double> >,
            normDiff_<short, SqrC1<int, double>, OpAdd<double> >,
            normDiff_<int, SqrC1<int, double>, OpAdd<double> >,
            normDiff_<float, SqrC1<float, double>, OpAdd<double> >,
            normDiff_<double, SqrC1<double, double>, OpAdd<double> >
        }
    };
    
    CV_Assert( a.type() == b.type() );
    bool isRelative = (normType & NORM_RELATIVE) != 0;
    normType &= 7;
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);
    NormDiffFunc func = tab[normType >> 1][a.depth()];
    CV_Assert(func != 0);
    
    double r = 0.;
    if( a.dims > 2 )
    {
        const Mat* arrays[] = {&a, &b, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            double n = func(it.planes[0], it.planes[1]);
                
            if( normType == NORM_INF )
                r = std::max(r, n);
            else
                r += n;
        }
    }
    else
    {
        CV_Assert( a.size() == b.size() );
        r = func( a, b );
    }
    if( normType == NORM_L2 )
        r = std::sqrt(r);
    if( isRelative )
        r /= norm(b, normType);
    return r;
}

double norm( const Mat& a, const Mat& b, int normType, const Mat& mask )
{
    static NormDiffMaskFunc tab[3][8] =
    {
        {
            normDiffMask_<uchar, OpAbs<int>, OpMax<int> >,
            normDiffMask_<schar, OpAbs<int>, OpMax<int> >,
            normDiffMask_<ushort, OpAbs<int>, OpMax<int> >,
            normDiffMask_<short, OpAbs<int>, OpMax<int> >,
            normDiffMask_<int, OpAbs<int>, OpMax<int> >,
            normDiffMask_<float, OpAbs<float>, OpMax<float> >,
            normDiffMask_<double, OpAbs<double>, OpMax<double> >
        },
        
        { 
            normDiffMaskBlock_<uchar, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normDiffMaskBlock_<schar, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<24>,
            normDiffMaskBlock_<ushort, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffMaskBlock_<short, OpAbs<int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffMask_<int, OpAbs<int>, OpAdd<double> >,
            normDiffMask_<float, OpAbs<float>, OpAdd<double> >,
            normDiffMask_<double, OpAbs<double>, OpAdd<double> >
        },

        { 
            normDiffMaskBlock_<uchar, SqrC1<int, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffMaskBlock_<schar, SqrC1<int, int>, OpAdd<unsigned>, OpAdd<double>, 1<<16>,
            normDiffMask_<ushort, SqrC1<int, double>, OpAdd<double> >,
            normDiffMask_<short, SqrC1<int, double>, OpAdd<double> >,
            normDiffMask_<int, SqrC1<int, double>, OpAdd<double> >,
            normDiffMask_<float, SqrC1<float, double>, OpAdd<double> >,
            normDiffMask_<double, SqrC1<double, double>, OpAdd<double> >
        }
    };
    
    if( !mask.data )
        return norm(a, b, normType);

    CV_Assert( a.type() == b.type() && mask.type() == CV_8U && a.channels() == 1);
    bool isRelative = (normType & NORM_RELATIVE) != 0;
    normType &= 7;
    CV_Assert(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2);

    NormDiffMaskFunc func = tab[normType >> 1][a.depth()];
    CV_Assert(func != 0);
    
    double r = 0.;
    if( a.dims > 2 )
    {
        const Mat* arrays[] = {&a, &b, &mask, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            double n = func(it.planes[0], it.planes[1], it.planes[2]);
            
            if( normType == NORM_INF )
                r = std::max(r, n);
            else
                r += n;
        }
    }
    else
    {
        CV_Assert( a.size() == b.size() && a.size() == mask.size() );
        r = func( a, b, mask );
    }
    
    if( normType == NORM_L2 )
        r = std::sqrt(r);
    if( isRelative )
        r /= std::max(norm(b, normType, mask), DBL_EPSILON);
    return r;
}

}


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
    return sum;
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
    return mean;
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
