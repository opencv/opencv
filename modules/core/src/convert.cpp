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

/****************************************************************************************\
*                                       split                                            *
\****************************************************************************************/

template<typename T> static void
splitC2_( const Mat& srcmat, Mat* dstmat )
{
    Size size = getContinuousSize( srcmat, dstmat[0], dstmat[1] );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        T* dst0 = (T*)(dstmat[0].data + dstmat[0].step*y);
        T* dst1 = (T*)(dstmat[1].data + dstmat[1].step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src[x*2], t1 = src[x*2+1];
            dst0[x] = t0; dst1[x] = t1;
        }
    }
}

template<typename T> static void
splitC3_( const Mat& srcmat, Mat* dstmat )
{
    Size size = getContinuousSize( srcmat, dstmat[0], dstmat[1], dstmat[2] );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        T* dst0 = (T*)(dstmat[0].data + dstmat[0].step*y);
        T* dst1 = (T*)(dstmat[1].data + dstmat[1].step*y);
        T* dst2 = (T*)(dstmat[2].data + dstmat[2].step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src[x*3], t1 = src[x*3+1], t2 = src[x*3+2];
            dst0[x] = t0; dst1[x] = t1; dst2[x] = t2;
        }
    }
}

template<typename T> static void
splitC4_( const Mat& srcmat, Mat* dstmat )
{
    Size size = getContinuousSize( srcmat, dstmat[0], dstmat[1], dstmat[2], dstmat[3] );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        T* dst0 = (T*)(dstmat[0].data + dstmat[0].step*y);
        T* dst1 = (T*)(dstmat[1].data + dstmat[1].step*y);
        T* dst2 = (T*)(dstmat[2].data + dstmat[2].step*y);
        T* dst3 = (T*)(dstmat[3].data + dstmat[3].step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src[x*4], t1 = src[x*4+1];
            dst0[x] = t0; dst1[x] = t1;
            t0 = src[x*4+2]; t1 = src[x*4+3];
            dst2[x] = t0; dst3[x] = t1;
        }
    }
}

typedef void (*SplitFunc)(const Mat& src, Mat* dst);

void split(const Mat& src, Mat* mv)
{
    static SplitFunc tab[] =
    {
        splitC2_<uchar>, splitC2_<ushort>, splitC2_<int>, 0, splitC2_<int64>,
        splitC3_<uchar>, splitC3_<ushort>, splitC3_<int>, 0, splitC3_<int64>,
        splitC4_<uchar>, splitC4_<ushort>, splitC4_<int>, 0, splitC4_<int64>
    };

    int i, depth = src.depth(), cn = src.channels();

    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    for( i = 0; i < cn; i++ )
        mv[i].create(src.dims, src.size, depth);

    if( cn <= 4 )
    {
        SplitFunc func = tab[(cn-2)*5 + (src.elemSize1()>>1)];
        CV_Assert( func != 0 );
        
        if( src.dims > 2 )
        {
            const Mat* arrays[5];
            Mat planes[5];
            arrays[0] = &src;
            for( i = 0; i < cn; i++ )
                arrays[i+1] = &mv[i];
            NAryMatIterator it(arrays, planes, cn+1);
            
            for( int i = 0; i < it.nplanes; i++, ++it )
                func( it.planes[0], &it.planes[1] );
        }
        else
            func( src, mv );
    }
    else
    {
        AutoBuffer<int> pairs(cn*2);

        for( i = 0; i < cn; i++ )
        {
            pairs[i*2] = i;
            pairs[i*2+1] = 0;
        }
        mixChannels( &src, 1, mv, cn, &pairs[0], cn );
    }
}
    
void split(const Mat& m, vector<Mat>& mv)
{
    mv.resize(!m.empty() ? m.channels() : 0);
    if(!m.empty())
        split(m, &mv[0]);
}

/****************************************************************************************\
*                                       merge                                            *
\****************************************************************************************/

// input vector is made non-const to make sure that we do not copy Mat on each access
template<typename T> static void
mergeC2_( const Mat* srcmat, Mat& dstmat )
{
    Size size = getContinuousSize( srcmat[0], srcmat[1], dstmat );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src0 = (const T*)(srcmat[0].data + srcmat[0].step*y);
        const T* src1 = (const T*)(srcmat[1].data + srcmat[1].step*y);
        T* dst = (T*)(dstmat.data + dstmat.step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src0[x], t1 = src1[x];
            dst[x*2] = t0; dst[x*2+1] = t1;
        }
    }
}

template<typename T> static void
mergeC3_( const Mat* srcmat, Mat& dstmat )
{
    Size size = getContinuousSize( srcmat[0], srcmat[1], srcmat[2], dstmat );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src0 = (const T*)(srcmat[0].data + srcmat[0].step*y);
        const T* src1 = (const T*)(srcmat[1].data + srcmat[1].step*y);
        const T* src2 = (const T*)(srcmat[2].data + srcmat[2].step*y);
        T* dst = (T*)(dstmat.data + dstmat.step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src0[x], t1 = src1[x], t2 = src2[x];
            dst[x*3] = t0; dst[x*3+1] = t1; dst[x*3+2] = t2;
        }
    }
}

template<typename T> static void
mergeC4_( const Mat* srcmat, Mat& dstmat )
{
    Size size = getContinuousSize( srcmat[0], srcmat[1], srcmat[2], srcmat[3], dstmat );
    for( int y = 0; y < size.height; y++ )
    {
        const T* src0 = (const T*)(srcmat[0].data + srcmat[0].step*y);
        const T* src1 = (const T*)(srcmat[1].data + srcmat[1].step*y);
        const T* src2 = (const T*)(srcmat[2].data + srcmat[2].step*y);
        const T* src3 = (const T*)(srcmat[3].data + srcmat[3].step*y);
        T* dst = (T*)(dstmat.data + dstmat.step*y);

        for( int x = 0; x < size.width; x++ )
        {
            T t0 = src0[x], t1 = src1[x];
            dst[x*4] = t0; dst[x*4+1] = t1;
            t0 = src2[x]; t1 = src3[x];
            dst[x*4+2] = t0; dst[x*4+3] = t1;
        }
    }
}

typedef void (*MergeFunc)(const Mat* src, Mat& dst);

void merge(const Mat* mv, size_t _n, Mat& dst)
{
    static MergeFunc tab[] =
    {
        mergeC2_<uchar>, mergeC2_<ushort>, mergeC2_<int>, 0, mergeC2_<int64>,
        mergeC3_<uchar>, mergeC3_<ushort>, mergeC3_<int>, 0, mergeC3_<int64>,
        mergeC4_<uchar>, mergeC4_<ushort>, mergeC4_<int>, 0, mergeC4_<int64>
    };

    CV_Assert( mv && _n > 0 );
    
    int depth = mv[0].depth();
    bool allch1 = true;
    int i, total = 0, n = (int)_n;
    
    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        total += mv[i].channels();
    }

    CV_Assert( 0 < total && total <= CV_CN_MAX );

    if( total == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, total));

    if( allch1 && total <= 4 )
    {
        MergeFunc func = tab[(total-2)*5 + (CV_ELEM_SIZE(depth)>>1)];
        CV_Assert( func != 0 );
        if( mv[0].dims > 2 )
        {
            const Mat* arrays[5];
            Mat planes[5];
            arrays[total] = &dst;
            for( i = 0; i < total; i++ )
                arrays[i] = &mv[i];
            NAryMatIterator it(arrays, planes, total+1);
            
            for( i = 0; i < it.nplanes; i++, ++it )
                func( &it.planes[0], it.planes[total] );
        }
        else
            func( mv, dst );
    }
    else
    {
        AutoBuffer<int> pairs(total*2);
        int j, k, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], total );
    }
}

void merge(const vector<Mat>& mv, Mat& dst)
{
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), dst);
}

/****************************************************************************************\
*                       Generalized split/merge: mixing channels                         *
\****************************************************************************************/

template<typename T> static void
mixChannels_( const void** _src, const int* sdelta0,
              const int* sdelta1, void** _dst,
              const int* ddelta0, const int* ddelta1,
              int n, Size size )
{
    const T** src = (const T**)_src;
    T** dst = (T**)_dst;
    int i, k;
    int block_size0 = n == 1 ? size.width : 1024;

    for( ; size.height--; )
    {
        int remaining = size.width;
        for( ; remaining > 0; )
        {
            int block_size = MIN( remaining, block_size0 );
            for( k = 0; k < n; k++ )
            {
                const T* s = src[k];
                T* d = dst[k];
                int ds = sdelta1[k], dd = ddelta1[k];
                if( s )
                {
                    for( i = 0; i <= block_size - 2; i += 2, s += ds*2, d += dd*2 )
                    {
                        T t0 = s[0], t1 = s[ds];
                        d[0] = t0; d[dd] = t1;
                    }
                    if( i < block_size )
                        d[0] = s[0], s += ds, d += dd;
                    src[k] = s;
                }
                else
                {
                    for( i=0; i <= block_size-2; i+=2, d+=dd*2 )
                        d[0] = d[dd] = 0;
                    if( i < block_size )
                        d[0] = 0, d += dd;
                }
                dst[k] = d;
            }
            remaining -= block_size;
        }
        for( k = 0; k < n; k++ )
            src[k] += sdelta0[k], dst[k] += ddelta0[k];
    }
}

typedef void (*MixChannelsFunc)( const void** src, const int* sdelta0,
        const int* sdelta1, void** dst, const int* ddelta0, const int* ddelta1, int n, Size size );

void mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );
    
    if( src[0].dims > 2 )
    {
        size_t k, m = nsrcs, n = ndsts;
        CV_Assert( n > 0 && m > 0 );
        AutoBuffer<const Mat*> v(m + n);
        AutoBuffer<Mat> planes(m + n);
        for( k = 0; k < m; k++ )
            v[k] = &src[k];
        for( k = 0; k < n; k++ )
            v[m + k] = &dst[k];
        NAryMatIterator it(v, planes, m + n);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            mixChannels( &it.planes[0], m, &it.planes[m], n, fromTo, npairs );
        return;
    }
    
    size_t i, j;
    int depth = dst[0].depth(), esz1 = (int)dst[0].elemSize1();
    Size size = dst[0].size();

    AutoBuffer<uchar> buf(npairs*(sizeof(void*)*2 + sizeof(int)*4));
    void** srcs = (void**)(uchar*)buf;
    void** dsts = srcs + npairs;
    int *s0 = (int*)(dsts + npairs), *s1 = s0 + npairs, *d0 = s1 + npairs, *d1 = d0 + npairs;
    bool isContinuous = true;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].size() == size && src[j].depth() == depth);
            isContinuous = isContinuous && src[j].isContinuous();
            srcs[i] = src[j].data + i0*esz1;
            s1[i] = src[j].channels(); s0[i] = (int)src[j].step/esz1 - size.width*src[j].channels();
        }
        else
        {
            srcs[i] = 0; s1[i] = s0[i] = 0;
        }
        
        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].size() == size && dst[j].depth() == depth);
        isContinuous = isContinuous && dst[j].isContinuous();
        dsts[i] = dst[j].data + i1*esz1;
        d1[i] = dst[j].channels(); d0[i] = (int)dst[j].step/esz1 - size.width*dst[j].channels();
    }

    MixChannelsFunc func = 0;
    if( esz1 == 1 )
        func = mixChannels_<uchar>;
    else if( esz1 == 2 )
        func = mixChannels_<ushort>;
    else if( esz1 == 4 )    
        func = mixChannels_<int>;
    else if( esz1 == 8 )    
        func = mixChannels_<int64>;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );    

    if( isContinuous )
    {
        size.width *= size.height;
        size.height = 1;
    }
    func( (const void**)srcs, s0, s1, dsts, d0, d1, (int)npairs, size );
}


void mixChannels(const vector<Mat>& src, vector<Mat>& dst,
                 const int* fromTo, int npairs)
{
    mixChannels(!src.empty() ? &src[0] : 0, src.size(),
                !dst.empty() ? &dst[0] : 0, dst.size(), fromTo, npairs);
}        

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

template<typename sT, typename dT> struct OpCvt
{
    typedef sT type1;
    typedef dT rtype;
    rtype operator()(type1 x) const { return saturate_cast<rtype>(x); }
};

template<typename sT, typename dT, int _fbits> struct OpCvtFixPt
{
    typedef sT type1;
    typedef dT rtype;
    enum { fbits = _fbits };
    rtype operator()(type1 x) const
    {
        return saturate_cast<rtype>((x + (1<<(fbits-1)))>>fbits);
    }
};

template<typename sT, typename dT> struct OpCvtAbs
{
    typedef sT type1;
    typedef dT rtype;
    rtype operator()(type1 x) const { return saturate_cast<rtype>(std::abs(x)); }
};

template<typename sT, typename dT, int _fbits> struct OpCvtAbsFixPt
{
    typedef sT type1;
    typedef dT rtype;
    enum { fbits = _fbits };

    rtype operator()(type1 x) const
    {
        return saturate_cast<rtype>((std::abs(x) + (1<<(fbits-1)))>>fbits);
    }
};

template<class Op> static void
cvtScaleLUT_( const Mat& srcmat, Mat& dstmat, double scale, double shift )
{
    Op op;
    typedef typename Op::rtype DT;
    DT lut[256];
    int i, sdepth = srcmat.depth(), ddepth = dstmat.depth();
    double val = shift;

    for( i = 0; i < 128; i++, val += scale )
        lut[i] = op(val);

    if( sdepth == CV_8S )
        val = shift*2 - val;

    for( ; i < 256; i++, val += scale )
        lut[i] = op(val);

    Mat _srcmat = srcmat;
    if( sdepth == CV_8S )
        _srcmat = Mat(srcmat.size(), CV_8UC(srcmat.channels()), srcmat.data, srcmat.step);
    LUT(_srcmat, Mat(1, 256, ddepth, lut), dstmat);
}

template<typename T, class Op> static void
cvtScale_( const Mat& srcmat, Mat& dstmat, double _scale, double _shift )
{
    Op op;
    typedef typename Op::type1 WT;
    typedef typename Op::rtype DT;
    Size size = getContinuousSize( srcmat, dstmat, srcmat.channels() );
    WT scale = saturate_cast<WT>(_scale), shift = saturate_cast<WT>(_shift);

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        DT* dst = (DT*)(dstmat.data + dstmat.step*y);
        int x = 0;

        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = op(src[x]*scale + shift);
            t1 = op(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = op(src[x+2]*scale + shift);
            t1 = op(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }

        for( ; x < size.width; x++ )
            dst[x] = op(src[x]*scale + shift);
    }
}

template<typename T, class OpFixPt, class Op, int MAX_SHIFT> static void
cvtScaleInt_( const Mat& srcmat, Mat& dstmat, double _scale, double _shift )
{
    if( std::abs(_scale) > 1 || std::abs(_shift) > MAX_SHIFT )
    {
        cvtScale_<T, Op>(srcmat, dstmat, _scale, _shift);
        return;
    }
    OpFixPt op;
    typedef typename OpFixPt::rtype DT;
    Size size = getContinuousSize( srcmat, dstmat, srcmat.channels() );
    int scale = saturate_cast<int>(_scale*(1<<OpFixPt::fbits)),
        shift = saturate_cast<int>(_shift*(1<<OpFixPt::fbits));

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        DT* dst = (DT*)(dstmat.data + dstmat.step*y);
        int x = 0;

        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = op(src[x]*scale + shift);
            t1 = op(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = op(src[x+2]*scale + shift);
            t1 = op(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }

        for( ; x < size.width; x++ )
            dst[x] = op(src[x]*scale + shift);
    }
}

template<typename T, typename DT> static void
cvt_( const Mat& srcmat, Mat& dstmat )
{
    Size size = getContinuousSize( srcmat, dstmat, srcmat.channels() );

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcmat.step*y);
        DT* dst = (DT*)(dstmat.data + dstmat.step*y);
        int x = 0;

        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]);
            t1 = saturate_cast<DT>(src[x+1]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]);
            t1 = saturate_cast<DT>(src[x+3]);
            dst[x+2] = t0; dst[x+3] = t1;
        }

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

static const int FBITS = 15;
#define ICV_SCALE(x) CV_DESCALE((x), FBITS)

typedef void (*CvtFunc)( const Mat& src, Mat& dst );
typedef void (*CvtScaleFunc)( const Mat& src, Mat& dst, double scale, double shift );

void convertScaleAbs( const Mat& src0, Mat& dst, double scale, double shift )
{
    static CvtScaleFunc tab[] =
    {
        cvtScaleLUT_<OpCvtAbs<double, uchar> >,
        cvtScaleLUT_<OpCvtAbs<double, uchar> >,
        cvtScaleInt_<ushort, OpCvtAbsFixPt<int, uchar, FBITS>, OpCvtAbs<float, uchar>, 0>,
        cvtScaleInt_<short, OpCvtAbsFixPt<int, uchar, FBITS>, OpCvtAbs<float, uchar>, 1<<15>,
        cvtScale_<int, OpCvtAbs<double, uchar> >,
        cvtScale_<float, OpCvtAbs<float, uchar> >,
        cvtScale_<double, OpCvtAbs<double, uchar> >, 0
    };

    Mat src = src0;
    dst.create( src.dims, src.size, CV_8UC(src.channels()) );
    CvtScaleFunc func = tab[src.depth()];
    CV_Assert( func != 0 );
    
    if( src.dims <= 2 )
    {
        func( src, dst, scale, shift );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func(it.planes[0], it.planes[1], scale, shift);
    }
}


void Mat::convertTo(Mat& dst, int _type, double alpha, double beta) const
{
    static CvtFunc tab[8][8] =
    {
        {0, cvt_<uchar, schar>, cvt_<uchar, ushort>, cvt_<uchar, short>,
        cvt_<uchar, int>, cvt_<uchar, float>, cvt_<uchar, double>, 0},

        {cvt_<schar, uchar>, 0, cvt_<schar, ushort>, cvt_<schar, short>,
        cvt_<schar, int>, cvt_<schar, float>, cvt_<schar, double>, 0},

        {cvt_<ushort, uchar>, cvt_<ushort, schar>, 0, cvt_<ushort, short>,
        cvt_<ushort, int>, cvt_<ushort, float>, cvt_<ushort, double>, 0},

        {cvt_<short, uchar>, cvt_<short, schar>, cvt_<short, ushort>, 0,
        cvt_<short, int>, cvt_<short, float>, cvt_<short, double>, 0},

        {cvt_<int, uchar>, cvt_<int, schar>, cvt_<int, ushort>,
        cvt_<int, short>, 0, cvt_<int, float>, cvt_<int, double>, 0},

        {cvt_<float, uchar>, cvt_<float, schar>, cvt_<float, ushort>,
        cvt_<float, short>, cvt_<float, int>, 0, cvt_<float, double>, 0},

        {cvt_<double, uchar>, cvt_<double, schar>, cvt_<double, ushort>,
        cvt_<double, short>, cvt_<double, int>, cvt_<double, float>, 0, 0},

        {0,0,0,0,0,0,0,0}
    };

    static CvtScaleFunc stab[8][8] =
    {
        {
            cvtScaleLUT_<OpCvt<double, uchar> >,
            cvtScaleLUT_<OpCvt<double, schar> >,
            cvtScaleLUT_<OpCvt<double, ushort> >,
            cvtScaleLUT_<OpCvt<double, short> >,
            cvtScaleLUT_<OpCvt<double, int> >,
            cvtScaleLUT_<OpCvt<double, float> >,
            cvtScaleLUT_<OpCvt<double, double> >, 0
        },

        {
            // this is copy of the above section,
            // since cvScaleLUT handles both 8u->? and 8s->? cases
            cvtScaleLUT_<OpCvt<double, uchar> >,
            cvtScaleLUT_<OpCvt<double, schar> >,
            cvtScaleLUT_<OpCvt<double, ushort> >,
            cvtScaleLUT_<OpCvt<double, short> >,
            cvtScaleLUT_<OpCvt<double, int> >,
            cvtScaleLUT_<OpCvt<double, float> >,
            cvtScaleLUT_<OpCvt<double, double> >, 0,
        },

        {
            cvtScaleInt_<ushort, OpCvtFixPt<int, uchar, FBITS>, OpCvt<float, uchar>, 0>,
            cvtScaleInt_<ushort, OpCvtFixPt<int, schar, FBITS>, OpCvt<float, schar>, 0>,
            cvtScaleInt_<ushort, OpCvtFixPt<int, ushort, FBITS>, OpCvt<float, ushort>, 0>,
            cvtScaleInt_<ushort, OpCvtFixPt<int, short, FBITS>, OpCvt<float, short>, 0>,
            cvtScale_<ushort, OpCvt<double, int> >,
            cvtScale_<ushort, OpCvt<double, float> >,
            cvtScale_<ushort, OpCvt<double, double> >, 0,
        },

        {
            cvtScaleInt_<short, OpCvtFixPt<int, uchar, FBITS>, OpCvt<float, uchar>, 1<<15>,
            cvtScaleInt_<short, OpCvtFixPt<int, schar, FBITS>, OpCvt<float, schar>, 1<<15>,
            cvtScaleInt_<short, OpCvtFixPt<int, ushort, FBITS>, OpCvt<float, ushort>, 1<<15>,
            cvtScaleInt_<short, OpCvtFixPt<int, short, FBITS>, OpCvt<float, short>, 1<<15>,
            cvtScale_<short, OpCvt<double, int> >,
            cvtScale_<short, OpCvt<double, float> >,
            cvtScale_<short, OpCvt<double, double> >, 0,
        },

        {
            cvtScale_<int, OpCvt<float, uchar> >,
            cvtScale_<int, OpCvt<float, schar> >,
            cvtScale_<int, OpCvt<double, ushort> >,
            cvtScale_<int, OpCvt<double, short> >,
            cvtScale_<int, OpCvt<double, int> >,
            cvtScale_<int, OpCvt<double, float> >,
            cvtScale_<int, OpCvt<double, double> >, 0,
        },

        {
            cvtScale_<float, OpCvt<float, uchar> >,
            cvtScale_<float, OpCvt<float, schar> >,
            cvtScale_<float, OpCvt<float, ushort> >,
            cvtScale_<float, OpCvt<float, short> >,
            cvtScale_<float, OpCvt<float, int> >,
            cvtScale_<float, OpCvt<float, float> >,
            cvtScale_<float, OpCvt<double, double> >, 0,
        },

        {
            cvtScale_<double, OpCvt<double, uchar> >,
            cvtScale_<double, OpCvt<double, schar> >,
            cvtScale_<double, OpCvt<double, ushort> >,
            cvtScale_<double, OpCvt<double, short> >,
            cvtScale_<double, OpCvt<double, int> >,
            cvtScale_<double, OpCvt<double, float> >,
            cvtScale_<double, OpCvt<double, double> >, 0,
        }
    };

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(dst);
        return;
    }

    Mat temp;
    const Mat* psrc = this;
    if( sdepth != ddepth && data == dst.data )
        psrc = &(temp = *this);
        
    CvtFunc func = 0;
    CvtScaleFunc scaleFunc = 0;
    
    if( noScale )
    {
        func = tab[sdepth][ddepth];
        CV_Assert( func != 0 );
    }
    else
    {
        scaleFunc = stab[sdepth][ddepth];
        CV_Assert( scaleFunc != 0 );
    }
    
    if( dims <= 2 )
    {
        dst.create( size(), _type );
        if( func )
            func( *psrc, dst );
        else
            scaleFunc( *psrc, dst, alpha, beta );
    }
    else
    {
        dst.create( dims, size, _type );
        const Mat* arrays[] = {psrc, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            if( func )
                func(it.planes[0], it.planes[1]);
            else
                scaleFunc(it.planes[0], it.planes[1], alpha, beta);
        }
    }    
}

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

template<typename T> static void
LUT8u( const Mat& srcmat, Mat& dstmat, const Mat& lut )
{
    int cn = lut.channels();
    int max_block_size = (1 << 10)*cn;
    const T* _lut = (const T*)lut.data;
    T lutp[4][256];
    int y, i, k;
    Size size = getContinuousSize( srcmat, dstmat, srcmat.channels() );

    if( cn == 1 )
    {
        for( y = 0; y < size.height; y++ )
        {
            const uchar* src = srcmat.data + srcmat.step*y;
            T* dst = (T*)(dstmat.data + dstmat.step*y);

            for( i = 0; i < size.width; i++ )
                dst[i] = _lut[src[i]];
        }
        return;
    }

    if( size.width*size.height < 256 )
    {
        for( y = 0; y < size.height; y++ )
        {
            const uchar* src = srcmat.data + srcmat.step*y;
            T* dst = (T*)(dstmat.data + dstmat.step*y);

            for( k = 0; k < cn; k++ )
                for( i = 0; i < size.width; i += cn )
                    dst[i+k] = _lut[src[i+k]*cn+k];
        }
        return;
    }

    /* repack the lut to planar layout */
    for( k = 0; k < cn; k++ )
        for( i = 0; i < 256; i++ )
            lutp[k][i] = _lut[i*cn+k];

    for( y = 0; y < size.height; y++ )
    {
        const uchar* src = srcmat.data + srcmat.step*y;
        T* dst = (T*)(dstmat.data + dstmat.step*y);

        for( i = 0; i < size.width; )
        {
            int j, limit = std::min(size.width, i + max_block_size);
            for( k = 0; k < cn; k++, src++, dst++ )
            {
                const T* lut = lutp[k];
                for( j = i; j <= limit - cn*2; j += cn*2 )
                {
                    T t0 = lut[src[j]];
                    T t1 = lut[src[j+cn]];
                    dst[j] = t0; dst[j+cn] = t1;
                }

                for( ; j < limit; j += cn )
                    dst[j] = lut[src[j]];
            }
            src -= cn;
            dst -= cn;
            i = limit;
        }
    }
}

typedef void (*LUTFunc)( const Mat& src, Mat& dst, const Mat& lut );

void LUT( const Mat& src, const Mat& lut, Mat& dst )
{
    int cn = src.channels(), esz1 = (int)lut.elemSize1();

    CV_Assert( (lut.channels() == cn || lut.channels() == 1) &&
        lut.rows*lut.cols == 256 && lut.isContinuous() &&
        (src.depth() == CV_8U || src.depth() == CV_8S) );
    dst.create( src.size(), CV_MAKETYPE(lut.depth(), cn));

    LUTFunc func = 0;
    if( esz1 == 1 )
        func = LUT8u<uchar>;
    else if( esz1 == 2 )
        func = LUT8u<ushort>;
    else if( esz1 == 4 )
        func = LUT8u<int>;
    else if( esz1 == 8 )
        func = LUT8u<int64>;
    else    
        CV_Error(CV_StsUnsupportedFormat, "");
    func( src, dst, lut );
}


void normalize( const Mat& src, Mat& dst, double a, double b,
                int norm_type, int rtype, const Mat& mask )
{
    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxLoc( src, &smin, &smax, 0, 0, mask );
        scale = (dmax - dmin)*(smax - smin > DBL_EPSILON ? 1./(smax - smin) : 0);
        shift = dmin - smin*scale;
    }
    else if( norm_type == CV_L2 || norm_type == CV_L1 || norm_type == CV_C )
    {
        scale = norm( src, norm_type, mask );
        scale = scale > DBL_EPSILON ? a/scale : 0.;
        shift = 0;
    }
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported norm type" );
    
    if( !mask.data )
        src.convertTo( dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( dst, mask );
    }
}

}

CV_IMPL void
cvSplit( const void* srcarr, void* dstarr0, void* dstarr1, void* dstarr2, void* dstarr3 )
{
    void* dptrs[] = { dstarr0, dstarr1, dstarr2, dstarr3 };
    cv::Mat src = cv::cvarrToMat(srcarr);
    int i, j, nz = 0;
    for( i = 0; i < 4; i++ )
        nz += dptrs[i] != 0;
    CV_Assert( nz > 0 );
    cv::vector<cv::Mat> dvec(nz);
    cv::vector<int> pairs(nz*2);

    for( i = j = 0; i < 4; i++ )
    {
        if( dptrs[i] != 0 )
        {
            dvec[j] = cv::cvarrToMat(dptrs[i]);
            CV_Assert( dvec[j].size() == src.size() &&
                dvec[j].depth() == src.depth() &&
                dvec[j].channels() == 1 && i < src.channels() );
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
    cv::vector<cv::Mat> svec(nz);
    cv::vector<int> pairs(nz*2);

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
    cv::AutoBuffer<cv::Mat, 32> buf;

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

/* End of file. */
