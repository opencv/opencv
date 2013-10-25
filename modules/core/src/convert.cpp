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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
*                                       split & merge                                    *
\****************************************************************************************/

template<typename T> static void
split_( const T* src, T** dst, int len, int cn )
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        T* dst0 = dst[0];
        for( i = j = 0; i < len; i++, j += cn )
            dst0[i] = src[j];
    }
    else if( k == 2 )
    {
        T *dst0 = dst[0], *dst1 = dst[1];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
        }
    }
    else if( k == 3 )
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
            dst2[i] = src[j+2];
        }
    }
    else
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }

    for( ; k < cn; k += 4 )
    {
        T *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }
}

template<typename T> static void
merge_( const T** src, T* dst, int len, int cn )
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        const T* src0 = src[0];
        for( i = j = 0; i < len; i++, j += cn )
            dst[j] = src0[i];
    }
    else if( k == 2 )
    {
        const T *src0 = src[0], *src1 = src[1];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        for( i = j = 0; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }

    for( ; k < cn; k += 4 )
    {
        const T *src0 = src[k], *src1 = src[k+1], *src2 = src[k+2], *src3 = src[k+3];
        for( i = 0, j = k; i < len; i++, j += cn )
        {
            dst[j] = src0[i]; dst[j+1] = src1[i];
            dst[j+2] = src2[i]; dst[j+3] = src3[i];
        }
    }
}

static void split8u(const uchar* src, uchar** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split16u(const ushort* src, ushort** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split32s(const int* src, int** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void split64s(const int64* src, int64** dst, int len, int cn )
{
    split_(src, dst, len, cn);
}

static void merge8u(const uchar** src, uchar* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge16u(const ushort** src, ushort* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge32s(const int** src, int* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

static void merge64s(const int64** src, int64* dst, int len, int cn )
{
    merge_(src, dst, len, cn);
}

typedef void (*SplitFunc)(const uchar* src, uchar** dst, int len, int cn);
typedef void (*MergeFunc)(const uchar** src, uchar* dst, int len, int cn);

static SplitFunc getSplitFunc(int depth)
{
    static SplitFunc splitTab[] =
    {
        (SplitFunc)GET_OPTIMIZED(split8u), (SplitFunc)GET_OPTIMIZED(split8u), (SplitFunc)GET_OPTIMIZED(split16u), (SplitFunc)GET_OPTIMIZED(split16u),
        (SplitFunc)GET_OPTIMIZED(split32s), (SplitFunc)GET_OPTIMIZED(split32s), (SplitFunc)GET_OPTIMIZED(split64s), 0
    };

    return splitTab[depth];
}

static MergeFunc getMergeFunc(int depth)
{
    static MergeFunc mergeTab[] =
    {
        (MergeFunc)GET_OPTIMIZED(merge8u), (MergeFunc)GET_OPTIMIZED(merge8u), (MergeFunc)GET_OPTIMIZED(merge16u), (MergeFunc)GET_OPTIMIZED(merge16u),
        (MergeFunc)GET_OPTIMIZED(merge32s), (MergeFunc)GET_OPTIMIZED(merge32s), (MergeFunc)GET_OPTIMIZED(merge64s), 0
    };

    return mergeTab[depth];
}

}

void cv::split(const Mat& src, Mat* mv)
{
    int k, depth = src.depth(), cn = src.channels();
    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    SplitFunc func = getSplitFunc(depth);
    CV_Assert( func != 0 );

    int esz = (int)src.elemSize(), esz1 = (int)src.elemSize1();
    int blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for( k = 0; k < cn; k++ )
    {
        mv[k].create(src.dims, src.size, depth);
        arrays[k+1] = &mv[k];
    }

    NAryMatIterator it(arrays, ptrs, cn+1);
    int total = (int)it.size, blocksize = cn <= 4 ? total : std::min(total, blocksize0);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < total; j += blocksize )
        {
            int bsz = std::min(total - j, blocksize);
            func( ptrs[0], &ptrs[1], bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( k = 0; k < cn; k++ )
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}

void cv::split(InputArray _m, OutputArrayOfArrays _mv)
{
    Mat m = _m.getMat();
    if( m.empty() )
    {
        _mv.release();
        return;
    }
    CV_Assert( !_mv.fixedType() || _mv.empty() || _mv.type() == m.depth() );
    _mv.create(m.channels(), 1, m.depth());
    Mat* dst = &_mv.getMatRef(0);
    split(m, dst);
}

void cv::merge(const Mat* mv, size_t n, OutputArray _dst)
{
    CV_Assert( mv && n > 0 );

    int depth = mv[0].depth();
    bool allch1 = true;
    int k, cn = 0;
    size_t i;

    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        cn += mv[i].channels();
    }

    CV_Assert( 0 < cn && cn <= CV_CN_MAX );
    _dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, cn));
    Mat dst = _dst.getMat();

    if( n == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    if( !allch1 )
    {
        AutoBuffer<int> pairs(cn*2);
        int j, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], cn );
        return;
    }

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    int blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    int total = (int)it.size, blocksize = cn <= 4 ? total : std::min(total, blocksize0);
    MergeFunc func = getMergeFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < total; j += blocksize )
        {
            int bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}

void cv::merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    std::vector<Mat> mv;
    _mv.getMatVector(mv);
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), _dst);
}

/****************************************************************************************\
*                       Generalized split/merge: mixing channels                         *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
mixChannels_( const T** src, const int* sdelta,
              T** dst, const int* ddelta,
              int len, int npairs )
{
    int i, k;
    for( k = 0; k < npairs; k++ )
    {
        const T* s = src[k];
        T* d = dst[k];
        int ds = sdelta[k], dd = ddelta[k];
        if( s )
        {
            for( i = 0; i <= len - 2; i += 2, s += ds*2, d += dd*2 )
            {
                T t0 = s[0], t1 = s[ds];
                d[0] = t0; d[dd] = t1;
            }
            if( i < len )
                d[0] = s[0];
        }
        else
        {
            for( i = 0; i <= len - 2; i += 2, d += dd*2 )
                d[0] = d[dd] = 0;
            if( i < len )
                d[0] = 0;
        }
    }
}


static void mixChannels8u( const uchar** src, const int* sdelta,
                           uchar** dst, const int* ddelta,
                           int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels16u( const ushort** src, const int* sdelta,
                            ushort** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels32s( const int** src, const int* sdelta,
                            int** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

static void mixChannels64s( const int64** src, const int* sdelta,
                            int64** dst, const int* ddelta,
                            int len, int npairs )
{
    mixChannels_(src, sdelta, dst, ddelta, len, npairs);
}

typedef void (*MixChannelsFunc)( const uchar** src, const int* sdelta,
        uchar** dst, const int* ddelta, int len, int npairs );

static MixChannelsFunc getMixchFunc(int depth)
{
    static MixChannelsFunc mixchTab[] =
    {
        (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels8u, (MixChannelsFunc)mixChannels16u,
        (MixChannelsFunc)mixChannels16u, (MixChannelsFunc)mixChannels32s, (MixChannelsFunc)mixChannels32s,
        (MixChannelsFunc)mixChannels64s, 0
    };

    return mixchTab[depth];
}

}

void cv::mixChannels( const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts, const int* fromTo, size_t npairs )
{
    if( npairs == 0 )
        return;
    CV_Assert( src && nsrcs > 0 && dst && ndsts > 0 && fromTo && npairs > 0 );

    size_t i, j, k, esz1 = dst[0].elemSize1();
    int depth = dst[0].depth();

    AutoBuffer<uchar> buf((nsrcs + ndsts + 1)*(sizeof(Mat*) + sizeof(uchar*)) + npairs*(sizeof(uchar*)*2 + sizeof(int)*6));
    const Mat** arrays = (const Mat**)(uchar*)buf;
    uchar** ptrs = (uchar**)(arrays + nsrcs + ndsts);
    const uchar** srcs = (const uchar**)(ptrs + nsrcs + ndsts + 1);
    uchar** dsts = (uchar**)(srcs + npairs);
    int* tab = (int*)(dsts + npairs);
    int *sdelta = (int*)(tab + npairs*4), *ddelta = sdelta + npairs;

    for( i = 0; i < nsrcs; i++ )
        arrays[i] = &src[i];
    for( i = 0; i < ndsts; i++ )
        arrays[i + nsrcs] = &dst[i];
    ptrs[nsrcs + ndsts] = 0;

    for( i = 0; i < npairs; i++ )
    {
        int i0 = fromTo[i*2], i1 = fromTo[i*2+1];
        if( i0 >= 0 )
        {
            for( j = 0; j < nsrcs; i0 -= src[j].channels(), j++ )
                if( i0 < src[j].channels() )
                    break;
            CV_Assert(j < nsrcs && src[j].depth() == depth);
            tab[i*4] = (int)j; tab[i*4+1] = (int)(i0*esz1);
            sdelta[i] = src[j].channels();
        }
        else
        {
            tab[i*4] = (int)(nsrcs + ndsts); tab[i*4+1] = 0;
            sdelta[i] = 0;
        }

        for( j = 0; j < ndsts; i1 -= dst[j].channels(), j++ )
            if( i1 < dst[j].channels() )
                break;
        CV_Assert(i1 >= 0 && j < ndsts && dst[j].depth() == depth);
        tab[i*4+2] = (int)(j + nsrcs); tab[i*4+3] = (int)(i1*esz1);
        ddelta[i] = dst[j].channels();
    }

    NAryMatIterator it(arrays, ptrs, (int)(nsrcs + ndsts));
    int total = (int)it.size, blocksize = std::min(total, (int)((BLOCK_SIZE + esz1-1)/esz1));
    MixChannelsFunc func = getMixchFunc(depth);

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( k = 0; k < npairs; k++ )
        {
            srcs[k] = ptrs[tab[k*4]] + tab[k*4+1];
            dsts[k] = ptrs[tab[k*4+2]] + tab[k*4+3];
        }

        for( int t = 0; t < total; t += blocksize )
        {
            int bsz = std::min(total - t, blocksize);
            func( srcs, sdelta, dsts, ddelta, bsz, (int)npairs );

            if( t + blocksize < total )
                for( k = 0; k < npairs; k++ )
                {
                    srcs[k] += blocksize*sdelta[k]*esz1;
                    dsts[k] += blocksize*ddelta[k]*esz1;
                }
        }
    }
}


void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                 const int* fromTo, size_t npairs)
{
    if(npairs == 0)
        return;
    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
                      src.kind() != _InputArray::STD_VECTOR_VECTOR;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
                      dst.kind() != _InputArray::STD_VECTOR_VECTOR;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, fromTo, npairs);
}

void cv::mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                     const std::vector<int>& fromTo)
{
    if(fromTo.empty())
        return;
    bool src_is_mat = src.kind() != _InputArray::STD_VECTOR_MAT &&
                      src.kind() != _InputArray::STD_VECTOR_VECTOR;
    bool dst_is_mat = dst.kind() != _InputArray::STD_VECTOR_MAT &&
                      dst.kind() != _InputArray::STD_VECTOR_VECTOR;
    int i;
    int nsrc = src_is_mat ? 1 : (int)src.total();
    int ndst = dst_is_mat ? 1 : (int)dst.total();

    CV_Assert(fromTo.size()%2 == 0 && nsrc > 0 && ndst > 0);
    cv::AutoBuffer<Mat> _buf(nsrc + ndst);
    Mat* buf = _buf;
    for( i = 0; i < nsrc; i++ )
        buf[i] = src.getMat(src_is_mat ? -1 : i);
    for( i = 0; i < ndst; i++ )
        buf[nsrc + i] = dst.getMat(dst_is_mat ? -1 : i);
    mixChannels(&buf[0], nsrc, &buf[nsrc], ndst, &fromTo[0], fromTo.size()/2);
}

void cv::extractChannel(InputArray _src, OutputArray _dst, int coi)
{
    Mat src = _src.getMat();
    CV_Assert( 0 <= coi && coi < src.channels() );
    _dst.create(src.dims, &src.size[0], src.depth());
    Mat dst = _dst.getMat();
    int ch[] = { coi, 0 };
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

void cv::insertChannel(InputArray _src, InputOutputArray _dst, int coi)
{
    Mat src = _src.getMat(), dst = _dst.getMat();
    CV_Assert( src.size == dst.size && src.depth() == dst.depth() );
    CV_Assert( 0 <= coi && coi < dst.channels() && src.channels() == 1 );
    int ch[] = { 0, coi };
    mixChannels(&src, 1, &dst, 1, ch, 1);
}

/****************************************************************************************\
*                                convertScale[Abs]                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename DT, typename WT> static void
cvtScaleAbs_( const T* src, size_t sstep,
              DT* dst, size_t dstep, Size size,
              WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(std::abs(src[x]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+1]*scale + shift));
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(std::abs(src[x+2]*scale + shift));
            t1 = saturate_cast<DT>(std::abs(src[x+3]*scale + shift));
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(std::abs(src[x]*scale + shift));
    }
}


template<typename T, typename DT, typename WT> static void
cvtScale_( const T* src, size_t sstep,
           DT* dst, size_t dstep, Size size,
           WT scale, WT shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x]*scale + shift);
            t1 = saturate_cast<DT>(src[x+1]*scale + shift);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<DT>(src[x+2]*scale + shift);
            t1 = saturate_cast<DT>(src[x+3]*scale + shift);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif

        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]*scale + shift);
    }
}

//vz optimized template specialization
template<> void
cvtScale_<short, short, float>( const short* src, size_t sstep,
           short* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_SSE2
            if(USE_SSE2)
            {
                __m128 scale128 = _mm_set1_ps (scale);
                __m128 shift128 = _mm_set1_ps (shift);
                for(; x <= size.width - 8; x += 8 )
                {
                    __m128i r0 = _mm_loadl_epi64((const __m128i*)(src + x));
                    __m128i r1 = _mm_loadl_epi64((const __m128i*)(src + x + 4));
                    __m128 rf0 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                    __m128 rf1 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));
                    rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                    rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);
                    r0 = _mm_cvtps_epi32(rf0);
                    r1 = _mm_cvtps_epi32(rf1);
                    r0 = _mm_packs_epi32(r0, r1);
                    _mm_storeu_si128((__m128i*)(dst + x), r0);
                }
            }
        #endif

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]*scale + shift);
    }
}

template<> void
cvtScale_<short, int, float>( const short* src, size_t sstep,
           int* dst, size_t dstep, Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;

         #if CV_SSE2
            if(USE_SSE2)//~5X
            {
                __m128 scale128 = _mm_set1_ps (scale);
                __m128 shift128 = _mm_set1_ps (shift);
                for(; x <= size.width - 8; x += 8 )
                {
                    __m128i r0 = _mm_loadl_epi64((const __m128i*)(src + x));
                    __m128i r1 = _mm_loadl_epi64((const __m128i*)(src + x + 4));
                    __m128 rf0 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                    __m128 rf1 =_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));
                    rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                    rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);
                    r0 = _mm_cvtps_epi32(rf0);
                    r1 = _mm_cvtps_epi32(rf1);

                    _mm_storeu_si128((__m128i*)(dst + x), r0);
                    _mm_storeu_si128((__m128i*)(dst + x + 4), r1);
                }
            }
        #endif

        //We will wait Haswell
        /*
        #if CV_AVX
            if(USE_AVX)//2X - bad variant
            {
                ////TODO:AVX implementation (optimization?) required
                __m256 scale256 = _mm256_set1_ps (scale);
                __m256 shift256 = _mm256_set1_ps (shift);
                for(; x <= size.width - 8; x += 8 )
                {
                    __m256i buf = _mm256_set_epi32((int)(*(src+x+7)),(int)(*(src+x+6)),(int)(*(src+x+5)),(int)(*(src+x+4)),(int)(*(src+x+3)),(int)(*(src+x+2)),(int)(*(src+x+1)),(int)(*(src+x)));
                    __m256 r0 = _mm256_add_ps( _mm256_mul_ps(_mm256_cvtepi32_ps (buf), scale256), shift256);
                    __m256i res = _mm256_cvtps_epi32(r0);
                    _mm256_storeu_si256 ((__m256i*)(dst+x), res);
                }
            }
        #endif*/

        for(; x < size.width; x++ )
            dst[x] = saturate_cast<int>(src[x]*scale + shift);
    }
}

template<typename T, typename DT> static void
cvt_( const T* src, size_t sstep,
      DT* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if CV_ENABLE_UNROLLED
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
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

//vz optimized template specialization, test Core_ConvertScale/ElemWiseTest
template<>  void
cvt_<float, short>( const float* src, size_t sstep,
     short* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;
        #if   CV_SSE2
        if(USE_SSE2){
              for( ; x <= size.width - 8; x += 8 )
            {
                __m128 src128 = _mm_loadu_ps (src + x);
                __m128i src_int128 = _mm_cvtps_epi32 (src128);

                src128 = _mm_loadu_ps (src + x + 4);
                __m128i src1_int128 = _mm_cvtps_epi32 (src128);

                src1_int128 = _mm_packs_epi32(src_int128, src1_int128);
                _mm_storeu_si128((__m128i*)(dst + x),src1_int128);
            }
        }
        #endif
        for( ; x < size.width; x++ )
            dst[x] = saturate_cast<short>(src[x]);
    }

}


template<typename T> static void
cpy_( const T* src, size_t sstep, T* dst, size_t dstep, Size size )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for( ; size.height--; src += sstep, dst += dstep )
        memcpy(dst, src, size.width*sizeof(src[0]));
}

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype) \
static void cvtScaleAbs##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}

#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype) \
static void cvtScale##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
dtype* dst, size_t dstep, Size size, double* scale) \
{ \
    cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]); \
}


#define DEF_CVT_FUNC(suffix, stype, dtype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CPY_FUNC(suffix, stype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
stype* dst, size_t dstep, Size size, double*) \
{ \
    cpy_(src, sstep, dst, dstep, size); \
}


DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float);
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float);

DEF_CVT_SCALE_FUNC(8u,     uchar, uchar, float);
DEF_CVT_SCALE_FUNC(8s8u,   schar, uchar, float);
DEF_CVT_SCALE_FUNC(16u8u,  ushort, uchar, float);
DEF_CVT_SCALE_FUNC(16s8u,  short, uchar, float);
DEF_CVT_SCALE_FUNC(32s8u,  int, uchar, float);
DEF_CVT_SCALE_FUNC(32f8u,  float, uchar, float);
DEF_CVT_SCALE_FUNC(64f8u,  double, uchar, float);

DEF_CVT_SCALE_FUNC(8u8s,   uchar, schar, float);
DEF_CVT_SCALE_FUNC(8s,     schar, schar, float);
DEF_CVT_SCALE_FUNC(16u8s,  ushort, schar, float);
DEF_CVT_SCALE_FUNC(16s8s,  short, schar, float);
DEF_CVT_SCALE_FUNC(32s8s,  int, schar, float);
DEF_CVT_SCALE_FUNC(32f8s,  float, schar, float);
DEF_CVT_SCALE_FUNC(64f8s,  double, schar, float);

DEF_CVT_SCALE_FUNC(8u16u,  uchar, ushort, float);
DEF_CVT_SCALE_FUNC(8s16u,  schar, ushort, float);
DEF_CVT_SCALE_FUNC(16u,    ushort, ushort, float);
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float);
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float);
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float);
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float);

DEF_CVT_SCALE_FUNC(8u16s,  uchar, short, float);
DEF_CVT_SCALE_FUNC(8s16s,  schar, short, float);
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float);
DEF_CVT_SCALE_FUNC(16s,    short, short, float);
DEF_CVT_SCALE_FUNC(32s16s, int, short, float);
DEF_CVT_SCALE_FUNC(32f16s, float, short, float);
DEF_CVT_SCALE_FUNC(64f16s, double, short, float);

DEF_CVT_SCALE_FUNC(8u32s,  uchar, int, float);
DEF_CVT_SCALE_FUNC(8s32s,  schar, int, float);
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float);
DEF_CVT_SCALE_FUNC(16s32s, short, int, float);
DEF_CVT_SCALE_FUNC(32s,    int, int, double);
DEF_CVT_SCALE_FUNC(32f32s, float, int, float);
DEF_CVT_SCALE_FUNC(64f32s, double, int, double);

DEF_CVT_SCALE_FUNC(8u32f,  uchar, float, float);
DEF_CVT_SCALE_FUNC(8s32f,  schar, float, float);
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float);
DEF_CVT_SCALE_FUNC(16s32f, short, float, float);
DEF_CVT_SCALE_FUNC(32s32f, int, float, double);
DEF_CVT_SCALE_FUNC(32f,    float, float, float);
DEF_CVT_SCALE_FUNC(64f32f, double, float, double);

DEF_CVT_SCALE_FUNC(8u64f,  uchar, double, double);
DEF_CVT_SCALE_FUNC(8s64f,  schar, double, double);
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double);
DEF_CVT_SCALE_FUNC(16s64f, short, double, double);
DEF_CVT_SCALE_FUNC(32s64f, int, double, double);
DEF_CVT_SCALE_FUNC(32f64f, float, double, double);
DEF_CVT_SCALE_FUNC(64f,    double, double, double);

DEF_CPY_FUNC(8u,     uchar);
DEF_CVT_FUNC(8s8u,   schar, uchar);
DEF_CVT_FUNC(16u8u,  ushort, uchar);
DEF_CVT_FUNC(16s8u,  short, uchar);
DEF_CVT_FUNC(32s8u,  int, uchar);
DEF_CVT_FUNC(32f8u,  float, uchar);
DEF_CVT_FUNC(64f8u,  double, uchar);

DEF_CVT_FUNC(8u8s,   uchar, schar);
DEF_CVT_FUNC(16u8s,  ushort, schar);
DEF_CVT_FUNC(16s8s,  short, schar);
DEF_CVT_FUNC(32s8s,  int, schar);
DEF_CVT_FUNC(32f8s,  float, schar);
DEF_CVT_FUNC(64f8s,  double, schar);

DEF_CVT_FUNC(8u16u,  uchar, ushort);
DEF_CVT_FUNC(8s16u,  schar, ushort);
DEF_CPY_FUNC(16u,    ushort);
DEF_CVT_FUNC(16s16u, short, ushort);
DEF_CVT_FUNC(32s16u, int, ushort);
DEF_CVT_FUNC(32f16u, float, ushort);
DEF_CVT_FUNC(64f16u, double, ushort);

DEF_CVT_FUNC(8u16s,  uchar, short);
DEF_CVT_FUNC(8s16s,  schar, short);
DEF_CVT_FUNC(16u16s, ushort, short);
DEF_CVT_FUNC(32s16s, int, short);
DEF_CVT_FUNC(32f16s, float, short);
DEF_CVT_FUNC(64f16s, double, short);

DEF_CVT_FUNC(8u32s,  uchar, int);
DEF_CVT_FUNC(8s32s,  schar, int);
DEF_CVT_FUNC(16u32s, ushort, int);
DEF_CVT_FUNC(16s32s, short, int);
DEF_CPY_FUNC(32s,    int);
DEF_CVT_FUNC(32f32s, float, int);
DEF_CVT_FUNC(64f32s, double, int);

DEF_CVT_FUNC(8u32f,  uchar, float);
DEF_CVT_FUNC(8s32f,  schar, float);
DEF_CVT_FUNC(16u32f, ushort, float);
DEF_CVT_FUNC(16s32f, short, float);
DEF_CVT_FUNC(32s32f, int, float);
DEF_CVT_FUNC(64f32f, double, float);

DEF_CVT_FUNC(8u64f,  uchar, double);
DEF_CVT_FUNC(8s64f,  schar, double);
DEF_CVT_FUNC(16u64f, ushort, double);
DEF_CVT_FUNC(16s64f, short, double);
DEF_CVT_FUNC(32s64f, int, double);
DEF_CVT_FUNC(32f64f, float, double);
DEF_CPY_FUNC(64s,    int64);

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] =
    {
        (BinaryFunc)cvtScaleAbs8u, (BinaryFunc)cvtScaleAbs8s8u, (BinaryFunc)cvtScaleAbs16u8u,
        (BinaryFunc)cvtScaleAbs16s8u, (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
        (BinaryFunc)cvtScaleAbs64f8u, 0
    };

    return cvtScaleAbsTab[depth];
}

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] =
    {
        {
            (BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
            (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u), (BinaryFunc)GET_OPTIMIZED(cvt32f16u),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
            (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
            (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
            (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
            (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f), (BinaryFunc)GET_OPTIMIZED(cvt32f64f),
            (BinaryFunc)(cvt64s), 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

static BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] =
    {
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u),
            (BinaryFunc)cvtScale64f8u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s),
            (BinaryFunc)cvtScale64f8s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16u),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
            (BinaryFunc)cvtScale64f16u, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
            (BinaryFunc)cvtScale64f16s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s),
            (BinaryFunc)cvtScale64f32s, 0
        },
        {
            (BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f),
            (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f),
            (BinaryFunc)cvtScale64f32f, 0
        },
        {
            (BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f,
            (BinaryFunc)cvtScale16s64f, (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f,
            (BinaryFunc)cvtScale64f, 0
        },
        {
            0, 0, 0, 0, 0, 0, 0, 0
        }
    };

    return cvtScaleTab[CV_MAT_DEPTH(ddepth)][CV_MAT_DEPTH(sdepth)];
}

}

void cv::convertScaleAbs( InputArray _src, OutputArray _dst, double alpha, double beta )
{
    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create( src.dims, src.size, CV_8UC(cn) );
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size*cn, 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale );
    }
}

void cv::Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        _dst.create( size(), _type );
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        _dst.create( dims, size, _type );
        Mat dst = _dst.getMat();
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale);
    }
}

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
LUT8u_( const uchar* src, const T* lut, T* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        for( int i = 0; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

static void LUT8u_8u( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_8s( const uchar* src, const schar* lut, schar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16u( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16s( const uchar* src, const short* lut, short* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32s( const uchar* src, const int* lut, int* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32f( const uchar* src, const float* lut, float* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_64f( const uchar* src, const double* lut, double* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc lutTab[] =
{
    (LUTFunc)LUT8u_8u, (LUTFunc)LUT8u_8s, (LUTFunc)LUT8u_16u, (LUTFunc)LUT8u_16s,
    (LUTFunc)LUT8u_32s, (LUTFunc)LUT8u_32f, (LUTFunc)LUT8u_64f, 0
};

}

void cv::LUT( InputArray _src, InputArray _lut, OutputArray _dst )
{
    Mat src = _src.getMat(), lut = _lut.getMat();
    int cn = src.channels();
    int lutcn = lut.channels();

    CV_Assert( (lutcn == cn || lutcn == 1) &&
        lut.total() == 256 && lut.isContinuous() &&
        (src.depth() == CV_8U || src.depth() == CV_8S) );
    _dst.create( src.dims, src.size, CV_MAKETYPE(lut.depth(), cn));
    Mat dst = _dst.getMat();

    LUTFunc func = lutTab[lut.depth()];
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.data, ptrs[1], len, cn, lutcn);
}


void cv::normalize( InputArray _src, OutputArray _dst, double a, double b,
                    int norm_type, int rtype, InputArray _mask )
{
    Mat src = _src.getMat(), mask = _mask.getMat();

    double scale = 1, shift = 0;
    if( norm_type == CV_MINMAX )
    {
        double smin = 0, smax = 0;
        double dmin = MIN( a, b ), dmax = MAX( a, b );
        minMaxLoc( _src, &smin, &smax, 0, 0, mask );
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

    if( rtype < 0 )
        rtype = _dst.fixedType() ? _dst.depth() : src.depth();

    _dst.create(src.dims, src.size, CV_MAKETYPE(rtype, src.channels()));
    Mat dst = _dst.getMat();

    if( !mask.data )
        src.convertTo( dst, rtype, scale, shift );
    else
    {
        Mat temp;
        src.convertTo( temp, rtype, scale, shift );
        temp.copyTo( dst, mask );
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

/* End of file. */
