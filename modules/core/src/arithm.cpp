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

/* ////////////////////////////////////////////////////////////////////
//
//  Matrix arithmetic and logical operations: +, -, *, /, &, |, ^, ~, abs ...
//
// */

#include "precomp.hpp"

namespace cv
{

#if CV_SSE2

enum { ARITHM_SIMD = CV_CPU_SSE2 };
    
template<class Op8> struct VBinOp8
{
    int operator()(const uchar* src1, const uchar* src2, uchar* dst, int len) const
    {
        int x = 0;
        for( ; x <= len - 32; x += 32 )
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src1 + x));
            __m128i r1 = _mm_loadu_si128((const __m128i*)(src1 + x + 16));
            r0 = op(r0,_mm_loadu_si128((const __m128i*)(src2 + x)));
            r1 = op(r1,_mm_loadu_si128((const __m128i*)(src2 + x + 16)));
            _mm_storeu_si128((__m128i*)(dst + x), r0);
            _mm_storeu_si128((__m128i*)(dst + x + 16), r1);
        }
        for( ; x <= len - 8; x += 8 )
        {
            __m128i r0 = _mm_loadl_epi64((const __m128i*)(src1 + x));
            r0 = op(r0,_mm_loadl_epi64((const __m128i*)(src2 + x)));
            _mm_storel_epi64((__m128i*)(dst + x), r0);
        }
        return x;
    }
    Op8 op;
};

template<typename T, class Op16> struct VBinOp16
{
    int operator()(const T* src1, const T* src2, T* dst, int len) const
    {
        int x = 0;
        for( ; x <= len - 16; x += 16 )
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src1 + x));
            __m128i r1 = _mm_loadu_si128((const __m128i*)(src1 + x + 8));
            r0 = op(r0,_mm_loadu_si128((const __m128i*)(src2 + x)));
            r1 = op(r1,_mm_loadu_si128((const __m128i*)(src2 + x + 8)));
            _mm_storeu_si128((__m128i*)(dst + x), r0);
            _mm_storeu_si128((__m128i*)(dst + x + 8), r1);
        }
        for( ; x <= len - 4; x += 4 )
        {
            __m128i r0 = _mm_loadl_epi64((const __m128i*)(src1 + x));
            r0 = op(r0,_mm_loadl_epi64((const __m128i*)(src2 + x)));
            _mm_storel_epi64((__m128i*)(dst + x), r0);
        }
        return x;
    }
    Op16 op;
};

template<class Op32f> struct VBinOp32f
{
    int operator()(const float* src1, const float* src2, float* dst, int len) const
    {
        int x = 0;
        if( (((size_t)src1|(size_t)src2|(size_t)dst)&15) == 0 )
            for( ; x <= len - 8; x += 8 )
            {
                __m128 r0 = _mm_load_ps(src1 + x);
                __m128 r1 = _mm_load_ps(src1 + x + 4);
                r0 = op(r0,_mm_load_ps(src2 + x));
                r1 = op(r1,_mm_load_ps(src2 + x + 4));
                _mm_store_ps(dst + x, r0);
                _mm_store_ps(dst + x + 4, r1);
            }
        else
            for( ; x <= len - 8; x += 8 )
            {
                __m128 r0 = _mm_loadu_ps(src1 + x);
                __m128 r1 = _mm_loadu_ps(src1 + x + 4);
                r0 = op(r0,_mm_loadu_ps(src2 + x));
                r1 = op(r1,_mm_loadu_ps(src2 + x + 4));
                _mm_storeu_ps(dst + x, r0);
                _mm_storeu_ps(dst + x + 4, r1);
            }
        return x;
    }
    Op32f op;
};

struct _VAdd8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_adds_epu8(a,b); }};
struct _VSub8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_subs_epu8(a,b); }};
struct _VMin8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_min_epu8(a,b); }};
struct _VMax8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_max_epu8(a,b); }};
struct _VCmpGT8u { __m128i operator()(const __m128i& a, const __m128i& b) const
{
    __m128i delta = _mm_set1_epi32(0x80808080);
    return _mm_cmpgt_epi8(_mm_xor_si128(a,delta),_mm_xor_si128(b,delta));
}};
struct _VCmpEQ8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_cmpeq_epi8(a,b); }};
struct _VAbsDiff8u
{
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_add_epi8(_mm_subs_epu8(a,b),_mm_subs_epu8(b,a)); }
};
struct _VAdd16u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_adds_epu16(a,b); }};
struct _VSub16u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_subs_epu16(a,b); }};
struct _VMin16u
{
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_subs_epu16(a,_mm_subs_epu16(a,b)); }
};
struct _VMax16u
{
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_adds_epu16(_mm_subs_epu16(a,b),b); }
};
struct _VAbsDiff16u
{
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_add_epi16(_mm_subs_epu16(a,b),_mm_subs_epu16(b,a)); }
};
struct _VAdd16s { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_adds_epi16(a,b); }};
struct _VSub16s { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_subs_epi16(a,b); }};
struct _VMin16s { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_min_epi16(a,b); }};
struct _VMax16s { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_max_epi16(a,b); }};
struct _VAbsDiff16s
{
    __m128i operator()(const __m128i& a, const __m128i& b) const
    {
        __m128i M = _mm_max_epi16(a,b), m = _mm_min_epi16(a,b);
        return _mm_subs_epi16(M, m);
    }
};
struct _VAdd32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_add_ps(a,b); }};
struct _VSub32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_sub_ps(a,b); }};
struct _VMin32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_min_ps(a,b); }};
struct _VMax32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_max_ps(a,b); }};
static int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
struct _VAbsDiff32f
{
    __m128 operator()(const __m128& a, const __m128& b) const
    {
        return _mm_and_ps(_mm_sub_ps(a,b), *(const __m128*)v32f_absmask);
    }
};

struct _VAnd8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_and_si128(a,b); }};
struct _VOr8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_or_si128(a,b); }};
struct _VXor8u { __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_xor_si128(a,b); }};

typedef VBinOp8<_VAdd8u> VAdd8u;
typedef VBinOp8<_VSub8u> VSub8u;
typedef VBinOp8<_VMin8u> VMin8u;
typedef VBinOp8<_VMax8u> VMax8u;
typedef VBinOp8<_VAbsDiff8u> VAbsDiff8u;
typedef VBinOp8<_VCmpEQ8u> VCmpEQ8u;
typedef VBinOp8<_VCmpGT8u> VCmpGT8u;

typedef VBinOp16<ushort, _VAdd16u> VAdd16u;
typedef VBinOp16<ushort, _VSub16u> VSub16u;
typedef VBinOp16<ushort, _VMin16u> VMin16u;
typedef VBinOp16<ushort, _VMax16u> VMax16u;
typedef VBinOp16<ushort, _VAbsDiff16u> VAbsDiff16u;

typedef VBinOp16<short, _VAdd16s> VAdd16s;
typedef VBinOp16<short, _VSub16s> VSub16s;
typedef VBinOp16<short, _VMin16s> VMin16s;
typedef VBinOp16<short, _VMax16s> VMax16s;
typedef VBinOp16<short, _VAbsDiff16s> VAbsDiff16s;

typedef VBinOp32f<_VAdd32f> VAdd32f;
typedef VBinOp32f<_VSub32f> VSub32f;
typedef VBinOp32f<_VMin32f> VMin32f;
typedef VBinOp32f<_VMax32f> VMax32f;
typedef VBinOp32f<_VAbsDiff32f> VAbsDiff32f;

typedef VBinOp8<_VAnd8u> VAnd8u;
typedef VBinOp8<_VOr8u> VOr8u;
typedef VBinOp8<_VXor8u> VXor8u;

#else

enum { ARITHM_SIMD = CV_CPU_NONE };    
    
typedef NoVec VAdd8u;
typedef NoVec VSub8u;
typedef NoVec VMin8u;
typedef NoVec VMax8u;
typedef NoVec VAbsDiff8u;
typedef NoVec VCmpEQ8u;
typedef NoVec VCmpGT8u;

typedef NoVec VAdd16u;
typedef NoVec VSub16u;
typedef NoVec VMin16u;
typedef NoVec VMax16u;
typedef NoVec VAbsDiff16u;

typedef NoVec VAdd16s;
typedef NoVec VSub16s;
typedef NoVec VMin16s;
typedef NoVec VMax16s;
typedef NoVec VAbsDiff16s;

typedef NoVec VAdd32f;
typedef NoVec VSub32f;
typedef NoVec VMin32f;
typedef NoVec VMax32f;
typedef NoVec VAbsDiff32f;

typedef NoVec VAnd8u;
typedef NoVec VOr8u;
typedef NoVec VXor8u;

#endif

/****************************************************************************************\
*                                   logical operations                                   *
\****************************************************************************************/

template<typename T> struct AndOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a & b; }
};

template<typename T> struct OrOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a | b; }
};

template<typename T> struct XorOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()( T a, T b ) const { return a ^ b; }
};

template<class OPB, class OPI, class OPV> static void
bitwiseOp_( const Mat& srcmat1, const Mat& srcmat2, Mat& dstmat )
{
    OPB opb; OPI opi; OPV opv;
    const uchar* src1 = srcmat1.data;
    const uchar* src2 = srcmat2.data;
    uchar* dst = dstmat.data;
    size_t step1 = srcmat1.step, step2 = srcmat2.step, step = dstmat.step;
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, (int)srcmat1.elemSize() );
    bool useSIMD = checkHardwareSupport(ARITHM_SIMD);

    for( ; size.height--; src1 += step1, src2 += step2, dst += step )
    {
        int i = useSIMD ? opv(src1, src2, dst, size.width) : 0;

        if( (((size_t)src1 | (size_t)src2 | (size_t)dst) & 3) == 0 )
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = opi(((const int*)(src1+i))[0], ((const int*)(src2+i))[0]);
                int t1 = opi(((const int*)(src1+i))[1], ((const int*)(src2+i))[1]);

                ((int*)(dst+i))[0] = t0;
                ((int*)(dst+i))[1] = t1;

                t0 = opi(((const int*)(src1+i))[2], ((const int*)(src2+i))[2]);
                t1 = opi(((const int*)(src1+i))[3], ((const int*)(src2+i))[3]);

                ((int*)(dst+i))[2] = t0;
                ((int*)(dst+i))[3] = t1;
            }

            for( ; i <= size.width - 4; i += 4 )
            {
                int t = opi(*(const int*)(src1+i), *(const int*)(src2+i));
                *(int*)(dst+i) = t;
            }
        }

        for( ; i < size.width; i++ )
            dst[i] = opb(src1[i], src2[i]);
    }
}


template<class OPB, class OPI, class OPV> static void
bitwiseSOp_( const Mat& srcmat, Mat& dstmat, const Scalar& _scalar )
{
    OPB opb; OPI opi; OPV opv;
    const uchar* src0 = srcmat.data;
    uchar* dst0 = dstmat.data;
    size_t step1 = srcmat.step, step = dstmat.step;
    Size size = getContinuousSize( srcmat, dstmat, (int)srcmat.elemSize() );
    const int delta = 96;
    uchar scalar[delta];
    scalarToRawData(_scalar, scalar, srcmat.type(), (int)(delta/srcmat.elemSize1()) );
    bool useSIMD = checkHardwareSupport(ARITHM_SIMD);

    for( ; size.height--; src0 += step1, dst0 += step )
    {
        const uchar* src = (const uchar*)src0;
        uchar* dst = dst0;
        int i, len = size.width;

        if( (((size_t)src|(size_t)dst) & 3) == 0 )
        {
            while( (len -= delta) >= 0 )
            {
                i = useSIMD ? opv(src, scalar, dst, delta) : 0;
                for( ; i < delta; i += 16 )
                {
                    int t0 = opi(((const int*)(src+i))[0], ((const int*)(scalar+i))[0]);
                    int t1 = opi(((const int*)(src+i))[1], ((const int*)(scalar+i))[1]);
                    ((int*)(dst+i))[0] = t0;
                    ((int*)(dst+i))[1] = t1;

                    t0 = opi(((const int*)(src+i))[2], ((const int*)(scalar+i))[2]);
                    t1 = opi(((const int*)(src+i))[3], ((const int*)(scalar+i))[3]);
                    ((int*)(dst+i))[2] = t0;
                    ((int*)(dst+i))[3] = t1;
                }
                src += delta;
                dst += delta;
            }
        }
        else
        {
            while( (len -= delta) >= 0 )
            {
                for( i = 0; i < delta; i += 4 )
                {
                    uchar t0 = opb(src[i], scalar[i]);
                    uchar t1 = opb(src[i+1], scalar[i+1]);
                    dst[i] = t0; dst[i+1] = t1;

                    t0 = opb(src[i+2], scalar[i+2]);
                    t1 = opb(src[i+3], scalar[i+3]);
                    dst[i+2] = t0; dst[i+3] = t1;
                }
                src += delta;
                dst += delta;
            }
        }

        for( len += delta, i = 0; i < len; i++ )
            dst[i] = opb(src[i],scalar[i]);
    }
}

    
static void
binaryOp( const Mat& src1, const Mat& src2, Mat& dst, BinaryFunc func, int dsttype=-1 )
{
    if( dsttype == -1 )
        dsttype = src1.type();
    CV_Assert( src1.type() == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, dsttype);
        const Mat* arrays[] = { &src1, &src2, &dst, 0 };
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func(it.planes[0], it.planes[1], it.planes[2]);
        return;
    }
    
    CV_Assert( src1.size() == src2.size() );
    dst.create( src1.size(), dsttype );
    func( src1, src2, dst );
}

    
static void
binaryMaskOp( const Mat& src1, const Mat& src2, Mat& dst,
              const Mat& mask, BinaryFunc func )
{
    CV_Assert( src1.type() == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = { &src1, &src2, &dst, &mask, 0 };
        Mat planes[4];
        NAryMatIterator it(arrays, planes);
        
        if( !mask.data )
            for( int i = 0; i < it.nplanes; i++, ++it )
                func(it.planes[0], it.planes[1], it.planes[2]);
        else
            for( int i = 0; i < it.nplanes; i++, ++it )
                binaryMaskOp(it.planes[0], it.planes[1],
                             it.planes[2], it.planes[3],
                             func);
        return;
    }
    
    CV_Assert( src1.size() == src2.size() );
    dst.create( src1.size(), src1.type() );

    if( !mask.data )
        func(src1, src2, dst);
    else
    {
        AutoBuffer<uchar> buf;
        size_t esz = dst.elemSize(), buf_step = dst.cols*esz;
        CopyMaskFunc copym_func = getCopyMaskFunc((int)esz);
        int y, dy;

        CV_Assert(mask.type() == CV_8UC1 && mask.size() == dst.size());
        dy = std::min(std::max((int)(CV_MAX_LOCAL_SIZE/buf_step), 1), dst.rows);
        buf.allocate( buf_step*dy );

        for( y = 0; y < dst.rows; y += dy )
        {
            dy = std::min(dy, dst.rows - y);
            Mat dstpart = dst.rowRange(y, y + dy);
            Mat temp(dy, dst.cols, dst.type(), (uchar*)buf );
            func( src1.rowRange(y, y + dy), src2.rowRange(y, y + dy), temp );
            copym_func( temp, dstpart, mask.rowRange(y, y + dy) );
        }
    }
}


static void
binarySMaskOp( const Mat& src1, const Scalar& s, Mat& dst,
               const Mat& mask, BinarySFuncCn func )
{
    CV_Assert( func != 0 );
    
    if( src1.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = { &src1, &dst, &mask, 0 };
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        if( !mask.data )
            for( int i = 0; i < it.nplanes; i++, ++it )
                func(it.planes[0], it.planes[1], s);
        else
            for( int i = 0; i < it.nplanes; i++, ++it )
                binarySMaskOp(it.planes[0], s, it.planes[1],
                              it.planes[2], func);
        return;
    }
    
    dst.create( src1.size(), src1.type() );

    if( !mask.data )
        func(src1, dst, s);
    else
    {
        AutoBuffer<uchar> buf;
        size_t esz = dst.elemSize(), buf_step = dst.cols*esz;
        CopyMaskFunc copym_func = getCopyMaskFunc((int)esz);
        int y, dy;

        CV_Assert(mask.type() == CV_8UC1 && mask.size() == dst.size());
        dy = std::min(std::max((int)(CV_MAX_LOCAL_SIZE/buf_step), 1), dst.rows);
        buf.allocate( buf_step*dy );

        for( y = 0; y < dst.rows; y += dy )
        {
            dy = std::min(dy, dst.rows - y);
            Mat dstpart = dst.rowRange(y, y + dy);
            Mat temp(dy, dst.cols, dst.type(), (uchar*)buf);
            func( src1.rowRange(y, y + dy), temp, s );
            copym_func( temp, dstpart, mask.rowRange(y, y + dy) );
        }
    }
}


void bitwise_and(const Mat& a, const Mat& b, Mat& c, const Mat& mask)
{
    binaryMaskOp(a, b, c, mask, bitwiseOp_<AndOp<uchar>, AndOp<int>, VAnd8u>);
}

void bitwise_or(const Mat& a, const Mat& b, Mat& c, const Mat& mask)
{
    binaryMaskOp(a, b, c, mask, bitwiseOp_<OrOp<uchar>, OrOp<int>, VOr8u>);
}

void bitwise_xor(const Mat& a, const Mat& b, Mat& c, const Mat& mask)
{
    binaryMaskOp(a, b, c, mask, bitwiseOp_<XorOp<uchar>, XorOp<int>, VXor8u>);
}

void bitwise_and(const Mat& a, const Scalar& s, Mat& c, const Mat& mask)
{
    binarySMaskOp(a, s, c, mask,
        bitwiseSOp_<AndOp<uchar>, AndOp<int>, VAnd8u>);
}

void bitwise_or(const Mat& a, const Scalar& s, Mat& c, const Mat& mask)
{
    binarySMaskOp(a, s, c, mask,
        bitwiseSOp_<OrOp<uchar>, OrOp<int>, VOr8u>);
}

void bitwise_xor(const Mat& a, const Scalar& s, Mat& c, const Mat& mask)
{
    binarySMaskOp(a, s, c, mask,
        bitwiseSOp_<XorOp<uchar>, XorOp<int>, VXor8u>);
}


void bitwise_not(const Mat& src, Mat& dst)
{
    if( src.dims > 2 )
    {
        dst.create(src.dims, src.size, src.type());
        const Mat* arrays[] = { &src, &dst, 0 };
        Mat planes[4];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            bitwise_not(it.planes[0], it.planes[1]);
        return;
    }
    
    const uchar* sptr = src.data;
    dst.create( src.size(), src.type() );
    uchar* dptr = dst.data;
    Size size = getContinuousSize( src, dst, (int)src.elemSize() );

    for( ; size.height--; sptr += src.step, dptr += dst.step )
    {
        int i = 0;
        if( (((size_t)sptr | (size_t)dptr) & 3) == 0 )
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = ~((const int*)(sptr+i))[0];
                int t1 = ~((const int*)(sptr+i))[1];

                ((int*)(dptr+i))[0] = t0;
                ((int*)(dptr+i))[1] = t1;

                t0 = ~((const int*)(sptr+i))[2];
                t1 = ~((const int*)(sptr+i))[3];

                ((int*)(dptr+i))[2] = t0;
                ((int*)(dptr+i))[3] = t1;
            }

            for( ; i <= size.width - 4; i += 4 )
                *(int*)(dptr+i) = ~*(const int*)(sptr+i);
        }

        for( ; i < size.width; i++ )
        {
            dptr[i] = (uchar)(~sptr[i]);
        }
    }
}

/****************************************************************************************\
*                                      add/subtract                                      *
\****************************************************************************************/

template<> inline uchar OpAdd<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a + b); }
template<> inline uchar OpSub<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a - b); }

static BinaryFunc addTab[] =
{
    binaryOpC1_<OpAdd<uchar>,VAdd8u>, 0,
    binaryOpC1_<OpAdd<ushort>,VAdd16u>,
    binaryOpC1_<OpAdd<short>,VAdd16s>,
    binaryOpC1_<OpAdd<int>,NoVec>,
    binaryOpC1_<OpAdd<float>,VAdd32f>,
    binaryOpC1_<OpAdd<double>,NoVec>, 0
};

static BinaryFunc subTab[] =
{
    binaryOpC1_<OpSub<uchar>,VSub8u>, 0,
    binaryOpC1_<OpSub<ushort>,VSub16u>,
    binaryOpC1_<OpSub<short>,VSub16s>,
    binaryOpC1_<OpSub<int>,NoVec>,
    binaryOpC1_<OpSub<float>,VSub32f>,
    binaryOpC1_<OpSub<double>,NoVec>, 0
};

void add( const Mat& src1, const Mat& src2, Mat& dst )
{
    int type = src1.type();
    BinaryFunc func = addTab[CV_MAT_DEPTH(type)];
    CV_Assert( type == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &src2, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], it.planes[2] );
        return;
    }
    
    Size size = src1.size();
    CV_Assert( size == src2.size() );
    dst.create( size, type );
    func(src1, src2, dst);
}

void subtract( const Mat& src1, const Mat& src2, Mat& dst )
{
    int type = src1.type();
    BinaryFunc func = subTab[CV_MAT_DEPTH(type)];
    CV_Assert( type == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &src2, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], it.planes[2] );
        return;
    }
    
    Size size = src1.size();
    CV_Assert( size == src2.size() );
    dst.create( size, type );
    func(src1, src2, dst);
}

void subtract(const Mat& a, const Scalar& s, Mat& c, const Mat& mask)
{
    add(a, -s, c, mask);
}

void add(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask)
{
    binaryMaskOp(src1, src2, dst, mask, addTab[src1.depth()] );
}

void subtract(const Mat& src1, const Mat& src2, Mat& dst, const Mat& mask)
{
    binaryMaskOp(src1, src2, dst, mask, subTab[src1.depth()] );
}

void add(const Mat& src1, const Scalar& s, Mat& dst, const Mat& mask)
{
    static BinarySFuncCn addSTab[] =
    {
        binarySOpCn_<OpAdd<uchar, int, uchar> >, 0,
        binarySOpCn_<OpAdd<ushort, int, ushort> >,
        binarySOpCn_<OpAdd<short, int, short> >,
        binarySOpCn_<OpAdd<int> >,
        binarySOpCn_<OpAdd<float> >,
        binarySOpCn_<OpAdd<double> >, 0
    };
    int depth = src1.depth();
    binarySMaskOp(src1, s, dst, mask, addSTab[depth]);
}

void subtract(const Scalar& s, const Mat& src1, Mat& dst, const Mat& mask)
{
    static BinarySFuncCn rsubSTab[] =
    {
        binarySOpCn_<OpRSub<uchar, int, uchar> >, 0,
        binarySOpCn_<OpRSub<ushort, int, ushort> >,
        binarySOpCn_<OpRSub<short, int, short> >,
        binarySOpCn_<OpRSub<int> >,
        binarySOpCn_<OpRSub<float> >,
        binarySOpCn_<OpRSub<double> >, 0
    };
    int depth = src1.depth();
    binarySMaskOp(src1, s, dst, mask, rsubSTab[depth]);
}

/****************************************************************************************\
*                                    multiply/divide                                     *
\****************************************************************************************/

template<typename T, typename WT> static void
mul_( const Mat& srcmat1, const Mat& srcmat2, Mat& dstmat, double _scale )
{
    const T* src1 = (const T*)srcmat1.data;
    const T* src2 = (const T*)srcmat2.data;
    T* dst = (T*)dstmat.data;
    size_t step1 = srcmat1.step/sizeof(src1[0]);
    size_t step2 = srcmat2.step/sizeof(src2[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, dstmat.channels() );

    if( fabs(_scale - 1.) < DBL_EPSILON )
    {
        for( ; size.height--; src1+=step1, src2+=step2, dst+=step )
        {
            int i;
            for( i = 0; i <= size.width - 4; i += 4 )
            {
                T t0 = saturate_cast<T>(src1[i] * src2[i]);
                T t1 = saturate_cast<T>(src1[i+1] * src2[i+1]);
                dst[i] = t0; dst[i+1] = t1;

                t0 = saturate_cast<T>(src1[i+2] * src2[i+2]);
                t1 = saturate_cast<T>(src1[i+3] * src2[i+3]);
                dst[i+2] = t0; dst[i+3] = t1;
            }

            for( ; i < size.width; i++ )
                dst[i] = saturate_cast<T>(src1[i] * src2[i]);
        }
    }
    else
    {
        WT scale = (WT)_scale;
        for( ; size.height--; src1+=step1, src2+=step2, dst+=step )
        {
            int i;
            for( i = 0; i <= size.width - 4; i += 4 )
            {
                T t0 = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
                T t1 = saturate_cast<T>(scale*(WT)src1[i+1]*src2[i+1]);
                dst[i] = t0; dst[i+1] = t1;

                t0 = saturate_cast<T>(scale*(WT)src1[i+2]*src2[i+2]);
                t1 = saturate_cast<T>(scale*(WT)src1[i+3]*src2[i+3]);
                dst[i+2] = t0; dst[i+3] = t1;
            }

            for( ; i < size.width; i++ )
                dst[i] = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
        }
    }
}

typedef void (*MulDivFunc)( const Mat& src1, const Mat& src2,
                            Mat& dst, double scale );

void multiply(const Mat& src1, const Mat& src2, Mat& dst, double scale)
{
    static MulDivFunc tab[] =
    {
        mul_<uchar, float>, 0, mul_<ushort, float>, mul_<short, float>,
        mul_<int, double>, mul_<float, float>, mul_<double, double>, 0
    };

    MulDivFunc func = tab[src1.depth()];
    CV_Assert( src1.type() == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &src2, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], it.planes[2], scale );
        return;
    }
    
    CV_Assert( src1.size() == src2.size() );
    dst.create( src1.size(), src1.type() );
    func( src1, src2, dst, scale );
}


template<typename T> static void
div_( const Mat& srcmat1, const Mat& srcmat2, Mat& dstmat, double scale )
{
    const T* src1 = (const T*)srcmat1.data;
    const T* src2 = (const T*)srcmat2.data;
    T* dst = (T*)dstmat.data;
    size_t step1 = srcmat1.step/sizeof(src1[0]);
    size_t step2 = srcmat2.step/sizeof(src2[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, dstmat.channels() );

    for( ; size.height--; src1+=step1, src2+=step2, dst+=step )
    {
        int i = 0;
        for( ; i <= size.width - 4; i += 4 )
        {
            if( src2[i] != 0 && src2[i+1] != 0 && src2[i+2] != 0 && src2[i+3] != 0 )
            {
                double a = (double)src2[i] * src2[i+1];
                double b = (double)src2[i+2] * src2[i+3];
                double d = scale/(a * b);
                b *= d;
                a *= d;

                T z0 = saturate_cast<T>(src2[i+1] * src1[i] * b);
                T z1 = saturate_cast<T>(src2[i] * src1[i+1] * b);
                T z2 = saturate_cast<T>(src2[i+3] * src1[i+2] * a);
                T z3 = saturate_cast<T>(src2[i+2] * src1[i+3] * a);

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
            else
            {
                T z0 = src2[i] != 0 ? saturate_cast<T>(src1[i]*scale/src2[i]) : 0;
                T z1 = src2[i+1] != 0 ? saturate_cast<T>(src1[i+1]*scale/src2[i+1]) : 0;
                T z2 = src2[i+2] != 0 ? saturate_cast<T>(src1[i+2]*scale/src2[i+2]) : 0;
                T z3 = src2[i+3] != 0 ? saturate_cast<T>(src1[i+3]*scale/src2[i+3]) : 0;

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
        }

        for( ; i < size.width; i++ )
            dst[i] = src2[i] != 0 ? saturate_cast<T>(src1[i]*scale/src2[i]) : 0;
    }
}


void divide(const Mat& src1, const Mat& src2, Mat& dst, double scale)
{
    static MulDivFunc tab[] =
    {
        div_<uchar>, 0, div_<ushort>, div_<short>,
        div_<int>, div_<float>, div_<double>, 0
    };

    MulDivFunc func = tab[src1.depth()];
    CV_Assert( src1.size() == src2.size() && src1.type() == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &src2, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], it.planes[2], scale );
        return;
    }
    
    CV_Assert( src1.size() == src2.size() );
    dst.create( src1.size(), src1.type() );
    func( src1, src2, dst, scale );
}

template<typename T> static void
recip_( double scale, const Mat& srcmat2, Mat& dstmat )
{
    const T* src2 = (const T*)srcmat2.data;
    T* dst = (T*)dstmat.data;
    size_t step2 = srcmat2.step/sizeof(src2[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = getContinuousSize( srcmat2, dstmat, dstmat.channels() );

    for( ; size.height--; src2+=step2, dst+=step )
    {
        int i = 0;
        for( ; i <= size.width - 4; i += 4 )
        {
            if( src2[i] != 0 && src2[i+1] != 0 && src2[i+2] != 0 && src2[i+3] != 0 )
            {
                double a = (double)src2[i] * src2[i+1];
                double b = (double)src2[i+2] * src2[i+3];
                double d = scale/(a * b);
                b *= d;
                a *= d;

                T z0 = saturate_cast<T>(src2[i+1] * b);
                T z1 = saturate_cast<T>(src2[i] * b);
                T z2 = saturate_cast<T>(src2[i+3] * a);
                T z3 = saturate_cast<T>(src2[i+2] * a);

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
            else
            {
                T z0 = src2[i] != 0 ? saturate_cast<T>(scale/src2[i]) : 0;
                T z1 = src2[i+1] != 0 ? saturate_cast<T>(scale/src2[i+1]) : 0;
                T z2 = src2[i+2] != 0 ? saturate_cast<T>(scale/src2[i+2]) : 0;
                T z3 = src2[i+3] != 0 ? saturate_cast<T>(scale/src2[i+3]) : 0;

                dst[i] = z0; dst[i+1] = z1;
                dst[i+2] = z2; dst[i+3] = z3;
            }
        }

        for( ; i < size.width; i++ )
            dst[i] = src2[i] != 0 ? saturate_cast<T>(scale/src2[i]) : 0;
    }
}

typedef void (*RecipFunc)( double scale, const Mat& src, Mat& dst );

void divide(double scale, const Mat& src, Mat& dst)
{
    static RecipFunc tab[] =
    {
        recip_<uchar>, 0, recip_<ushort>, recip_<short>,
        recip_<int>, recip_<float>, recip_<double>, 0
    };

    RecipFunc func = tab[src.depth()];
    CV_Assert( func != 0 );
    
    if( src.dims > 2 )
    {
        dst.create(src.dims, src.size, src.type());
        const Mat* arrays[] = {&src, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( scale, it.planes[0], it.planes[1] );
        return;
    }
    
    dst.create( src.size(), src.type() );
    func( scale, src, dst );
}

/****************************************************************************************\
*                                      addWeighted                                       *
\****************************************************************************************/

template<typename T, typename WT> static void
addWeighted_( const Mat& srcmat1, double _alpha, const Mat& srcmat2,
              double _beta, double _gamma, Mat& dstmat )
{
    const T* src1 = (const T*)srcmat1.data;
    const T* src2 = (const T*)srcmat2.data;
    T* dst = (T*)dstmat.data;
    size_t step1 = srcmat1.step/sizeof(src1[0]);
    size_t step2 = srcmat2.step/sizeof(src2[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, dstmat.channels() );
    WT alpha = (WT)_alpha, beta = (WT)_beta, gamma = (WT)_gamma;

    for( ; size.height--; src1+=step1, src2+=step2, dst+=step )
    {
        int i = 0;
        for( ; i <= size.width - 4; i += 4 )
        {
            T t0 = saturate_cast<T>(src1[i]*alpha + src2[i]*beta + gamma);
            T t1 = saturate_cast<T>(src1[i+1]*alpha + src2[i+1]*beta + gamma);
            dst[i] = t0; dst[i+1] = t1;

            t0 = saturate_cast<T>(src1[i+2]*alpha + src2[i+2]*beta + gamma);
            t1 = saturate_cast<T>(src1[i+3]*alpha + src2[i+3]*beta + gamma);
            dst[i+2] = t0; dst[i+3] = t1;
        }

        for( ; i < size.width; i++ )
            dst[i] = saturate_cast<T>(src1[i]*alpha + src2[i]*beta + gamma);
    }
}


static void
addWeighted8u( const Mat& srcmat1, double alpha,
               const Mat& srcmat2, double beta,
               double gamma, Mat& dstmat )
{
    const int shift = 14;
    if( srcmat1.rows*srcmat1.cols*srcmat1.channels() <= 256 ||
        fabs(alpha) > 256 || fabs(beta) > 256 || fabs(gamma) > 256*256 )
    {
        addWeighted_<uchar, float>(srcmat1, alpha, srcmat2, beta, gamma, dstmat);
        return;
    }
    const uchar* src1 = srcmat1.data;
    const uchar* src2 = srcmat2.data;
    uchar* dst = dstmat.data;
    size_t step1 = srcmat1.step;
    size_t step2 = srcmat2.step;
    size_t step = dstmat.step;
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, dstmat.channels() );

    int tab1[256], tab2[256];
    double t = 0;
    int j, t0, t1, t2, t3;

    alpha *= 1 << shift;
    gamma = gamma*(1 << shift) + (1 << (shift - 1));
    beta *= 1 << shift;

    for( j = 0; j < 256; j++ )
    {
        tab1[j] = cvRound(t);
        tab2[j] = cvRound(gamma);
        t += alpha;
        gamma += beta;
    }

    t0 = (tab1[0] + tab2[0]) >> shift;
    t1 = (tab1[0] + tab2[255]) >> shift;
    t2 = (tab1[255] + tab2[0]) >> shift;
    t3 = (tab1[255] + tab2[255]) >> shift;

    if( (unsigned)(t0+256) < 768 && (unsigned)(t1+256) < 768 &&
        (unsigned)(t2+256) < 768 && (unsigned)(t3+256) < 768 )
    {
        // use faster table-based convertion back to 8u
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int i;

            for( i = 0; i <= size.width - 4; i += 4 )
            {
                t0 = CV_FAST_CAST_8U((tab1[src1[i]] + tab2[src2[i]]) >> shift);
                t1 = CV_FAST_CAST_8U((tab1[src1[i+1]] + tab2[src2[i+1]]) >> shift);

                dst[i] = (uchar)t0;
                dst[i+1] = (uchar)t1;

                t0 = CV_FAST_CAST_8U((tab1[src1[i+2]] + tab2[src2[i+2]]) >> shift);
                t1 = CV_FAST_CAST_8U((tab1[src1[i+3]] + tab2[src2[i+3]]) >> shift);

                dst[i+2] = (uchar)t0;
                dst[i+3] = (uchar)t1;
            }

            for( ; i < size.width; i++ )
            {
                t0 = CV_FAST_CAST_8U((tab1[src1[i]] + tab2[src2[i]]) >> shift);
                dst[i] = (uchar)t0;
            }
        }
    }
    else
    {
        // use universal macro for convertion back to 8u
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
        {
            int i;

            for( i = 0; i <= size.width - 4; i += 4 )
            {
                t0 = (tab1[src1[i]] + tab2[src2[i]]) >> shift;
                t1 = (tab1[src1[i+1]] + tab2[src2[i+1]]) >> shift;

                dst[i] = CV_CAST_8U( t0 );
                dst[i+1] = CV_CAST_8U( t1 );

                t0 = (tab1[src1[i+2]] + tab2[src2[i+2]]) >> shift;
                t1 = (tab1[src1[i+3]] + tab2[src2[i+3]]) >> shift;

                dst[i+2] = CV_CAST_8U( t0 );
                dst[i+3] = CV_CAST_8U( t1 );
            }

            for( ; i < size.width; i++ )
            {
                t0 = (tab1[src1[i]] + tab2[src2[i]]) >> shift;
                dst[i] = CV_CAST_8U( t0 );
            }
        }
    }
}

typedef void (*AddWeightedFunc)( const Mat& src1, double alpha, const Mat& src2,
                                 double beta, double gamma, Mat& dst );

void addWeighted( const Mat& src1, double alpha, const Mat& src2,
                  double beta, double gamma, Mat& dst )
{
    static AddWeightedFunc tab[]=
    {
        addWeighted8u, 0, addWeighted_<ushort, float>, addWeighted_<short, float>,
        addWeighted_<int, double>, addWeighted_<float, float>, addWeighted_<double, double>, 0
    };

    AddWeightedFunc func = tab[src1.depth()];
    CV_Assert( src1.type() == src2.type() && func != 0 );
    
    if( src1.dims > 2 || src2.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &src2, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], alpha, it.planes[1], beta, gamma, it.planes[2] );
        return;
    }
    
    CV_Assert( src1.size() == src2.size() );
    dst.create( src1.size(), src1.type() );
    func( src1, alpha, src2, beta, gamma, dst );
}


/****************************************************************************************\
*                                      absdiff                                           *
\****************************************************************************************/

template<typename T> struct OpAbsDiff
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator()(T a, T b) { return (T)std::abs(a - b); }
};

template<> inline short OpAbsDiff<short>::operator ()(short a, short b)
{ return saturate_cast<short>(std::abs(a - b)); }

template<typename T, typename WT=T> struct OpAbsDiffS
{
    typedef T type1;
    typedef WT type2;
    typedef T rtype;
    T operator()(T a, WT b) { return saturate_cast<T>(std::abs(a - b)); }
};

void absdiff( const Mat& src1, const Mat& src2, Mat& dst )
{
    static BinaryFunc tab[] =
    {
        binaryOpC1_<OpAbsDiff<uchar>,VAbsDiff8u>, 0,
        binaryOpC1_<OpAbsDiff<ushort>,VAbsDiff16u>,
        binaryOpC1_<OpAbsDiff<short>,VAbsDiff16s>,
        binaryOpC1_<OpAbsDiff<int>,NoVec>,
        binaryOpC1_<OpAbsDiff<float>,VAbsDiff32f>,
        binaryOpC1_<OpAbsDiff<double>,NoVec>, 0
    };

    binaryOp(src1, src2, dst, tab[src1.depth()]);
}


void absdiff( const Mat& src1, const Scalar& s, Mat& dst )
{
    static BinarySFuncCn tab[] =
    {
        binarySOpCn_<OpAbsDiffS<uchar, int> >, 0,
        binarySOpCn_<OpAbsDiffS<ushort, int> >,
        binarySOpCn_<OpAbsDiffS<short, int> >,
        binarySOpCn_<OpAbsDiffS<int> >,
        binarySOpCn_<OpAbsDiffS<float> >,
        binarySOpCn_<OpAbsDiffS<double> >, 0
    };

    BinarySFuncCn func = tab[src1.depth()];
    CV_Assert(src1.channels() <= 4 && func != 0);
    
    if( src1.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &dst, 0};
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], s );
        return;
    }

    dst.create(src1.size(), src1.type());
    func( src1, dst, s );
}

/****************************************************************************************\
*                                      inRange[S]                                        *
\****************************************************************************************/

template<typename T, typename WT> struct InRangeC1
{
    typedef T xtype;
    typedef WT btype;
    uchar operator()(xtype x, btype a, btype b) const
    { return (uchar)-(a <= x && x < b); }
};

template<typename T, typename WT> struct InRangeC2
{
    typedef Vec<T,2> xtype;
    typedef Vec<WT,2> btype;
    uchar operator()(const xtype& x, const btype& a, const btype& b) const
    {
        return (uchar)-(a[0] <= x[0] && x[0] < b[0] &&
                        a[1] <= x[1] && x[1] < b[1]);
    }
};

template<typename T, typename WT> struct InRangeC3
{
    typedef Vec<T,3> xtype;
    typedef Vec<WT,3> btype;
    uchar operator()(const xtype& x, const btype& a, const btype& b) const
    {
        return (uchar)-(a[0] <= x[0] && x[0] < b[0] &&
                        a[1] <= x[1] && x[1] < b[1] &&
                        a[2] <= x[2] && x[2] < b[2]);
    }
};

template<typename T, typename WT> struct InRangeC4
{
    typedef Vec<T,4> xtype;
    typedef Vec<WT,4> btype;
    uchar operator()(const xtype& x, const btype& a, const btype& b) const
    {
        return (uchar)-(a[0] <= x[0] && x[0] < b[0] &&
                        a[1] <= x[1] && x[1] < b[1] &&
                        a[2] <= x[2] && x[2] < b[2] &&
                        a[3] <= x[3] && x[3] < b[3]);
    }
};

template<class Op> static void
inRange_( const Mat& srcmat1, const Mat& srcmat2, const Mat& srcmat3, Mat& dstmat )
{
    Op op;
    uchar* dst = dstmat.data;
    size_t dstep = dstmat.step;
    Size size = getContinuousSize( srcmat1, srcmat2, srcmat3, dstmat );

    for( int y = 0; y < size.height; y++, dst += dstep )
    {
        const typename Op::xtype* src1 = (const typename Op::xtype*)(srcmat1.data + srcmat1.step*y);
        const typename Op::xtype* src2 = (const typename Op::xtype*)(srcmat2.data + srcmat2.step*y);
        const typename Op::xtype* src3 = (const typename Op::xtype*)(srcmat3.data + srcmat3.step*y);
        for( int x = 0; x < size.width; x++ )
            dst[x] = op( src1[x], src2[x], src3[x] );
    }
}

template<class Op> static void
inRangeS_( const Mat& srcmat1, const Scalar& _a, const Scalar& _b, Mat& dstmat )
{
    Op op;
    typedef typename Op::btype WT;
    typedef typename DataType<WT>::channel_type WT1;
    WT a, b;
    uchar* dst = dstmat.data;
    size_t dstep = dstmat.step;
    Size size = getContinuousSize( srcmat1, dstmat );
    int cn = srcmat1.channels();
    scalarToRawData(_a, &a, CV_MAKETYPE(DataType<WT>::depth, cn));
    scalarToRawData(_b, &b, CV_MAKETYPE(DataType<WT>::depth, cn));

    for( int y = 0; y < size.height; y++, dst += dstep )
    {
        const typename Op::xtype* src1 = (const typename Op::xtype*)(srcmat1.data + srcmat1.step*y);
        for( int x = 0; x < size.width; x++ )
            dst[x] = op( src1[x], a, b );
    }
}

typedef void (*InRangeFunc)( const Mat& src1, const Mat& src2, const Mat& src3, Mat& dst );
typedef void (*InRangeSFunc)( const Mat& src1, const Scalar& a, const Scalar& b, Mat& dst );

void inRange(const Mat& src, const Mat& lowerb,
             const Mat& upperb, Mat& dst)
{
    static InRangeFunc tab[] =
    {
        inRange_<InRangeC1<uchar, uchar> >, 0,
        inRange_<InRangeC1<ushort, ushort> >,
        inRange_<InRangeC1<short, short> >,
        inRange_<InRangeC1<int, int> >,
        inRange_<InRangeC1<float, float> >,
        inRange_<InRangeC1<double, double> >, 0,

        inRange_<InRangeC2<uchar, uchar> >, 0,
        inRange_<InRangeC2<ushort, ushort> >,
        inRange_<InRangeC2<short, short> >,
        inRange_<InRangeC2<int, int> >,
        inRange_<InRangeC2<float, float> >,
        inRange_<InRangeC2<double, double> >, 0,

        inRange_<InRangeC3<uchar, uchar> >, 0,
        inRange_<InRangeC3<ushort, ushort> >,
        inRange_<InRangeC3<short, short> >,
        inRange_<InRangeC3<int, int> >,
        inRange_<InRangeC3<float, float> >,
        inRange_<InRangeC3<double, double> >, 0,

        inRange_<InRangeC4<uchar, uchar> >, 0,
        inRange_<InRangeC4<ushort, ushort> >,
        inRange_<InRangeC4<short, short> >,
        inRange_<InRangeC4<int, int> >,
        inRange_<InRangeC4<float, float> >,
        inRange_<InRangeC4<double, double> >, 0
    };

    CV_Assert( src.type() == lowerb.type() && src.type() == upperb.type() && src.channels() <= 4 );

    InRangeFunc func = tab[src.type()];
    CV_Assert( func != 0 );

    if( src.dims > 2 || lowerb.dims > 2 || upperb.dims > 2 )
    {
        dst.create(src.dims, src.size, CV_8U);
        const Mat* arrays[] = {&src, &lowerb, &upperb, &dst, 0};
        Mat planes[4];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], it.planes[2], it.planes[3] );
        return;
    }
    
    CV_Assert( src.size() == lowerb.size() && src.size() == upperb.size() );
    dst.create(src.size(), CV_8U);
    func( src, lowerb, upperb, dst );
}

void inRange(const Mat& src, const Scalar& lowerb,
             const Scalar& upperb, Mat& dst)
{
    static InRangeSFunc tab[] =
    {
        inRangeS_<InRangeC1<uchar, int> >, 0,
        inRangeS_<InRangeC1<ushort, int> >,
        inRangeS_<InRangeC1<short, int> >,
        inRangeS_<InRangeC1<int, int> >,
        inRangeS_<InRangeC1<float, float> >,
        inRangeS_<InRangeC1<double, double> >, 0,

        inRangeS_<InRangeC2<uchar, int> >, 0,
        inRangeS_<InRangeC2<ushort, int> >,
        inRangeS_<InRangeC2<short, int> >,
        inRangeS_<InRangeC2<int, int> >,
        inRangeS_<InRangeC2<float, float> >,
        inRangeS_<InRangeC2<double, double> >, 0,

        inRangeS_<InRangeC3<uchar, int> >, 0,
        inRangeS_<InRangeC3<ushort, int> >,
        inRangeS_<InRangeC3<short, int> >,
        inRangeS_<InRangeC3<int, int> >,
        inRangeS_<InRangeC3<float, float> >,
        inRangeS_<InRangeC3<double, double> >, 0,

        inRangeS_<InRangeC4<uchar, int> >, 0,
        inRangeS_<InRangeC4<ushort, int> >,
        inRangeS_<InRangeC4<short, int> >,
        inRangeS_<InRangeC4<int, int> >,
        inRangeS_<InRangeC4<float, float> >,
        inRangeS_<InRangeC4<double, double> >, 0
    };

    CV_Assert( src.channels() <= 4 );

    InRangeSFunc func = tab[src.type()];
    CV_Assert( func != 0 );
    
    if( src.dims > 2 )
    {
        dst.create(src.dims, src.size, CV_8U);
        const Mat* arrays[] = {&src, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], lowerb, upperb, it.planes[1] );
        return;
    }

    dst.create(src.size(), CV_8U);
    func( src, lowerb, upperb, dst );
}

/****************************************************************************************\
*                                          compare                                       *
\****************************************************************************************/

template<typename T, typename WT=T> struct CmpEQ
{
    typedef T type1;
    typedef WT type2;
    typedef uchar rtype;
    uchar operator()(T a, WT b) const { return (uchar)-(a == b); }
};

template<typename T, typename WT=T> struct CmpGT
{
    typedef T type1;
    typedef WT type2;
    typedef uchar rtype;
    uchar operator()(T a, WT b) const { return (uchar)-(a > b); }
};

template<typename T, typename WT=T> struct CmpGE
{
    typedef T type1;
    typedef WT type2;
    typedef uchar rtype;
    uchar operator()(T a, WT b) const { return (uchar)-(a >= b); }
};

void compare( const Mat& src1, const Mat& src2, Mat& dst, int cmpOp )
{
    static BinaryFunc tab[][8] =
    {
        {binaryOpC1_<CmpGT<uchar>,VCmpGT8u>, 0,
        binaryOpC1_<CmpGT<ushort>,NoVec>,
        binaryOpC1_<CmpGT<short>,NoVec>,
        binaryOpC1_<CmpGT<int>,NoVec>,
        binaryOpC1_<CmpGT<float>,NoVec>,
        binaryOpC1_<CmpGT<double>,NoVec>, 0},

        {binaryOpC1_<CmpEQ<uchar>,VCmpEQ8u>, 0,
        binaryOpC1_<CmpEQ<ushort>,NoVec>,
        binaryOpC1_<CmpEQ<ushort>,NoVec>, // same function as for ushort's
        binaryOpC1_<CmpEQ<int>,NoVec>,
        binaryOpC1_<CmpEQ<float>,NoVec>,
        binaryOpC1_<CmpEQ<double>,NoVec>, 0},
    };

    CV_Assert(src1.channels() == 1);

    int depth = src1.depth();
    const Mat *psrc1 = &src1, *psrc2 = &src2;
    bool invflag = false;

    switch( cmpOp )
    {
    case CMP_GT:
    case CMP_EQ:
        break;
    case CMP_GE:
        std::swap( psrc1, psrc2 );
        invflag = true;
        break;
    case CMP_LT:
        std::swap( psrc1, psrc2 );
        break;
    case CMP_LE:
        invflag = true;
        break;
    case CMP_NE:
        cmpOp = CMP_EQ;
        invflag = true;
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown comparison method");
    }

    BinaryFunc func = tab[cmpOp == CMP_EQ][depth];
    binaryOp(*psrc1, *psrc2, dst, func, CV_8U);
    if( invflag )
        bitwise_not(dst, dst);
}


void compare( const Mat& src1, double value, Mat& dst, int cmpOp )
{
    static BinarySFuncC1 tab[][8] =
    {
        {binarySOpC1_<CmpEQ<uchar, int> >, 0,
        binarySOpC1_<CmpEQ<ushort, int> >,
        binarySOpC1_<CmpEQ<short, int> >,
        binarySOpC1_<CmpEQ<int> >,
        binarySOpC1_<CmpEQ<float> >,
        binarySOpC1_<CmpEQ<double> >, 0},

        {binarySOpC1_<CmpGT<uchar, int> >, 0,
        binarySOpC1_<CmpGT<ushort, int> >,
        binarySOpC1_<CmpGT<short, int> >,
        binarySOpC1_<CmpGT<int> >,
        binarySOpC1_<CmpGT<float> >,
        binarySOpC1_<CmpGT<double> >, 0},

        {binarySOpC1_<CmpGE<uchar, int> >, 0,
        binarySOpC1_<CmpGE<ushort, int> >,
        binarySOpC1_<CmpGE<short, int> >,
        binarySOpC1_<CmpGE<int> >,
        binarySOpC1_<CmpGE<float> >,
        binarySOpC1_<CmpGE<double> >, 0},
    };

    int depth = src1.depth();
    bool invflag = false;

    switch( cmpOp )
    {
    case CMP_GT:
    case CMP_EQ:
    case CMP_GE:
        break;
    case CMP_LT:
        invflag = true;
        cmpOp = CMP_GE;
        break;
    case CMP_LE:
        invflag = true;
        cmpOp = CMP_GT;
        break;
    case CMP_NE:
        invflag = true;
        cmpOp = CMP_EQ;
        break;
    default:
        CV_Error(CV_StsBadArg, "Unknown comparison method");
    }

    BinarySFuncC1 func = tab[cmpOp == CMP_EQ ? 0 : cmpOp == CMP_GT ? 1 : 2][depth];
    CV_Assert( func != 0 );
    
    if( src1.dims > 2 )
    {
        dst.create(src1.dims, src1.size, CV_8UC(src1.channels()));
        const Mat* arrays[] = {&src1, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
        {
            func( it.planes[0], it.planes[1], value );
            if( invflag )
                bitwise_not(it.planes[2], it.planes[2]);
        }
        return;
    }
    
    dst.create(src1.rows, src1.cols, CV_8UC(src1.channels()));
    func( src1, dst, value );
    if( invflag )
        bitwise_not(dst, dst);
}

/****************************************************************************************\
*                                       min/max                                          *
\****************************************************************************************/

template<typename T> struct MinOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::min(a, b); }
};

template<typename T> struct MaxOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::max(a, b); }
};

template<> inline uchar MinOp<uchar>::operator ()(uchar a, uchar b) const { return CV_MIN_8U(a, b); }
template<> inline uchar MaxOp<uchar>::operator ()(uchar a, uchar b) const { return CV_MAX_8U(a, b); }

void min( const Mat& src1, const Mat& src2, Mat& dst )
{
    static BinaryFunc tab[] =
    {
        binaryOpC1_<MinOp<uchar>,VMin8u>, 0, binaryOpC1_<MinOp<ushort>,VMin16u>,
        binaryOpC1_<MinOp<short>,VMin16s>, binaryOpC1_<MinOp<int>,NoVec>,
        binaryOpC1_<MinOp<float>,VMin32f>, binaryOpC1_<MinOp<double>,NoVec>, 0
    };

    binaryOp(src1, src2, dst, tab[src1.depth()]);
}

void max( const Mat& src1, const Mat& src2, Mat& dst )
{
    static BinaryFunc tab[] =
    {
        binaryOpC1_<MaxOp<uchar>,VMax8u>, 0, binaryOpC1_<MaxOp<ushort>,VMax16u>,
        binaryOpC1_<MaxOp<short>,VMax16s>, binaryOpC1_<MaxOp<int>,NoVec>,
        binaryOpC1_<MaxOp<float>,VMax32f>, binaryOpC1_<MaxOp<double>,NoVec>, 0
    };

    binaryOp(src1, src2, dst, tab[src1.depth()]);
}

void min( const Mat& src1, double value, Mat& dst )
{
    static BinarySFuncC1 tab[] =
    {
        binarySOpC1_<MinOp<uchar> >, 0,
        binarySOpC1_<MinOp<ushort> >,
        binarySOpC1_<MinOp<short> >,
        binarySOpC1_<MinOp<int> >,
        binarySOpC1_<MinOp<float> >,
        binarySOpC1_<MinOp<double> >, 0
    };

    BinarySFuncC1 func = tab[src1.depth()];
    CV_Assert(func != 0);
    
    if( src1.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], value );
        return;
    }
    
    dst.create(src1.size(), src1.type());
    return func( src1, dst, value );
}

void max( const Mat& src1, double value, Mat& dst )
{
    static BinarySFuncC1 tab[] =
    {
        binarySOpC1_<MaxOp<uchar> >, 0,
        binarySOpC1_<MaxOp<ushort> >,
        binarySOpC1_<MaxOp<short> >,
        binarySOpC1_<MaxOp<int> >,
        binarySOpC1_<MaxOp<float> >,
        binarySOpC1_<MaxOp<double> >, 0
    };

    BinarySFuncC1 func = tab[src1.depth()];
    CV_Assert(func != 0);
    
    if( src1.dims > 2 )
    {
        dst.create(src1.dims, src1.size, src1.type());
        const Mat* arrays[] = {&src1, &dst, 0};
        Mat planes[2];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            func( it.planes[0], it.planes[1], value );
        return;
    }
    
    dst.create(src1.size(), src1.type());
    return func( src1, dst, value );
}

}

/****************************************************************************************\
*                                Earlier API: cvAdd etc.                                 *
\****************************************************************************************/

CV_IMPL void
cvNot( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    cv::bitwise_not( src, dst );
}


CV_IMPL void
cvAnd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src1, src2, dst, mask );
}

CV_IMPL void
cvOr( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src1, src2, dst, mask );
}


CV_IMPL void
cvXor( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src1, src2, dst, mask );
}


CV_IMPL void
cvAndS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_and( src, s, dst, mask );
}


CV_IMPL void
cvOrS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_or( src, s, dst, mask );
}


CV_IMPL void
cvXorS( const CvArr* srcarr, CvScalar s, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src.size == dst.size && src.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::bitwise_xor( src, s, dst, mask );
}

CV_IMPL void cvAdd( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, src2, dst, mask );
}

CV_IMPL void cvSub( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( src1, src2, dst, mask );
}

CV_IMPL void cvAddS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::add( src1, value, dst, mask );
}

CV_IMPL void cvSubRS( const CvArr* srcarr1, CvScalar value, CvArr* dstarr, const CvArr* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::subtract( value, src1, dst, mask );
}

CV_IMPL void cvMul( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    cv::multiply( src1, src2, dst, scale );
}

CV_IMPL void cvDiv( const CvArr* srcarr1, const CvArr* srcarr2,
                    CvArr* dstarr, double scale )
{
    cv::Mat src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr), mask;
    CV_Assert( src2.size == dst.size && src2.type() == dst.type() );

    if( srcarr1 )
        cv::divide( cv::cvarrToMat(srcarr1), src2, dst, scale );
    else
        cv::divide( scale, src2, dst );
}


CV_IMPL void
cvAddWeighted( const CvArr* srcarr1, double alpha,
               const CvArr* srcarr2, double beta,
               double gamma, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), src2 = cv::cvarrToMat(srcarr2),
        dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    cv::addWeighted( src1, alpha, src2, beta, gamma, dst );
}


CV_IMPL  void
cvAbsDiff( const CvArr* srcarr1, const CvArr* srcarr2, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvAbsDiffS( const CvArr* srcarr1, CvArr* dstarr, CvScalar scalar )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::absdiff( src1, scalar, dst );
}

CV_IMPL void
cvInRange( const void* srcarr1, const void* srcarr2,
           const void* srcarr3, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, cv::cvarrToMat(srcarr2), cv::cvarrToMat(srcarr3), dst );
}

CV_IMPL void
cvInRangeS( const void* srcarr1, CvScalar lowerb, CvScalar upperb, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::inRange( src1, lowerb, upperb, dst );
}


CV_IMPL void
cvCmp( const void* srcarr1, const void* srcarr2, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, cv::cvarrToMat(srcarr2), dst, cmp_op );
}


CV_IMPL void
cvCmpS( const void* srcarr1, double value, void* dstarr, int cmp_op )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && dst.type() == CV_8U );

    cv::compare( src1, value, dst, cmp_op );
}


CV_IMPL void
cvMin( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvMax( const void* srcarr1, const void* srcarr2, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, cv::cvarrToMat(srcarr2), dst );
}

CV_IMPL void
cvMinS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::min( src1, value, dst );
}


CV_IMPL void
cvMaxS( const void* srcarr1, double value, void* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );

    cv::max( src1, value, dst );
}


/* End of file. */
