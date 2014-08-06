/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
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

/* ////////////////////////////////////////////////////////////////////
//
//  Mat basic operations: Copy, Set
//
// */

#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

namespace cv
{

template<typename T> static void
copyMask_(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size)
{
    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const T* src = (const T*)_src;
        T* dst = (T*)_dst;
        int x = 0;
         #if CV_ENABLE_UNROLLED
        for( ; x <= size.width - 4; x += 4 )
        {
            if( mask[x] )
                dst[x] = src[x];
            if( mask[x+1] )
                dst[x+1] = src[x+1];
            if( mask[x+2] )
                dst[x+2] = src[x+2];
            if( mask[x+3] )
                dst[x+3] = src[x+3];
        }
        #endif
        for( ; x < size.width; x++ )
            if( mask[x] )
                dst[x] = src[x];
    }
}

template<> void
copyMask_<uchar>(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size)
{
#if defined HAVE_IPP
    if (ippiCopy_8u_C1MR(_src, (int)sstep, _dst, (int)dstep, ippiSize(size), mask, (int)mstep) >= 0)
        return;
    setIppErrorStatus();
#endif

    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const uchar* src = (const uchar*)_src;
        uchar* dst = (uchar*)_dst;
        int x = 0;
        #if CV_SSE4_2
        if(USE_SSE4_2)//
        {
            __m128i zero = _mm_setzero_si128 ();

             for( ; x <= size.width - 16; x += 16 )
             {
                 const __m128i rSrc = _mm_lddqu_si128((const __m128i*)(src+x));
                 __m128i _mask = _mm_lddqu_si128((const __m128i*)(mask+x));
                 __m128i rDst = _mm_lddqu_si128((__m128i*)(dst+x));
                 __m128i _negMask = _mm_cmpeq_epi8(_mask, zero);
                 rDst = _mm_blendv_epi8(rSrc, rDst, _negMask);
                 _mm_storeu_si128((__m128i*)(dst + x), rDst);
             }
        }
        #endif
        for( ; x < size.width; x++ )
            if( mask[x] )
                dst[x] = src[x];
    }
}

template<> void
copyMask_<ushort>(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size)
{
#if defined HAVE_IPP
    if (ippiCopy_16u_C1MR((const Ipp16u *)_src, (int)sstep, (Ipp16u *)_dst, (int)dstep, ippiSize(size), mask, (int)mstep) >= 0)
        return;
    setIppErrorStatus();
#endif

    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const ushort* src = (const ushort*)_src;
        ushort* dst = (ushort*)_dst;
        int x = 0;
        #if CV_SSE4_2
        if(USE_SSE4_2)//
        {
            __m128i zero = _mm_setzero_si128 ();
            for( ; x <= size.width - 8; x += 8 )
            {
                 const __m128i rSrc =_mm_lddqu_si128((const __m128i*)(src+x));
                 __m128i _mask = _mm_loadl_epi64((const __m128i*)(mask+x));
                 _mask = _mm_unpacklo_epi8(_mask, _mask);
                 __m128i rDst = _mm_lddqu_si128((const __m128i*)(dst+x));
                 __m128i _negMask = _mm_cmpeq_epi8(_mask, zero);
                 rDst = _mm_blendv_epi8(rSrc, rDst, _negMask);
                 _mm_storeu_si128((__m128i*)(dst + x), rDst);
             }
        }
        #endif
        for( ; x < size.width; x++ )
            if( mask[x] )
                dst[x] = src[x];
    }
}

static void
copyMaskGeneric(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size, void* _esz)
{
    size_t k, esz = *(size_t*)_esz;
    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const uchar* src = _src;
        uchar* dst = _dst;
        int x = 0;
        for( ; x < size.width; x++, src += esz, dst += esz )
        {
            if( !mask[x] )
                continue;
            for( k = 0; k < esz; k++ )
                dst[k] = src[k];
        }
    }
}


#define DEF_COPY_MASK(suffix, type) \
static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, \
                             uchar* dst, size_t dstep, Size size, void*) \
{ \
    copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size); \
}

#if defined HAVE_IPP
#define DEF_COPY_MASK_F(suffix, type, ippfavor, ipptype) \
static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, \
                             uchar* dst, size_t dstep, Size size, void*) \
{ \
    if (ippiCopy_##ippfavor((const ipptype *)src, (int)sstep, (ipptype *)dst, (int)dstep, ippiSize(size), (const Ipp8u *)mask, (int)mstep) >= 0) \
        return; \
    setIppErrorStatus(); \
    copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size); \
}
#else
#define DEF_COPY_MASK_F(suffix, type, ippfavor, ipptype) \
static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, \
                             uchar* dst, size_t dstep, Size size, void*) \
{ \
    copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size); \
}
#endif


DEF_COPY_MASK(8u, uchar)
DEF_COPY_MASK(16u, ushort)
DEF_COPY_MASK_F(8uC3, Vec3b, 8u_C3MR, Ipp8u)
DEF_COPY_MASK_F(32s, int, 32s_C1MR, Ipp32s)
DEF_COPY_MASK_F(16uC3, Vec3s, 16u_C3MR, Ipp16u)
DEF_COPY_MASK(32sC2, Vec2i)
DEF_COPY_MASK_F(32sC3, Vec3i, 32s_C3MR, Ipp32s)
DEF_COPY_MASK_F(32sC4, Vec4i, 32s_C4MR, Ipp32s)
DEF_COPY_MASK(32sC6, Vec6i)
DEF_COPY_MASK(32sC8, Vec8i)

BinaryFunc copyMaskTab[] =
{
    0,
    copyMask8u,
    copyMask16u,
    copyMask8uC3,
    copyMask32s,
    0,
    copyMask16uC3,
    0,
    copyMask32sC2,
    0, 0, 0,
    copyMask32sC3,
    0, 0, 0,
    copyMask32sC4,
    0, 0, 0, 0, 0, 0, 0,
    copyMask32sC6,
    0, 0, 0, 0, 0, 0, 0,
    copyMask32sC8
};

BinaryFunc getCopyMaskFunc(size_t esz)
{
    return esz <= 32 && copyMaskTab[esz] ? copyMaskTab[esz] : copyMaskGeneric;
}

/* dst = src */
void Mat::copyTo( OutputArray _dst ) const
{
    int dtype = _dst.type();
    if( _dst.fixedType() && dtype != type() )
    {
        CV_Assert( channels() == CV_MAT_CN(dtype) );
        convertTo( _dst, dtype );
        return;
    }

    if( empty() )
    {
        _dst.release();
        return;
    }

    if( _dst.isUMat() )
    {
        _dst.create( dims, size.p, type() );
        UMat dst = _dst.getUMat();

        size_t i, sz[CV_MAX_DIM], dstofs[CV_MAX_DIM], esz = elemSize();
        for( i = 0; i < (size_t)dims; i++ )
            sz[i] = size.p[i];
        sz[dims-1] *= esz;
        dst.ndoffset(dstofs);
        dstofs[dims-1] *= esz;
        dst.u->currAllocator->upload(dst.u, data, dims, sz, dstofs, dst.step.p, step.p);
        return;
    }

    if( dims <= 2 )
    {
        _dst.create( rows, cols, type() );
        Mat dst = _dst.getMat();
        if( data == dst.data )
            return;

        if( rows > 0 && cols > 0 )
        {
            const uchar* sptr = data;
            uchar* dptr = dst.data;

            Size sz = getContinuousSize(*this, dst);
            size_t len = sz.width*elemSize();

#if defined HAVE_IPP
            if (ippiCopy_8u_C1R(sptr, (int)step, dptr, (int)dst.step, ippiSize((int)len, sz.height)) >= 0)
                return;
            setIppErrorStatus();
#endif

            for( ; sz.height--; sptr += step, dptr += dst.step )
                memcpy( dptr, sptr, len );
        }
        return;
    }

    _dst.create( dims, size, type() );
    Mat dst = _dst.getMat();
    if( data == dst.data )
        return;

    if( total() != 0 )
    {
        const Mat* arrays[] = { this, &dst };
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs, 2);
        size_t sz = it.size*elemSize();

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            memcpy(ptrs[1], ptrs[0], sz);
    }
}

void Mat::copyTo( OutputArray _dst, InputArray _mask ) const
{
    Mat mask = _mask.getMat();
    if( !mask.data )
    {
        copyTo(_dst);
        return;
    }

    int cn = channels(), mcn = mask.channels();
    CV_Assert( mask.depth() == CV_8U && (mcn == 1 || mcn == cn) );
    bool colorMask = mcn > 1;

    size_t esz = colorMask ? elemSize1() : elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);

    uchar* data0 = _dst.getMat().data;
    _dst.create( dims, size, type() );
    Mat dst = _dst.getMat();

    if( dst.data != data0 ) // do not leave dst uninitialized
        dst = Scalar(0);

    if( dims <= 2 )
    {
        CV_Assert( size() == mask.size() );
        Size sz = getContinuousSize(*this, dst, mask, mcn);
        copymask(data, step, mask.data, mask.step, dst.data, dst.step, sz, &esz);
        return;
    }

    const Mat* arrays[] = { this, &dst, &mask, 0 };
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    Size sz((int)(it.size*mcn), 1);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        copymask(ptrs[0], 0, ptrs[2], 0, ptrs[1], 0, sz, &esz);
}

Mat& Mat::operator = (const Scalar& s)
{
    const Mat* arrays[] = { this };
    uchar* dptr;
    NAryMatIterator it(arrays, &dptr, 1);
    size_t elsize = it.size*elemSize();
    const int64* is = (const int64*)&s.val[0];

    if( is[0] == 0 && is[1] == 0 && is[2] == 0 && is[3] == 0 )
    {
#if defined HAVE_IPP && !defined HAVE_IPP_ICV_ONLY && 0
        if (dims <= 2 || isContinuous())
        {
            IppiSize roisize = { cols, rows };
            if (isContinuous())
            {
                roisize.width = (int)total();
                roisize.height = 1;

                if (ippsZero_8u(data, static_cast<int>(roisize.width * elemSize())) >= 0)
                    return *this;
                setIppErrorStatus();
            }
            roisize.width *= (int)elemSize();

            if (ippiSet_8u_C1R(0, data, (int)step, roisize) >= 0)
                return *this;
            setIppErrorStatus();
        }
#endif

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            memset( dptr, 0, elsize );
    }
    else
    {
        if( it.nplanes > 0 )
        {
            double scalar[12];
            scalarToRawData(s, scalar, type(), 12);
            size_t blockSize = 12*elemSize1();

            for( size_t j = 0; j < elsize; j += blockSize )
            {
                size_t sz = MIN(blockSize, elsize - j);
                memcpy( dptr + j, scalar, sz );
            }
        }

        for( size_t i = 1; i < it.nplanes; i++ )
        {
            ++it;
            memcpy( dptr, data, elsize );
        }
    }
    return *this;
}


Mat& Mat::setTo(InputArray _value, InputArray _mask)
{
    if( empty() )
        return *this;

    Mat value = _value.getMat(), mask = _mask.getMat();

    CV_Assert( checkScalar(value, type(), _value.kind(), _InputArray::MAT ));
    CV_Assert( mask.empty() || (mask.type() == CV_8U && size == mask.size) );

#if defined HAVE_IPP
    int cn = channels(), depth0 = depth();

    if (!mask.empty() && (dims <= 2 || (isContinuous() && mask.isContinuous())) &&
            (/*depth0 == CV_8U ||*/ depth0 == CV_16U || depth0 == CV_16S || depth0 == CV_32S || depth0 == CV_32F) &&
            (cn == 1 || cn == 3 || cn == 4))
    {
        uchar _buf[32];
        void * buf = _buf;
        convertAndUnrollScalar( value, type(), _buf, 1 );

        IppStatus status = (IppStatus)-1;
        IppiSize roisize = { cols, rows };
        int mstep = (int)mask.step[0], dstep = (int)step[0];

        if (isContinuous() && mask.isContinuous())
        {
            roisize.width = (int)total();
            roisize.height = 1;
        }

        if (cn == 1)
        {
            /*if (depth0 == CV_8U)
                status = ippiSet_8u_C1MR(*(Ipp8u *)buf, (Ipp8u *)data, dstep, roisize, mask.data, mstep);
            else*/ if (depth0 == CV_16U)
                status = ippiSet_16u_C1MR(*(Ipp16u *)buf, (Ipp16u *)data, dstep, roisize, mask.data, mstep);
            else if (depth0 == CV_16S)
                status = ippiSet_16s_C1MR(*(Ipp16s *)buf, (Ipp16s *)data, dstep, roisize, mask.data, mstep);
            else if (depth0 == CV_32S)
                status = ippiSet_32s_C1MR(*(Ipp32s *)buf, (Ipp32s *)data, dstep, roisize, mask.data, mstep);
            else if (depth0 == CV_32F)
                status = ippiSet_32f_C1MR(*(Ipp32f *)buf, (Ipp32f *)data, dstep, roisize, mask.data, mstep);
        }
        else if (cn == 3 || cn == 4)
        {
#define IPP_SET(ippfavor, ippcn) \
    do \
    { \
        typedef Ipp##ippfavor ipptype; \
        ipptype ippvalue[4] = { ((ipptype *)buf)[0], ((ipptype *)buf)[1], ((ipptype *)buf)[2], ((ipptype *)buf)[3] }; \
        status = ippiSet_##ippfavor##_C##ippcn##MR(ippvalue, (ipptype *)data, dstep, roisize, mask.data, mstep); \
    } while ((void)0, 0)

#define IPP_SET_CN(ippcn) \
    do \
    { \
        if (cn == ippcn) \
        { \
            /*if (depth0 == CV_8U) \
                IPP_SET(8u, ippcn); \
            else*/ if (depth0 == CV_16U) \
                IPP_SET(16u, ippcn); \
            else if (depth0 == CV_16S) \
                IPP_SET(16s, ippcn); \
            else if (depth0 == CV_32S) \
                IPP_SET(32s, ippcn); \
            else if (depth0 == CV_32F) \
                IPP_SET(32f, ippcn); \
        } \
    } while ((void)0, 0)

            IPP_SET_CN(3);
            IPP_SET_CN(4);

#undef IPP_SET_CN
#undef IPP_SET
        }

        if (status >= 0)
            return *this;
        setIppErrorStatus();
    }
#endif

    size_t esz = elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);

    const Mat* arrays[] = { this, !mask.empty() ? &mask : 0, 0 };
    uchar* ptrs[2]={0,0};
    NAryMatIterator it(arrays, ptrs);
    int totalsz = (int)it.size, blockSize0 = std::min(totalsz, (int)((BLOCK_SIZE + esz-1)/esz));
    AutoBuffer<uchar> _scbuf(blockSize0*esz + 32);
    uchar* scbuf = alignPtr((uchar*)_scbuf, (int)sizeof(double));
    convertAndUnrollScalar( value, type(), scbuf, blockSize0 );

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < totalsz; j += blockSize0 )
        {
            Size sz(std::min(blockSize0, totalsz - j), 1);
            size_t blockSize = sz.width*esz;
            if( ptrs[1] )
            {
                copymask(scbuf, 0, ptrs[1], 0, ptrs[0], 0, sz, &esz);
                ptrs[1] += sz.width;
            }
            else
                memcpy(ptrs[0], scbuf, blockSize);
            ptrs[0] += blockSize;
        }
    }
    return *this;
}


static void
flipHoriz( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
    int i, j, limit = (int)(((size.width + 1)/2)*esz);
    AutoBuffer<int> _tab(size.width*esz);
    int* tab = _tab;

    for( i = 0; i < size.width; i++ )
        for( size_t k = 0; k < esz; k++ )
            tab[i*esz + k] = (int)((size.width - i - 1)*esz + k);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for( i = 0; i < limit; i++ )
        {
            j = tab[i];
            uchar t0 = src[i], t1 = src[j];
            dst[i] = t1; dst[j] = t0;
        }
    }
}

static void
flipVert( const uchar* src0, size_t sstep, uchar* dst0, size_t dstep, Size size, size_t esz )
{
    const uchar* src1 = src0 + (size.height - 1)*sstep;
    uchar* dst1 = dst0 + (size.height - 1)*dstep;
    size.width *= (int)esz;

    for( int y = 0; y < (size.height + 1)/2; y++, src0 += sstep, src1 -= sstep,
                                                  dst0 += dstep, dst1 -= dstep )
    {
        int i = 0;
        if( ((size_t)src0|(size_t)dst0|(size_t)src1|(size_t)dst1) % sizeof(int) == 0 )
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;

                t0 = ((int*)(src0 + i))[1];
                t1 = ((int*)(src1 + i))[1];

                ((int*)(dst0 + i))[1] = t1;
                ((int*)(dst1 + i))[1] = t0;

                t0 = ((int*)(src0 + i))[2];
                t1 = ((int*)(src1 + i))[2];

                ((int*)(dst0 + i))[2] = t1;
                ((int*)(dst1 + i))[2] = t0;

                t0 = ((int*)(src0 + i))[3];
                t1 = ((int*)(src1 + i))[3];

                ((int*)(dst0 + i))[3] = t1;
                ((int*)(dst1 + i))[3] = t0;
            }

            for( ; i <= size.width - 4; i += 4 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;
            }
        }

        for( ; i < size.width; i++ )
        {
            uchar t0 = src0[i];
            uchar t1 = src1[i];

            dst0[i] = t1;
            dst1[i] = t0;
        }
    }
}

#ifdef HAVE_OPENCL

enum { FLIP_COLS = 1 << 0, FLIP_ROWS = 1 << 1, FLIP_BOTH = FLIP_ROWS | FLIP_COLS };

static bool ocl_flip(InputArray _src, OutputArray _dst, int flipCode )
{
    CV_Assert(flipCode >= -1 && flipCode <= 1);
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            flipType, kercn = std::min(ocl::predictOptimalVectorWidth(_src, _dst), 4);

    if (cn > 4)
        return false;

    const char * kernelName;
    if (flipCode == 0)
        kernelName = "arithm_flip_rows", flipType = FLIP_ROWS;
    else if (flipCode > 0)
        kernelName = "arithm_flip_cols", flipType = FLIP_COLS;
    else
        kernelName = "arithm_flip_rows_cols", flipType = FLIP_BOTH;

    ocl::Device dev = ocl::Device::getDefault();
    int pxPerWIy = (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU)) ? 4 : 1;
    kercn = (cn!=3 || flipType == FLIP_ROWS) ? std::max(kercn, cn) : cn;

    ocl::Kernel k(kernelName, ocl::core::flip_oclsrc,
        format( "-D T=%s -D T1=%s -D cn=%d -D PIX_PER_WI_Y=%d -D kercn=%d",
                ocl::memopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                ocl::memopTypeToStr(depth), cn, pxPerWIy, kercn));
    if (k.empty())
        return false;

    Size size = _src.size();
    _dst.create(size, type);
    UMat src = _src.getUMat(), dst = _dst.getUMat();

    int cols = size.width * cn / kercn, rows = size.height;
    cols = flipType == FLIP_COLS ? (cols + 1) >> 1 : cols;
    rows = flipType & FLIP_ROWS ? (rows + 1) >> 1 : rows;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src),
           ocl::KernelArg::WriteOnly(dst, cn, kercn), rows, cols);

    size_t maxWorkGroupSize = dev.maxWorkGroupSize();
    CV_Assert(maxWorkGroupSize % 4 == 0);

    size_t globalsize[2] = { cols, (rows + pxPerWIy - 1) / pxPerWIy },
            localsize[2] = { maxWorkGroupSize / 4, 4 };
    return k.run(2, globalsize, (flipType == FLIP_COLS) && !dev.isIntel() ? localsize : NULL, false);
}

#endif

void flip( InputArray _src, OutputArray _dst, int flip_mode )
{
    CV_Assert( _src.dims() <= 2 );
    Size size = _src.size();

    if (flip_mode < 0)
    {
        if (size.width == 1)
            flip_mode = 0;
        if (size.height == 1)
            flip_mode = 1;
    }

    if ((size.width == 1 && flip_mode > 0) ||
        (size.height == 1 && flip_mode == 0) ||
        (size.height == 1 && size.width == 1 && flip_mode < 0))
    {
        return _src.copyTo(_dst);
    }

    CV_OCL_RUN( _dst.isUMat(), ocl_flip(_src, _dst, flip_mode))

    Mat src = _src.getMat();
    int type = src.type();
    _dst.create( size, type );
    Mat dst = _dst.getMat();
    size_t esz = CV_ELEM_SIZE(type);

#if defined HAVE_IPP
    typedef IppStatus (CV_STDCALL * ippiMirror)(const void * pSrc, int srcStep, void * pDst, int dstStep, IppiSize roiSize, IppiAxis flip);
    typedef IppStatus (CV_STDCALL * ippiMirrorI)(const void * pSrcDst, int srcDstStep, IppiSize roiSize, IppiAxis flip);
    ippiMirror ippFunc = 0;
    ippiMirrorI ippFuncI = 0;

    if (src.data == dst.data)
    {
        CV_SUPPRESS_DEPRECATED_START
        ippFuncI =
            type == CV_8UC1 ? (ippiMirrorI)ippiMirror_8u_C1IR :
            type == CV_8UC3 ? (ippiMirrorI)ippiMirror_8u_C3IR :
            type == CV_8UC4 ? (ippiMirrorI)ippiMirror_8u_C4IR :
            type == CV_16UC1 ? (ippiMirrorI)ippiMirror_16u_C1IR :
            type == CV_16UC3 ? (ippiMirrorI)ippiMirror_16u_C3IR :
            type == CV_16UC4 ? (ippiMirrorI)ippiMirror_16u_C4IR :
            type == CV_16SC1 ? (ippiMirrorI)ippiMirror_16s_C1IR :
            type == CV_16SC3 ? (ippiMirrorI)ippiMirror_16s_C3IR :
            type == CV_16SC4 ? (ippiMirrorI)ippiMirror_16s_C4IR :
            type == CV_32SC1 ? (ippiMirrorI)ippiMirror_32s_C1IR :
            type == CV_32SC3 ? (ippiMirrorI)ippiMirror_32s_C3IR :
            type == CV_32SC4 ? (ippiMirrorI)ippiMirror_32s_C4IR :
            type == CV_32FC1 ? (ippiMirrorI)ippiMirror_32f_C1IR :
            type == CV_32FC3 ? (ippiMirrorI)ippiMirror_32f_C3IR :
            type == CV_32FC4 ? (ippiMirrorI)ippiMirror_32f_C4IR : 0;
        CV_SUPPRESS_DEPRECATED_END
    }
    else
    {
        ippFunc =
            type == CV_8UC1 ? (ippiMirror)ippiMirror_8u_C1R :
            type == CV_8UC3 ? (ippiMirror)ippiMirror_8u_C3R :
            type == CV_8UC4 ? (ippiMirror)ippiMirror_8u_C4R :
            type == CV_16UC1 ? (ippiMirror)ippiMirror_16u_C1R :
            type == CV_16UC3 ? (ippiMirror)ippiMirror_16u_C3R :
            type == CV_16UC4 ? (ippiMirror)ippiMirror_16u_C4R :
            type == CV_16SC1 ? (ippiMirror)ippiMirror_16s_C1R :
            type == CV_16SC3 ? (ippiMirror)ippiMirror_16s_C3R :
            type == CV_16SC4 ? (ippiMirror)ippiMirror_16s_C4R :
            type == CV_32SC1 ? (ippiMirror)ippiMirror_32s_C1R :
            type == CV_32SC3 ? (ippiMirror)ippiMirror_32s_C3R :
            type == CV_32SC4 ? (ippiMirror)ippiMirror_32s_C4R :
            type == CV_32FC1 ? (ippiMirror)ippiMirror_32f_C1R :
            type == CV_32FC3 ? (ippiMirror)ippiMirror_32f_C3R :
            type == CV_32FC4 ? (ippiMirror)ippiMirror_32f_C4R : 0;
    }
    IppiAxis axis = flip_mode == 0 ? ippAxsHorizontal :
        flip_mode > 0 ? ippAxsVertical : ippAxsBoth;
    IppiSize roisize = { dst.cols, dst.rows };

    if (ippFunc != 0)
    {
        if (ippFunc(src.data, (int)src.step, dst.data, (int)dst.step, ippiSize(src.cols, src.rows), axis) >= 0)
            return;
        setIppErrorStatus();
    }
    else if (ippFuncI != 0)
    {
        if (ippFuncI(dst.data, (int)dst.step, roisize, axis) >= 0)
            return;
        setIppErrorStatus();
    }
#endif

    if( flip_mode <= 0 )
        flipVert( src.data, src.step, dst.data, dst.step, src.size(), esz );
    else
        flipHoriz( src.data, src.step, dst.data, dst.step, src.size(), esz );

    if( flip_mode < 0 )
        flipHoriz( dst.data, dst.step, dst.data, dst.step, dst.size(), esz );
}

/*#ifdef HAVE_OPENCL

static bool ocl_repeat(InputArray _src, int ny, int nx, OutputArray _dst)
{
    if (ny == 1 && nx == 1)
    {
        _src.copyTo(_dst);
        return true;
    }

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1,
            kercn = std::min(ocl::predictOptimalVectorWidth(_src, _dst), 4);

    ocl::Kernel k("repeat", ocl::core::repeat_oclsrc,
                  format("-D T=%s -D nx=%d -D ny=%d -D rowsPerWI=%d -D cn=%d",
                         ocl::memopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                         nx, ny, rowsPerWI, kercn));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), dst = _dst.getUMat();
    k.args(ocl::KernelArg::ReadOnly(src, cn, kercn), ocl::KernelArg::WriteOnlyNoSize(dst));

    size_t globalsize[] = { src.cols * cn / kercn, (src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif*/

void repeat(InputArray _src, int ny, int nx, OutputArray _dst)
{
    CV_Assert( _src.dims() <= 2 );
    CV_Assert( ny > 0 && nx > 0 );

    Size ssize = _src.size();
    _dst.create(ssize.height*ny, ssize.width*nx, _src.type());

    /*CV_OCL_RUN(_dst.isUMat(),
               ocl_repeat(_src, ny, nx, _dst))*/

    Mat src = _src.getMat(), dst = _dst.getMat();
    Size dsize = dst.size();
    int esz = (int)src.elemSize();
    int x, y;
    ssize.width *= esz; dsize.width *= esz;

    for( y = 0; y < ssize.height; y++ )
    {
        for( x = 0; x < dsize.width; x += ssize.width )
            memcpy( dst.data + y*dst.step + x, src.data + y*src.step, ssize.width );
    }

    for( ; y < dsize.height; y++ )
        memcpy( dst.data + y*dst.step, dst.data + (y - ssize.height)*dst.step, dsize.width );
}

Mat repeat(const Mat& src, int ny, int nx)
{
    if( nx == 1 && ny == 1 )
        return src;
    Mat dst;
    repeat(src, ny, nx, dst);
    return dst;
}


} // cv


/*
 Various border types, image boundaries are denoted with '|'

 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
 * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg
 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
 */
int cv::borderInterpolate( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        CV_Assert(len > 0);
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == BORDER_CONSTANT )
        p = -1;
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported border type" );
    return p;
}

namespace
{

void copyMakeBorder_8u( const uchar* src, size_t srcstep, cv::Size srcroi,
                        uchar* dst, size_t dststep, cv::Size dstroi,
                        int top, int left, int cn, int borderType )
{
    const int isz = (int)sizeof(int);
    int i, j, k, elemSize = 1;
    bool intMode = false;

    if( (cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0 )
    {
        cn /= isz;
        elemSize = isz;
        intMode = true;
    }

    cv::AutoBuffer<int> _tab((dstroi.width - srcroi.width)*cn);
    int* tab = _tab;
    int right = dstroi.width - srcroi.width - left;
    int bottom = dstroi.height - srcroi.height - top;

    for( i = 0; i < left; i++ )
    {
        j = cv::borderInterpolate(i - left, srcroi.width, borderType)*cn;
        for( k = 0; k < cn; k++ )
            tab[i*cn + k] = j + k;
    }

    for( i = 0; i < right; i++ )
    {
        j = cv::borderInterpolate(srcroi.width + i, srcroi.width, borderType)*cn;
        for( k = 0; k < cn; k++ )
            tab[(i+left)*cn + k] = j + k;
    }

    srcroi.width *= cn;
    dstroi.width *= cn;
    left *= cn;
    right *= cn;

    uchar* dstInner = dst + dststep*top + left*elemSize;

    for( i = 0; i < srcroi.height; i++, dstInner += dststep, src += srcstep )
    {
        if( dstInner != src )
            memcpy(dstInner, src, srcroi.width*elemSize);

        if( intMode )
        {
            const int* isrc = (int*)src;
            int* idstInner = (int*)dstInner;
            for( j = 0; j < left; j++ )
                idstInner[j - left] = isrc[tab[j]];
            for( j = 0; j < right; j++ )
                idstInner[j + srcroi.width] = isrc[tab[j + left]];
        }
        else
        {
            for( j = 0; j < left; j++ )
                dstInner[j - left] = src[tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + srcroi.width] = src[tab[j + left]];
        }
    }

    dstroi.width *= elemSize;
    dst += dststep*top;

    for( i = 0; i < top; i++ )
    {
        j = cv::borderInterpolate(i - top, srcroi.height, borderType);
        memcpy(dst + (i - top)*dststep, dst + j*dststep, dstroi.width);
    }

    for( i = 0; i < bottom; i++ )
    {
        j = cv::borderInterpolate(i + srcroi.height, srcroi.height, borderType);
        memcpy(dst + (i + srcroi.height)*dststep, dst + j*dststep, dstroi.width);
    }
}


void copyMakeConstBorder_8u( const uchar* src, size_t srcstep, cv::Size srcroi,
                             uchar* dst, size_t dststep, cv::Size dstroi,
                             int top, int left, int cn, const uchar* value )
{
    int i, j;
    cv::AutoBuffer<uchar> _constBuf(dstroi.width*cn);
    uchar* constBuf = _constBuf;
    int right = dstroi.width - srcroi.width - left;
    int bottom = dstroi.height - srcroi.height - top;

    for( i = 0; i < dstroi.width; i++ )
    {
        for( j = 0; j < cn; j++ )
            constBuf[i*cn + j] = value[j];
    }

    srcroi.width *= cn;
    dstroi.width *= cn;
    left *= cn;
    right *= cn;

    uchar* dstInner = dst + dststep*top + left;

    for( i = 0; i < srcroi.height; i++, dstInner += dststep, src += srcstep )
    {
        if( dstInner != src )
            memcpy( dstInner, src, srcroi.width );
        memcpy( dstInner - left, constBuf, left );
        memcpy( dstInner + srcroi.width, constBuf, right );
    }

    dst += dststep*top;

    for( i = 0; i < top; i++ )
        memcpy(dst + (i - top)*dststep, constBuf, dstroi.width);

    for( i = 0; i < bottom; i++ )
        memcpy(dst + (i + srcroi.height)*dststep, constBuf, dstroi.width);
}

}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_copyMakeBorder( InputArray _src, OutputArray _dst, int top, int bottom,
                                int left, int right, int borderType, const Scalar& value )
{
    int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    borderType &= ~cv::BORDER_ISOLATED;

    if ( !(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE || borderType == BORDER_REFLECT ||
           borderType == BORDER_WRAP || borderType == BORDER_REFLECT_101) ||
         cn > 4)
        return false;

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP", "BORDER_REFLECT_101" };
    int scalarcn = cn == 3 ? 4 : cn;
    int sctype = CV_MAKETYPE(depth, scalarcn);
    String buildOptions = format("-D T=%s -D %s -D T1=%s -D cn=%d -D ST=%s -D rowsPerWI=%d",
                                 ocl::memopTypeToStr(type), borderMap[borderType],
                                 ocl::memopTypeToStr(depth), cn,
                                 ocl::memopTypeToStr(sctype), rowsPerWI);

    ocl::Kernel k("copyMakeBorder", ocl::core::copymakeborder_oclsrc, buildOptions);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    if( src.isSubmatrix() && !isolated )
    {
        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        int dtop = std::min(ofs.y, top);
        int dbottom = std::min(wholeSize.height - src.rows - ofs.y, bottom);
        int dleft = std::min(ofs.x, left);
        int dright = std::min(wholeSize.width - src.cols - ofs.x, right);
        src.adjustROI(dtop, dbottom, dleft, dright);
        top -= dtop;
        left -= dleft;
        bottom -= dbottom;
        right -= dright;
    }

    _dst.create(src.rows + top + bottom, src.cols + left + right, type);
    UMat dst = _dst.getUMat();

    if (top == 0 && left == 0 && bottom == 0 && right == 0)
    {
        if(src.u != dst.u || src.step != dst.step)
            src.copyTo(dst);
        return true;
    }

    k.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnly(dst),
           top, left, ocl::KernelArg::Constant(Mat(1, 1, sctype, value)));

    size_t globalsize[2] = { dst.cols, (dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::copyMakeBorder( InputArray _src, OutputArray _dst, int top, int bottom,
                         int left, int right, int borderType, const Scalar& value )
{
    CV_Assert( top >= 0 && bottom >= 0 && left >= 0 && right >= 0 );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_copyMakeBorder(_src, _dst, top, bottom, left, right, borderType, value))

    Mat src = _src.getMat();
    int type = src.type();

    if( src.isSubmatrix() && (borderType & BORDER_ISOLATED) == 0 )
    {
        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        int dtop = std::min(ofs.y, top);
        int dbottom = std::min(wholeSize.height - src.rows - ofs.y, bottom);
        int dleft = std::min(ofs.x, left);
        int dright = std::min(wholeSize.width - src.cols - ofs.x, right);
        src.adjustROI(dtop, dbottom, dleft, dright);
        top -= dtop;
        left -= dleft;
        bottom -= dbottom;
        right -= dright;
    }

    _dst.create( src.rows + top + bottom, src.cols + left + right, type );
    Mat dst = _dst.getMat();

    if(top == 0 && left == 0 && bottom == 0 && right == 0)
    {
        if(src.data != dst.data || src.step != dst.step)
            src.copyTo(dst);
        return;
    }

    borderType &= ~BORDER_ISOLATED;

#if defined HAVE_IPP && 0
    typedef IppStatus (CV_STDCALL * ippiCopyMakeBorder)(const void * pSrc, int srcStep, IppiSize srcRoiSize, void * pDst,
                                                        int dstStep, IppiSize dstRoiSize, int topBorderHeight, int leftBorderWidth);
    typedef IppStatus (CV_STDCALL * ippiCopyMakeBorderI)(const void * pSrc, int srcDstStep, IppiSize srcRoiSize, IppiSize dstRoiSize,
                                                         int topBorderHeight, int leftborderwidth);
    typedef IppStatus (CV_STDCALL * ippiCopyConstBorder)(const void * pSrc, int srcStep, IppiSize srcRoiSize, void * pDst, int dstStep,
                                                         IppiSize dstRoiSize, int topBorderHeight, int leftBorderWidth, void * value);

    IppiSize srcRoiSize = { src.cols, src.rows }, dstRoiSize = { dst.cols, dst.rows };
    ippiCopyMakeBorder ippFunc = 0;
    ippiCopyMakeBorderI ippFuncI = 0;
    ippiCopyConstBorder ippFuncConst = 0;
    bool inplace = dst.datastart == src.datastart;

    if (borderType == BORDER_CONSTANT)
    {
         ippFuncConst =
//             type == CV_8UC1 ? (ippiCopyConstBorder)ippiCopyConstBorder_8u_C1R : bug in IPP 8.1
             type == CV_16UC1 ? (ippiCopyConstBorder)ippiCopyConstBorder_16u_C1R :
//             type == CV_16SC1 ? (ippiCopyConstBorder)ippiCopyConstBorder_16s_C1R : bug in IPP 8.1
//             type == CV_32SC1 ? (ippiCopyConstBorder)ippiCopyConstBorder_32s_C1R : bug in IPP 8.1
//             type == CV_32FC1 ? (ippiCopyConstBorder)ippiCopyConstBorder_32f_C1R : bug in IPP 8.1
             type == CV_8UC3 ? (ippiCopyConstBorder)ippiCopyConstBorder_8u_C3R :
             type == CV_16UC3 ? (ippiCopyConstBorder)ippiCopyConstBorder_16u_C3R :
             type == CV_16SC3 ? (ippiCopyConstBorder)ippiCopyConstBorder_16s_C3R :
             type == CV_32SC3 ? (ippiCopyConstBorder)ippiCopyConstBorder_32s_C3R :
             type == CV_32FC3 ? (ippiCopyConstBorder)ippiCopyConstBorder_32f_C3R :
             type == CV_8UC4 ? (ippiCopyConstBorder)ippiCopyConstBorder_8u_C4R :
             type == CV_16UC4 ? (ippiCopyConstBorder)ippiCopyConstBorder_16u_C4R :
             type == CV_16SC4 ? (ippiCopyConstBorder)ippiCopyConstBorder_16s_C4R :
             type == CV_32SC4 ? (ippiCopyConstBorder)ippiCopyConstBorder_32s_C4R :
             type == CV_32FC4 ? (ippiCopyConstBorder)ippiCopyConstBorder_32f_C4R : 0;
    }
    else if (borderType == BORDER_WRAP)
    {
        if (inplace)
        {
            CV_SUPPRESS_DEPRECATED_START
            ippFuncI =
                type == CV_32SC1 ? (ippiCopyMakeBorderI)ippiCopyWrapBorder_32s_C1IR :
                type == CV_32FC1 ? (ippiCopyMakeBorderI)ippiCopyWrapBorder_32s_C1IR : 0;
            CV_SUPPRESS_DEPRECATED_END
        }
        else
        {
            ippFunc =
                type == CV_32SC1 ? (ippiCopyMakeBorder)ippiCopyWrapBorder_32s_C1R :
                type == CV_32FC1 ? (ippiCopyMakeBorder)ippiCopyWrapBorder_32s_C1R : 0;
        }
    }
    else if (borderType == BORDER_REPLICATE)
    {
        if (inplace)
        {
            CV_SUPPRESS_DEPRECATED_START
            ippFuncI =
                type == CV_8UC1 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_8u_C1IR :
                type == CV_16UC1 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16u_C1IR :
                type == CV_16SC1 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16s_C1IR :
                type == CV_32SC1 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32s_C1IR :
                type == CV_32FC1 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32f_C1IR :
                type == CV_8UC3 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_8u_C3IR :
                type == CV_16UC3 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16u_C3IR :
                type == CV_16SC3 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16s_C3IR :
                type == CV_32SC3 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32s_C3IR :
                type == CV_32FC3 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32f_C3IR :
                type == CV_8UC4 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_8u_C4IR :
                type == CV_16UC4 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16u_C4IR :
                type == CV_16SC4 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_16s_C4IR :
                type == CV_32SC4 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32s_C4IR :
                type == CV_32FC4 ? (ippiCopyMakeBorderI)ippiCopyReplicateBorder_32f_C4IR : 0;
            CV_SUPPRESS_DEPRECATED_END
        }
        else
        {
             ippFunc =
                 type == CV_8UC1 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_8u_C1R :
                 type == CV_16UC1 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16u_C1R :
                 type == CV_16SC1 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16s_C1R :
                 type == CV_32SC1 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32s_C1R :
                 type == CV_32FC1 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32f_C1R :
                 type == CV_8UC3 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_8u_C3R :
                 type == CV_16UC3 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16u_C3R :
                 type == CV_16SC3 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16s_C3R :
                 type == CV_32SC3 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32s_C3R :
                 type == CV_32FC3 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32f_C3R :
                 type == CV_8UC4 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_8u_C4R :
                 type == CV_16UC4 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16u_C4R :
                 type == CV_16SC4 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_16s_C4R :
                 type == CV_32SC4 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32s_C4R :
                 type == CV_32FC4 ? (ippiCopyMakeBorder)ippiCopyReplicateBorder_32f_C4R : 0;
        }
    }

    if (ippFunc || ippFuncI || ippFuncConst)
    {
        uchar scbuf[32];
        scalarToRawData(value, scbuf, type);

        if ( (ippFunc && ippFunc(src.data, (int)src.step, srcRoiSize, dst.data, (int)dst.step, dstRoiSize, top, left) >= 0) ||
             (ippFuncI && ippFuncI(src.data, (int)src.step, srcRoiSize, dstRoiSize, top, left) >= 0) ||
             (ippFuncConst && ippFuncConst(src.data, (int)src.step, srcRoiSize, dst.data, (int)dst.step,
                                           dstRoiSize, top, left, scbuf) >= 0))
            return;

        setIppErrorStatus();
    }
#endif

    if( borderType != BORDER_CONSTANT )
        copyMakeBorder_8u( src.data, src.step, src.size(),
                           dst.data, dst.step, dst.size(),
                           top, left, (int)src.elemSize(), borderType );
    else
    {
        int cn = src.channels(), cn1 = cn;
        AutoBuffer<double> buf(cn);
        if( cn > 4 )
        {
            CV_Assert( value[0] == value[1] && value[0] == value[2] && value[0] == value[3] );
            cn1 = 1;
        }
        scalarToRawData(value, buf, CV_MAKETYPE(src.depth(), cn1), cn);
        copyMakeConstBorder_8u( src.data, src.step, src.size(),
                                dst.data, dst.step, dst.size(),
                                top, left, (int)src.elemSize(), (uchar*)(double*)buf );
    }
}

/* dst = src */
CV_IMPL void
cvCopy( const void* srcarr, void* dstarr, const void* maskarr )
{
    if( CV_IS_SPARSE_MAT(srcarr) && CV_IS_SPARSE_MAT(dstarr))
    {
        CV_Assert( maskarr == 0 );
        CvSparseMat* src1 = (CvSparseMat*)srcarr;
        CvSparseMat* dst1 = (CvSparseMat*)dstarr;
        CvSparseMatIterator iterator;
        CvSparseNode* node;

        dst1->dims = src1->dims;
        memcpy( dst1->size, src1->size, src1->dims*sizeof(src1->size[0]));
        dst1->valoffset = src1->valoffset;
        dst1->idxoffset = src1->idxoffset;
        cvClearSet( dst1->heap );

        if( src1->heap->active_count >= dst1->hashsize*CV_SPARSE_HASH_RATIO )
        {
            cvFree( &dst1->hashtable );
            dst1->hashsize = src1->hashsize;
            dst1->hashtable =
                (void**)cvAlloc( dst1->hashsize*sizeof(dst1->hashtable[0]));
        }

        memset( dst1->hashtable, 0, dst1->hashsize*sizeof(dst1->hashtable[0]));

        for( node = cvInitSparseMatIterator( src1, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            CvSparseNode* node_copy = (CvSparseNode*)cvSetNew( dst1->heap );
            int tabidx = node->hashval & (dst1->hashsize - 1);
            memcpy( node_copy, node, dst1->heap->elem_size );
            node_copy->next = (CvSparseNode*)dst1->hashtable[tabidx];
            dst1->hashtable[tabidx] = node_copy;
        }
        return;
    }
    cv::Mat src = cv::cvarrToMat(srcarr, false, true, 1), dst = cv::cvarrToMat(dstarr, false, true, 1);
    CV_Assert( src.depth() == dst.depth() && src.size == dst.size );

    int coi1 = 0, coi2 = 0;
    if( CV_IS_IMAGE(srcarr) )
        coi1 = cvGetImageCOI((const IplImage*)srcarr);
    if( CV_IS_IMAGE(dstarr) )
        coi2 = cvGetImageCOI((const IplImage*)dstarr);

    if( coi1 || coi2 )
    {
        CV_Assert( (coi1 != 0 || src.channels() == 1) &&
            (coi2 != 0 || dst.channels() == 1) );

        int pair[] = { std::max(coi1-1, 0), std::max(coi2-1, 0) };
        cv::mixChannels( &src, 1, &dst, 1, pair, 1 );
        return;
    }
    else
        CV_Assert( src.channels() == dst.channels() );

    if( !maskarr )
        src.copyTo(dst);
    else
        src.copyTo(dst, cv::cvarrToMat(maskarr));
}

CV_IMPL void
cvSet( void* arr, CvScalar value, const void* maskarr )
{
    cv::Mat m = cv::cvarrToMat(arr);
    if( !maskarr )
        m = value;
    else
        m.setTo(cv::Scalar(value), cv::cvarrToMat(maskarr));
}

CV_IMPL void
cvSetZero( CvArr* arr )
{
    if( CV_IS_SPARSE_MAT(arr) )
    {
        CvSparseMat* mat1 = (CvSparseMat*)arr;
        cvClearSet( mat1->heap );
        if( mat1->hashtable )
            memset( mat1->hashtable, 0, mat1->hashsize*sizeof(mat1->hashtable[0]));
        return;
    }
    cv::Mat m = cv::cvarrToMat(arr);
    m = cv::Scalar(0);
}

CV_IMPL void
cvFlip( const CvArr* srcarr, CvArr* dstarr, int flip_mode )
{
    cv::Mat src = cv::cvarrToMat(srcarr);
    cv::Mat dst;

    if (!dstarr)
      dst = src;
    else
      dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.type() == dst.type() && src.size() == dst.size() );
    cv::flip( src, dst, flip_mode );
}

CV_IMPL void
cvRepeat( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() &&
        dst.rows % src.rows == 0 && dst.cols % src.cols == 0 );
    cv::repeat(src, dst.rows/src.rows, dst.cols/src.cols, dst);
}

/* End of file. */
