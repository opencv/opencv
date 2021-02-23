// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

namespace cv {

////////////////////////////////////// transpose /////////////////////////////////////////

template<typename T> static void
transpose_( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz )
{
    int i=0, j, m = sz.width, n = sz.height;

    #if CV_ENABLE_UNROLLED
    for(; i <= m - 4; i += 4 )
    {
        T* d0 = (T*)(dst + dstep*i);
        T* d1 = (T*)(dst + dstep*(i+1));
        T* d2 = (T*)(dst + dstep*(i+2));
        T* d3 = (T*)(dst + dstep*(i+3));

        for( j = 0; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
            d1[j] = s0[1]; d1[j+1] = s1[1]; d1[j+2] = s2[1]; d1[j+3] = s3[1];
            d2[j] = s0[2]; d2[j+1] = s1[2]; d2[j+2] = s2[2]; d2[j+3] = s3[2];
            d3[j] = s0[3]; d3[j+1] = s1[3]; d3[j+2] = s2[3]; d3[j+3] = s3[3];
        }

        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0]; d1[j] = s0[1]; d2[j] = s0[2]; d3[j] = s0[3];
        }
    }
    #endif
    for( ; i < m; i++ )
    {
        T* d0 = (T*)(dst + dstep*i);
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= n - 4; j += 4 )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + sstep*j);
            const T* s1 = (const T*)(src + i*sizeof(T) + sstep*(j+1));
            const T* s2 = (const T*)(src + i*sizeof(T) + sstep*(j+2));
            const T* s3 = (const T*)(src + i*sizeof(T) + sstep*(j+3));

            d0[j] = s0[0]; d0[j+1] = s1[0]; d0[j+2] = s2[0]; d0[j+3] = s3[0];
        }
        #endif
        for( ; j < n; j++ )
        {
            const T* s0 = (const T*)(src + i*sizeof(T) + j*sstep);
            d0[j] = s0[0];
        }
    }
}

template<typename T> static void
transposeI_( uchar* data, size_t step, int n )
{
    for( int i = 0; i < n; i++ )
    {
        T* row = (T*)(data + step*i);
        uchar* data1 = data + i*sizeof(T);
        for( int j = i+1; j < n; j++ )
            std::swap( row[j], *(T*)(data1 + step*j) );
    }
}

typedef void (*TransposeFunc)( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz );
typedef void (*TransposeInplaceFunc)( uchar* data, size_t step, int n );

#define DEF_TRANSPOSE_FUNC(suffix, type) \
static void transpose_##suffix( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz ) \
{ transpose_<type>(src, sstep, dst, dstep, sz); } \
\
static void transposeI_##suffix( uchar* data, size_t step, int n ) \
{ transposeI_<type>(data, step, n); }

DEF_TRANSPOSE_FUNC(8u, uchar)
DEF_TRANSPOSE_FUNC(16u, ushort)
DEF_TRANSPOSE_FUNC(8uC3, Vec3b)
DEF_TRANSPOSE_FUNC(32s, int)
DEF_TRANSPOSE_FUNC(16uC3, Vec3s)
DEF_TRANSPOSE_FUNC(32sC2, Vec2i)
DEF_TRANSPOSE_FUNC(32sC3, Vec3i)
DEF_TRANSPOSE_FUNC(32sC4, Vec4i)
DEF_TRANSPOSE_FUNC(32sC6, Vec6i)
DEF_TRANSPOSE_FUNC(32sC8, Vec8i)

static TransposeFunc transposeTab[] =
{
    0, transpose_8u, transpose_16u, transpose_8uC3, transpose_32s, 0, transpose_16uC3, 0,
    transpose_32sC2, 0, 0, 0, transpose_32sC3, 0, 0, 0, transpose_32sC4,
    0, 0, 0, 0, 0, 0, 0, transpose_32sC6, 0, 0, 0, 0, 0, 0, 0, transpose_32sC8
};

static TransposeInplaceFunc transposeInplaceTab[] =
{
    0, transposeI_8u, transposeI_16u, transposeI_8uC3, transposeI_32s, 0, transposeI_16uC3, 0,
    transposeI_32sC2, 0, 0, 0, transposeI_32sC3, 0, 0, 0, transposeI_32sC4,
    0, 0, 0, 0, 0, 0, 0, transposeI_32sC6, 0, 0, 0, 0, 0, 0, 0, transposeI_32sC8
};

#ifdef HAVE_OPENCL

static bool ocl_transpose( InputArray _src, OutputArray _dst )
{
    const ocl::Device & dev = ocl::Device::getDefault();
    const int TILE_DIM = 32, BLOCK_ROWS = 8;
    int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type),
        rowsPerWI = dev.isIntel() ? 4 : 1;

    UMat src = _src.getUMat();
    _dst.create(src.cols, src.rows, type);
    UMat dst = _dst.getUMat();

    String kernelName("transpose");
    bool inplace = dst.u == src.u;

    if (inplace)
    {
        CV_Assert(dst.cols == dst.rows);
        kernelName += "_inplace";
    }
    else
    {
        // check required local memory size
        size_t required_local_memory = (size_t) TILE_DIM*(TILE_DIM+1)*CV_ELEM_SIZE(type);
        if (required_local_memory > ocl::Device::getDefault().localMemSize())
            return false;
    }

    ocl::Kernel k(kernelName.c_str(), ocl::core::transpose_oclsrc,
                  format("-D T=%s -D T1=%s -D cn=%d -D TILE_DIM=%d -D BLOCK_ROWS=%d -D rowsPerWI=%d%s",
                         ocl::memopTypeToStr(type), ocl::memopTypeToStr(depth),
                         cn, TILE_DIM, BLOCK_ROWS, rowsPerWI, inplace ? " -D INPLACE" : ""));
    if (k.empty())
        return false;

    if (inplace)
        k.args(ocl::KernelArg::ReadWriteNoSize(dst), dst.rows);
    else
        k.args(ocl::KernelArg::ReadOnly(src),
               ocl::KernelArg::WriteOnlyNoSize(dst));

    size_t localsize[2]  = { TILE_DIM, BLOCK_ROWS };
    size_t globalsize[2] = { (size_t)src.cols, inplace ? ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI : (divUp((size_t)src.rows, TILE_DIM) * BLOCK_ROWS) };

    if (inplace && dev.isIntel())
    {
        localsize[0] = 16;
        localsize[1] = dev.maxWorkGroupSize() / localsize[0];
    }

    return k.run(2, globalsize, localsize, false);
}

#endif

#ifdef HAVE_IPP
static bool ipp_transpose( Mat &src, Mat &dst )
{
    CV_INSTRUMENT_REGION_IPP();

    int type = src.type();
    typedef IppStatus (CV_STDCALL * IppiTranspose)(const void * pSrc, int srcStep, void * pDst, int dstStep, IppiSize roiSize);
    typedef IppStatus (CV_STDCALL * IppiTransposeI)(const void * pSrcDst, int srcDstStep, IppiSize roiSize);
    IppiTranspose ippiTranspose = 0;
    IppiTransposeI ippiTranspose_I = 0;

    if (dst.data == src.data && dst.cols == dst.rows)
    {
        CV_SUPPRESS_DEPRECATED_START
        ippiTranspose_I =
            type == CV_8UC1 ? (IppiTransposeI)ippiTranspose_8u_C1IR :
            type == CV_8UC3 ? (IppiTransposeI)ippiTranspose_8u_C3IR :
            type == CV_8UC4 ? (IppiTransposeI)ippiTranspose_8u_C4IR :
            type == CV_16UC1 ? (IppiTransposeI)ippiTranspose_16u_C1IR :
            type == CV_16UC3 ? (IppiTransposeI)ippiTranspose_16u_C3IR :
            type == CV_16UC4 ? (IppiTransposeI)ippiTranspose_16u_C4IR :
            type == CV_16SC1 ? (IppiTransposeI)ippiTranspose_16s_C1IR :
            type == CV_16SC3 ? (IppiTransposeI)ippiTranspose_16s_C3IR :
            type == CV_16SC4 ? (IppiTransposeI)ippiTranspose_16s_C4IR :
            type == CV_32SC1 ? (IppiTransposeI)ippiTranspose_32s_C1IR :
            type == CV_32SC3 ? (IppiTransposeI)ippiTranspose_32s_C3IR :
            type == CV_32SC4 ? (IppiTransposeI)ippiTranspose_32s_C4IR :
            type == CV_32FC1 ? (IppiTransposeI)ippiTranspose_32f_C1IR :
            type == CV_32FC3 ? (IppiTransposeI)ippiTranspose_32f_C3IR :
            type == CV_32FC4 ? (IppiTransposeI)ippiTranspose_32f_C4IR : 0;
        CV_SUPPRESS_DEPRECATED_END
    }
    else
    {
        ippiTranspose =
            type == CV_8UC1 ? (IppiTranspose)ippiTranspose_8u_C1R :
            type == CV_8UC3 ? (IppiTranspose)ippiTranspose_8u_C3R :
            type == CV_8UC4 ? (IppiTranspose)ippiTranspose_8u_C4R :
            type == CV_16UC1 ? (IppiTranspose)ippiTranspose_16u_C1R :
            type == CV_16UC3 ? (IppiTranspose)ippiTranspose_16u_C3R :
            type == CV_16UC4 ? (IppiTranspose)ippiTranspose_16u_C4R :
            type == CV_16SC1 ? (IppiTranspose)ippiTranspose_16s_C1R :
            type == CV_16SC3 ? (IppiTranspose)ippiTranspose_16s_C3R :
            type == CV_16SC4 ? (IppiTranspose)ippiTranspose_16s_C4R :
            type == CV_32SC1 ? (IppiTranspose)ippiTranspose_32s_C1R :
            type == CV_32SC3 ? (IppiTranspose)ippiTranspose_32s_C3R :
            type == CV_32SC4 ? (IppiTranspose)ippiTranspose_32s_C4R :
            type == CV_32FC1 ? (IppiTranspose)ippiTranspose_32f_C1R :
            type == CV_32FC3 ? (IppiTranspose)ippiTranspose_32f_C3R :
            type == CV_32FC4 ? (IppiTranspose)ippiTranspose_32f_C4R : 0;
    }

    IppiSize roiSize = { src.cols, src.rows };
    if (ippiTranspose != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose, src.ptr(), (int)src.step, dst.ptr(), (int)dst.step, roiSize) >= 0)
            return true;
    }
    else if (ippiTranspose_I != 0)
    {
        if (CV_INSTRUMENT_FUN_IPP(ippiTranspose_I, dst.ptr(), (int)dst.step, roiSize) >= 0)
            return true;
    }
    return false;
}
#endif


void transpose( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), esz = CV_ELEM_SIZE(type);
    CV_Assert( _src.dims() <= 2 && esz <= 32 );

    CV_OCL_RUN(_dst.isUMat(),
               ocl_transpose(_src, _dst))

    Mat src = _src.getMat();
    if( src.empty() )
    {
        _dst.release();
        return;
    }

    _dst.create(src.cols, src.rows, src.type());
    Mat dst = _dst.getMat();

    // handle the case of single-column/single-row matrices, stored in STL vectors.
    if( src.rows != dst.cols || src.cols != dst.rows )
    {
        CV_Assert( src.size() == dst.size() && (src.cols == 1 || src.rows == 1) );
        src.copyTo(dst);
        return;
    }

    CV_IPP_RUN_FAST(ipp_transpose(src, dst))

    if( dst.data == src.data )
    {
        TransposeInplaceFunc func = transposeInplaceTab[esz];
        CV_Assert( func != 0 );
        CV_Assert( dst.cols == dst.rows );
        func( dst.ptr(), dst.step, dst.rows );
    }
    else
    {
        TransposeFunc func = transposeTab[esz];
        CV_Assert( func != 0 );
        func( src.ptr(), src.step, dst.ptr(), dst.step, src.size() );
    }
}


#if CV_SIMD128
template<typename V> CV_ALWAYS_INLINE void flipHoriz_single( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
    typedef typename V::lane_type T;
    int end = (int)(size.width*esz);
    int width = (end + 1)/2;
    int width_1 = width & -v_uint8x16::nlanes;
    int i, j;

#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(T)>(src, dst));
#endif

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for( i = 0, j = end; i < width_1; i += v_uint8x16::nlanes, j -= v_uint8x16::nlanes )
        {
            V t0, t1;

            t0 = v_load((T*)((uchar*)src + i));
            t1 = v_load((T*)((uchar*)src + j - v_uint8x16::nlanes));
            t0 = v_reverse(t0);
            t1 = v_reverse(t1);
            v_store((T*)(dst + j - v_uint8x16::nlanes), t0);
            v_store((T*)(dst + i), t1);
        }
        if (isAligned<sizeof(T)>(src, dst))
        {
            for ( ; i < width; i += sizeof(T), j -= sizeof(T) )
            {
                T t0, t1;

                t0 = *((T*)((uchar*)src + i));
                t1 = *((T*)((uchar*)src + j - sizeof(T)));
                *((T*)(dst + j - sizeof(T))) = t0;
                *((T*)(dst + i)) = t1;
            }
        }
        else
        {
            for ( ; i < width; i += sizeof(T), j -= sizeof(T) )
            {
                for (int k = 0; k < (int)sizeof(T); k++)
                {
                    uchar t0, t1;

                    t0 = *((uchar*)src + i + k);
                    t1 = *((uchar*)src + j + k - sizeof(T));
                    *(dst + j + k - sizeof(T)) = t0;
                    *(dst + i + k) = t1;
                }
            }
        }
    }
}

template<typename T1, typename T2> CV_ALWAYS_INLINE void flipHoriz_double( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
    int end = (int)(size.width*esz);
    int width = (end + 1)/2;

#if CV_STRONG_ALIGNMENT
    CV_Assert(isAligned<sizeof(T1)>(src, dst));
    CV_Assert(isAligned<sizeof(T2)>(src, dst));
#endif

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for ( int i = 0, j = end; i < width; i += sizeof(T1) + sizeof(T2), j -= sizeof(T1) + sizeof(T2) )
        {
            T1 t0, t1;
            T2 t2, t3;

            t0 = *((T1*)((uchar*)src + i));
            t2 = *((T2*)((uchar*)src + i + sizeof(T1)));
            t1 = *((T1*)((uchar*)src + j - sizeof(T1) - sizeof(T2)));
            t3 = *((T2*)((uchar*)src + j - sizeof(T2)));
            *((T1*)(dst + j - sizeof(T1) - sizeof(T2))) = t0;
            *((T2*)(dst + j - sizeof(T2))) = t2;
            *((T1*)(dst + i)) = t1;
            *((T2*)(dst + i + sizeof(T1))) = t3;
        }
    }
}
#endif

static void
flipHoriz( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
#if CV_SIMD
#if CV_STRONG_ALIGNMENT
    size_t alignmentMark = ((size_t)src)|((size_t)dst)|sstep|dstep;
#endif
    if (esz == 2 * v_uint8x16::nlanes)
    {
        int end = (int)(size.width*esz);
        int width = end/2;

        for( ; size.height--; src += sstep, dst += dstep )
        {
            for( int i = 0, j = end - 2 * v_uint8x16::nlanes; i < width; i += 2 * v_uint8x16::nlanes, j -= 2 * v_uint8x16::nlanes )
            {
#if CV_SIMD256
                v_uint8x32 t0, t1;

                t0 = v256_load((uchar*)src + i);
                t1 = v256_load((uchar*)src + j);
                v_store(dst + j, t0);
                v_store(dst + i, t1);
#else
                v_uint8x16 t0, t1, t2, t3;

                t0 = v_load((uchar*)src + i);
                t1 = v_load((uchar*)src + i + v_uint8x16::nlanes);
                t2 = v_load((uchar*)src + j);
                t3 = v_load((uchar*)src + j + v_uint8x16::nlanes);
                v_store(dst + j, t0);
                v_store(dst + j + v_uint8x16::nlanes, t1);
                v_store(dst + i, t2);
                v_store(dst + i + v_uint8x16::nlanes, t3);
#endif
            }
        }
    }
    else if (esz == v_uint8x16::nlanes)
    {
        int end = (int)(size.width*esz);
        int width = end/2;

        for( ; size.height--; src += sstep, dst += dstep )
        {
            for( int i = 0, j = end - v_uint8x16::nlanes; i < width; i += v_uint8x16::nlanes, j -= v_uint8x16::nlanes )
            {
                v_uint8x16 t0, t1;

                t0 = v_load((uchar*)src + i);
                t1 = v_load((uchar*)src + j);
                v_store(dst + j, t0);
                v_store(dst + i, t1);
            }
        }
    }
    else if (esz == 8
#if CV_STRONG_ALIGNMENT
            && isAligned<sizeof(uint64)>(alignmentMark)
#endif
    )
    {
        flipHoriz_single<v_uint64x2>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 4
#if CV_STRONG_ALIGNMENT
            && isAligned<sizeof(unsigned)>(alignmentMark)
#endif
    )
    {
        flipHoriz_single<v_uint32x4>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 2
#if CV_STRONG_ALIGNMENT
            && isAligned<sizeof(ushort)>(alignmentMark)
#endif
    )
    {
        flipHoriz_single<v_uint16x8>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 1)
    {
        flipHoriz_single<v_uint8x16>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 24
#if CV_STRONG_ALIGNMENT
            && isAligned<sizeof(uint64_t)>(alignmentMark)
#endif
    )
    {
        int end = (int)(size.width*esz);
        int width = (end + 1)/2;

        for( ; size.height--; src += sstep, dst += dstep )
        {
            for ( int i = 0, j = end; i < width; i += v_uint8x16::nlanes + sizeof(uint64_t), j -= v_uint8x16::nlanes + sizeof(uint64_t) )
            {
                v_uint8x16 t0, t1;
                uint64_t t2, t3;

                t0 = v_load((uchar*)src + i);
                t2 = *((uint64_t*)((uchar*)src + i + v_uint8x16::nlanes));
                t1 = v_load((uchar*)src + j - v_uint8x16::nlanes - sizeof(uint64_t));
                t3 = *((uint64_t*)((uchar*)src + j - sizeof(uint64_t)));
                v_store(dst + j - v_uint8x16::nlanes - sizeof(uint64_t), t0);
                *((uint64_t*)(dst + j - sizeof(uint64_t))) = t2;
                v_store(dst + i, t1);
                *((uint64_t*)(dst + i + v_uint8x16::nlanes)) = t3;
            }
        }
    }
#if !CV_STRONG_ALIGNMENT
    else if (esz == 12)
    {
        flipHoriz_double<uint64_t,uint>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 6)
    {
        flipHoriz_double<uint,ushort>(src, sstep, dst, dstep, size, esz);
    }
    else if (esz == 3)
    {
        flipHoriz_double<ushort,uchar>(src, sstep, dst, dstep, size, esz);
    }
#endif
    else
#endif // CV_SIMD
    {
        int i, j, limit = (int)(((size.width + 1)/2)*esz);
        AutoBuffer<int> _tab(size.width*esz);
        int* tab = _tab.data();

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
#if CV_SIMD
#if CV_STRONG_ALIGNMENT
        if (isAligned<sizeof(int)>(src0, src1, dst0, dst1))
#endif
        {
            for (; i <= size.width - CV_SIMD_WIDTH; i += CV_SIMD_WIDTH)
            {
                v_int32 t0 = vx_load((int*)(src0 + i));
                v_int32 t1 = vx_load((int*)(src1 + i));
                vx_store((int*)(dst0 + i), t1);
                vx_store((int*)(dst1 + i), t0);
            }
        }
#if CV_STRONG_ALIGNMENT
        else
        {
            for (; i <= size.width - CV_SIMD_WIDTH; i += CV_SIMD_WIDTH)
            {
                v_uint8 t0 = vx_load(src0 + i);
                v_uint8 t1 = vx_load(src1 + i);
                vx_store(dst0 + i, t1);
                vx_store(dst1 + i, t0);
            }
        }
#endif
#endif

        if (isAligned<sizeof(int)>(src0, src1, dst0, dst1))
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

    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            flipType, kercn = std::min(ocl::predictOptimalVectorWidth(_src, _dst), 4);

    bool doubleSupport = dev.doubleFPConfig() > 0;
    if (!doubleSupport && depth == CV_64F)
        kercn = cn;

    if (cn > 4)
        return false;

    const char * kernelName;
    if (flipCode == 0)
        kernelName = "arithm_flip_rows", flipType = FLIP_ROWS;
    else if (flipCode > 0)
        kernelName = "arithm_flip_cols", flipType = FLIP_COLS;
    else
        kernelName = "arithm_flip_rows_cols", flipType = FLIP_BOTH;

    int pxPerWIy = (dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU)) ? 4 : 1;
    kercn = (cn!=3 || flipType == FLIP_ROWS) ? std::max(kercn, cn) : cn;

    ocl::Kernel k(kernelName, ocl::core::flip_oclsrc,
        format( "-D T=%s -D T1=%s -D DEPTH=%d -D cn=%d -D PIX_PER_WI_Y=%d -D kercn=%d",
                kercn != cn ? ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)) : ocl::vecopTypeToStr(CV_MAKE_TYPE(depth, kercn)),
                kercn != cn ? ocl::typeToStr(depth) : ocl::vecopTypeToStr(depth), depth, cn, pxPerWIy, kercn));
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

    size_t globalsize[2] = { (size_t)cols, ((size_t)rows + pxPerWIy - 1) / pxPerWIy },
            localsize[2] = { maxWorkGroupSize / 4, 4 };
    return k.run(2, globalsize, (flipType == FLIP_COLS) && !dev.isIntel() ? localsize : NULL, false);
}

#endif

#if defined HAVE_IPP
static bool ipp_flip(Mat &src, Mat &dst, int flip_mode)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    // Details: https://github.com/opencv/opencv/issues/12943
    if (flip_mode <= 0 /* swap rows */
        && cv::ipp::getIppTopFeatures() != ippCPUID_SSE42
        && (int64_t)(src.total()) * src.elemSize() >= CV_BIG_INT(0x80000000)/*2Gb*/
    )
        return false;

    IppiAxis ippMode;
    if(flip_mode < 0)
        ippMode = ippAxsBoth;
    else if(flip_mode == 0)
        ippMode = ippAxsHorizontal;
    else
        ippMode = ippAxsVertical;

    try
    {
        ::ipp::IwiImage iwSrc = ippiGetImage(src);
        ::ipp::IwiImage iwDst = ippiGetImage(dst);

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiMirror, iwSrc, iwDst, ippMode);
    }
    catch(const ::ipp::IwException &)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(flip_mode);
    return false;
#endif
}
#endif


void flip( InputArray _src, OutputArray _dst, int flip_mode )
{
    CV_INSTRUMENT_REGION();

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
        (size.height == 1 && flip_mode == 0))
    {
        return _src.copyTo(_dst);
    }

    CV_OCL_RUN( _dst.isUMat(), ocl_flip(_src, _dst, flip_mode))

    Mat src = _src.getMat();
    int type = src.type();
    _dst.create( size, type );
    Mat dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_flip(src, dst, flip_mode));

    size_t esz = CV_ELEM_SIZE(type);

    if( flip_mode <= 0 )
        flipVert( src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz );
    else
        flipHoriz( src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz );

    if( flip_mode < 0 )
        flipHoriz( dst.ptr(), dst.step, dst.ptr(), dst.step, dst.size(), esz );
}

void rotate(InputArray _src, OutputArray _dst, int rotateMode)
{
    CV_Assert(_src.dims() <= 2);

    switch (rotateMode)
    {
    case ROTATE_90_CLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 1);
        break;
    case ROTATE_180:
        flip(_src, _dst, -1);
        break;
    case ROTATE_90_COUNTERCLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 0);
        break;
    default:
        break;
    }
}

}  // namespace
