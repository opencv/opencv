// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

namespace cv { namespace hal {

#if CV_NEON
template<typename T> struct VSplit2;
template<typename T> struct VSplit3;
template<typename T> struct VSplit4;

#define SPLIT2_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0,                    \
                        data_type* dst1) const                                    \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
        }                                                                         \
    }

#define SPLIT3_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0, data_type* dst1,   \
                        data_type* dst2) const                                    \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
            store_func(dst2, r.val[2]);                                           \
        }                                                                         \
    }

#define SPLIT4_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>                                                        \
    {                                                                             \
        void operator()(const data_type* src, data_type* dst0, data_type* dst1,   \
                        data_type* dst2, data_type* dst3) const                   \
        {                                                                         \
            reg_type r = load_func(src);                                          \
            store_func(dst0, r.val[0]);                                           \
            store_func(dst1, r.val[1]);                                           \
            store_func(dst2, r.val[2]);                                           \
            store_func(dst3, r.val[3]);                                           \
        }                                                                         \
    }

SPLIT2_KERNEL_TEMPLATE(VSplit2, uchar ,  uint8x16x2_t, vld2q_u8 , vst1q_u8 );
SPLIT2_KERNEL_TEMPLATE(VSplit2, ushort,  uint16x8x2_t, vld2q_u16, vst1q_u16);
SPLIT2_KERNEL_TEMPLATE(VSplit2, int   ,   int32x4x2_t, vld2q_s32, vst1q_s32);
SPLIT2_KERNEL_TEMPLATE(VSplit2, int64 ,   int64x1x2_t, vld2_s64 , vst1_s64 );

SPLIT3_KERNEL_TEMPLATE(VSplit3, uchar ,  uint8x16x3_t, vld3q_u8 , vst1q_u8 );
SPLIT3_KERNEL_TEMPLATE(VSplit3, ushort,  uint16x8x3_t, vld3q_u16, vst1q_u16);
SPLIT3_KERNEL_TEMPLATE(VSplit3, int   ,   int32x4x3_t, vld3q_s32, vst1q_s32);
SPLIT3_KERNEL_TEMPLATE(VSplit3, int64 ,   int64x1x3_t, vld3_s64 , vst1_s64 );

SPLIT4_KERNEL_TEMPLATE(VSplit4, uchar ,  uint8x16x4_t, vld4q_u8 , vst1q_u8 );
SPLIT4_KERNEL_TEMPLATE(VSplit4, ushort,  uint16x8x4_t, vld4q_u16, vst1q_u16);
SPLIT4_KERNEL_TEMPLATE(VSplit4, int   ,   int32x4x4_t, vld4q_s32, vst1q_s32);
SPLIT4_KERNEL_TEMPLATE(VSplit4, int64 ,   int64x1x4_t, vld4_s64 , vst1_s64 );

#elif CV_SSE2

template <typename T>
struct VSplit2
{
    VSplit2() : support(false) { }
    void operator()(const T *, T *, T *) const { }

    bool support;
};

template <typename T>
struct VSplit3
{
    VSplit3() : support(false) { }
    void operator()(const T *, T *, T *, T *) const { }

    bool support;
};

template <typename T>
struct VSplit4
{
    VSplit4() : support(false) { }
    void operator()(const T *, T *, T *, T *, T *) const { }

    bool support;
};

#define SPLIT2_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit2<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit2()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src,                                                 \
                    data_type * dst0, data_type * dst1) const                              \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2, v_src3);                                  \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define SPLIT3_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit3<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit3()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src,                                                 \
                    data_type * dst0, data_type * dst1, data_type * dst2) const            \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
        reg_type v_src4 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 4)); \
        reg_type v_src5 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 5)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2,                                           \
                         v_src3, v_src4, v_src5);                                          \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
        _mm_storeu_##flavor((cast_type *)(dst2), v_src4);                                  \
        _mm_storeu_##flavor((cast_type *)(dst2 + ELEMS_IN_VEC), v_src5);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define SPLIT4_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_deinterleave, flavor)   \
template <>                                                                                \
struct VSplit4<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VSplit4()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(CV_CPU_SSE2);                                       \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src, data_type * dst0, data_type * dst1,             \
                    data_type * dst2, data_type * dst3) const                              \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((cast_type const *)(src));                    \
        reg_type v_src1 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC));     \
        reg_type v_src2 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 2)); \
        reg_type v_src3 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 3)); \
        reg_type v_src4 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 4)); \
        reg_type v_src5 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 5)); \
        reg_type v_src6 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 6)); \
        reg_type v_src7 = _mm_loadu_##flavor((cast_type const *)(src + ELEMS_IN_VEC * 7)); \
                                                                                           \
        _mm_deinterleave(v_src0, v_src1, v_src2, v_src3,                                   \
                         v_src4, v_src5, v_src6, v_src7);                                  \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst0), v_src0);                                  \
        _mm_storeu_##flavor((cast_type *)(dst0 + ELEMS_IN_VEC), v_src1);                   \
        _mm_storeu_##flavor((cast_type *)(dst1), v_src2);                                  \
        _mm_storeu_##flavor((cast_type *)(dst1 + ELEMS_IN_VEC), v_src3);                   \
        _mm_storeu_##flavor((cast_type *)(dst2), v_src4);                                  \
        _mm_storeu_##flavor((cast_type *)(dst2 + ELEMS_IN_VEC), v_src5);                   \
        _mm_storeu_##flavor((cast_type *)(dst3), v_src6);                                  \
        _mm_storeu_##flavor((cast_type *)(dst3 + ELEMS_IN_VEC), v_src7);                   \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

SPLIT2_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT2_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT2_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

SPLIT3_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT3_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT3_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

SPLIT4_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_deinterleave_epi8, si128);
SPLIT4_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_deinterleave_epi16, si128);
SPLIT4_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_deinterleave_ps, ps);

#endif

template<typename T> static void
split_( const T* src, T** dst, int len, int cn )
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if( k == 1 )
    {
        T* dst0 = dst[0];

        if(cn == 1)
        {
            memcpy(dst0, src, len * sizeof(T));
        }
        else
        {
            for( i = 0, j = 0 ; i < len; i++, j += cn )
                dst0[i] = src[j];
        }
    }
    else if( k == 2 )
    {
        T *dst0 = dst[0], *dst1 = dst[1];
        i = j = 0;

#if CV_NEON
        if(cn == 2)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 2 * inc_i;

            VSplit2<T> vsplit;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i);
        }
#elif CV_SSE2
        if (cn == 2)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 2 * inc_i;

            VSplit2<T> vsplit;
            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
        }
    }
    else if( k == 3 )
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        i = j = 0;

#if CV_NEON
        if(cn == 3)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 3 * inc_i;

            VSplit3<T> vsplit;
            for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i, dst2 + i);
        }
#elif CV_SSE2
        if (cn == 3)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 3 * inc_i;

            VSplit3<T> vsplit;

            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i, dst2 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
            dst2[i] = src[j+2];
        }
    }
    else
    {
        T *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        i = j = 0;

#if CV_NEON
        if(cn == 4)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 4 * inc_i;

            VSplit4<T> vsplit;
            for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                vsplit(src + j, dst0 + i, dst1 + i, dst2 + i, dst3 + i);
        }
#elif CV_SSE2
        if (cn == 4)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 4 * inc_i;

            VSplit4<T> vsplit;
            if (vsplit.support)
            {
                for( ; i <= len - inc_i; i += inc_i, j += inc_j)
                    vsplit(src + j, dst0 + i, dst1 + i, dst2 + i, dst3 + i);
            }
        }
#endif
        for( ; i < len; i++, j += cn )
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

void split8u(const uchar* src, uchar** dst, int len, int cn )
{
    CALL_HAL(split8u, cv_hal_split8u, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split16u(const ushort* src, ushort** dst, int len, int cn )
{
    CALL_HAL(split16u, cv_hal_split16u, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split32s(const int* src, int** dst, int len, int cn )
{
    CALL_HAL(split32s, cv_hal_split32s, src,dst, len, cn)
    split_(src, dst, len, cn);
}

void split64s(const int64* src, int64** dst, int len, int cn )
{
    CALL_HAL(split64s, cv_hal_split64s, src,dst, len, cn)
    split_(src, dst, len, cn);
}

}} // cv::hal::

/****************************************************************************************\
*                                       split & merge                                    *
\****************************************************************************************/

typedef void (*SplitFunc)(const uchar* src, uchar** dst, int len, int cn);

static SplitFunc getSplitFunc(int depth)
{
    static SplitFunc splitTab[] =
    {
        (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split8u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u), (SplitFunc)GET_OPTIMIZED(cv::hal::split16u),
        (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split32s), (SplitFunc)GET_OPTIMIZED(cv::hal::split64s), 0
    };

    return splitTab[depth];
}

#ifdef HAVE_IPP

namespace cv {
static bool ipp_split(const Mat& src, Mat* mv, int channels)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    if(channels != 3 && channels != 4)
        return false;

    if(src.dims <= 2)
    {
        IppiSize size       = ippiSize(src.size());
        void    *dstPtrs[4] = {NULL};
        size_t   dstStep    = mv[0].step;
        for(int i = 0; i < channels; i++)
        {
            dstPtrs[i] = mv[i].ptr();
            if(dstStep != mv[i].step)
                return false;
        }

        return CV_INSTRUMENT_FUN_IPP(llwiCopySplit, src.ptr(), (int)src.step, dstPtrs, (int)dstStep, size, (int)src.elemSize1(), channels, 0) >= 0;
    }
    else
    {
        const Mat *arrays[5] = {NULL};
        uchar     *ptrs[5]   = {NULL};
        arrays[0] = &src;

        for(int i = 1; i < channels; i++)
        {
            arrays[i] = &mv[i-1];
        }

        NAryMatIterator it(arrays, ptrs);
        IppiSize size = { (int)it.size, 1 };

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopySplit, ptrs[0], 0, (void**)&ptrs[1], 0, size, (int)src.elemSize1(), channels, 0) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(src); CV_UNUSED(mv); CV_UNUSED(channels);
    return false;
#endif
}
}
#endif

void cv::split(const Mat& src, Mat* mv)
{
    CV_INSTRUMENT_REGION()

    int k, depth = src.depth(), cn = src.channels();
    if( cn == 1 )
    {
        src.copyTo(mv[0]);
        return;
    }

    for( k = 0; k < cn; k++ )
    {
        mv[k].create(src.dims, src.size, depth);
    }

    CV_IPP_RUN_FAST(ipp_split(src, mv, cn));

    SplitFunc func = getSplitFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = src.elemSize(), esz1 = src.elemSize1();
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for( k = 0; k < cn; k++ )
    {
        arrays[k+1] = &mv[k];
    }

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( ptrs[0], &ptrs[1], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( k = 0; k < cn; k++ )
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_split( InputArray _m, OutputArrayOfArrays _mv )
{
    int type = _m.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;

    String dstargs, processelem, indexdecl;
    for (int i = 0; i < cn; ++i)
    {
        dstargs += format("DECLARE_DST_PARAM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
    }

    ocl::Kernel k("split", ocl::core::split_merge_oclsrc,
                  format("-D T=%s -D OP_SPLIT -D cn=%d -D DECLARE_DST_PARAMS=%s"
                         " -D PROCESS_ELEMS_N=%s -D DECLARE_INDEX_N=%s",
                         ocl::memopTypeToStr(depth), cn, dstargs.c_str(),
                         processelem.c_str(), indexdecl.c_str()));
    if (k.empty())
        return false;

    Size size = _m.size();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(size, depth, i);

    std::vector<UMat> dst;
    _mv.getUMatVector(dst);

    int argidx = k.set(0, ocl::KernelArg::ReadOnly(_m.getUMat()));
    for (int i = 0; i < cn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::WriteOnlyNoSize(dst[i]));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { (size_t)size.width, ((size_t)size.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::split(InputArray _m, OutputArrayOfArrays _mv)
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_m.dims() <= 2 && _mv.isUMatVector(),
               ocl_split(_m, _mv))

    Mat m = _m.getMat();
    if( m.empty() )
    {
        _mv.release();
        return;
    }

    CV_Assert( !_mv.fixedType() || _mv.empty() || _mv.type() == m.depth() );

    int depth = m.depth(), cn = m.channels();
    _mv.create(cn, 1, depth);
    for (int i = 0; i < cn; ++i)
        _mv.create(m.dims, m.size.p, depth, i);

    std::vector<Mat> dst;
    _mv.getMatVector(dst);

    split(m, &dst[0]);
}
