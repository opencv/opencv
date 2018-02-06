// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

namespace cv { namespace hal {

#if CV_NEON
template<typename T> struct VMerge2;
template<typename T> struct VMerge3;
template<typename T> struct VMerge4;

#define MERGE2_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        data_type* dst){                                          \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

#define MERGE3_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        const data_type* src2, data_type* dst){                   \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            r.val[2] = load_func(src2);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

#define MERGE4_KERNEL_TEMPLATE(name, data_type, reg_type, load_func, store_func)  \
    template<>                                                                    \
    struct name<data_type>{                                                       \
        void operator()(const data_type* src0, const data_type* src1,             \
                        const data_type* src2, const data_type* src3,             \
                        data_type* dst){                                          \
            reg_type r;                                                           \
            r.val[0] = load_func(src0);                                           \
            r.val[1] = load_func(src1);                                           \
            r.val[2] = load_func(src2);                                           \
            r.val[3] = load_func(src3);                                           \
            store_func(dst, r);                                                   \
        }                                                                         \
    }

MERGE2_KERNEL_TEMPLATE(VMerge2, uchar ,  uint8x16x2_t, vld1q_u8 , vst2q_u8 );
MERGE2_KERNEL_TEMPLATE(VMerge2, ushort,  uint16x8x2_t, vld1q_u16, vst2q_u16);
MERGE2_KERNEL_TEMPLATE(VMerge2, int   ,   int32x4x2_t, vld1q_s32, vst2q_s32);
MERGE2_KERNEL_TEMPLATE(VMerge2, int64 ,   int64x1x2_t, vld1_s64 , vst2_s64 );

MERGE3_KERNEL_TEMPLATE(VMerge3, uchar ,  uint8x16x3_t, vld1q_u8 , vst3q_u8 );
MERGE3_KERNEL_TEMPLATE(VMerge3, ushort,  uint16x8x3_t, vld1q_u16, vst3q_u16);
MERGE3_KERNEL_TEMPLATE(VMerge3, int   ,   int32x4x3_t, vld1q_s32, vst3q_s32);
MERGE3_KERNEL_TEMPLATE(VMerge3, int64 ,   int64x1x3_t, vld1_s64 , vst3_s64 );

MERGE4_KERNEL_TEMPLATE(VMerge4, uchar ,  uint8x16x4_t, vld1q_u8 , vst4q_u8 );
MERGE4_KERNEL_TEMPLATE(VMerge4, ushort,  uint16x8x4_t, vld1q_u16, vst4q_u16);
MERGE4_KERNEL_TEMPLATE(VMerge4, int   ,   int32x4x4_t, vld1q_s32, vst4q_s32);
MERGE4_KERNEL_TEMPLATE(VMerge4, int64 ,   int64x1x4_t, vld1_s64 , vst4_s64 );

#elif CV_SSE2

template <typename T>
struct VMerge2
{
    VMerge2() : support(false) { }
    void operator()(const T *, const T *, T *) const { }

    bool support;
};

template <typename T>
struct VMerge3
{
    VMerge3() : support(false) { }
    void operator()(const T *, const T *, const T *, T *) const { }

    bool support;
};

template <typename T>
struct VMerge4
{
    VMerge4() : support(false) { }
    void operator()(const T *, const T *, const T *, const T *, T *) const { }

    bool support;
};

#define MERGE2_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge2<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge2()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1,                        \
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2, v_src3);                                    \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define MERGE3_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge3<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge3()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1, const data_type * src2,\
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
        reg_type v_src4 = _mm_loadu_##flavor((const cast_type *)(src2));                   \
        reg_type v_src5 = _mm_loadu_##flavor((const cast_type *)(src2 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2,                                             \
                       v_src3, v_src4, v_src5);                                            \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 4), v_src4);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 5), v_src5);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

#define MERGE4_KERNEL_TEMPLATE(data_type, reg_type, cast_type, _mm_interleave, flavor, se) \
template <>                                                                                \
struct VMerge4<data_type>                                                                  \
{                                                                                          \
    enum                                                                                   \
    {                                                                                      \
        ELEMS_IN_VEC = 16 / sizeof(data_type)                                              \
    };                                                                                     \
                                                                                           \
    VMerge4()                                                                              \
    {                                                                                      \
        support = checkHardwareSupport(se);                                                \
    }                                                                                      \
                                                                                           \
    void operator()(const data_type * src0, const data_type * src1,                        \
                    const data_type * src2, const data_type * src3,                        \
                    data_type * dst) const                                                 \
    {                                                                                      \
        reg_type v_src0 = _mm_loadu_##flavor((const cast_type *)(src0));                   \
        reg_type v_src1 = _mm_loadu_##flavor((const cast_type *)(src0 + ELEMS_IN_VEC));    \
        reg_type v_src2 = _mm_loadu_##flavor((const cast_type *)(src1));                   \
        reg_type v_src3 = _mm_loadu_##flavor((const cast_type *)(src1 + ELEMS_IN_VEC));    \
        reg_type v_src4 = _mm_loadu_##flavor((const cast_type *)(src2));                   \
        reg_type v_src5 = _mm_loadu_##flavor((const cast_type *)(src2 + ELEMS_IN_VEC));    \
        reg_type v_src6 = _mm_loadu_##flavor((const cast_type *)(src3));                   \
        reg_type v_src7 = _mm_loadu_##flavor((const cast_type *)(src3 + ELEMS_IN_VEC));    \
                                                                                           \
        _mm_interleave(v_src0, v_src1, v_src2, v_src3,                                     \
                       v_src4, v_src5, v_src6, v_src7);                                    \
                                                                                           \
        _mm_storeu_##flavor((cast_type *)(dst), v_src0);                                   \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC), v_src1);                    \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 2), v_src2);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 3), v_src3);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 4), v_src4);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 5), v_src5);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 6), v_src6);                \
        _mm_storeu_##flavor((cast_type *)(dst + ELEMS_IN_VEC * 7), v_src7);                \
    }                                                                                      \
                                                                                           \
    bool support;                                                                          \
}

MERGE2_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);
MERGE3_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);
MERGE4_KERNEL_TEMPLATE( uchar, __m128i, __m128i, _mm_interleave_epi8, si128, CV_CPU_SSE2);

#if CV_SSE4_1
MERGE2_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
MERGE3_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
MERGE4_KERNEL_TEMPLATE(ushort, __m128i, __m128i, _mm_interleave_epi16, si128, CV_CPU_SSE4_1);
#endif

MERGE2_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);
MERGE3_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);
MERGE4_KERNEL_TEMPLATE(   int,  __m128,   float, _mm_interleave_ps, ps, CV_CPU_SSE2);

#endif

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
        i = j = 0;
#if CV_NEON
        if(cn == 2)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 2 * inc_i;

            VMerge2<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 2)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 2 * inc_i;

            VMerge2<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
        }
    }
    else if( k == 3 )
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2];
        i = j = 0;
#if CV_NEON
        if(cn == 3)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 3 * inc_i;

            VMerge3<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, src2 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 3)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 3 * inc_i;

            VMerge3<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, src2 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
        {
            dst[j] = src0[i];
            dst[j+1] = src1[i];
            dst[j+2] = src2[i];
        }
    }
    else
    {
        const T *src0 = src[0], *src1 = src[1], *src2 = src[2], *src3 = src[3];
        i = j = 0;
#if CV_NEON
        if(cn == 4)
        {
            int inc_i = (sizeof(T) == 8)? 1: 16/sizeof(T);
            int inc_j = 4 * inc_i;

            VMerge4<T> vmerge;
            for( ; i < len - inc_i; i += inc_i, j += inc_j)
                vmerge(src0 + i, src1 + i, src2 + i, src3 + i, dst + j);
        }
#elif CV_SSE2
        if(cn == 4)
        {
            int inc_i = 32/sizeof(T);
            int inc_j = 4 * inc_i;

            VMerge4<T> vmerge;
            if (vmerge.support)
                for( ; i < len - inc_i; i += inc_i, j += inc_j)
                    vmerge(src0 + i, src1 + i, src2 + i, src3 + i, dst + j);
        }
#endif
        for( ; i < len; i++, j += cn )
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


void merge8u(const uchar** src, uchar* dst, int len, int cn )
{
    CALL_HAL(merge8u, cv_hal_merge8u, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge16u(const ushort** src, ushort* dst, int len, int cn )
{
    CALL_HAL(merge16u, cv_hal_merge16u, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge32s(const int** src, int* dst, int len, int cn )
{
    CALL_HAL(merge32s, cv_hal_merge32s, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

void merge64s(const int64** src, int64* dst, int len, int cn )
{
    CALL_HAL(merge64s, cv_hal_merge64s, src, dst, len, cn)
    merge_(src, dst, len, cn);
}

}} // cv::hal::


typedef void (*MergeFunc)(const uchar** src, uchar* dst, int len, int cn);

static MergeFunc getMergeFunc(int depth)
{
    static MergeFunc mergeTab[] =
    {
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge8u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u), (MergeFunc)GET_OPTIMIZED(cv::hal::merge16u),
        (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge32s), (MergeFunc)GET_OPTIMIZED(cv::hal::merge64s), 0
    };

    return mergeTab[depth];
}

#ifdef HAVE_IPP

namespace cv {
static bool ipp_merge(const Mat* mv, Mat& dst, int channels)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    if(channels != 3 && channels != 4)
        return false;

    if(mv[0].dims <= 2)
    {
        IppiSize    size       = ippiSize(mv[0].size());
        const void *srcPtrs[4] = {NULL};
        size_t      srcStep    = mv[0].step;
        for(int i = 0; i < channels; i++)
        {
            srcPtrs[i] = mv[i].ptr();
            if(srcStep != mv[i].step)
                return false;
        }

        return CV_INSTRUMENT_FUN_IPP(llwiCopyMerge, srcPtrs, (int)srcStep, dst.ptr(), (int)dst.step, size, (int)mv[0].elemSize1(), channels, 0) >= 0;
    }
    else
    {
        const Mat *arrays[5] = {NULL};
        uchar     *ptrs[5]   = {NULL};
        arrays[0] = &dst;

        for(int i = 1; i < channels; i++)
        {
            arrays[i] = &mv[i-1];
        }

        NAryMatIterator it(arrays, ptrs);
        IppiSize size = { (int)it.size, 1 };

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if(CV_INSTRUMENT_FUN_IPP(llwiCopyMerge, (const void**)&ptrs[1], 0, ptrs[0], 0, size, (int)mv[0].elemSize1(), channels, 0) < 0)
                return false;
        }
        return true;
    }
#else
    CV_UNUSED(dst); CV_UNUSED(mv); CV_UNUSED(channels);
    return false;
#endif
}
}
#endif

void cv::merge(const Mat* mv, size_t n, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

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

    CV_IPP_RUN_FAST(ipp_merge(mv, dst, (int)n));

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

    MergeFunc func = getMergeFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    size_t blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)(uchar*)_buf;
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = (int)it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_merge( InputArrayOfArrays _mv, OutputArray _dst )
{
    std::vector<UMat> src, ksrc;
    _mv.getUMatVector(src);
    CV_Assert(!src.empty());

    int type = src[0].type(), depth = CV_MAT_DEPTH(type),
            rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    Size size = src[0].size();

    for (size_t i = 0, srcsize = src.size(); i < srcsize; ++i)
    {
        int itype = src[i].type(), icn = CV_MAT_CN(itype), idepth = CV_MAT_DEPTH(itype),
                esz1 = CV_ELEM_SIZE1(idepth);
        if (src[i].dims > 2)
            return false;

        CV_Assert(size == src[i].size() && depth == idepth);

        for (int cn = 0; cn < icn; ++cn)
        {
            UMat tsrc = src[i];
            tsrc.offset += cn * esz1;
            ksrc.push_back(tsrc);
        }
    }
    int dcn = (int)ksrc.size();

    String srcargs, processelem, cndecl, indexdecl;
    for (int i = 0; i < dcn; ++i)
    {
        srcargs += format("DECLARE_SRC_PARAM(%d)", i);
        processelem += format("PROCESS_ELEM(%d)", i);
        indexdecl += format("DECLARE_INDEX(%d)", i);
        cndecl += format(" -D scn%d=%d", i, ksrc[i].channels());
    }

    ocl::Kernel k("merge", ocl::core::split_merge_oclsrc,
                  format("-D OP_MERGE -D cn=%d -D T=%s -D DECLARE_SRC_PARAMS_N=%s"
                         " -D DECLARE_INDEX_N=%s -D PROCESS_ELEMS_N=%s%s",
                         dcn, ocl::memopTypeToStr(depth), srcargs.c_str(),
                         indexdecl.c_str(), processelem.c_str(), cndecl.c_str()));
    if (k.empty())
        return false;

    _dst.create(size, CV_MAKE_TYPE(depth, dcn));
    UMat dst = _dst.getUMat();

    int argidx = 0;
    for (int i = 0; i < dcn; ++i)
        argidx = k.set(argidx, ocl::KernelArg::ReadOnlyNoSize(ksrc[i]));
    argidx = k.set(argidx, ocl::KernelArg::WriteOnly(dst));
    k.set(argidx, rowsPerWI);

    size_t globalsize[2] = { (size_t)dst.cols, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

}

#endif

void cv::merge(InputArrayOfArrays _mv, OutputArray _dst)
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_mv.isUMatVector() && _dst.isUMat(),
               ocl_merge(_mv, _dst))

    std::vector<Mat> mv;
    _mv.getMatVector(mv);
    merge(!mv.empty() ? &mv[0] : 0, mv.size(), _dst);
}
