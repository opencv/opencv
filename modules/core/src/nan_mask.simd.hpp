// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"

namespace cv {

typedef void (*PatchNanFunc)(uchar* tptr, size_t len, double newVal);
typedef void (*FiniteMaskFunc)(const uchar *src, uchar *dst, size_t total);

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

PatchNanFunc getPatchNanFunc(bool isDouble);
FiniteMaskFunc getFiniteMaskFunc(bool isDouble, int cn);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

static void patchNaNs_32f(uchar* ptr, size_t len, double newVal)
{
    int32_t* tptr = (int32_t*)ptr;
    Cv32suf val;
    val.f = (float)newVal;

    size_t j = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_int32 v_mask1 = vx_setall_s32(0x7fffffff), v_mask2 = vx_setall_s32(0x7f800000);
    v_int32 v_val = vx_setall_s32(val.i);

    size_t cWidth = (size_t)VTraits<v_int32>::vlanes();
    for (; j + cWidth <= len; j += cWidth)
    {
        v_int32 v_src = vx_load(tptr + j);
        v_int32 v_cmp_mask = v_lt(v_mask2, v_and(v_src, v_mask1));
        v_int32 v_dst = v_select(v_cmp_mask, v_val, v_src);
        v_store(tptr + j, v_dst);
    }
    vx_cleanup();
#endif

    for (; j < len; j++)
    {
        if ((tptr[j] & 0x7fffffff) > 0x7f800000)
        {
            tptr[j] = val.i;
        }
    }
}


static void patchNaNs_64f(uchar* ptr, size_t len, double newVal)
{
    int64_t* tptr = (int64_t*)ptr;
    Cv64suf val;
    val.f = newVal;

    size_t j = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_int64 v_exp_mask = vx_setall_s64(0x7FF0000000000000);
    v_int64 v_pos_mask = vx_setall_s64(0x7FFFFFFFFFFFFFFF);
    v_int64 v_val = vx_setall_s64(val.i);

    size_t cWidth = (size_t)VTraits<v_int64>::vlanes();
    for (; j + cWidth <= len; j += cWidth)
    {
        v_int64 v_src = vx_load(tptr + j);
        
        v_int64 v_cmp_mask = v_lt(v_exp_mask, v_and(v_src, v_pos_mask));
        // v_select is not available for v_int64, emulating it
        v_int64 v_dst = v_or(v_and(v_cmp_mask, v_val), v_and(v_not(v_cmp_mask), v_src));
        v_store(tptr + j, v_dst);
    }
    vx_cleanup();
#endif

    for (; j < len; j++)
        if ((tptr[j] & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000)
            tptr[j] = val.i;
}

PatchNanFunc getPatchNanFunc(bool isDouble)
{
    return isDouble ? (PatchNanFunc)GET_OPTIMIZED(patchNaNs_64f)
                    : (PatchNanFunc)GET_OPTIMIZED(patchNaNs_32f);
}

////// finiteMask //////

#if (CV_SIMD || CV_SIMD_SCALABLE)

//TODO: make true SIMD code instead for the rest
template <typename _Tp, int cn>
int finiteMaskSIMD_(const _Tp *src, uchar *dst, size_t total)
{
    const int osize = 8;
    int i = 0;
    for (; i <= (int)total - osize; i += osize)
    {
        for (int j = 0; j < osize; j++)
        {
            bool finite = true;
            for (int c = 0; c < cn; c++)
            {
                _Tp val = src[i * cn + j * cn + c];
                finite = finite && !cvIsNaN(val) && !cvIsInf(val);
            }
            dst[i + j] = finite ? 255 : 0;
        }
    }

    return i;
}


template <>
int finiteMaskSIMD_<float, 1>(const float *fsrc, uchar *dst, size_t total)
{
    const uint32_t* src = (const uint32_t*)fsrc;
    const int osize = VTraits<v_uint8>::vlanes();
    v_uint32 vmaskExp = vx_setall_u32(0x7f800000);

    int i = 0;
    for(; i <= (int)total - osize; i += osize )
    {
        v_uint32 vv0, vv1, vv2, vv3;
        vv0 = v_ne(v_and(vx_load(src + i              ), vmaskExp), vmaskExp);
        vv1 = v_ne(v_and(vx_load(src + i +   (osize/4)), vmaskExp), vmaskExp);
        vv2 = v_ne(v_and(vx_load(src + i + 2*(osize/4)), vmaskExp), vmaskExp);
        vv3 = v_ne(v_and(vx_load(src + i + 3*(osize/4)), vmaskExp), vmaskExp);

        v_store(dst + i, v_pack_b(vv0, vv1, vv2, vv3));
    }
    vx_cleanup();

    return i;
}


template <>
int finiteMaskSIMD_<float, 2>(const float *fsrc, uchar *dst, size_t total)
{
    const uint32_t* src = (const uint32_t*)fsrc;
    const int size8 = VTraits<v_uint8>::vlanes();
    v_uint32 vmaskExp = vx_setall_u32(0x7f800000);
    v_uint16 vmaskBoth = vx_setall_u16(0xffff);

    int i = 0;
    for(; i <= (int)total - (size8 / 2); i += (size8 / 2) )
    {
        v_uint32 vv0, vv1, vv2, vv3;
        vv0 = v_ne(v_and(vx_load(src + i*2                ), vmaskExp), vmaskExp);
        vv1 = v_ne(v_and(vx_load(src + i*2 +   (size8 / 4)), vmaskExp), vmaskExp);
        vv2 = v_ne(v_and(vx_load(src + i*2 + 2*(size8 / 4)), vmaskExp), vmaskExp);
        vv3 = v_ne(v_and(vx_load(src + i*2 + 3*(size8 / 4)), vmaskExp), vmaskExp);
        v_uint8 velems = v_pack_b(vv0, vv1, vv2, vv3);
        v_uint16 vfinite = v_eq(v_reinterpret_as_u16(velems), vmaskBoth);

        // 2nd argument in vfinite is useless
        v_store_low(dst + i, v_pack(vfinite, vfinite));
    }
    vx_cleanup();

    return i;
}


template <>
int finiteMaskSIMD_<float, 3>(const float *fsrc, uchar *dst, size_t total)
{
    const uint32_t* src = (const uint32_t*)fsrc;
    const int npixels = VTraits<v_float32>::vlanes();
    v_uint32 vmaskExp = vx_setall_u32(0x7f800000);
    v_uint32 z = vx_setzero_u32();

    int i = 0;
    for (; i <= (int)total - npixels; i += npixels)
    {
        v_uint32 vv0, vv1, vv2;
        vv0 = v_ne(v_and(vx_load(src + i*3            ), vmaskExp), vmaskExp);
        vv1 = v_ne(v_and(vx_load(src + i*3 +   npixels), vmaskExp), vmaskExp);
        vv2 = v_ne(v_and(vx_load(src + i*3 + 2*npixels), vmaskExp), vmaskExp);

        v_uint8 velems = v_pack_b(vv0, vv1, vv2, z);

        // 2nd arg is useless
        v_uint8 vsh1 = v_extract<1>(velems, velems);
        v_uint8 vsh2 = v_extract<2>(velems, velems);

        v_uint8 vres3 = v_and(v_and(velems, vsh1), vsh2);
        for (int j = 0; j < npixels; j++)
        {
            dst[i + j] = v_get0(vres3);
            // 2nd arg is useless
            vres3 = v_extract<3>(vres3, vres3);
        }
    }

    return i;
}


template <>
int finiteMaskSIMD_<float, 4>(const float *fsrc, uchar *dst, size_t total)
{
    const uint32_t* src = (const uint32_t*)fsrc;
    CV_StaticAssert(VTraits<v_uint8>::max_nlanes*8 <= 1024, "SIMD registers should be not more than 1024 bit wide");
    const int npixels = VTraits<v_uint32>::vlanes()/4;

    v_uint32 vmaskExp = vx_setall_u32(0x7f800000);
    v_uint32 z = vx_setzero_u32();
    v_uint32 vmaskAll4 = vx_setall_u32(0xFFFFFFFF);

    int i = 0;
    for(; i <= (int)total - npixels; i += npixels )
    {
        v_uint32 vv = v_ne(v_and(vx_load(src + i*4), vmaskExp), vmaskExp);
        v_uint8 velems = v_pack_b(vv, z, z, z);

        v_uint32 vresWide = v_eq(v_reinterpret_as_u32(velems), vmaskAll4);

        v_uint8 vres = v_pack_b(vresWide, z, z, z);
        if (npixels == 1) // 128 bit wide
        {
            dst[i] = v_get0(vres);
        }
        else if (npixels == 2) // 256 bit wide
        {
            *((uint16_t*)(dst + i)) = v_get0(v_reinterpret_as_u16(vres));
        }
        else if (npixels == 4) // 512 bit wide
        {
            *((uint32_t*)(dst + i)) = v_get0(v_reinterpret_as_u32(vres));
        }
        else if (npixels == 8) // 1024 bit wide
        {
            *((uint64_t*)(dst + i)) = v_get0(v_reinterpret_as_u64(vres));
        }
    }
    vx_cleanup();

    return i;
}


template <>
int finiteMaskSIMD_<double, 1>(const double *dsrc, uchar *dst, size_t total)
{
    const uint64_t* src = (const uint64_t*)dsrc;
    const int size8 = VTraits<v_uint8>::vlanes();
    int i = 0;

    v_uint64 vmaskExp = vx_setall_u64(0x7ff0000000000000);
    v_uint64 z = vx_setzero_u64();
    
    for(; i <= (int)total - (size8 / 2); i += (size8 / 2) )
    {
        v_uint64 vv0, vv1, vv2, vv3;
        vv0 = v_ne(v_and(vx_load(src + i                ), vmaskExp), vmaskExp);
        vv1 = v_ne(v_and(vx_load(src + i +   (size8 / 8)), vmaskExp), vmaskExp);
        vv2 = v_ne(v_and(vx_load(src + i + 2*(size8 / 8)), vmaskExp), vmaskExp);
        vv3 = v_ne(v_and(vx_load(src + i + 3*(size8 / 8)), vmaskExp), vmaskExp);

        v_uint8 v = v_pack_b(vv0, vv1, vv2, vv3, z, z, z, z);

        v_store_low(dst + i, v);
    }
    vx_cleanup();

    return i;
}

template <>
int finiteMaskSIMD_<double, 2>(const double *dsrc, uchar *dst, size_t total)
{
    const uint64_t* src = (const uint64_t*)dsrc;
    const int size8 = VTraits<v_uint8>::vlanes();
    const int npixels = size8 / 16;
    v_uint64 vmaskExp = vx_setall_u64(0x7ff0000000000000);
    v_uint64 z = vx_setzero_u64();

    int i = 0;
    for(; i <= (int)total - npixels; i += npixels )
    {
        v_uint64 vv = v_ne(v_and(vx_load(src + i*2), vmaskExp), vmaskExp);

        v_uint8 velems = v_pack_b(vv, z, z, z, z, z, z, z);

        v_uint16 vmaskBoth = vx_setall_u16(0xffff);
        v_uint16 vfinite = v_eq(v_reinterpret_as_u16(velems), vmaskBoth);

        // 2nd arg is useless
        v_uint8 vres = v_pack(vfinite, vfinite);

        for (int j = 0; j < npixels; j++)
        {
            dst[i+j] = v_get0(vres);
            // 2nd arg is useless
            vres = v_extract<1>(vres, vres);
        }
    }
    vx_cleanup();

    return i;
}


template <>
int finiteMaskSIMD_<double, 3>(const double *fsrc, uchar *dst, size_t total)
{
    const uint64_t* src = (const uint64_t*)fsrc;
    const int npixels = VTraits<v_float64>::vlanes();
    v_uint64 vmaskExp = vx_setall_u64(0x7ff0000000000000);
    v_uint64 z = vx_setzero_u64();

    int i = 0;
    for (; i <= (int)total - npixels; i += npixels)
    {
        v_uint64 vv0, vv1, vv2;
        vv0 = v_ne(v_and(vx_load(src + i*3            ), vmaskExp), vmaskExp);
        vv1 = v_ne(v_and(vx_load(src + i*3 +   npixels), vmaskExp), vmaskExp);
        vv2 = v_ne(v_and(vx_load(src + i*3 + 2*npixels), vmaskExp), vmaskExp);

        v_uint8 velems = v_pack_b(vv0, vv1, vv2, z, z, z, z, z);

        // 2nd arg is useless
        v_uint8 vsh1 = v_extract<1>(velems, velems);
        v_uint8 vsh2 = v_extract<2>(velems, velems);

        v_uint8 vres3 = v_and(v_and(velems, vsh1), vsh2);
        for (int j = 0; j < npixels; j++)
        {
            dst[i + j] = v_get0(vres3);
            // 2nd arg is useless
            vres3 = v_extract<3>(vres3, vres3);
        }
    }

    return i;
}

#endif


template <typename _Tp, int cn>
void finiteMask_(const uchar *src, uchar *dst, size_t total)
{
    size_t i = 0;
    const _Tp* tsrc = (const _Tp*) src;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    i = finiteMaskSIMD_<_Tp, cn>(tsrc, dst, total);
#endif

    for(; i < total; i++ )
    {
        bool finite = true;
        for (int c = 0; c < cn; c++)
        {
            _Tp val = tsrc[i * cn + c];
            finite = finite && !cvIsNaN(val) && !cvIsInf(val);
        }
        dst[i] = finite ? 255 : 0;
    }
}

FiniteMaskFunc getFiniteMaskFunc(bool isDouble, int cn)
{
    static FiniteMaskFunc tab[] =
    {
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<float,  1>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<float,  2>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<float,  3>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<float,  4>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<double, 1>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<double, 2>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<double, 3>)),
        (FiniteMaskFunc)GET_OPTIMIZED((finiteMask_<double, 4>)),
    };

    int idx = (isDouble ? 4 : 0) + cn - 1;
    return tab[idx];
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
