/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_UNPACK_H
#define OPENEXR_CORE_UNPACK_H

#include "openexr_config.h"
#include "openexr_decode.h"
#include "openexr_encode.h"

#include "internal_structs.h"

/* only recently has imath supported half in C (C++ only before),
 * allow an older version to still work, and if that is available, we
 * will favor the implementation there as it will be the latest
 * up-to-date optimizations */
#if (IMATH_VERSION_MAJOR > 3) ||                                               \
    (IMATH_VERSION_MAJOR == 3 && IMATH_VERSION_MINOR >= 1)
#    define IMATH_HALF_SAFE_FOR_C
/* avoid the library dependency */
#    define IMATH_HALF_NO_LOOKUP_TABLE
#    include <half.h>
#endif

#ifdef _WIN32
#    include <intrin.h>
#elif defined(__x86_64__)
#    include <x86intrin.h>
#endif

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef exr_result_t (*internal_exr_unpack_fn) (exr_decode_pipeline_t*);

internal_exr_unpack_fn internal_exr_match_decode (
    exr_decode_pipeline_t* decode,
    int                    isdeep,
    int                    chanstofill,
    int                    chanstounpack,
    int                    sametype,
    int                    sameouttype,
    int                    samebpc,
    int                    sameoutbpc,
    int                    hassampling,
    int                    hastypechange,
    int                    sameoutinc,
    int                    simpinterleave,
    int                    simpinterleaverev,
    int                    simplineoff);

typedef exr_result_t (*internal_exr_pack_fn) (exr_encode_pipeline_t*);

internal_exr_pack_fn
internal_exr_match_encode (exr_encode_pipeline_t* encode, int isdeep);

exr_result_t internal_coding_fill_channel_info (
    exr_coding_channel_info_t**         channels,
    int16_t*                            num_chans,
    exr_coding_channel_info_t*          builtinextras,
    const exr_chunk_info_t*             cinfo,
    const struct _internal_exr_context* pctxt,
    const struct _internal_exr_part*    part);

exr_result_t internal_coding_update_channel_info (
    exr_coding_channel_info_t*          channels,
    int16_t                             num_chans,
    const exr_chunk_info_t*             cinfo,
    const struct _internal_exr_context* pctxt,
    const struct _internal_exr_part*    part);

exr_result_t internal_validate_next_chunk (
    exr_encode_pipeline_t*              encode,
    const struct _internal_exr_context* pctxt,
    const struct _internal_exr_part*    part);

/**************************************/

exr_result_t internal_encode_free_buffer (
    exr_encode_pipeline_t*               encode,
    exr_transcoding_pipeline_buffer_id_t bufid,
    void**                               buf,
    size_t*                              sz);

exr_result_t internal_encode_alloc_buffer (
    exr_encode_pipeline_t*               encode,
    exr_transcoding_pipeline_buffer_id_t bufid,
    void**                               buf,
    size_t*                              cursz,
    size_t                               newsz);

exr_result_t internal_decode_free_buffer (
    exr_decode_pipeline_t*               decode,
    exr_transcoding_pipeline_buffer_id_t bufid,
    void**                               buf,
    size_t*                              sz);

exr_result_t internal_decode_alloc_buffer (
    exr_decode_pipeline_t*               decode,
    exr_transcoding_pipeline_buffer_id_t bufid,
    void**                               buf,
    size_t*                              cursz,
    size_t                               newsz);

/**************************************/

static inline float
half_to_float (uint16_t hv)
{
#ifdef IMATH_HALF_SAFE_FOR_C
    return imath_half_to_float (hv);
#else
    /* replicate the code here from imath 3.1 since we are on an older
     * version which doesn't have a half that is safe for C code. Same
     * author, so free to do so. */
#    if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#        define OUR_LIKELY(x) (__builtin_expect ((x), 1))
#        define OUR_UNLIKELY(x) (__builtin_expect ((x), 0))
#    else
#        define OUR_LIKELY(x) (x)
#        define OUR_UNLIKELY(x) (x)
#    endif
    union
    {
        uint32_t i;
        float    f;
    } v;
    uint32_t hexpmant = ((uint32_t) (hv) << 17) >> 4;
    v.i               = ((uint32_t) (hv >> 15)) << 31;
    if (OUR_LIKELY ((hexpmant >= 0x00800000)))
    {
        v.i |= hexpmant;
        if (OUR_LIKELY ((hexpmant < 0x0f800000)))
            v.i += 0x38000000;
        else
            v.i |= 0x7f800000;
    }
    else if (hexpmant != 0)
    {
        uint32_t lc;
#    if defined(_MSC_VER) && (_M_IX86 || _M_X64)
        lc = __lzcnt (hexpmant);
#    elif defined(__GNUC__) || defined(__clang__)
        lc = (uint32_t) __builtin_clz (hexpmant);
#    else
        lc = 0;
        while (0 == ((hexpmant << lc) & 0x80000000))
            ++lc;
#    endif
        lc -= 8;
        v.i |= 0x38800000;
        v.i |= (hexpmant << lc);
        v.i -= (lc << 23);
    }
    return v.f;
#endif
}

static inline uint32_t
half_to_float_int (uint16_t hv)
{
    union
    {
        uint32_t i;
        float    f;
    } v;
    v.f = half_to_float (hv);
    return v.i;
}

static inline uint16_t
float_to_half (float fv)
{
#ifdef IMATH_HALF_SAFE_FOR_C
    return imath_float_to_half (fv);
#else
    union
    {
        uint32_t i;
        float    f;
    } v;
    uint16_t ret;
    uint32_t e, m, ui, r, shift;

    v.f = fv;
    ui  = (v.i & ~0x80000000);
    ret = ((v.i >> 16) & 0x8000);

    if (ui >= 0x38800000)
    {
        if (OUR_UNLIKELY (ui >= 0x7f800000))
        {
            ret |= 0x7c00;
            if (ui == 0x7f800000) return ret;
            m = (ui & 0x7fffff) >> 13;
            return (uint16_t) (ret | m | (m == 0));
        }

        if (OUR_UNLIKELY (ui > 0x477fefff)) return ret | 0x7c00;

        ui -= 0x38000000;
        ui = ((ui + 0x00000fff + ((ui >> 13) & 1)) >> 13);
        return (uint16_t) (ret | ui);
    }

    // zero or flush to 0
    if (ui < 0x33000001) return ret;

    // produce a denormalized half
    e     = (ui >> 23);
    shift = 0x7e - e;
    m     = 0x800000 | (ui & 0x7fffff);
    r     = m << (32 - shift);
    ret |= (m >> shift);
    if (r > 0x80000000 || (r == 0x80000000 && (ret & 0x1) != 0)) ++ret;
    return ret;
#endif
}

static inline uint16_t
float_to_half_int (uint32_t fiv)
{
    union
    {
        uint32_t i;
        float    f;
    } v;
    v.i = fiv;
    return float_to_half (v.f);
}

/**************************************/

static inline uint32_t
half_to_uint (uint16_t hv)
{
    /* replicating logic from imfmisc if negative or nan, return 0, if
     * inf, return uint32 max otherwise convert to float and cast to
     * uint */
    if (hv & 0x8000) return 0;
    if ((hv & 0x7c00) == 0x7c00)
    {
        if ((hv & 0x3ff) != 0) return 0;
        return UINT32_MAX;
    }
    return (uint32_t) (half_to_float (hv));
}

static inline uint32_t
float_to_uint (float fv)
{
    if (fv < 0.f || isnan (fv)) return 0;
    if (isinf (fv) || fv > (float) (UINT32_MAX)) return UINT32_MAX;
    return (uint32_t) (fv);
}

static inline uint32_t
float_to_uint_int (uint32_t fint)
{
    union
    {
        uint32_t i;
        float    f;
    } v;
    v.i = fint;
    return float_to_uint (v.f);
}

static inline uint16_t
uint_to_half (uint32_t ui)
{
    if (ui > 65504) return 0x7c00;

    return float_to_half ((float) (ui));
}

static inline float
uint_to_float (uint32_t ui)
{
    return (float) ui;
}

static inline uint32_t
uint_to_float_int (uint32_t ui)
{
    union
    {
        uint32_t i;
        float    f;
    } v;
    v.f = uint_to_float (ui);
    return v.i;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_UNPACK_H */
