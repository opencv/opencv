/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_coding.h"
#include "internal_xdr.h"
#include "internal_cpuid.h"

#include "openexr_attr.h"

#include <string.h>

/**************************************/

#ifndef __F16C__
static inline void
half_to_float4 (float* out, const uint16_t* src)
{
    out[0] = half_to_float (src[0]);
    out[1] = half_to_float (src[1]);
    out[2] = half_to_float (src[2]);
    out[3] = half_to_float (src[3]);
}

static inline void
half_to_float8 (float* out, const uint16_t* src)
{
    half_to_float4 (out, src);
    half_to_float4 (out + 4, src + 4);
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX__) &&            \
    (defined(__F16C__) || defined(__GNUC__) || defined(__clang__))

#    if defined(__F16C__)
static inline void
half_to_float_buffer (float* out, const uint16_t* in, int w)
#    elif defined(__GNUC__) || defined(__clang__)
__attribute__ ((target ("f16c"))) static void
half_to_float_buffer_f16c (float* out, const uint16_t* in, int w)
#    endif
{
    while (w >= 8)
    {
        _mm256_storeu_ps (
            out, _mm256_cvtph_ps (_mm_loadu_si128 ((const __m128i*) in)));
        out += 8;
        in += 8;
        w -= 8;
    }
    // gcc < 9 does not have loadu_si64
#    if defined(__clang__) || (__GNUC__ >= 9)
    switch (w)
    {
        case 7:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in)));
            out[4] = half_to_float (in[4]);
            break;
        case 4: _mm_storeu_ps (out, _mm_cvtph_ps (_mm_loadu_si64 (in))); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
#    else
    while (w > 0)
    {
        *out++ = half_to_float (*in++);
        --w;
    }
#    endif
}

#    ifndef __F16C__
static void
half_to_float_buffer_impl (float* out, const uint16_t* in, int w)
{
    while (w >= 8)
    {
        half_to_float8 (out, in);
        out += 8;
        in += 8;
        w -= 8;
    }
    switch (w)
    {
        case 7:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            break;
        case 4: half_to_float4 (out, in); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
}

static void (*half_to_float_buffer) (float*, const uint16_t*, int) =
    &half_to_float_buffer_impl;

static inline void
choose_half_to_float_impl (void)
{
    if (has_native_half ()) half_to_float_buffer = &half_to_float_buffer_f16c;
}
#    else
/* when we explicitly compile against f16, force it in */
static inline void
choose_half_to_float_impl (void)
{}

#    endif /* F16C */

#else

static inline void
half_to_float_buffer (float* out, const uint16_t* in, int w)
{
#    if EXR_HOST_IS_NOT_LITTLE_ENDIAN
    for (int x = 0; x < w; ++x)
        out[x] = half_to_float (one_to_native16 (in[x]));
#    else
    while (w >= 8)
    {
        half_to_float8 (out, in);
        out += 8;
        in += 8;
        w -= 8;
    }
    switch (w)
    {
        case 7:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            out[6] = half_to_float (in[6]);
            break;
        case 6:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            out[5] = half_to_float (in[5]);
            break;
        case 5:
            half_to_float4 (out, in);
            out[4] = half_to_float (in[4]);
            break;
        case 4: half_to_float4 (out, in); break;
        case 3:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            out[2] = half_to_float (in[2]);
            break;
        case 2:
            out[0] = half_to_float (in[0]);
            out[1] = half_to_float (in[1]);
            break;
        case 1: out[0] = half_to_float (in[0]); break;
    }
#    endif
}

static void
choose_half_to_float_impl (void)
{}

#endif

/**************************************/

static exr_result_t
unpack_16bit_3chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        uint16_t* out = (uint16_t*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = one_to_native16 (in0[x]);
            out[1] = one_to_native16 (in1[x]);
            out[2] = one_to_native16 (in2[x]);
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[2].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        uint16_t* out = (uint16_t*) out0;

        in0 = (const uint16_t*) srcbuffer; // B
        in1 = in0 + w;                     // G
        in2 = in1 + w;                     // R

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = one_to_native16 (in2[x]);
            out[1] = one_to_native16 (in1[x]);
            out[2] = one_to_native16 (in0[x]);
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        float* out = (float*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in0[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in2[x]));
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[2].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        float* out = (float*) out0;

        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;

        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in2[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in0[x]));
            out += 3;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h;
    int             linc0, linc1, linc2;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    // planar output
    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
                            /* specialise to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out0) + x) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out1) + x) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out2) + x) = one_to_native16 (in2[x]);
#else
        memcpy (out0, in0, (size_t) (w) * sizeof (uint16_t));
        memcpy (out1, in1, (size_t) (w) * sizeof (uint16_t));
        memcpy (out2, in2, (size_t) (w) * sizeof (uint16_t));
#endif
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_3chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h;
    int             linc0, linc1, linc2;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    // planar output
    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
                            /* specialise to memcpy if we can */
        half_to_float_buffer ((float*) out0, in0, w);
        half_to_float_buffer ((float*) out1, in1, w);
        half_to_float_buffer ((float*) out2, in2, w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_3chan (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2;
    uint8_t *       out0, *out1, *out2;
    int             w, h;
    int             inc0, inc1, inc2;
    int             linc0, linc1, linc2;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    inc0  = decode->channels[0].user_pixel_stride;
    inc1  = decode->channels[1].user_pixel_stride;
    inc2  = decode->channels[2].user_pixel_stride;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;

    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        srcbuffer += w * 6; // 3 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out0 + x * inc0)) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out1 + x * inc1)) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out2 + x * inc2)) = one_to_native16 (in2[x]);
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;
    /* TODO: can do this with sse and do 2 outpixels at once */
    union
    {
        struct
        {
            uint16_t a;
            uint16_t b;
            uint16_t g;
            uint16_t r;
        };
        uint64_t allc;
    } combined;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        uint64_t* outall = (uint64_t*) out0;
        in0              = (const uint16_t*) srcbuffer;
        in1              = in0 + w;
        in2              = in1 + w;
        in3              = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            combined.a = one_to_native16 (in0[x]);
            combined.b = one_to_native16 (in1[x]);
            combined.g = one_to_native16 (in2[x]);
            combined.r = one_to_native16 (in3[x]);
            outall[x]  = combined.allc;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;
    /* TODO: can do this with sse and do 2 outpixels at once */
    union
    {
        struct
        {
            uint16_t r;
            uint16_t g;
            uint16_t b;
            uint16_t a;
        };
        uint64_t allc;
    } combined;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[3].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        uint64_t* outall = (uint64_t*) out0;
        in0              = (const uint16_t*) srcbuffer;
        in1              = in0 + w;
        in2              = in1 + w;
        in3              = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            combined.a = one_to_native16 (in0[x]);
            combined.b = one_to_native16 (in1[x]);
            combined.g = one_to_native16 (in2[x]);
            combined.r = one_to_native16 (in3[x]);
            outall[x]  = combined.allc;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_interleave (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        float* out = (float*) out0;
        in0        = (const uint16_t*) srcbuffer;
        in1        = in0 + w;
        in2        = in1 + w;
        in3        = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in3[x]));
            out[1] = half_to_float (one_to_native16 (in2[x]));
            out[2] = half_to_float (one_to_native16 (in1[x]));
            out[3] = half_to_float (one_to_native16 (in0[x]));
            out += 4;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_interleave_rev (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t*        out0;
    int             w, h;
    int             linc0;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;

    out0 = decode->channels[3].decode_to_ptr;

    /* interleaving case, we can do this! */
    for (int y = 0; y < h; ++y)
    {
        float* out = (float*) out0;
        in0        = (const uint16_t*) srcbuffer;
        in1        = in0 + w;
        in2        = in1 + w;
        in3        = in2 + w;

        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
        {
            out[0] = half_to_float (one_to_native16 (in0[x]));
            out[1] = half_to_float (one_to_native16 (in1[x]));
            out[2] = half_to_float (one_to_native16 (in2[x]));
            out[3] = half_to_float (one_to_native16 (in3[x]));
            out += 4;
        }
        out0 += linc0;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h;
    int             linc0, linc1, linc2, linc3;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    // planar output
    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        in3 = in2 + w;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
                            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out0) + x) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out1) + x) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out2) + x) = one_to_native16 (in2[x]);
        for (int x = 0; x < w; ++x)
            *(((uint16_t*) out3) + x) = one_to_native16 (in3[x]);
#else
        memcpy (out0, in0, (size_t) (w) * sizeof (uint16_t));
        memcpy (out1, in1, (size_t) (w) * sizeof (uint16_t));
        memcpy (out2, in2, (size_t) (w) * sizeof (uint16_t));
        memcpy (out3, in3, (size_t) (w) * sizeof (uint16_t));
#endif
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_half_to_float_4chan_planar (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h;
    int             linc0, linc1, linc2, linc3;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    // planar output
    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        in3 = in2 + w;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion

        half_to_float_buffer ((float*) out0, in0, w);
        half_to_float_buffer ((float*) out1, in1, w);
        half_to_float_buffer ((float*) out2, in2, w);
        half_to_float_buffer ((float*) out3, in3, w);

        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit_4chan (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t*  srcbuffer = decode->unpacked_buffer;
    const uint16_t *in0, *in1, *in2, *in3;
    uint8_t *       out0, *out1, *out2, *out3;
    int             w, h;
    int             inc0, inc1, inc2, inc3;
    int             linc0, linc1, linc2, linc3;

    w     = decode->channels[0].width;
    h     = decode->chunk.height;
    inc0  = decode->channels[0].user_pixel_stride;
    inc1  = decode->channels[1].user_pixel_stride;
    inc2  = decode->channels[2].user_pixel_stride;
    inc3  = decode->channels[3].user_pixel_stride;
    linc0 = decode->channels[0].user_line_stride;
    linc1 = decode->channels[1].user_line_stride;
    linc2 = decode->channels[2].user_line_stride;
    linc3 = decode->channels[3].user_line_stride;

    out0 = decode->channels[0].decode_to_ptr;
    out1 = decode->channels[1].decode_to_ptr;
    out2 = decode->channels[2].decode_to_ptr;
    out3 = decode->channels[3].decode_to_ptr;

    for (int y = 0; y < h; ++y)
    {
        in0 = (const uint16_t*) srcbuffer;
        in1 = in0 + w;
        in2 = in1 + w;
        in3 = in2 + w;
        srcbuffer += w * 8; // 4 * sizeof(uint16_t), avoid type conversion
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out0 + x * inc0)) = one_to_native16 (in0[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out1 + x * inc1)) = one_to_native16 (in1[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out2 + x * inc2)) = one_to_native16 (in2[x]);
        for (int x = 0; x < w; ++x)
            *((uint16_t*) (out3 + x * inc3)) = one_to_native16 (in3[x]);
        out0 += linc0;
        out1 += linc1;
        out2 += linc2;
        out3 += linc3;
    }
    return EXR_ERR_SUCCESS;
}

/**************************************/

static exr_result_t
unpack_16bit (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int            w, h, pixincrement;

    h = decode->chunk.height;
    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata        = decc->decode_to_ptr;
            w            = decc->width;
            pixincrement = decc->user_pixel_stride;
            cdata += (uint64_t) y * (uint64_t) decc->user_line_stride;
            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
            if (pixincrement == 2)
            {
                uint16_t*       tmp = (uint16_t*) cdata;
                const uint16_t* src = (const uint16_t*) srcbuffer;
                uint16_t*       end = tmp + w;

                while (tmp < end)
                    *tmp++ = one_to_native16 (*src++);
            }
            else
            {
                const uint16_t* src = (const uint16_t*) srcbuffer;
                for (int x = 0; x < w; ++x)
                {
                    *((uint16_t*) cdata) = one_to_native16 (*src++);
                    cdata += pixincrement;
                }
            }
#else
            if (pixincrement == 2)
            {
                memcpy (cdata, srcbuffer, (size_t) (w) *2);
            }
            else
            {
                const uint16_t* src = (const uint16_t*) srcbuffer;
                for (int x = 0; x < w; ++x)
                {
                    *((uint16_t*) cdata) = *src++;
                    cdata += pixincrement;
                }
            }
#endif
            srcbuffer += w * 2;
        }
    }
    return EXR_ERR_SUCCESS;
}

//static exr_result_t unpack_32bit_3chan (exr_decode_pipeline_t* decode);
//static exr_result_t unpack_32bit_4chan (exr_decode_pipeline_t* decode);

static exr_result_t
unpack_32bit (exr_decode_pipeline_t* decode)
{
    /* we know we're unpacking all the channels and there is no subsampling */
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int64_t        w, h, pixincrement;
    int            chans = decode->channel_count;

    h = (int64_t) decode->chunk.height;

    for (int64_t y = 0; y < h; ++y)
    {
        for (int c = 0; c < chans; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata        = decc->decode_to_ptr;
            w            = decc->width;
            pixincrement = decc->user_pixel_stride;
            cdata += y * (int64_t) decc->user_line_stride;
            /* specialize to memcpy if we can */
#if EXR_HOST_IS_NOT_LITTLE_ENDIAN
            if (pixincrement == 4)
            {
                uint32_t*       tmp = (uint32_t*) cdata;
                const uint32_t* src = (const uint32_t*) srcbuffer;
                uint32_t*       end = tmp + w;

                while (tmp < end)
                    *tmp++ = le32toh (*src++);
            }
            else
            {
                const uint32_t* src = (const uint32_t*) srcbuffer;
                for (int64_t x = 0; x < w; ++x)
                {
                    *((uint32_t*) cdata) = le32toh (*src++);
                    cdata += pixincrement;
                }
            }
#else
            if (pixincrement == 4)
            {
                memcpy (cdata, srcbuffer, (size_t) (w) *4);
            }
            else
            {
                const uint32_t* src = (const uint32_t*) srcbuffer;
                for (int64_t x = 0; x < w; ++x)
                {
                    *((uint32_t*) cdata) = *src++;
                    cdata += pixincrement;
                }
            }
#endif
            srcbuffer += w * 4;
        }
    }
    return EXR_ERR_SUCCESS;
}

#define UNPACK_SAMPLES(samps)                                                  \
    switch (decc->data_type)                                                   \
    {                                                                          \
        case EXR_PIXEL_HALF:                                                   \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint16_t*) cdata) = unaligned_load16 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint16_t cval = unaligned_load16 (src);                \
                        ++src;                                                 \
                        *((float*) cdata) = half_to_float (cval);              \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    const uint16_t* src = (const uint16_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint16_t cval = unaligned_load16 (src);                \
                        ++src;                                                 \
                        *((uint32_t*) cdata) = half_to_uint (cval);            \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        case EXR_PIXEL_FLOAT:                                                  \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint16_t*) cdata) = float_to_half_int (fint);       \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint32_t*) cdata) = unaligned_load32 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint32_t*) cdata) = float_to_uint_int (fint);       \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        case EXR_PIXEL_UINT:                                                   \
            switch (decc->user_data_type)                                      \
            {                                                                  \
                case EXR_PIXEL_HALF: {                                         \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((uint16_t*) cdata) = uint_to_half (fint);            \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_FLOAT: {                                        \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        uint32_t fint = unaligned_load32 (src);                \
                        ++src;                                                 \
                        *((float*) cdata) = uint_to_float (fint);              \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                case EXR_PIXEL_UINT: {                                         \
                    const uint32_t* src = (const uint32_t*) srcbuffer;         \
                    for (int s = 0; s < samps; ++s)                            \
                    {                                                          \
                        *((uint32_t*) cdata) = unaligned_load32 (src);         \
                        ++src;                                                 \
                        cdata += ubpc;                                         \
                    }                                                          \
                    break;                                                     \
                }                                                              \
                default: return EXR_ERR_INVALID_ARGUMENT;                      \
            }                                                                  \
            break;                                                             \
        default: return EXR_ERR_INVALID_ARGUMENT;                              \
    }

static exr_result_t
generic_unpack (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer = decode->unpacked_buffer;
    uint8_t*       cdata;
    int            w, bpc, ubpc;

    for (int y = 0; y < decode->chunk.height; ++y)
    {
        int cury = y + decode->chunk.start_y;

        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc = (decode->channels + c);

            cdata = decc->decode_to_ptr;
            w     = decc->width;
            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_pixel_stride;

            if (decc->y_samples > 1)
            {
                if ((cury % decc->y_samples) != 0) continue;
                if (cdata)
                    cdata +=
                        ((uint64_t) (y / decc->y_samples) *
                         (uint64_t) decc->user_line_stride);
                else
                {
                    srcbuffer += w * bpc;
                    continue;
                }
            }
            else if (cdata)
            {
                cdata += ((uint64_t) y) * ((uint64_t) decc->user_line_stride);
            }
            else
            {
                srcbuffer += w * bpc;
                continue;
            }

            UNPACK_SAMPLES (w)
            srcbuffer += w * bpc;
        }
    }
    return EXR_ERR_SUCCESS;
}

static exr_result_t
generic_unpack_deep_pointers (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer  = decode->unpacked_buffer;
    const int32_t* sampbuffer = decode->sample_count_table;
    void**         pdata;
    int            w, h, bpc, ubpc;

    w = decode->chunk.width;
    h = decode->chunk.height;

    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc      = (decode->channels + c);
            int32_t                    prevsamps = 0;
            size_t                     pixstride;
            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_bytes_per_element;
            pdata = (void**) decc->decode_to_ptr;

            if (!pdata)
            {
                prevsamps = 0;
                if ((decode->decode_flags &
                     EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    for (int x = 0; x < w; ++x)
                        prevsamps += sampbuffer[x];
                }
                else
                    prevsamps = sampbuffer[w - 1];
                srcbuffer += ((size_t) bpc) * ((size_t) prevsamps);
                continue;
            }

            pdata += ((size_t) y) *
                     (((size_t) decc->user_line_stride) / sizeof (void*));
            pixstride = ((size_t) decc->user_pixel_stride) / sizeof (void*);

            for (int x = 0; x < w; ++x)
            {
                void*   outpix = *pdata;
                int32_t samps  = sampbuffer[x];
                if (0 == (decode->decode_flags &
                          EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    int32_t tmp = samps - prevsamps;
                    prevsamps   = samps;
                    samps       = tmp;
                }

                pdata += pixstride;
                if (outpix)
                {
                    uint8_t* cdata = outpix;

                    UNPACK_SAMPLES (samps)
                }
                srcbuffer += ((size_t) bpc) * ((size_t) samps);
            }
        }
        sampbuffer += w;
    }
    return EXR_ERR_SUCCESS;
}

static exr_result_t
generic_unpack_deep (exr_decode_pipeline_t* decode)
{
    const uint8_t* srcbuffer  = decode->unpacked_buffer;
    const int32_t* sampbuffer = decode->sample_count_table;
    uint8_t*       cdata;
    int            w, h, bpc, ubpc;
    size_t         totsamps = 0;

    w = decode->chunk.width;
    h = decode->chunk.height;

    for (int y = 0; y < h; ++y)
    {
        for (int c = 0; c < decode->channel_count; ++c)
        {
            exr_coding_channel_info_t* decc      = (decode->channels + c);
            int32_t                    prevsamps = 0;

            int incr_tot = ((c + 1) == decode->channel_count);

            bpc   = decc->bytes_per_element;
            ubpc  = decc->user_bytes_per_element;
            cdata = decc->decode_to_ptr;

            if (!cdata)
            {
                prevsamps = 0;
                if ((decode->decode_flags &
                     EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    for (int x = 0; x < w; ++x)
                        prevsamps += sampbuffer[x];
                }
                else
                    prevsamps = sampbuffer[w - 1];

                srcbuffer += ((size_t) bpc) * ((size_t) prevsamps);

                if (incr_tot) totsamps += (size_t) prevsamps;

                continue;
            }

            cdata += totsamps * ((size_t) ubpc);

            for (int x = 0; x < w; ++x)
            {
                int32_t samps = sampbuffer[x];
                if (0 == (decode->decode_flags &
                          EXR_DECODE_SAMPLE_COUNTS_AS_INDIVIDUAL))
                {
                    int32_t tmp = samps - prevsamps;
                    prevsamps   = samps;
                    samps       = tmp;
                }

                UNPACK_SAMPLES (samps)

                srcbuffer += ((size_t) bpc) * ((size_t) samps);
                if (incr_tot) totsamps += (size_t) samps;
            }
        }
        sampbuffer += w;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

internal_exr_unpack_fn
internal_exr_match_decode (
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
    int                    simplineoff)
{
    static int init_cpu_check = 1;
    if (init_cpu_check)
    {
        choose_half_to_float_impl ();
        init_cpu_check = 0;
    }

    if (isdeep)
    {
        if ((decode->decode_flags & EXR_DECODE_NON_IMAGE_DATA_AS_POINTERS))
            return &generic_unpack_deep_pointers;
        return &generic_unpack_deep;
    }

    if (hastypechange > 0)
    {
        /* other optimizations would not be difficult, but this will
         * be the common one (where on encode / pack we want to do the
         * opposite) */
        if (sametype == (int) EXR_PIXEL_HALF &&
            sameouttype == (int) EXR_PIXEL_FLOAT)
        {
            if (simpinterleave > 0)
            {
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_interleave;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_interleave;
            }

            if (simpinterleaverev > 0)
            {
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_interleave_rev;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_interleave_rev;
            }

            if (sameoutinc == 4)
            {
                if (decode->channel_count == 4)
                    return &unpack_half_to_float_4chan_planar;
                if (decode->channel_count == 3)
                    return &unpack_half_to_float_3chan_planar;
            }
        }

        return &generic_unpack;
    }

    if (hassampling || chanstofill != decode->channel_count || samebpc <= 0 ||
        sameoutbpc <= 0)
        return &generic_unpack;

    (void) chanstounpack;
    (void) simplineoff;

    if (samebpc == 2)
    {
        if (simpinterleave > 0)
        {
            if (decode->channel_count == 4)
                return &unpack_16bit_4chan_interleave;
            if (decode->channel_count == 3)
                return &unpack_16bit_3chan_interleave;
        }

        if (simpinterleaverev > 0)
        {
            if (decode->channel_count == 4)
                return &unpack_16bit_4chan_interleave_rev;
            if (decode->channel_count == 3)
                return &unpack_16bit_3chan_interleave_rev;
        }

        if (sameoutinc == 2)
        {
            if (decode->channel_count == 4) return &unpack_16bit_4chan_planar;
            if (decode->channel_count == 3) return &unpack_16bit_3chan_planar;
        }

        if (decode->channel_count == 4) return &unpack_16bit_4chan;
        if (decode->channel_count == 3) return &unpack_16bit_3chan;

        return &unpack_16bit;
    }

    if (samebpc == 4)
    {
        //if (decode->channel_count == 4) return &unpack_32bit_4chan;
        //if (decode->channel_count == 3) return &unpack_32bit_3chan;
        return &unpack_32bit;
    }

    return &generic_unpack;
}
