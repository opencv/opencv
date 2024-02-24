/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "openexr_encode.h"

#include "internal_coding.h"
#include "internal_xdr.h"

/**************************************/

static exr_result_t
default_pack_deep (exr_encode_pipeline_t* encode)
{
    (void) encode;
    return EXR_ERR_INVALID_ARGUMENT;
}

static exr_result_t
default_pack (exr_encode_pipeline_t* encode)
{
    uint8_t*       dstbuffer = encode->packed_buffer;
    const uint8_t* cdata;
    int            w, bpc, pixincrement;
    uint64_t       packed_bytes = 0;
    uint64_t       chan_bytes   = 0;

    for (int y = 0; y < encode->chunk.height; ++y)
    {
        int cury = y + encode->chunk.start_y;

        for (int c = 0; c < encode->channel_count; ++c)
        {
            exr_coding_channel_info_t* encc = (encode->channels + c);

            if (encc->height == 0) continue;

            cdata      = encc->encode_from_ptr;
            w          = encc->width;
            bpc        = encc->bytes_per_element;
            chan_bytes = (uint64_t) (w) * (uint64_t) (bpc);

            if (encc->y_samples > 1)
            {
                if ((cury % encc->y_samples) != 0) continue;
                if (cdata)
                    cdata +=
                        ((uint64_t) (y / encc->y_samples) *
                         (uint64_t) encc->user_line_stride);
            }
            else { cdata += (uint64_t) y * (uint64_t) encc->user_line_stride; }

            pixincrement = encc->user_pixel_stride;
            switch (encc->data_type)
            {
                case EXR_PIXEL_HALF:
                    switch (encc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            uint16_t* dst = (uint16_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                unaligned_store16 (
                                    dst, *((const uint16_t*) cdata));
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            uint16_t* dst = (uint16_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                uint16_t cval =
                                    float_to_half (*((const float*) cdata));
                                unaligned_store16 (dst, cval);
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            uint16_t* dst = (uint16_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                uint16_t cval =
                                    uint_to_half (*((const uint32_t*) cdata));
                                unaligned_store16 (dst, cval);
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_FLOAT:
                    switch (encc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                uint32_t fint = half_to_float_int (
                                    *((const uint16_t*) cdata));
                                unaligned_store32 (dst, fint);
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                unaligned_store32 (
                                    dst, *((const uint32_t*) cdata));
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                uint32_t fint = uint_to_float_int (
                                    *((const uint32_t*) cdata));
                                unaligned_store32 (dst, fint);
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                case EXR_PIXEL_UINT:
                    switch (encc->user_data_type)
                    {
                        case EXR_PIXEL_HALF: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                uint16_t tmp = *((const uint16_t*) cdata);
                                unaligned_store32 (dst, half_to_uint (tmp));
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_FLOAT: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                float tmp = *((const float*) cdata);
                                unaligned_store32 (dst, float_to_uint (tmp));
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        case EXR_PIXEL_UINT: {
                            uint32_t* dst = (uint32_t*) dstbuffer;
                            for (int x = 0; x < w; ++x)
                            {
                                unaligned_store32 (
                                    dst, *((const uint32_t*) cdata));
                                ++dst;
                                cdata += pixincrement;
                            }
                            break;
                        }
                        default: return EXR_ERR_INVALID_ARGUMENT;
                    }
                    break;
                default: return EXR_ERR_INVALID_ARGUMENT;
            }
            dstbuffer += chan_bytes;
            packed_bytes += chan_bytes;
        }
    }

    encode->packed_bytes = packed_bytes;

    return EXR_ERR_SUCCESS;
}

internal_exr_pack_fn
internal_exr_match_encode (exr_encode_pipeline_t* encode, int isdeep)
{
    (void) encode;
    if (isdeep) return &default_pack_deep;

    return &default_pack;
}
