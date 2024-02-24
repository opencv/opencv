/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_CODING_H
#define OPENEXR_CORE_CODING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/**
 * Enum for use in a custom allocator in the encode/decode pipelines
 * (that is, so the implementor knows whether to allocate on which
 * device based on the buffer disposition).
 */
typedef enum exr_transcoding_pipeline_buffer_id
{
    EXR_TRANSCODE_BUFFER_PACKED,
    EXR_TRANSCODE_BUFFER_UNPACKED,
    EXR_TRANSCODE_BUFFER_COMPRESSED,
    EXR_TRANSCODE_BUFFER_SCRATCH1,
    EXR_TRANSCODE_BUFFER_SCRATCH2,
    EXR_TRANSCODE_BUFFER_PACKED_SAMPLES,
    EXR_TRANSCODE_BUFFER_SAMPLES
} exr_transcoding_pipeline_buffer_id_t;

/** @brief Struct for negotiating buffers when decoding/encoding
 * chunks of data.
 *
 * This is generic and meant to negotiate exr data bi-directionally,
 * in that the same structure is used for both decoding and encoding
 * chunks for read and write, respectively.
 *
 * The first half of the structure will be filled by the library, and
 * the caller is expected to fill the second half appropriately.
 */
typedef struct
{
    /**************************************************
     * Elements below are populated by the library when
     * decoding is initialized/updated and must be left
     * untouched when using the default decoder routines.
     **************************************************/

    /** Channel name.
     *
     * This is provided as a convenient reference. Do not free, this
     * refers to the internal data structure in the context.
     */
    const char* channel_name;

    /** Number of lines for this channel in this chunk.
     *
     * May be 0 or less than overall image height based on sampling
     * (i.e. when in 4:2:0 type sampling)
     */
    int32_t height;

    /** Width in pixel count.
     *
     * May be 0 or less than overall image width based on sampling
     * (i.e. 4:2:2 will have some channels have fewer values).
     */
    int32_t width;

    /** Horizontal subsampling information. */
    int32_t x_samples;
    /** Vertical subsampling information. */
    int32_t y_samples;

    /** Linear flag from channel definition (used by b44). */
    uint8_t p_linear;

    /** How many bytes per pixel this channel consumes (2 for float16,
     * 4 for float32/uint32).
     */
    int8_t bytes_per_element;

    /** Small form of exr_pixel_type_t enum (EXR_PIXEL_UINT/HALF/FLOAT). */
    uint16_t data_type;

    /**************************************************
     * Elements below must be edited by the caller
     * to control encoding/decoding.
     **************************************************/

    /** How many bytes per pixel the input is or output should be
     * (2 for float16, 4 for float32/uint32). Defaults to same
     * size as input.
     */
    int16_t user_bytes_per_element;

    /** Small form of exr_pixel_type_t enum
     * (EXR_PIXEL_UINT/HALF/FLOAT). Defaults to same type as input.
     */
    uint16_t user_data_type;

    /** Increment to get to next pixel.
     *
     * This is in bytes. Must be specified when the decode pointer is
     * specified (and always for encode).
     *
     * This is useful for implementing transcoding generically of
     * planar or interleaved data. For planar data, where the layout
     * is RRRRRGGGGGBBBBB, you can pass in 1 * bytes per component.
     */

    int32_t user_pixel_stride;

    /** When \c lines > 1 for a chunk, this is the increment used to get
     * from beginning of line to beginning of next line.
     *
     * This is in bytes. Must be specified when the decode pointer is
     * specified (and always for encode).
     */
    int32_t user_line_stride;

    /** This data member has different requirements reading vs
     * writing. When reading, if this is left as `NULL`, the channel
     * will be skipped during read and not filled in.  During a write
     * operation, this pointer is considered const and not
     * modified. To make this more clear, a union is used here.
     */
    union
    {
        uint8_t*       decode_to_ptr;
        const uint8_t* encode_from_ptr;
    };
} exr_coding_channel_info_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_CODING_H */
