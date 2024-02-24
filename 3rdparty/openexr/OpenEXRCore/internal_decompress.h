/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_DECOMPRESS_H
#define OPENEXR_CORE_DECOMPRESS_H

#include "openexr_decode.h"

/*
 * for uncompressing, we might be doing either the deep sample count
 * table or the actual pixel data so need to receive the destination
 * pointers
 */

uint64_t internal_rle_decompress (
    uint8_t* out, uint64_t outbytes, const uint8_t* src, uint64_t srcbytes);

exr_result_t internal_exr_undo_rle (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_zip (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_piz (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_pxr24 (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_b44 (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_b44a (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_dwaa (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

exr_result_t internal_exr_undo_dwab (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size);

#endif /* OPENEXR_CORE_DECOMPRESS_H */
