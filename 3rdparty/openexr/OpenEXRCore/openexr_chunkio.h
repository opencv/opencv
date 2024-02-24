/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_CHUNKIO_H
#define OPENEXR_CORE_CHUNKIO_H

#include "openexr_part.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/**
 * Struct describing raw data information about a chunk.
 *
 * A chunk is the generic term for a pixel data block in an EXR file,
 * as described in the OpenEXR File Layout documentation. This is
 * common between all different forms of data that can be stored.
 */
typedef struct
{
    int32_t idx;

    /** For tiles, this is the tilex; for scans it is the x. */
    int32_t start_x;
    /** For tiles, this is the tiley; for scans it is the scanline y. */
    int32_t start_y;
    int32_t height; /**< For this chunk. */
    int32_t width;  /**< For this chunk. */

    uint8_t level_x; /**< For tiled files. */
    uint8_t level_y; /**< For tiled files. */

    uint8_t type;
    uint8_t compression;

    uint64_t data_offset;
    uint64_t packed_size;
    uint64_t unpacked_size;

    uint64_t sample_count_data_offset;
    uint64_t sample_count_table_size;
} exr_chunk_info_t;

EXR_EXPORT
exr_result_t exr_read_scanline_chunk_info (
    exr_const_context_t ctxt, int part_index, int y, exr_chunk_info_t* cinfo);

EXR_EXPORT
exr_result_t exr_read_tile_chunk_info (
    exr_const_context_t ctxt,
    int                 part_index,
    int                 tilex,
    int                 tiley,
    int                 levelx,
    int                 levely,
    exr_chunk_info_t*   cinfo);

/** Read the packed data block for a chunk.
 *
 * This assumes that the buffer pointed to by @p packed_data is
 * large enough to hold the chunk block info packed_size bytes.
 */
EXR_EXPORT
exr_result_t exr_read_chunk (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    void*                   packed_data);

/**
 * Read chunk for deep data.
 *
 * This allows one to read the packed data, the sample count data, or both.
 * \c exr_read_chunk also works to read deep data packed data,
 * but this is a routine to get the sample count table and the packed
 * data in one go, or if you want to pre-read the sample count data,
 * you can get just that buffer.
 */
EXR_EXPORT
exr_result_t exr_read_deep_chunk (
    exr_const_context_t     ctxt,
    int                     part_index,
    const exr_chunk_info_t* cinfo,
    void*                   packed_data,
    void*                   sample_data);

/**************************************/

/** Initialize a \c exr_chunk_info_t structure when encoding scanline
 * data (similar to read but does not do anything with a chunk
 * table).
 */
EXR_EXPORT
exr_result_t exr_write_scanline_chunk_info (
    exr_context_t ctxt, int part_index, int y, exr_chunk_info_t* cinfo);

/** Initialize a \c exr_chunk_info_t structure when encoding tiled data
 * (similar to read but does not do anything with a chunk table).
 */
EXR_EXPORT
exr_result_t exr_write_tile_chunk_info (
    exr_context_t     ctxt,
    int               part_index,
    int               tilex,
    int               tiley,
    int               levelx,
    int               levely,
    exr_chunk_info_t* cinfo);

/**
 * @p y must the appropriate starting y for the specified chunk.
 */
EXR_EXPORT
exr_result_t exr_write_scanline_chunk (
    exr_context_t ctxt,
    int           part_index,
    int           y,
    const void*   packed_data,
    uint64_t      packed_size);

/**
 * @p y must the appropriate starting y for the specified chunk.
 */
EXR_EXPORT
exr_result_t exr_write_deep_scanline_chunk (
    exr_context_t ctxt,
    int           part_index,
    int           y,
    const void*   packed_data,
    uint64_t      packed_size,
    uint64_t      unpacked_size,
    const void*   sample_data,
    uint64_t      sample_data_size);

EXR_EXPORT
exr_result_t exr_write_tile_chunk (
    exr_context_t ctxt,
    int           part_index,
    int           tilex,
    int           tiley,
    int           levelx,
    int           levely,
    const void*   packed_data,
    uint64_t      packed_size);

EXR_EXPORT
exr_result_t exr_write_deep_tile_chunk (
    exr_context_t ctxt,
    int           part_index,
    int           tilex,
    int           tiley,
    int           levelx,
    int           levely,
    const void*   packed_data,
    uint64_t      packed_size,
    uint64_t      unpacked_size,
    const void*   sample_data,
    uint64_t      sample_data_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_CHUNKIO_H */
