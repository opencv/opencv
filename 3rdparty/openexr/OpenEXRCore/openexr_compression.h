/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_COMPRESSION_H
#define OPENEXR_CORE_COMPRESSION_H

#include "openexr_context.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @file */

/** Computes a buffer that will be large enough to hold the compressed
 * data. This may include some extra padding for headers / scratch */
EXR_EXPORT
size_t exr_compress_max_buffer_size (size_t in_bytes);

/** Compresses a buffer using a zlib style compression.
 *
 * If the level is -1, will use the default compression set to the library
 * \ref exr_set_default_zip_compression_level
 * data. This may include some extra padding for headers / scratch */
EXR_EXPORT
exr_result_t exr_compress_buffer (
    exr_const_context_t ctxt,
    int                 level,
    const void*         in,
    size_t              in_bytes,
    void*               out,
    size_t              out_bytes_avail,
    size_t*             actual_out);

EXR_EXPORT
exr_result_t exr_uncompress_buffer (
    exr_const_context_t ctxt,
    const void*         in,
    size_t              in_bytes,
    void*               out,
    size_t              out_bytes_avail,
    size_t*             actual_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENEXR_CORE_COMPRESSION_H */
