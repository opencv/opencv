/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_HUF_CODING_H
#define OPENEXR_CORE_HUF_CODING_H

#include "openexr_errors.h"
#include "openexr_decode.h"

uint64_t internal_exr_huf_compress_spare_bytes (void);
uint64_t internal_exr_huf_decompress_spare_bytes (void);

exr_result_t internal_huf_compress (
    uint64_t*       encbytes,
    void*           out,
    uint64_t        outsz,
    const uint16_t* raw,
    uint64_t        nRaw,
    void*           spare,
    uint64_t        sparebytes);

exr_result_t internal_huf_decompress (
    exr_decode_pipeline_t* decode,
    const uint8_t*         compressed,
    uint64_t               nCompressed,
    uint16_t*              raw,
    uint64_t               nRaw,
    void*                  spare,
    uint64_t               sparebytes);

#endif /* OPENEXR_CORE_HUF_CODING_H */
