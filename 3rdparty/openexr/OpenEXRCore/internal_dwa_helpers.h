//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED
#define IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED

/* TODO: move to here when ready */
#include "../OpenEXR/dwaLookups.h"

/**************************************/

typedef enum _AcCompression
{
    STATIC_HUFFMAN,
    DEFLATE,
} AcCompression;

typedef enum _CompressorScheme
{
    UNKNOWN = 0,
    LOSSY_DCT,
    RLE,

    NUM_COMPRESSOR_SCHEMES
} CompressorScheme;

//
// Per-chunk compressed data sizes, one value per chunk
//

typedef enum _DataSizesSingle
{
    VERSION = 0, // Version number:
    //   0: classic
    //   1: adds "end of block" to the AC RLE

    UNKNOWN_UNCOMPRESSED_SIZE, // Size of leftover data, uncompressed.
    UNKNOWN_COMPRESSED_SIZE,   // Size of leftover data, zlib compressed.

    AC_COMPRESSED_SIZE,    // AC RLE + Huffman size
    DC_COMPRESSED_SIZE,    // DC + Deflate size
    RLE_COMPRESSED_SIZE,   // RLE + Deflate data size
    RLE_UNCOMPRESSED_SIZE, // RLE'd data size
    RLE_RAW_SIZE,          // Un-RLE'd data size

    AC_UNCOMPRESSED_COUNT, // AC RLE number of elements
    DC_UNCOMPRESSED_COUNT, // DC number of elements

    AC_COMPRESSION, // AC compression strategy
    NUM_SIZES_SINGLE
} DataSizesSingle;

static inline size_t
std_max (size_t a, size_t b)
{
    return a < b ? b : a;
}

#include "internal_dwa_simd.h"
#include "internal_dwa_channeldata.h"
#include "internal_dwa_classifier.h"
#include "internal_dwa_decoder.h"
#include "internal_dwa_encoder.h"
#include "internal_dwa_compressor.h"

#endif /* IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED */
