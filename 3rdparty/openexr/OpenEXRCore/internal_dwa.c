/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright (c) DreamWorks Animation LLC and Contributors of the OpenEXR Project
*/

/*
 *---------------------------------------------------
 *
 * class DwaCompressor -- Store lossy RGB data by quantizing
 *                          DCT components.
 *
 * First, we try and figure out what compression strategy to take
 * based in channel name. For RGB channels, we want a lossy method
 * described below. But, if we have alpha, we should do something
 * different (and probably using RLE). If we have depth, or velocity,
 * or something else, just fall back to ZIP. The rules for deciding
 * which strategy to use are setup in initializeDefaultChannelRules().
 * When writing a file, the relevant rules needed to decode are written
 * into the start of the data block, making a self-contained file.
 * If initializeDefaultChannelRules() doesn't quite suite your naming
 * conventions, you can adjust the rules without breaking decoder
 * compatibility.
 *
 * If we're going to lossy compress R, G, or B channels, it's easier
 * to toss bits in a more perceptual uniform space. One could argue
 * at length as to what constitutes perceptually uniform, especially
 * when storing either scene/input/focal plane referred and output referred
 * data.
 *
 * We'll compromise. For values <= 1, we use a traditional power function
 * (without any of that straight-line business at the bottom). For values > 1,
 * we want something more like a log function, since power functions blow
 * up. At 1, we want a smooth blend between the functions. So, we use a
 * piecewise function that does just that - see dwaLookups.cpp for
 * a little more detail.
 *
 * Also, if we find that we have R, G, and B channels from the same layer,
 * we can get a bit more compression efficiency by transforming to a Y'CbCr
 * space. We use the 709 transform, but with Cb,Cr = 0 for an input of
 * (0, 0, 0), instead of the traditional Cb,Cr = .5. Shifting the zero point
 * makes no sense with large range data. Transforms are done to from
 * the perceptual space data, not the linear-light space data (R'G'B' ->
 * (Y'CbCr, not RGB -> YCbCr).
 *
 * Next, we forward DCT the data. This is done with a floating
 * point DCT, as we don't really have control over the src range. The
 * resulting values are dropped to half-float precision.
 *
 * Now, we need to quantize. Quantization departs from the usual way
 * of dividing and rounding. Instead, we start with some floating
 * point "base-error" value. From this, we can derive quantization
 * error for each DCT component. Take the standard JPEG quantization
 * tables and normalize them by the smallest value. Then, multiply
 * the normalized quant tables by our base-error value. This gives
 * a range of errors for each DCT component.
 *
 * For each DCT component, we want to find a quantized value that
 * is within +- the per-component error. Pick the quantized value
 * that has the fewest bits set in its' binary representation.
 * Brute-forcing the search would make for extremely inefficient
 * compression. Fortunately, we can precompute a table to assist
 * with this search.
 *
 * For each 16-bit float value, there are at most 15 other values with
 * fewer bits set. We can precompute these values in a compact form, since
 * many source values have far fewer that 15 possible quantized values.
 * Now, instead of searching the entire range +- the component error,
 * we can just search at most 15 quantization candidates. The search can
 * be accelerated a bit more by sorting the candidates by the
 * number of bits set, in increasing order. Then, the search can stop
 * once a candidate is found w/i the per-component quantization
 * error range.
 *
 * The quantization strategy has the side-benefit that there is no
 * de-quantization step upon decode, so we don't bother recording
 * the quantization table.
 *
 * Ok. So we now have quantized values. Time for entropy coding. We
 * can use either static Huffman or zlib/DEFLATE. The static Huffman
 * is more efficient at compacting data, but can have a greater
 * overhead, especially for smaller tile/strip sizes.
 *
 * There is some additional fun, like ZIP compressing the DC components
 * instead of Huffman/zlib, which helps make things slightly smaller.
 *
 * Compression level is controlled by setting an int/float/double attribute
 * on the header named "dwaCompressionLevel". This is a thinly veiled name for
 * the "base-error" value mentioned above. The "base-error" is just
 * dwaCompressionLevel / 100000. The default value of 45.0 is generally
 * pretty good at generating "visually lossless" values at reasonable
 * data rates. Setting dwaCompressionLevel to 0 should result in no additional
 * quantization at the quantization stage (though there may be
 * quantization in practice at the CSC/DCT steps). But if you really
 * want lossless compression, there are pleanty of other choices
 * of compressors ;)
 *
 * When dealing with FLOAT source buffers, we first quantize the source
 * to HALF and continue down as we would for HALF source.
 *
 *---------------------------------------------------
 */

#include "internal_compress.h"
#include "internal_decompress.h"
#include "internal_memory.h"
#include "internal_coding.h"

#include "openexr_compression.h"

#include "internal_cpuid.h"
#include "internal_huf.h"
#include "internal_structs.h"
#include "internal_xdr.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "OpenEXRConfigInternal.h"

#include "internal_dwa_helpers.h"

/**************************************/

exr_result_t
internal_exr_apply_dwaa (exr_encode_pipeline_t* encode)
{
    exr_result_t  rv;
    DwaCompressor dwaa;
    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        internal_exr_huf_compress_spare_bytes ());
    if (rv == EXR_ERR_SUCCESS)
    {
        rv = DwaCompressor_construct (&dwaa, DEFLATE, encode, NULL);
        if (rv == EXR_ERR_SUCCESS) rv = DwaCompressor_compress (&dwaa);

        DwaCompressor_destroy (&dwaa);
    }

    return rv;
}

/**************************************/

exr_result_t
internal_exr_apply_dwab (exr_encode_pipeline_t* encode)
{
    exr_result_t  rv;
    DwaCompressor dwab;

    rv = internal_encode_alloc_buffer (
        encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(encode->scratch_buffer_1),
        &(encode->scratch_alloc_size_1),
        internal_exr_huf_compress_spare_bytes ());
    if (rv == EXR_ERR_SUCCESS)
    {
        rv = DwaCompressor_construct (&dwab, STATIC_HUFFMAN, encode, NULL);
        if (rv == EXR_ERR_SUCCESS) rv = DwaCompressor_compress (&dwab);

        DwaCompressor_destroy (&dwab);
    }

    return rv;
}

exr_result_t
internal_exr_undo_dwaa (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size)
{
    exr_result_t  rv;
    DwaCompressor dwaa;

    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        internal_exr_huf_decompress_spare_bytes ());
    if (rv == EXR_ERR_SUCCESS)
    {
        rv = DwaCompressor_construct (&dwaa, STATIC_HUFFMAN, NULL, decode);
        if (rv == EXR_ERR_SUCCESS)
            rv = DwaCompressor_uncompress (
                &dwaa,
                compressed_data,
                comp_buf_size,
                uncompressed_data,
                uncompressed_size);

        DwaCompressor_destroy (&dwaa);
    }

    return rv;
}

exr_result_t
internal_exr_undo_dwab (
    exr_decode_pipeline_t* decode,
    const void*            compressed_data,
    uint64_t               comp_buf_size,
    void*                  uncompressed_data,
    uint64_t               uncompressed_size)
{
    exr_result_t  rv;
    DwaCompressor dwaa;

    rv = internal_decode_alloc_buffer (
        decode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(decode->scratch_buffer_1),
        &(decode->scratch_alloc_size_1),
        internal_exr_huf_decompress_spare_bytes ());
    if (rv == EXR_ERR_SUCCESS)
    {
        rv = DwaCompressor_construct (&dwaa, STATIC_HUFFMAN, NULL, decode);
        if (rv == EXR_ERR_SUCCESS)
            rv = DwaCompressor_uncompress (
                &dwaa,
                compressed_data,
                comp_buf_size,
                uncompressed_data,
                uncompressed_size);

        DwaCompressor_destroy (&dwaa);
    }

    return rv;
}
