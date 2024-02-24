//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED
#    error "only include internal_dwa_helpers.h"
#endif

//
// 'class' for the LOSSY_DCT decoder classes
//

typedef struct _LossyDctDecoder
{
    //
    // if NATIVE and XDR are really the same values, we can
    // skip some processing and speed things along
    //

    //
    // Counts of how many items have been packed into the
    // AC and DC buffers
    //

    uint64_t _packedAcCount;
    uint64_t _packedDcCount;

    //
    // AC and DC buffers to pack
    //

    uint8_t* _packedAc;
    uint8_t* _packedAcEnd;
    uint8_t* _packedDc;

    uint64_t _remDcCount;

    //
    // half -> half LUT to transform from nonlinear to linear
    //

    const uint16_t* _toLinear;

    //
    // image dimensions
    //

    int _width;
    int _height;

    DctCoderChannelData* _channel_decode_data[3];
    int                  _channel_decode_data_count;
    uint8_t              _pad[4];
} LossyDctDecoder;

static exr_result_t LossyDctDecoder_base_construct (
    LossyDctDecoder* d,
    uint8_t*         packedAc,
    uint8_t*         packedAcEnd,
    uint8_t*         packedDc,
    uint64_t         remDcCount,
    const uint16_t*  toLinear,
    int              width,
    int              height);

static exr_result_t LossyDctDecoder_construct (
    LossyDctDecoder*     d,
    DctCoderChannelData* rowPtrs,
    uint8_t*             packedAc,
    uint8_t*             packedAcEnd,
    uint8_t*             packedDc,
    uint64_t             remDcCount,
    const uint16_t*      toLinear,
    int                  width,
    int                  height);

static exr_result_t LossyDctDecoderCsc_construct (
    LossyDctDecoder*     d,
    DctCoderChannelData* rowPtrsR,
    DctCoderChannelData* rowPtrsG,
    DctCoderChannelData* rowPtrsB,
    uint8_t*             packedAc,
    uint8_t*             packedAcEnd,
    uint8_t*             packedDc,
    uint64_t             remDcCount,
    const uint16_t*      toLinear,
    int                  width,
    int                  height);

static exr_result_t LossyDctDecoder_execute (
    void* (*alloc_fn) (size_t), void (*free_fn) (void*), LossyDctDecoder* d);

//
// Un-RLE the packed AC components into
// a half buffer. The half block should
// be the full 8x8 block (in zig-zag order
// still), not the first AC component.
//
// currAcComp is advanced as bytes are decoded.
//
// This returns the index of the last non-zero
// value in the buffer - with the index into zig zag
// order data. If we return 0, we have DC only data.
//
static int LossyDctDecoder_unRleAc (
    LossyDctDecoder* d,
    int*             lastNonZero,
    uint16_t**       currAcComp,
    uint16_t*        acBufferEnd,
    uint16_t*        halfZigBlock);

//
// Used to decode a single channel of LOSSY_DCT data.
//
exr_result_t
LossyDctDecoder_construct (
    LossyDctDecoder*     d,
    DctCoderChannelData* rowPtrs,
    uint8_t*             packedAc,
    uint8_t*             packedAcEnd,
    uint8_t*             packedDc,
    uint64_t             remDcCount,
    const uint16_t*      toLinear,
    int                  width,
    int                  height)
{
    exr_result_t rv;
    //
    // toLinear is a half-float LUT to convert the encoded values
    // back to linear light. If you want to skip this step, pass
    // in NULL here.
    //

    rv = LossyDctDecoder_base_construct (
        d, packedAc, packedAcEnd, packedDc, remDcCount, toLinear, width, height);

    d->_channel_decode_data[0]    = rowPtrs;
    d->_channel_decode_data_count = 1;

    return rv;
}

/**************************************/

//
// Used to decode 3 channels of LOSSY_DCT data that
// are grouped together and color space converted.
//
//
// toLinear is a half-float LUT to convert the encoded values
// back to linear light. If you want to skip this step, pass
// in NULL here.
//
exr_result_t
LossyDctDecoderCsc_construct (
    LossyDctDecoder*     d,
    DctCoderChannelData* rowPtrsR,
    DctCoderChannelData* rowPtrsG,
    DctCoderChannelData* rowPtrsB,
    uint8_t*             packedAc,
    uint8_t*             packedAcEnd,
    uint8_t*             packedDc,
    uint64_t             remDcCount,
    const uint16_t*      toLinear,
    int                  width,
    int                  height)
{
    exr_result_t rv;
    rv = LossyDctDecoder_base_construct (
        d, packedAc, packedAcEnd, packedDc, remDcCount, toLinear, width, height);
    if (rv != EXR_ERR_SUCCESS) return rv;

    d->_channel_decode_data[0]    = rowPtrsR;
    d->_channel_decode_data[1]    = rowPtrsG;
    d->_channel_decode_data[2]    = rowPtrsB;
    d->_channel_decode_data_count = 3;

    return rv;
}

/**************************************/

exr_result_t
LossyDctDecoder_base_construct (
    LossyDctDecoder* d,
    uint8_t*         packedAc,
    uint8_t*         packedAcEnd,
    uint8_t*         packedDc,
    uint64_t         remDcCount,
    const uint16_t*  toLinear,
    int              width,
    int              height)
{
    d->_packedAcCount = 0;
    d->_packedDcCount = 0;
    d->_packedAc      = packedAc;
    d->_packedAcEnd   = packedAcEnd;
    d->_packedDc      = packedDc;
    d->_remDcCount    = remDcCount;
    d->_toLinear      = toLinear;
    d->_width         = width;
    d->_height        = height;
    if (d->_toLinear == NULL) d->_toLinear = dwaCompressorNoOp;

    //d->_isNativeXdr = GLOBAL_SYSTEM_LITTLE_ENDIAN;

    d->_channel_decode_data[0]    = NULL;
    d->_channel_decode_data[1]    = NULL;
    d->_channel_decode_data[2]    = NULL;
    d->_channel_decode_data_count = 0;

    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
LossyDctDecoder_execute (
    void* (*alloc_fn) (size_t), void (*free_fn) (void*), LossyDctDecoder* d)
{
    exr_result_t         rv;
    int                  numComp = d->_channel_decode_data_count;
    DctCoderChannelData* chanData[3];
    int                  lastNonZero = 0;
    int                  numBlocksX  = (d->_width + 7) / 8;
    int                  numBlocksY = (d->_height + 7) / 8;
    int                  leftoverX  = d->_width - (numBlocksX - 1) * 8;
    int                  leftoverY  = d->_height - (numBlocksY - 1) * 8;

    int numFullBlocksX = d->_width / 8;

    uint16_t* currAcComp = (uint16_t*) (d->_packedAc);
    uint16_t* acCompEnd  = (uint16_t*) (d->_packedAcEnd);
    uint16_t* currDcComp[3];
    uint8_t*  rowBlockHandle;
    uint16_t* rowBlock[3];

    if (d->_remDcCount < ((uint64_t)numComp * (uint64_t)numBlocksX * (uint64_t)numBlocksY))
    {
        return EXR_ERR_CORRUPT_CHUNK;
    }

    for (int chan = 0; chan < numComp; ++chan)
    {
        chanData[chan] = d->_channel_decode_data[chan];
    }

    //
    // Allocate a temp aligned buffer to hold a rows worth of full
    // 8x8 half-float blocks
    //

    rowBlockHandle = alloc_fn (
        (size_t) numComp * (size_t) numBlocksX * 64 * sizeof (uint16_t) +
        _SSE_ALIGNMENT);
    if (!rowBlockHandle) return EXR_ERR_OUT_OF_MEMORY;

    rowBlock[0] = (uint16_t*) rowBlockHandle;

    for (int i = 0; i < _SSE_ALIGNMENT; ++i)
    {
        if (((uintptr_t) (rowBlockHandle + i) & _SSE_ALIGNMENT_MASK) == 0)
            rowBlock[0] = (uint16_t*) (rowBlockHandle + i);
    }

    for (int comp = 1; comp < numComp; ++comp)
        rowBlock[comp] = rowBlock[comp - 1] + numBlocksX * 64;

    //
    // Pack DC components together by common plane, so we can get
    // a little more out of differencing them. We'll always have
    // one component per block, so we can computed offsets.
    //

    currDcComp[0] = (uint16_t*) d->_packedDc;
    for (int comp = 1; comp < numComp; ++comp)
        currDcComp[comp] = currDcComp[comp - 1] + numBlocksX * numBlocksY;

    for (int blocky = 0; blocky < numBlocksY; ++blocky)
    {
        int maxY = 8, maxX = 8;
        if (blocky == numBlocksY - 1) maxY = leftoverY;

        for (int blockx = 0; blockx < numBlocksX; ++blockx)
        {
            uint8_t blockIsConstant = DWA_CLASSIFIER_TRUE;

            if (blockx == numBlocksX - 1) maxX = leftoverX;

            //
            // If we can detect that the block is constant values
            // (all components only have DC values, and all AC is 0),
            // we can do everything only on 1 value, instead of all
            // 64.
            //
            // This won't really help for regular images, but it is
            // meant more for layers with large swaths of black
            //
            for (int comp = 0; comp < numComp; ++comp)
            {
                uint16_t* halfZigData = chanData[comp]->_halfZigData;
                float*    dctData     = chanData[comp]->_dctData;
                //
                // DC component is stored separately
                //

#ifdef IMF_HAVE_SSE2
                {
                    __m128i* dst = (__m128i*) halfZigData;

                    dst[7] = _mm_setzero_si128 ();
                    dst[6] = _mm_setzero_si128 ();
                    dst[5] = _mm_setzero_si128 ();
                    dst[4] = _mm_setzero_si128 ();
                    dst[3] = _mm_setzero_si128 ();
                    dst[2] = _mm_setzero_si128 ();
                    dst[1] = _mm_setzero_si128 ();
                    dst[0] = _mm_insert_epi16 (
                        _mm_setzero_si128 (), *currDcComp[comp]++, 0);
                }
#else /* IMF_HAVE_SSE2 */

                memset (halfZigData, 0, 64 * 2);
                halfZigData[0] = *currDcComp[comp]++;

#endif /* IMF_HAVE_SSE2 */

                d->_packedDcCount++;

                //
                // UnRLE the AC. This will modify currAcComp
                //

                rv = LossyDctDecoder_unRleAc (
                    d, &lastNonZero, &currAcComp, acCompEnd, halfZigData);
                if (rv != EXR_ERR_SUCCESS)
                {
                    free_fn (rowBlockHandle);
                    return rv;
                }

                //
                // Convert from XDR to NATIVE
                //

                priv_to_native16 (halfZigData, 64);

                if (lastNonZero == 0)
                {
                    //
                    // DC only case - AC components are all 0
                    //
                    dctData[0] = half_to_float (halfZigData[0]);

                    dctInverse8x8DcOnly (dctData);
                }
                else
                {
                    //
                    // We have some AC components that are non-zero.
                    // Can't use the 'constant block' optimization
                    //

                    blockIsConstant = DWA_CLASSIFIER_FALSE;

                    //
                    // Un-Zig zag
                    //

                    (*fromHalfZigZag) (halfZigData, dctData);

                    //
                    // Zig-Zag indices in normal layout are as follows:
                    //
                    // 0   1   5   6   14  15  27  28
                    // 2   4   7   13  16  26  29  42
                    // 3   8   12  17  25  30  41  43
                    // 9   11  18  24  31  40  44  53
                    // 10  19  23  32  39  45  52  54
                    // 20  22  33  38  46  51  55  60
                    // 21  34  37  47  50  56  59  61
                    // 35  36  48  49  57  58  62  63
                    //
                    // If lastNonZero is less than the first item on
                    // each row, we know that the whole row is zero and
                    // can be skipped in the row-oriented part of the
                    // iDCT.
                    //
                    // The unrolled logic here is:
                    //
                    //    if lastNonZero < rowStartIdx[i],
                    //    zeroedRows = rowsEmpty[i]
                    //
                    // where:
                    //
                    //    const int rowStartIdx[] = {2, 3, 9, 10, 20, 21, 35};
                    //    const int rowsEmpty[]   = {7, 6, 5,  4,  3,  2,  1};
                    //

                    if (lastNonZero < 2)
                        dctInverse8x8_7 (dctData);
                    else if (lastNonZero < 3)
                        dctInverse8x8_6 (dctData);
                    else if (lastNonZero < 9)
                        dctInverse8x8_5 (dctData);
                    else if (lastNonZero < 10)
                        dctInverse8x8_4 (dctData);
                    else if (lastNonZero < 20)
                        dctInverse8x8_3 (dctData);
                    else if (lastNonZero < 21)
                        dctInverse8x8_2 (dctData);
                    else if (lastNonZero < 35)
                        dctInverse8x8_1 (dctData);
                    else
                        dctInverse8x8_0 (dctData);
                }
            }

            //
            // Perform the CSC
            //

            if (numComp == 3)
            {
                if (!blockIsConstant)
                {
                    csc709Inverse64 (
                        chanData[0]->_dctData,
                        chanData[1]->_dctData,
                        chanData[2]->_dctData);
                }
                else
                {
                    csc709Inverse (
                        chanData[0]->_dctData,
                        chanData[1]->_dctData,
                        chanData[2]->_dctData);
                }
            }

            //
            // Float -> Half conversion.
            //
            // If the block has a constant value, just convert the first pixel.
            //

            for (int comp = 0; comp < numComp; ++comp)
            {
                if (!blockIsConstant)
                {
                    (*convertFloatToHalf64) (
                        &rowBlock[comp][blockx * 64], chanData[comp]->_dctData);
                }
                else
                {
#ifdef IMF_HAVE_SSE2

                    __m128i* dst = (__m128i*) &rowBlock[comp][blockx * 64];

                    dst[0] = _mm_set1_epi16 (
                        (short) float_to_half (chanData[comp]->_dctData[0]));

                    dst[1] = dst[0];
                    dst[2] = dst[0];
                    dst[3] = dst[0];
                    dst[4] = dst[0];
                    dst[5] = dst[0];
                    dst[6] = dst[0];
                    dst[7] = dst[0];

#else /* IMF_HAVE_SSE2 */

                    uint16_t* dst = &rowBlock[comp][blockx * 64];

                    dst[0] = float_to_half (chanData[comp]->_dctData[0]);

                    for (int i = 1; i < 64; ++i)
                    {
                        dst[i] = dst[0];
                    }

#endif            /* IMF_HAVE_SSE2 */
                } // blockIsConstant
            }     // comp
        }         // blockx

        //
        // At this point, we have half-float nonlinear value blocked
        // in rowBlock[][]. We need to unblock the data, transfer
        // back to linear, and write the results in the _rowPtrs[].
        //
        // There is a fast-path for aligned rows, which helps
        // things a little. Since this fast path is only valid
        // for full 8-element wide blocks, the partial x blocks
        // are broken into a separate loop below.
        //
        // At the moment, the fast path requires:
        //   * sse support
        //   * aligned row pointers
        //   * full 8-element wide blocks
        //

        for (int comp = 0; comp < numComp; ++comp)
        {
            //
            // Test if we can use the fast path
            //

#ifdef IMF_HAVE_SSE2

            uint8_t fastPath = DWA_CLASSIFIER_TRUE;

            for (int y = 8 * blocky; y < 8 * blocky + maxY; ++y)
            {
                if ((uintptr_t) (chanData[comp]->_rows[y]) &
                    _SSE_ALIGNMENT_MASK)
                    fastPath = DWA_CLASSIFIER_FALSE;
            }

            if (fastPath)
            {
                //
                // Handle all the full X blocks, in a fast path with sse2 and
                // aligned row pointers
                //

                for (int y = 8 * blocky; y < 8 * blocky + maxY; ++y)
                {
                    __m128i* dst = (__m128i*) chanData[comp]->_rows[y];
                    __m128i* src = (__m128i*) &rowBlock[comp][(y & 0x7) * 8];

                    for (int blockx = 0; blockx < numFullBlocksX; ++blockx)
                    {
                        uint16_t i0, i1, i2, i3, i4, i5, i6, i7;
                        //
                        // These may need some twiddling.
                        // Run with multiples of 8
                        //

                        _mm_prefetch ((char*) (src + 16), _MM_HINT_NTA);

                        i0 = (uint16_t) _mm_extract_epi16 (*src, 0);
                        i1 = (uint16_t) _mm_extract_epi16 (*src, 1);
                        i2 = (uint16_t) _mm_extract_epi16 (*src, 2);
                        i3 = (uint16_t) _mm_extract_epi16 (*src, 3);

                        i4 = (uint16_t) _mm_extract_epi16 (*src, 4);
                        i5 = (uint16_t) _mm_extract_epi16 (*src, 5);
                        i6 = (uint16_t) _mm_extract_epi16 (*src, 6);
                        i7 = (uint16_t) _mm_extract_epi16 (*src, 7);

                        i0 = d->_toLinear[i0];
                        i1 = d->_toLinear[i1];
                        i2 = d->_toLinear[i2];
                        i3 = d->_toLinear[i3];

                        i4 = d->_toLinear[i4];
                        i5 = d->_toLinear[i5];
                        i6 = d->_toLinear[i6];
                        i7 = d->_toLinear[i7];

                        *dst = _mm_insert_epi16 (_mm_setzero_si128 (), i0, 0);
                        *dst = _mm_insert_epi16 (*dst, i1, 1);
                        *dst = _mm_insert_epi16 (*dst, i2, 2);
                        *dst = _mm_insert_epi16 (*dst, i3, 3);

                        *dst = _mm_insert_epi16 (*dst, i4, 4);
                        *dst = _mm_insert_epi16 (*dst, i5, 5);
                        *dst = _mm_insert_epi16 (*dst, i6, 6);
                        *dst = _mm_insert_epi16 (*dst, i7, 7);

                        src += 8;
                        dst++;
                    }
                }
            }
            else
            {

#endif /* IMF_HAVE_SSE2 */

                //
                // Basic scalar kinda slow path for handling the full X blocks
                //

                for (int y = 8 * blocky; y < 8 * blocky + maxY; ++y)
                {
                    uint16_t* dst = (uint16_t*) chanData[comp]->_rows[y];

                    for (int blockx = 0; blockx < numFullBlocksX; ++blockx)
                    {
                        uint16_t* src =
                            &rowBlock[comp][blockx * 64 + ((y & 0x7) * 8)];

                        dst[0] = d->_toLinear[src[0]];
                        dst[1] = d->_toLinear[src[1]];
                        dst[2] = d->_toLinear[src[2]];
                        dst[3] = d->_toLinear[src[3]];

                        dst[4] = d->_toLinear[src[4]];
                        dst[5] = d->_toLinear[src[5]];
                        dst[6] = d->_toLinear[src[6]];
                        dst[7] = d->_toLinear[src[7]];

                        dst += 8;
                    }
                }

#ifdef IMF_HAVE_SSE2
            }

#endif /* IMF_HAVE_SSE2 */

            //
            // If we have partial X blocks, deal with all those now
            // Since this should be minimal work, there currently
            // is only one path that should work for everyone.
            //

            if (numFullBlocksX != numBlocksX)
            {
                for (int y = 8 * blocky; y < 8 * blocky + maxY; ++y)
                {
                    uint16_t* src = (uint16_t*) &rowBlock[comp]
                                                         [numFullBlocksX * 64 +
                                                          ((y & 0x7) * 8)];

                    uint16_t* dst = (uint16_t*) chanData[comp]->_rows[y];

                    dst += 8 * numFullBlocksX;

                    for (int x = 0; x < maxX; ++x)
                    {
                        *dst++ = d->_toLinear[*src++];
                    }
                }
            }
        } // comp
    }     // blocky

    //
    // Walk over all the channels that are of type FLOAT.
    // Convert from HALF XDR back to FLOAT XDR.
    //

    for (int chan = 0; chan < numComp; ++chan)
    {
        if (chanData[chan]->_type != EXR_PIXEL_FLOAT) continue;

        /* process in place in reverse to avoid temporary buffer */
        for (int y = 0; y < d->_height; ++y)
        {
            float*    floatXdrPtr = (float*) chanData[chan]->_rows[y];
            uint16_t* halfXdr     = (uint16_t*) floatXdrPtr;

            for (int x = d->_width - 1; x >= 0; --x)
            {
                floatXdrPtr[x] = one_from_native_float (
                    half_to_float (one_to_native16 (halfXdr[x])));
            }
        }
    }

    free_fn (rowBlockHandle);

    return EXR_ERR_SUCCESS;
}

/**************************************/

//
// Un-RLE the packed AC components into
// a half buffer. The half block should
// be the full 8x8 block (in zig-zag order
// still), not the first AC component.
//
// currAcComp is advanced as bytes are decoded.
//
// This returns the index of the last non-zero
// value in the buffer - with the index into zig zag
// order data. If we return 0, we have DC only data.
//
// This is assuminging that halfZigBlock is zero'ed
// prior to calling
//
exr_result_t
LossyDctDecoder_unRleAc (
    LossyDctDecoder* d,
    int*             lastNonZero,
    uint16_t**       currAcComp,
    uint16_t*        packedAcEnd,
    uint16_t*        halfZigBlock)
{
    //
    // Un-RLE the RLE'd blocks. If we find an item whose
    // high byte is 0xff, then insert the number of 0's
    // as indicated by the low byte.
    //
    // Otherwise, just copy the number verbatim.
    //
    int       dctComp = 1;
    uint16_t* acComp  = *currAcComp;
    uint16_t  val;
    int       lnz      = 0;
    uint64_t  ac_count = 0;

    //
    // Start with a zero'ed block, so we don't have to
    // write when we hit a run symbol
    //

    while (dctComp < 64)
    {
        if (acComp >= packedAcEnd) { return EXR_ERR_CORRUPT_CHUNK; }
        val = *acComp;
        if (val == 0xff00)
        {
            //
            // End of block
            //

            dctComp = 64;
        }
        else if ((val >> 8) == 0xff)
        {
            //
            // Run detected! Insert 0's.
            //
            // Since the block has been zeroed, just advance the ptr
            //

            dctComp += val & 0xff;
        }
        else
        {
            //
            // Not a run, just copy over the value
            //
            lnz                   = dctComp;
            halfZigBlock[dctComp] = val;

            dctComp++;
        }

        ac_count++;
        acComp++;
    }

    d->_packedAcCount += ac_count;
    *lastNonZero = lnz;
    *currAcComp  = acComp;
    return EXR_ERR_SUCCESS;
}
