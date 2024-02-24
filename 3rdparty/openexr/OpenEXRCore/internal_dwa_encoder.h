//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED
#    error "only include internal_dwa_helpers.h"
#endif

//
// Base 'class' for encoding using the lossy DCT scheme
//

typedef struct _LossyDctEncoder
{
    const uint16_t* _toNonlinear;

    uint64_t _numAcComp, _numDcComp;

    DctCoderChannelData* _channel_encode_data[3];
    int                  _channel_encode_data_count;

    int   _width, _height;
    float _quantBaseError;

    //
    // Pointers to the buffers where AC and DC
    // DCT components should be packed for
    // lossless compression downstream
    //

    uint8_t* _packedAc;
    uint8_t* _packedDc;

    //
    // Our "quantization tables" - the example JPEG tables,
    // normalized so that the smallest value in each is 1.0.
    // This gives us a relationship between error in DCT
    // components
    //

    float _quantTableY[64];
    float _quantTableCbCr[64];
} LossyDctEncoder;

static exr_result_t LossyDctEncoder_base_construct (
    LossyDctEncoder* e,
    float            quantBaseError,
    uint8_t*         packedAc,
    uint8_t*         packedDc,
    const uint16_t*  toNonlinear,
    int              width,
    int              height);

static exr_result_t LossyDctEncoder_construct (
    LossyDctEncoder*     e,
    float                quantBaseError,
    DctCoderChannelData* rowPtrs,
    uint8_t*             packedAc,
    uint8_t*             packedDc,
    const uint16_t*      toNonlinear,
    int                  width,
    int                  height);

static exr_result_t LossyDctEncoderCsc_construct (
    LossyDctEncoder*     e,
    float                quantBaseError,
    DctCoderChannelData* rowPtrsR,
    DctCoderChannelData* rowPtrsG,
    DctCoderChannelData* rowPtrsB,
    uint8_t*             packedAc,
    uint8_t*             packedDc,
    const uint16_t*      toNonlinear,
    int                  width,
    int                  height);

static exr_result_t LossyDctEncoder_execute (
    void* (*alloc_fn) (size_t), void (*free_fn) (void*), LossyDctEncoder* e);

static void
LossyDctEncoder_rleAc (LossyDctEncoder* e, uint16_t* block, uint16_t** acPtr);

/**************************************/

exr_result_t
LossyDctEncoder_base_construct (
    LossyDctEncoder* e,
    float            quantBaseError,
    uint8_t*         packedAc,
    uint8_t*         packedDc,
    const uint16_t*  toNonlinear,
    int              width,
    int              height)
{
    //
    // Here, we take the generic JPEG quantization tables and
    // normalize them by the smallest component in each table.
    // This gives us a relationship amongst the DCT components,
    // in terms of how sensitive each component is to
    // error.
    //
    // A higher normalized value means we can quantize more,
    // and a small normalized value means we can quantize less.
    //
    // Eventually, we will want an acceptable quantization
    // error range for each component. We find this by
    // multiplying some user-specified level (_quantBaseError)
    // by the normalized table (_quantTableY, _quantTableCbCr) to
    // find the acceptable quantization error range.
    //
    // The quantization table is not needed for decoding, and
    // is not transmitted. So, if you want to get really fancy,
    // you could derive some content-dependent quantization
    // table, and the decoder would not need to be changed. But,
    // for now, we'll just use static quantization tables.
    //
    int jpegQuantTableY[] = {
        16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
        14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
        18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

    int jpegQuantTableYMin = 10;

    int jpegQuantTableCbCr[] = {
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

    int jpegQuantTableCbCrMin = 17;

    e->_quantBaseError = quantBaseError;
    e->_width          = width;
    e->_height         = height;
    e->_toNonlinear    = toNonlinear;
    e->_numAcComp      = 0;
    e->_numDcComp      = 0;
    e->_packedAc       = packedAc;
    e->_packedDc       = packedDc;
    if (e->_quantBaseError < 0) e->_quantBaseError = 0;

    for (int idx = 0; idx < 64; ++idx)
    {
        e->_quantTableY[idx] =
            (e->_quantBaseError * (float) (jpegQuantTableY[idx]) /
             (float) (jpegQuantTableYMin));

        e->_quantTableCbCr[idx] =
            (e->_quantBaseError * (float) (jpegQuantTableCbCr[idx]) /
             (float) (jpegQuantTableCbCrMin));
    }

    e->_channel_encode_data[0]    = NULL;
    e->_channel_encode_data[1]    = NULL;
    e->_channel_encode_data[2]    = NULL;
    e->_channel_encode_data_count = 0;

    return EXR_ERR_SUCCESS;
}

/**************************************/

//
// Single channel lossy DCT encoder
//

exr_result_t
LossyDctEncoder_construct (
    LossyDctEncoder*     e,
    float                quantBaseError,
    DctCoderChannelData* rowPtrs,
    uint8_t*             packedAc,
    uint8_t*             packedDc,
    const uint16_t*      toNonlinear,
    int                  width,
    int                  height)
{
    exr_result_t rv;

    rv = LossyDctEncoder_base_construct (
        e, quantBaseError, packedAc, packedDc, toNonlinear, width, height);
    e->_channel_encode_data[0]    = rowPtrs;
    e->_channel_encode_data_count = 1;

    return rv;
}

/**************************************/

//
// RGB channel lossy DCT encoder
//
exr_result_t
LossyDctEncoderCsc_construct (
    LossyDctEncoder*     e,
    float                quantBaseError,
    DctCoderChannelData* rowPtrsR,
    DctCoderChannelData* rowPtrsG,
    DctCoderChannelData* rowPtrsB,
    uint8_t*             packedAc,
    uint8_t*             packedDc,
    const uint16_t*      toNonlinear,
    int                  width,
    int                  height)
{
    exr_result_t rv;

    rv = LossyDctEncoder_base_construct (
        e, quantBaseError, packedAc, packedDc, toNonlinear, width, height);
    e->_channel_encode_data[0]    = rowPtrsR;
    e->_channel_encode_data[1]    = rowPtrsG;
    e->_channel_encode_data[2]    = rowPtrsB;
    e->_channel_encode_data_count = 3;

    return rv;
}

/**************************************/

//
// Reorder from zig-zag order to normal ordering
//
static void
toZigZag (uint16_t* dst, uint16_t* src)
{
    static const int remap[] = {
        0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

    for (int i = 0; i < 64; ++i)
        dst[i] = src[remap[i]];
}

//
// Precomputing the bit count runs faster than using
// the builtin instruction, at least in one case..
//
// Precomputing 8-bits is no slower than 16-bits,
// and saves a fair bit of overhead..
//
static inline int
countSetBits (uint16_t src)
{
    static const uint16_t numBitsSet[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4,
        2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

    return numBitsSet[src & 0xff] + numBitsSet[src >> 8];
}

//
// Take a DCT coefficient, as well as an acceptable error. Search
// nearby values within the error tolerance, that have fewer
// bits set.
//
// The list of candidates has been pre-computed and sorted
// in order of increasing numbers of bits set. This way, we
// can stop searching as soon as we find a candidate that
// is within the error tolerance.
//
static inline uint16_t
quantize (float dctval, float errorTolerance)
{
    uint16_t tmp;
    // pre-quantize float -> half and back
    uint16_t src      = float_to_half (dctval);
    float    srcFloat = half_to_float (src);

    int             numSetBits = countSetBits (src);
    const uint16_t* closest    = closestData + closestDataOffset[src];

    for (int targetNumSetBits = numSetBits - 1; targetNumSetBits >= 0;
         --targetNumSetBits)
    {
        tmp = *closest;

        if (fabsf (half_to_float (tmp) - srcFloat) < errorTolerance) return tmp;

        closest++;
    }

    return src;
}

/**************************************/

//
// Given three channels of source data, encoding by first applying
// a color space conversion to a YCbCr space.  Otherwise, if we only
// have one channel, just encode it as is.
//
// Other numbers of channels are somewhat unexpected at this point
//
exr_result_t
LossyDctEncoder_execute (
    void* (*alloc_fn) (size_t), void (*free_fn) (void*), LossyDctEncoder* e)
{
    int                  numComp = e->_channel_encode_data_count;
    DctCoderChannelData* chanData[3];

    int numBlocksX = (int) (ceilf ((float) e->_width / 8.0f));
    int numBlocksY = (int) (ceilf ((float) e->_height / 8.0f));

    uint16_t halfZigCoef[64];
    uint16_t halfCoef[64];

    uint16_t* currAcComp            = (uint16_t*) e->_packedAc;
    int       tmpHalfBufferElements = 0;
    uint16_t* tmpHalfBuffer         = NULL;
    uint16_t* tmpHalfBufferPtr      = NULL;

    e->_numAcComp = 0;
    e->_numDcComp = 0;

    //
    // Allocate a temp half buffer to quantize into for
    // any FLOAT source channels.
    //

    for (int chan = 0; chan < numComp; ++chan)
    {
        chanData[chan] = e->_channel_encode_data[chan];
        if (chanData[chan]->_type == EXR_PIXEL_FLOAT)
            tmpHalfBufferElements += e->_width * e->_height;
    }

    if (tmpHalfBufferElements)
    {
        tmpHalfBuffer = (uint16_t*) alloc_fn (
            (size_t) tmpHalfBufferElements * sizeof (uint16_t));
        if (!tmpHalfBuffer) return EXR_ERR_OUT_OF_MEMORY;
        tmpHalfBufferPtr = tmpHalfBuffer;
    }

    //
    // Run over all the float scanlines, quantizing,
    // and re-assigning _rowPtr[y]. We need to translate
    // FLOAT XDR to HALF XDR.
    //

    for (int chan = 0; chan < numComp; ++chan)
    {
        if (chanData[chan]->_type != EXR_PIXEL_FLOAT) continue;

        for (int y = 0; y < e->_height; ++y)
        {
            const float* srcXdr = (const float*) chanData[chan]->_rows[y];

            for (int x = 0; x < e->_width; ++x)
            {
                //
                // Clamp to half ranges, instead of just casting. This
                // avoids introducing Infs which end up getting zeroed later
                //
                float src = one_to_native_float (srcXdr[x]);
                if (src > 65504.f)
                    src = 65504.f;
                else if (src < -65504.f)
                    src = -65504.f;
                tmpHalfBufferPtr[x] = one_from_native16 (float_to_half (src));
            }

            chanData[chan]->_rows[y] = (uint8_t*) tmpHalfBufferPtr;
            tmpHalfBufferPtr += e->_width;
        }
    }

    //
    // Pack DC components together by common plane, so we can get
    // a little more out of differencing them. We'll always have
    // one component per block, so we can computed offsets.
    //

    chanData[0]->_dc_comp = (uint16_t*) e->_packedDc;
    for (int chan = 1; chan < numComp; ++chan)
        chanData[chan]->_dc_comp =
            chanData[chan - 1]->_dc_comp + numBlocksX * numBlocksY;

    for (int blocky = 0; blocky < numBlocksY; ++blocky)
    {
        for (int blockx = 0; blockx < numBlocksX; ++blockx)
        {
            uint16_t     h;
            const float* quantTable;

            for (int chan = 0; chan < numComp; ++chan)
            {
                //
                // Break the source into 8x8 blocks. If we don't
                // fit at the edges, mirror.
                //
                // Also, convert from linear to nonlinear representation.
                // Our source is assumed to be XDR, and we need to convert
                // to NATIVE prior to converting to float.
                //
                // If we're converting linear -> nonlinear, assume that the
                // XDR -> NATIVE conversion is built into the lookup. Otherwise,
                // we'll need to explicitly do it.
                //

                for (int y = 0; y < 8; ++y)
                {
                    for (int x = 0; x < 8; ++x)
                    {
                        int vx = 8 * blockx + x;
                        int vy = 8 * blocky + y;

                        if (vx >= e->_width)
                            vx = e->_width - (vx - (e->_width - 1));

                        if (vx < 0) vx = e->_width - 1;

                        if (vy >= e->_height)
                            vy = e->_height - (vy - (e->_height - 1));

                        if (vy < 0) vy = e->_height - 1;

                        h = ((const uint16_t*) (chanData[chan]->_rows)[vy])[vx];

                        if (e->_toNonlinear) { h = e->_toNonlinear[h]; }
                        else { h = one_to_native16 (h); }

                        chanData[chan]->_dctData[y * 8 + x] = half_to_float (h);
                    } // x
                }     // y
            }         // chan

            //
            // Color space conversion
            //

            if (numComp == 3)
            {
                csc709Forward64 (
                    chanData[0]->_dctData,
                    chanData[1]->_dctData,
                    chanData[2]->_dctData);
            }

            quantTable = e->_quantTableY;
            for (int chan = 0; chan < numComp; ++chan)
            {
                //
                // Forward DCT
                //
                dctForward8x8 (chanData[chan]->_dctData);

                //
                // Quantize to half, and zigzag
                //

                for (int i = 0; i < 64; ++i)
                {
                    halfCoef[i] =
                        quantize (chanData[chan]->_dctData[i], quantTable[i]);
                }

                toZigZag (halfZigCoef, halfCoef);

                //
                // Convert from NATIVE back to XDR, before we write out
                //
                priv_from_native16 (halfZigCoef, 64);

                //
                // Save the DC component separately, to be compressed on
                // its own.
                //

                *(chanData[chan]->_dc_comp)++ = halfZigCoef[0];
                e->_numDcComp++;

                //
                // Then RLE the AC components (which will record the count
                // of the resulting number of items)
                //

                LossyDctEncoder_rleAc (e, halfZigCoef, &currAcComp);
                quantTable = e->_quantTableCbCr;
            } // chan
        }     // blockx
    }         // blocky

    if (tmpHalfBuffer) free_fn (tmpHalfBuffer);

    return EXR_ERR_SUCCESS;
}

/**************************************/

//
// RLE the zig-zag of the AC components + copy over
// into another tmp buffer
//
// Try to do a simple RLE scheme to reduce run's of 0's. This
// differs from the jpeg EOB case, since EOB just indicates that
// the rest of the block is zero. In our case, we have lots of
// NaN symbols, which shouldn't be allowed to occur in DCT
// coefficients - so we'll use them for encoding runs.
//
// If the high byte is 0xff, then we have a run of 0's, of length
// given by the low byte. For example, 0xff03 would be a run
// of 3 0's, starting at the current location.
//
// block is our block of 64 coefficients
// acPtr a pointer to back the RLE'd values into.
//
// This will advance the counter, _numAcComp.
//

void
LossyDctEncoder_rleAc (LossyDctEncoder* e, uint16_t* block, uint16_t** acPtr)
{
    int       dctComp   = 1;
    uint16_t  rleSymbol = 0x0;
    uint16_t* curAC     = *acPtr;

    while (dctComp < 64)
    {
        uint16_t runLen = 1;

        //
        // If we don't have a 0, output verbatim
        //

        if (block[dctComp] != rleSymbol)
        {
            *curAC++ = block[dctComp];
            e->_numAcComp++;

            dctComp += runLen;
            continue;
        }

        //
        // We're sitting on a 0, so see how big the run is.
        //

        while ((dctComp + runLen < 64) &&
               (block[dctComp + runLen] == rleSymbol))
        {
            runLen++;
        }

        //
        // If the run len is too small, just output verbatim
        // otherwise output our run token
        //
        // Originally, we wouldn't have a separate symbol for
        // "end of block". But in some experimentation, it looks
        // like using 0xff00 for "end of block" can save a bit
        // of space.
        //

        if (runLen == 1)
        {
            runLen   = 1;
            *curAC++ = block[dctComp];
            e->_numAcComp++;

            //
            // Using 0xff00 for "end of block"
            //
        }
        else if (runLen + dctComp == 64)
        {
            //
            // Signal EOB
            //

            *curAC++ = 0xff00;
            e->_numAcComp++;
        }
        else
        {
            //
            // Signal normal run
            //

            *curAC++ = (uint16_t) 0xff00 | runLen;
            e->_numAcComp++;
        }

        //
        // Advance by runLen
        //

        dctComp += runLen;
    }
    *acPtr = curAC;
}
