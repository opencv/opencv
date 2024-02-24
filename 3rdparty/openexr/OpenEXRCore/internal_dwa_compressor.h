//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMF_INTERNAL_DWA_HELPERS_H_HAS_BEEN_INCLUDED
#    error "only include internal_dwa_helpers.h"
#endif

/**************************************/

typedef struct _DwaCompressor
{
    exr_encode_pipeline_t* _encode;
    exr_decode_pipeline_t* _decode;

    AcCompression _acCompression;

    int _numScanLines;
    int _min[2], _max[2];

    int            _numChannels;
    int            _numCscChannelSets;
    ChannelData*   _channelData;
    CscChannelSet* _cscChannelSets;
    void*          _channel_mem;

    Classifier* _channelRules;
    size_t      _channelRuleCount;

    uint8_t* _packedAcBuffer;
    uint64_t _packedAcBufferSize;
    uint8_t* _packedDcBuffer;
    uint64_t _packedDcBufferSize;
    uint8_t* _rleBuffer;
    uint64_t _rleBufferSize;
    uint8_t* _planarUncBuffer[NUM_COMPRESSOR_SCHEMES];
    uint64_t _planarUncBufferSize[NUM_COMPRESSOR_SCHEMES];

    exr_memory_allocation_func_t alloc_fn;
    exr_memory_free_func_t       free_fn;

    int   _zipLevel;
    float _dwaCompressionLevel;
} DwaCompressor;

static exr_result_t DwaCompressor_construct (
    DwaCompressor*         me,
    AcCompression          acCompression,
    exr_encode_pipeline_t* encode,
    exr_decode_pipeline_t* decode);

static void DwaCompressor_destroy (DwaCompressor* me);

static exr_result_t DwaCompressor_compress (DwaCompressor* me);

static exr_result_t DwaCompressor_uncompress (
    DwaCompressor* me,
    const uint8_t* inPtr,
    uint64_t       iSize,
    void*          uncompressed_data,
    uint64_t       uncompressed_size);

static exr_result_t
DwaCompressor_initializeBuffers (DwaCompressor* me, size_t*);

static exr_result_t DwaCompressor_writeRelevantChannelRules (
    DwaCompressor* me, uint8_t** outPtr, uint64_t nAvail, uint64_t* nWritten);
static exr_result_t DwaCompressor_readChannelRules (
    DwaCompressor*  me,
    const uint8_t** inPtr,
    uint64_t*       nAvail,
    uint64_t*       outRuleSize);

//
// Populate our cached version of the channel data with
// data from the real channel list. We want to
// copy over attributes, determine compression schemes
// relevant for the channel type, and find sets of
// channels to be compressed from Y'CbCr data instead
// of R'G'B'.
//
static exr_result_t DwaCompressor_classifyChannels (DwaCompressor* me);

//
// Compute various buffer pointers for each channel
//

static exr_result_t DwaCompressor_setupChannelData (DwaCompressor* me);

/**************************************/

exr_result_t
DwaCompressor_construct (
    DwaCompressor*         me,
    AcCompression          acCompression,
    exr_encode_pipeline_t* encode,
    exr_decode_pipeline_t* decode)
{
    exr_result_t rv = EXR_ERR_SUCCESS;

    initializeFuncs ();

    memset (me, 0, sizeof (DwaCompressor));

    me->_acCompression = acCompression;

    me->_encode = encode;
    me->_decode = decode;

    if (encode)
    {
        const struct _internal_exr_context* pctxt = EXR_CCTXT (encode->context);

        me->alloc_fn = pctxt ? pctxt->alloc_fn : internal_exr_alloc;
        me->free_fn  = pctxt ? pctxt->free_fn : internal_exr_free;

        me->_channelData = internal_exr_alloc_aligned (
            me->alloc_fn,
            &(me->_channel_mem),
            sizeof (ChannelData) * (size_t) encode->channel_count,
            _SSE_ALIGNMENT);
        if (!me->_channelData) return EXR_ERR_OUT_OF_MEMORY;

        memset (
            me->_channelData,
            0,
            sizeof (ChannelData) * (size_t) encode->channel_count);

        me->_numChannels = encode->channel_count;
        for (int c = 0; c < encode->channel_count; ++c)
        {
            me->_channelData[c].chan        = encode->channels + c;
            me->_channelData[c].compression = UNKNOWN;
            DctCoderChannelData_construct (
                &(me->_channelData[c]._dctData),
                me->_channelData[c].chan->data_type);
        }

        // DWAA should be 32, DWAB should be 256
        me->_numScanLines = encode->chunk.height;

        me->_min[0] = encode->chunk.start_x;
        me->_min[1] = encode->chunk.start_y;
        me->_max[0] = me->_min[0] + encode->chunk.width - 1;
        me->_max[1] = me->_min[1] + encode->chunk.height - 1;

        rv = exr_get_zip_compression_level (
            encode->context, encode->part_index, &(me->_zipLevel));
        if (rv != EXR_ERR_SUCCESS) return rv;
        rv = exr_get_dwa_compression_level (
            encode->context, encode->part_index, &(me->_dwaCompressionLevel));
        if (rv != EXR_ERR_SUCCESS) return rv;
    }
    else
    {
        const struct _internal_exr_context* pctxt = EXR_CCTXT (decode->context);

        me->alloc_fn = pctxt ? pctxt->alloc_fn : internal_exr_alloc;
        me->free_fn  = pctxt ? pctxt->free_fn : internal_exr_free;

        me->_channelData = internal_exr_alloc_aligned (
            me->alloc_fn,
            &(me->_channel_mem),
            sizeof (ChannelData) * (size_t) decode->channel_count,
            _SSE_ALIGNMENT);
        if (!me->_channelData) return EXR_ERR_OUT_OF_MEMORY;

        memset (
            me->_channelData,
            0,
            sizeof (ChannelData) * (size_t) decode->channel_count);

        me->_numChannels = decode->channel_count;
        for (int c = 0; c < decode->channel_count; ++c)
        {
            me->_channelData[c].chan        = decode->channels + c;
            me->_channelData[c].compression = UNKNOWN;
        }

        me->_numScanLines = decode->chunk.height;

        me->_min[0] = decode->chunk.start_x;
        me->_min[1] = decode->chunk.start_y;
        me->_max[0] = me->_min[0] + decode->chunk.width - 1;
        me->_max[1] = me->_min[1] + decode->chunk.height - 1;
    }
    return rv;
}

/**************************************/

static void
DwaCompressor_destroy (DwaCompressor* me)
{
    if (me->_packedAcBuffer) me->free_fn (me->_packedAcBuffer);
    if (me->_packedDcBuffer) me->free_fn (me->_packedDcBuffer);
    if (me->_rleBuffer) me->free_fn (me->_rleBuffer);

    if (me->_channel_mem)
    {
        for (int c = 0; c < me->_numChannels; ++c)
            DctCoderChannelData_destroy (
                me->free_fn, &(me->_channelData[c]._dctData));

        me->free_fn (me->_channel_mem);
    }

    if (me->_cscChannelSets) me->free_fn (me->_cscChannelSets);
    if (me->_channelRules != sLegacyChannelRules &&
        me->_channelRules != sDefaultChannelRules)
    {
        for (size_t i = 0; i < me->_channelRuleCount; ++i)
            Classifier_destroy (me->free_fn, &(me->_channelRules[i]));
        me->free_fn (me->_channelRules);
    }

    for (int i = 0; i < NUM_COMPRESSOR_SCHEMES; ++i)
    {
        if (me->_planarUncBuffer[i]) me->free_fn (me->_planarUncBuffer[i]);
    }
}

/**************************************/

exr_result_t
DwaCompressor_compress (DwaCompressor* me)
{
    exr_result_t rv;
    uint8_t*     outPtr;
    uint64_t*    sizes;
    size_t       outBufferSize = 0;
    uint64_t     dataBytes, nWritten = 0;
    uint64_t     nAvail;
    uint64_t     fileVersion = 2;
    uint64_t*    version;
    uint64_t*    unknownUncompressedSize;
    uint64_t*    unknownCompressedSize;
    uint64_t*    acCompressedSize;
    uint64_t*    dcCompressedSize;
    uint64_t*    rleCompressedSize;
    uint64_t*    rleUncompressedSize;
    uint64_t*    rleRawSize;

    uint64_t* totalAcUncompressedCount;
    uint64_t* totalDcUncompressedCount;

    uint64_t* acCompression;
    uint8_t*  packedAcEnd;
    uint8_t*  packedDcEnd;
    uint8_t*  outDataPtr;
    uint8_t*  inDataPtr;

    // Starting with 2, we write the channel
    // classification rules into the file
    if (fileVersion < 2)
    {
        me->_channelRules = sLegacyChannelRules;
        me->_channelRuleCount =
            sizeof (sLegacyChannelRules) / sizeof (Classifier);
    }
    else
    {
        me->_channelRules = sDefaultChannelRules;
        me->_channelRuleCount =
            sizeof (sDefaultChannelRules) / sizeof (Classifier);
    }

    rv = DwaCompressor_initializeBuffers (me, &outBufferSize);

    nAvail = me->_encode->compressed_alloc_size;
    if (nAvail < (NUM_SIZES_SINGLE * sizeof (uint64_t)))
        return EXR_ERR_OUT_OF_MEMORY;

    rv = internal_encode_alloc_buffer (
        me->_encode,
        EXR_TRANSCODE_BUFFER_SCRATCH1,
        &(me->_encode->compressed_buffer),
        &(me->_encode->compressed_alloc_size),
        outBufferSize);
    if (rv != EXR_ERR_SUCCESS) return rv;

    nAvail = outBufferSize;
    sizes  = (uint64_t*) me->_encode->compressed_buffer;

    //
    // Zero all the numbers in the chunk header
    //
    //    memset (sizes, 0, NUM_SIZES_SINGLE * sizeof (uint64_t));
    memset (sizes, 0, me->_encode->compressed_alloc_size);

#define OBIDX(x) (uint64_t*) (sizes + x)

    version                 = OBIDX (VERSION);
    unknownUncompressedSize = OBIDX (UNKNOWN_UNCOMPRESSED_SIZE);
    unknownCompressedSize   = OBIDX (UNKNOWN_COMPRESSED_SIZE);
    acCompressedSize        = OBIDX (AC_COMPRESSED_SIZE);
    dcCompressedSize        = OBIDX (DC_COMPRESSED_SIZE);
    rleCompressedSize       = OBIDX (RLE_COMPRESSED_SIZE);
    rleUncompressedSize     = OBIDX (RLE_UNCOMPRESSED_SIZE);
    rleRawSize              = OBIDX (RLE_RAW_SIZE);

    totalAcUncompressedCount = OBIDX (AC_UNCOMPRESSED_COUNT);
    totalDcUncompressedCount = OBIDX (DC_UNCOMPRESSED_COUNT);

    acCompression = OBIDX (AC_COMPRESSION);
    packedAcEnd   = NULL;
    packedDcEnd   = NULL;

    // Now write in the channel rules...
    outPtr = (uint8_t*) (sizes + NUM_SIZES_SINGLE);
    if (rv == EXR_ERR_SUCCESS && fileVersion >= 2)
    {
        rv = DwaCompressor_writeRelevantChannelRules (
            me, &outPtr, nAvail, &nWritten);
    }

    // post add this so we have a 0 value for the relevant channel
    // rules to fill up
    nWritten += NUM_SIZES_SINGLE * sizeof (uint64_t);

    if (rv != EXR_ERR_SUCCESS || nWritten >= me->_encode->compressed_alloc_size)
        return EXR_ERR_OUT_OF_MEMORY;

    outDataPtr = outPtr;

    //
    // We might not be dealing with any color data, in which
    // case the AC buffer size will be 0, and dereferencing
    // a vector will not be a good thing to do.
    //

    if (me->_packedAcBuffer) packedAcEnd = me->_packedAcBuffer;
    if (me->_packedDcBuffer) packedDcEnd = me->_packedDcBuffer;

    //
    // Setup the AC compression strategy and the version in the data block,
    // then write the relevant channel classification rules if needed
    //
    *version       = fileVersion;
    *acCompression = me->_acCompression;

    rv = DwaCompressor_setupChannelData (me);
    if (rv != EXR_ERR_SUCCESS) return rv;

    //
    // Determine the start of each row in the input buffer
    // Channels are interleaved by scanline
    //
    for (int c = 0; c < me->_numChannels; ++c)
    {
        me->_channelData[c].processed = 0;
    }

    inDataPtr = me->_encode->packed_buffer;

    for (int y = me->_min[1]; y <= me->_max[1]; ++y)
    {
        for (int c = 0; c < me->_numChannels; ++c)
        {
            ChannelData*               cd   = &(me->_channelData[c]);
            exr_coding_channel_info_t* chan = cd->chan;

            if ((y % chan->y_samples) != 0) continue;

            rv = DctCoderChannelData_push_row (
                me->alloc_fn, me->free_fn, &(cd->_dctData), inDataPtr);
            if (rv != EXR_ERR_SUCCESS) return rv;

            inDataPtr += chan->width * chan->bytes_per_element;
        }
    }

    //
    // Make a pass over all our CSC sets and try to encode them first
    //

    for (int csc = 0; csc < me->_numCscChannelSets; ++csc)
    {
        LossyDctEncoder enc;
        CscChannelSet*  cset = &(me->_cscChannelSets[csc]);

        rv = LossyDctEncoderCsc_construct (
            &enc,
            me->_dwaCompressionLevel / 100000.f,
            &(me->_channelData[cset->idx[0]]._dctData),
            &(me->_channelData[cset->idx[1]]._dctData),
            &(me->_channelData[cset->idx[2]]._dctData),
            packedAcEnd,
            packedDcEnd,
            dwaCompressorToNonlinear,
            me->_channelData[cset->idx[0]].chan->width,
            me->_channelData[cset->idx[0]].chan->height);

        if (rv == EXR_ERR_SUCCESS)
            rv = LossyDctEncoder_execute (me->alloc_fn, me->free_fn, &enc);

        *totalAcUncompressedCount = *totalAcUncompressedCount + enc._numAcComp;
        *totalDcUncompressedCount = *totalDcUncompressedCount + enc._numDcComp;

        packedAcEnd += enc._numAcComp * sizeof (uint16_t);
        packedDcEnd += enc._numDcComp * sizeof (uint16_t);

        me->_channelData[cset->idx[0]].processed = 1;
        me->_channelData[cset->idx[1]].processed = 1;
        me->_channelData[cset->idx[2]].processed = 1;

        if (rv != EXR_ERR_SUCCESS) return rv;
    }

    for (int chan = 0; chan < me->_numChannels; ++chan)
    {
        ChannelData*               cd    = &(me->_channelData[chan]);
        exr_coding_channel_info_t* pchan = cd->chan;

        if (cd->processed) continue;

        switch (cd->compression)
        {
            case LOSSY_DCT:
                //
                // For LOSSY_DCT, treat this just like the CSC'd case,
                // but only operate on one channel
                //
                {
                    LossyDctEncoder       enc;
                    const unsigned short* nonlinearLut = NULL;

                    if (!pchan->p_linear)
                        nonlinearLut = dwaCompressorToNonlinear;

                    rv = LossyDctEncoder_construct (
                        &enc,
                        me->_dwaCompressionLevel / 100000.f,
                        &(cd->_dctData),
                        packedAcEnd,
                        packedDcEnd,
                        nonlinearLut,
                        pchan->width,
                        pchan->height);

                    if (rv == EXR_ERR_SUCCESS)
                        rv = LossyDctEncoder_execute (
                            me->alloc_fn, me->free_fn, &enc);

                    *totalAcUncompressedCount =
                        *totalAcUncompressedCount + enc._numAcComp;
                    *totalDcUncompressedCount =
                        *totalDcUncompressedCount + enc._numDcComp;

                    packedAcEnd += enc._numAcComp * sizeof (uint16_t);
                    packedDcEnd += enc._numDcComp * sizeof (uint16_t);

                    if (rv != EXR_ERR_SUCCESS) return rv;
                }
                break;

            case RLE: {
                //
                // For RLE, bash the bytes up so that the first bytes of each
                // pixel are contiguous, as are the second bytes, and so on.
                //
                DctCoderChannelData* dcd = &(cd->_dctData);
                for (size_t y = 0; y < dcd->_size; ++y)
                {
                    const uint8_t* row = dcd->_rows[y];

                    for (int x = 0; x < pchan->width; ++x)
                    {
                        for (int byte = 0; byte < pchan->bytes_per_element;
                             ++byte)
                        {
                            *cd->planarUncRleEnd[byte]++ = *row++;
                        }
                    }

                    *rleRawSize += (uint64_t) pchan->width *
                                   (uint64_t) pchan->bytes_per_element;
                }

                break;
            }

            case UNKNOWN:

                //
                // Otherwise, just copy data over verbatim
                //

                {
                    size_t scanlineSize = (size_t) pchan->width *
                                          (size_t) pchan->bytes_per_element;
                    DctCoderChannelData* dcd = &(cd->_dctData);
                    for (size_t y = 0; y < dcd->_size; ++y)
                    {
                        memcpy (
                            cd->planarUncBufferEnd,
                            dcd->_rows[y],
                            scanlineSize);

                        cd->planarUncBufferEnd += scanlineSize;
                    }

                    *unknownUncompressedSize += cd->planarUncSize;
                }

                break;

            case NUM_COMPRESSOR_SCHEMES:
            default: return EXR_ERR_INVALID_ARGUMENT;
        }

        cd->processed = DWA_CLASSIFIER_TRUE;
    }

    //
    // Pack the Unknown data into the output buffer first. Instead of
    // just copying it uncompressed, try zlib compression at least.
    //

    if (*unknownUncompressedSize > 0)
    {
        size_t outSize;

        rv = exr_compress_buffer (
            me->_encode->context,
            9, // TODO: use default??? the old call to zlib had 9 hardcoded
            me->_planarUncBuffer[UNKNOWN],
            *unknownUncompressedSize,
            outDataPtr,
            exr_compress_max_buffer_size (*unknownUncompressedSize),
            &outSize);
        if (rv != EXR_ERR_SUCCESS) return rv;

        outDataPtr += outSize;
        *unknownCompressedSize = outSize;
        nWritten += outSize;
    }

    //
    // Now, pack all the Lossy DCT coefficients into our output
    // buffer, with Huffman encoding.
    //
    // Also, record the compressed size and the number of
    // uncompressed componentns we have.
    //

    if (*totalAcUncompressedCount > 0)
    {
        switch (me->_acCompression)
        {
            case STATIC_HUFFMAN: {
                size_t outDataSize =
                    outBufferSize -
                    (size_t) ((uintptr_t) outDataPtr - (uintptr_t) sizes);

                rv = internal_huf_compress (
                    acCompressedSize,
                    outDataPtr,
                    outDataSize,
                    (const uint16_t*) me->_packedAcBuffer,
                    *totalAcUncompressedCount,
                    me->_encode->scratch_buffer_1,
                    me->_encode->scratch_alloc_size_1);
                if (rv != EXR_ERR_SUCCESS)
                {
                    if (rv == EXR_ERR_ARGUMENT_OUT_OF_RANGE)
                    {
                        memcpy (
                            me->_encode->compressed_buffer,
                            me->_encode->packed_buffer,
                            me->_encode->packed_alloc_size);
                        me->_encode->compressed_bytes =
                            me->_encode->packed_alloc_size;
                        return EXR_ERR_SUCCESS;
                    }
                    return rv;
                }
                break;
            }

            case DEFLATE: {
                size_t sourceLen =
                    *totalAcUncompressedCount * sizeof (uint16_t);
                size_t destLen;

                rv = exr_compress_buffer (
                    me->_encode->context,
                    9, // TODO: use default??? the old call to zlib had 9 hardcoded
                    me->_packedAcBuffer,
                    sourceLen,
                    outDataPtr,
                    exr_compress_max_buffer_size (sourceLen),
                    &destLen);
                if (rv != EXR_ERR_SUCCESS) return rv;

                *acCompressedSize = destLen;
                break;
            }

            default:
                return EXR_ERR_INVALID_ARGUMENT;
                //assert (false);
        }

        outDataPtr += *acCompressedSize;
        nWritten += *acCompressedSize;
    }

    //
    // Handle the DC components separately
    //

    if (*totalDcUncompressedCount > 0)
    {
        size_t compBytes;
        size_t uncompBytes = *totalDcUncompressedCount * sizeof (uint16_t);

        rv = internal_encode_alloc_buffer (
            me->_encode,
            EXR_TRANSCODE_BUFFER_SCRATCH1,
            &(me->_encode->scratch_buffer_1),
            &(me->_encode->scratch_alloc_size_1),
            uncompBytes);

        if (rv != EXR_ERR_SUCCESS) return rv;

        internal_zip_deconstruct_bytes (
            me->_encode->scratch_buffer_1, me->_packedDcBuffer, uncompBytes);

        rv = exr_compress_buffer (
            me->_encode->context,
            me->_zipLevel,
            me->_encode->scratch_buffer_1,
            uncompBytes,
            outDataPtr,
            exr_compress_max_buffer_size (uncompBytes),
            &compBytes);

        if (rv != EXR_ERR_SUCCESS) return rv;

        *dcCompressedSize = compBytes;
        outDataPtr += compBytes;
        nWritten += compBytes;
    }

    //
    // If we have RLE data, first RLE encode it and set the uncompressed
    // size. Then, deflate the results and set the compressed size.
    //

    if (*rleRawSize > 0)
    {
        size_t compBytes;
        *rleUncompressedSize = internal_rle_compress (
            me->_rleBuffer,
            me->_rleBufferSize,
            me->_planarUncBuffer[RLE],
            *rleRawSize);

        rv = exr_compress_buffer (
            me->_encode->context,
            9, // TODO: use default??? the old call to zlib had 9 hardcoded
            me->_rleBuffer,
            *rleUncompressedSize,
            outDataPtr,
            exr_compress_max_buffer_size (*rleUncompressedSize),
            &compBytes);

        if (rv != EXR_ERR_SUCCESS) return rv;

        *rleCompressedSize = compBytes;
        outDataPtr += compBytes;
        nWritten += compBytes;
    }

    //
    // Flip the counters to XDR format
    //
    priv_from_native64 (sizes, NUM_SIZES_SINGLE);

    dataBytes =
        (uintptr_t) outDataPtr - (uintptr_t) me->_encode->compressed_buffer;
    if (nWritten != dataBytes) { return EXR_ERR_CORRUPT_CHUNK; }

    if (nWritten >= me->_encode->packed_bytes)
    {
        memcpy (
            me->_encode->compressed_buffer,
            me->_encode->packed_buffer,
            me->_encode->packed_bytes);
        me->_encode->compressed_bytes = me->_encode->packed_bytes;
    }
    else { me->_encode->compressed_bytes = nWritten; }
    return rv;
}

/**************************************/

exr_result_t
DwaCompressor_uncompress (
    DwaCompressor* me,
    const uint8_t* inPtr,
    uint64_t       iSize,
    void*          uncompressed_data,
    uint64_t       uncompressed_size)
{
    uint64_t     headerSize = NUM_SIZES_SINGLE * sizeof (uint64_t);
    exr_result_t rv         = EXR_ERR_SUCCESS;
    uint64_t     counters[NUM_SIZES_SINGLE];
    uint64_t     version;
    uint64_t     unknownUncompressedSize;
    uint64_t     unknownCompressedSize;
    uint64_t     acCompressedSize;
    uint64_t     dcCompressedSize;
    uint64_t     rleCompressedSize;
    uint64_t     rleUncompressedSize;
    uint64_t     rleRawSize;

    uint64_t totalAcUncompressedCount;
    uint64_t totalDcUncompressedCount;

    uint64_t acCompression;

    size_t         outBufferSize;
    uint64_t       compressedSize;
    const uint8_t* dataPtr;
    uint64_t       dataLeft;
    uint8_t*       outBufferEnd;
    uint8_t*       packedAcBufferEnd;
    uint8_t*       packedDcBufferEnd;
    const uint8_t* dataPtrEnd;
    const uint8_t* compressedUnknownBuf;
    const uint8_t* compressedAcBuf;
    const uint8_t* compressedDcBuf;
    const uint8_t* compressedRleBuf;

    if (iSize < headerSize) return EXR_ERR_CORRUPT_CHUNK;

    //
    // Flip the counters from XDR to NATIVE
    //

    memset (uncompressed_data, 0, uncompressed_size);

    memcpy (counters, inPtr, headerSize);
    priv_to_native64 (counters, NUM_SIZES_SINGLE);

    //
    // Unwind all the counter info
    //
    version                 = counters[VERSION];
    unknownUncompressedSize = counters[UNKNOWN_UNCOMPRESSED_SIZE];
    unknownCompressedSize   = counters[UNKNOWN_COMPRESSED_SIZE];
    acCompressedSize        = counters[AC_COMPRESSED_SIZE];
    dcCompressedSize        = counters[DC_COMPRESSED_SIZE];
    rleCompressedSize       = counters[RLE_COMPRESSED_SIZE];
    rleUncompressedSize     = counters[RLE_UNCOMPRESSED_SIZE];
    rleRawSize              = counters[RLE_RAW_SIZE];

    totalAcUncompressedCount = counters[AC_UNCOMPRESSED_COUNT];
    totalDcUncompressedCount = counters[DC_UNCOMPRESSED_COUNT];

    acCompression = counters[AC_COMPRESSION];

    compressedSize = unknownCompressedSize + acCompressedSize +
                     dcCompressedSize + rleCompressedSize;

    dataPtrEnd = inPtr + iSize;
    dataPtr  = inPtr + headerSize;
    dataLeft = iSize - headerSize;

    /* Both the sum and individual sizes are checked in case of overflow. */
    if (iSize < (headerSize + compressedSize) ||
        iSize < unknownCompressedSize || iSize < acCompressedSize ||
        iSize < dcCompressedSize || iSize < rleCompressedSize)
    {
        return EXR_ERR_CORRUPT_CHUNK;
    }

    if ((int64_t) unknownUncompressedSize < 0 ||
        (int64_t) unknownCompressedSize < 0 || (int64_t) acCompressedSize < 0 ||
        (int64_t) dcCompressedSize < 0 || (int64_t) rleCompressedSize < 0 ||
        (int64_t) rleUncompressedSize < 0 || (int64_t) rleRawSize < 0 ||
        (int64_t) totalAcUncompressedCount < 0 ||
        (int64_t) totalDcUncompressedCount < 0)
    {
        return EXR_ERR_CORRUPT_CHUNK;
    }

    if (version < 2)
    {
        me->_channelRules = sLegacyChannelRules;
        me->_channelRuleCount =
            sizeof (sLegacyChannelRules) / sizeof (Classifier);
    }
    else
    {
        uint64_t ruleSize;
        rv =
            DwaCompressor_readChannelRules (me, &dataPtr, &dataLeft, &ruleSize);

        headerSize += ruleSize;
    }

    if (rv != EXR_ERR_SUCCESS) return rv;

    outBufferSize = 0;
    rv            = DwaCompressor_initializeBuffers (me, &outBufferSize);
    if (rv != EXR_ERR_SUCCESS) return rv;

    //
    // Allocate _outBuffer, if we haven't done so already
    //

    // the C++ classes used to have one buffer size for compress / uncompress
    // but here we want to do zero-ish copy...
    outBufferEnd  = me->_decode->unpacked_buffer;
    outBufferSize = me->_decode->unpacked_alloc_size;

    //
    // Find the start of the RLE packed AC components and
    // the DC components for each channel. This will be handy
    // if you want to decode the channels in parallel later on.
    //

    packedAcBufferEnd = NULL;

    if (me->_packedAcBuffer) packedAcBufferEnd = me->_packedAcBuffer;

    packedDcBufferEnd = NULL;

    if (me->_packedDcBuffer) packedDcBufferEnd = me->_packedDcBuffer;

    //
    // UNKNOWN data is packed first, followed by the
    // Huffman-compressed AC, then the DC values,
    // and then the zlib compressed RLE data.
    //

    compressedUnknownBuf = dataPtr;

    compressedAcBuf =
        compressedUnknownBuf + (ptrdiff_t) (unknownCompressedSize);
    compressedDcBuf  = compressedAcBuf + (ptrdiff_t) (acCompressedSize);
    compressedRleBuf = compressedDcBuf + (ptrdiff_t) (dcCompressedSize);

    if (compressedUnknownBuf > dataPtrEnd ||
        dataPtr > compressedAcBuf ||
        compressedAcBuf > dataPtrEnd ||
        dataPtr > compressedDcBuf ||
        compressedDcBuf > dataPtrEnd ||
        dataPtr > compressedRleBuf ||
        compressedRleBuf > dataPtrEnd ||
        (compressedRleBuf + rleCompressedSize) > dataPtrEnd)
    {
        return EXR_ERR_CORRUPT_CHUNK;
    }

    //
    // Sanity check that the version is something we expect. Right now,
    // we can decode version 0, 1, and 2. v1 adds 'end of block' symbols
    // to the AC RLE. v2 adds channel classification rules at the
    // start of the data block.
    //

    if (version > 2) { return EXR_ERR_BAD_CHUNK_LEADER; }

    rv = DwaCompressor_setupChannelData (me);

    //
    // Uncompress the UNKNOWN data into _planarUncBuffer[UNKNOWN]
    //

    if (unknownCompressedSize > 0)
    {
        if (unknownUncompressedSize > me->_planarUncBufferSize[UNKNOWN])
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        if (EXR_ERR_SUCCESS != exr_uncompress_buffer (
                                   me->_decode->context,
                                   compressedUnknownBuf,
                                   unknownCompressedSize,
                                   me->_planarUncBuffer[UNKNOWN],
                                   unknownUncompressedSize,
                                   NULL))
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }
    }

    //
    // Uncompress the AC data into _packedAcBuffer
    //

    if (acCompressedSize > 0)
    {
        if (!me->_packedAcBuffer ||
            totalAcUncompressedCount * sizeof (uint16_t) >
                me->_packedAcBufferSize)
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        //
        // Don't trust the user to get it right, look in the file.
        //

        switch (acCompression)
        {
            case STATIC_HUFFMAN:
                rv = internal_huf_decompress (
                    me->_decode,
                    compressedAcBuf,
                    acCompressedSize,
                    (uint16_t*) me->_packedAcBuffer,
                    totalAcUncompressedCount,
                    me->_decode->scratch_buffer_1,
                    me->_decode->scratch_alloc_size_1);
                if (rv != EXR_ERR_SUCCESS) { return rv; }
                break;

            case DEFLATE: {
                size_t destLen;

                rv = exr_uncompress_buffer (
                    me->_decode->context,
                    compressedAcBuf,
                    acCompressedSize,
                    me->_packedAcBuffer,
                    totalAcUncompressedCount * sizeof (uint16_t),
                    &destLen);
                if (rv != EXR_ERR_SUCCESS) return rv;

                if (totalAcUncompressedCount * sizeof (uint16_t) != destLen)
                {
                    return EXR_ERR_CORRUPT_CHUNK;
                }
            }
            break;

            default: return EXR_ERR_CORRUPT_CHUNK; break;
        }
    }

    //
    // Uncompress the DC data into _packedDcBuffer
    //

    if (dcCompressedSize > 0)
    {
        size_t destLen;
        size_t uncompBytes = totalDcUncompressedCount * sizeof (uint16_t);
        if (uncompBytes > me->_packedDcBufferSize)
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        rv = internal_decode_alloc_buffer (
            me->_decode,
            EXR_TRANSCODE_BUFFER_SCRATCH1,
            &(me->_decode->scratch_buffer_1),
            &(me->_decode->scratch_alloc_size_1),
            uncompBytes);

        if (rv != EXR_ERR_SUCCESS) return rv;

        rv = exr_uncompress_buffer (
            me->_decode->context,
            compressedDcBuf,
            dcCompressedSize,
            me->_decode->scratch_buffer_1,
            uncompBytes,
            &destLen);
        if (rv != EXR_ERR_SUCCESS || (uncompBytes != destLen))
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        internal_zip_reconstruct_bytes (
            me->_packedDcBuffer, me->_decode->scratch_buffer_1, uncompBytes);
    }
    else
    {
        // if the compressed size is 0, then the uncompressed size must also be zero
        if (totalDcUncompressedCount != 0) { return EXR_ERR_CORRUPT_CHUNK; }
    }

    //
    // Uncompress the RLE data into _rleBuffer, then unRLE the results
    // into _planarUncBuffer[RLE]
    //

    if (rleRawSize > 0)
    {
        size_t dstLen;

        if (rleUncompressedSize > me->_rleBufferSize ||
            rleRawSize > me->_planarUncBufferSize[RLE])
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        if (EXR_ERR_SUCCESS != exr_uncompress_buffer (
                                   me->_decode->context,
                                   compressedRleBuf,
                                   rleCompressedSize,
                                   me->_rleBuffer,
                                   rleUncompressedSize,
                                   &dstLen))
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        if (dstLen != rleUncompressedSize) { return EXR_ERR_CORRUPT_CHUNK; }

        if (internal_rle_decompress (
                me->_planarUncBuffer[RLE],
                rleRawSize,
                (const uint8_t*) me->_rleBuffer,
                rleUncompressedSize) != rleRawSize)
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }
    }

    //
    // Determine the start of each row in the output buffer
    //
    for (int c = 0; c < me->_numChannels; ++c)
    {
        me->_channelData[c].processed = 0;
    }

    for (int y = me->_min[1]; y <= me->_max[1]; ++y)
    {
        for (int c = 0; c < me->_numChannels; ++c)
        {
            ChannelData*               cd   = &(me->_channelData[c]);
            exr_coding_channel_info_t* chan = cd->chan;

            if ((y % chan->y_samples) != 0) continue;

            rv = DctCoderChannelData_push_row (
                me->alloc_fn, me->free_fn, &(cd->_dctData), outBufferEnd);
            if (rv != EXR_ERR_SUCCESS) return rv;

            outBufferEnd += chan->width * chan->bytes_per_element;
        }
    }

    //
    // Setup to decode each block of 3 channels that need to
    // be handled together
    //

    for (int csc = 0; csc < me->_numCscChannelSets; ++csc)
    {
        LossyDctDecoder decoder;
        CscChannelSet*  cset = &(me->_cscChannelSets[csc]);

        int rChan = cset->idx[0];
        int gChan = cset->idx[1];
        int bChan = cset->idx[2];

        if (me->_channelData[rChan].compression != LOSSY_DCT ||
            me->_channelData[gChan].compression != LOSSY_DCT ||
            me->_channelData[bChan].compression != LOSSY_DCT)
        {
            return EXR_ERR_CORRUPT_CHUNK;
        }

        rv = LossyDctDecoderCsc_construct (
            &decoder,
            &(me->_channelData[rChan]._dctData),
            &(me->_channelData[gChan]._dctData),
            &(me->_channelData[bChan]._dctData),
            packedAcBufferEnd,
            packedAcBufferEnd + totalAcUncompressedCount * sizeof (uint16_t),
            packedDcBufferEnd,
            totalDcUncompressedCount,
            dwaCompressorToLinear,
            me->_channelData[rChan].chan->width,
            me->_channelData[rChan].chan->height);

        if (rv == EXR_ERR_SUCCESS)
            rv = LossyDctDecoder_execute (me->alloc_fn, me->free_fn, &decoder);

        packedAcBufferEnd += decoder._packedAcCount * sizeof (uint16_t);

        packedDcBufferEnd += decoder._packedDcCount * sizeof (uint16_t);
        totalDcUncompressedCount -= decoder._packedDcCount;

        me->_channelData[rChan].processed = 1;
        me->_channelData[gChan].processed = 1;
        me->_channelData[bChan].processed = 1;

        if (rv != EXR_ERR_SUCCESS) { return rv; }
    }

    //
    // Setup to handle the remaining channels by themselves
    //

    for (int c = 0; c < me->_numChannels; ++c)
    {
        ChannelData*               cd        = &(me->_channelData[c]);
        exr_coding_channel_info_t* chan      = cd->chan;
        DctCoderChannelData*       dcddata   = &(cd->_dctData);
        int                        pixelSize = chan->bytes_per_element;

        if (cd->processed) continue;

        switch (cd->compression)
        {
            case LOSSY_DCT:

                //
                // Setup a single-channel lossy DCT decoder pointing
                // at the output buffer
                //

                {
                    const uint16_t* linearLut = NULL;
                    LossyDctDecoder decoder;

                    if (!chan->p_linear) linearLut = dwaCompressorToLinear;

                    rv = LossyDctDecoder_construct (
                        &decoder,
                        dcddata,
                        packedAcBufferEnd,
                        packedAcBufferEnd +
                            totalAcUncompressedCount * sizeof (uint16_t),
                        packedDcBufferEnd,
                        totalDcUncompressedCount,
                        linearLut,
                        chan->width,
                        chan->height);

                    if (rv == EXR_ERR_SUCCESS)
                        rv = LossyDctDecoder_execute (
                            me->alloc_fn, me->free_fn, &decoder);

                    packedAcBufferEnd +=
                        (size_t) decoder._packedAcCount * sizeof (uint16_t);

                    packedDcBufferEnd +=
                        (size_t) decoder._packedDcCount * sizeof (uint16_t);

                    totalDcUncompressedCount -= decoder._packedDcCount;
                    if (rv != EXR_ERR_SUCCESS) { return rv; }
                }

                break;

            case RLE:

                //
                // For the RLE case, the data has been un-RLE'd into
                // planarUncRleEnd[], but is still split out by bytes.
                // We need to rearrange the bytes back into the correct
                // order in the output buffer;
                //

                {
                    int row = 0;

                    for (int y = me->_min[1]; y <= me->_max[1]; ++y)
                    {
                        uint8_t* dst;
                        if ((y % chan->y_samples) != 0) continue;

                        dst = dcddata->_rows[row];

                        if (pixelSize == 2)
                        {
                            interleaveByte2 (
                                dst,
                                cd->planarUncRleEnd[0],
                                cd->planarUncRleEnd[1],
                                chan->width);

                            cd->planarUncRleEnd[0] += chan->width;
                            cd->planarUncRleEnd[1] += chan->width;
                        }
                        else
                        {
                            for (int x = 0; x < chan->width; ++x)
                            {
                                for (int byte = 0; byte < pixelSize; ++byte)
                                {
                                    *dst++ = *cd->planarUncRleEnd[byte]++;
                                }
                            }
                        }

                        row++;
                    }
                }

                break;

            case UNKNOWN:

                //
                // In the UNKNOWN case, data is already in planarUncBufferEnd
                // and just needs to copied over to the output buffer
                //

                {
                    int    row = 0;
                    size_t dstScanlineSize =
                        (size_t) chan->width * (size_t) pixelSize;

                    for (int y = me->_min[1]; y <= me->_max[1]; ++y)
                    {
                        if ((y % chan->y_samples) != 0) continue;

                        //
                        // sanity check for buffer data lying within range
                        //
                        if ((cd->planarUncBufferEnd +
                             (size_t) (dstScanlineSize)) >
                            (me->_planarUncBuffer[UNKNOWN] +
                             me->_planarUncBufferSize[UNKNOWN]))
                        {
                            return EXR_ERR_CORRUPT_CHUNK;
                        }

                        memcpy (
                            dcddata->_rows[row],
                            cd->planarUncBufferEnd,
                            dstScanlineSize);

                        cd->planarUncBufferEnd += dstScanlineSize;
                        row++;
                    }
                }

                break;

            case NUM_COMPRESSOR_SCHEMES:
            default: return EXR_ERR_CORRUPT_CHUNK; break;
        }

        cd->processed = 1;
    }

    return rv;
}

/**************************************/

exr_result_t
DwaCompressor_initializeBuffers (DwaCompressor* me, size_t* bufferSize)
{
    exr_result_t rv = EXR_ERR_SUCCESS;

    //
    // _outBuffer needs to be big enough to hold all our
    // compressed data - which could vary depending on what sort
    // of channels we have.
    //

    uint64_t maxOutBufferSize  = 0;
    uint64_t numLossyDctChans  = 0;
    uint64_t unknownBufferSize = 0;
    uint64_t rleBufferSize     = 0;

    uint64_t maxLossyDctAcSize =
        (uint64_t) (ceilf ((float) me->_numScanLines / 8.0f)) *
        (uint64_t) (ceilf ((float) (me->_max[0] - me->_min[0] + 1) / 8.0f)) *
        63 * sizeof (uint16_t);

    uint64_t maxLossyDctDcSize =
        (uint64_t) (ceilf ((float) me->_numScanLines / 8.0f)) *
        (uint64_t) (ceilf ((float) (me->_max[0] - me->_min[0] + 1) / 8.0f)) *
        sizeof (uint16_t);

    uint64_t pixelCount = (uint64_t) (me->_numScanLines) *
                          (uint64_t) (me->_max[0] - me->_min[0] + 1);

    uint64_t planarUncBufferSize[NUM_COMPRESSOR_SCHEMES];

    for (int i = 0; i < NUM_COMPRESSOR_SCHEMES; ++i)
        planarUncBufferSize[i] = 0;

    for (size_t i = 0; i < me->_channelRuleCount; ++i)
    {
        maxOutBufferSize += Classifier_size (&(me->_channelRules[i]));
    }

    rv = DwaCompressor_classifyChannels (me);
    if (rv != EXR_ERR_SUCCESS) return rv;

    for (int c = 0; c < me->_numChannels; ++c)
    {
        const exr_coding_channel_info_t* curc = me->_channelData[c].chan;
        switch (me->_channelData[c].compression)
        {
            case LOSSY_DCT:

                //
                // This is the size of the number of packed
                // components, plus the requirements for
                // maximum Huffman encoding size (for STATIC_HUFFMAN)
                // or for zlib compression (for DEFLATE)
                //

                maxOutBufferSize += std_max (
                    2lu * maxLossyDctAcSize + 65536lu,
                    exr_compress_max_buffer_size (maxLossyDctAcSize));
                numLossyDctChans++;
                break;

            case RLE:
                //
                // RLE, if gone horribly wrong, could double the size
                // of the source data.
                //
                rleBufferSize +=
                    2 * pixelCount * (uint64_t) curc->bytes_per_element;

                planarUncBufferSize[RLE] +=
                    2 * pixelCount * (uint64_t) curc->bytes_per_element;
                break;

            case UNKNOWN:
                unknownBufferSize +=
                    pixelCount * (uint64_t) curc->bytes_per_element;
                planarUncBufferSize[UNKNOWN] +=
                    pixelCount * (uint64_t) curc->bytes_per_element;
                break;

            case NUM_COMPRESSOR_SCHEMES:
            default: return EXR_ERR_INVALID_ARGUMENT;
        }
    }

    //
    // Also, since the results of the RLE are packed into
    // the output buffer, we need the extra room there. But
    // we're going to zlib compress() the data we pack,
    // which could take slightly more space
    //

    maxOutBufferSize += exr_compress_max_buffer_size (rleBufferSize);

    //
    // And the same goes for the UNKNOWN data
    //

    maxOutBufferSize += exr_compress_max_buffer_size (unknownBufferSize);

    //
    // Reserve space big enough to hold the DC data
    // and include its compressed results in the size requirements
    // for our output buffer
    //

    maxOutBufferSize +=
        exr_compress_max_buffer_size (maxLossyDctDcSize * numLossyDctChans);

    //
    // We also need to reserve space at the head of the buffer to
    // write out the size of our various packed and compressed data.
    //

    maxOutBufferSize += NUM_SIZES_SINGLE * sizeof (uint64_t);

    //
    // Later, we're going to hijack outBuffer for the result of
    // both encoding and decoding. So it needs to be big enough
    // to hold either a buffers' worth of uncompressed or
    // compressed data
    //
    // For encoding, we'll need _outBuffer to hold maxOutBufferSize bytes,
    // but for decoding, we only need it to be maxScanLineSize*numScanLines.
    // Cache the max size for now, and alloc the buffer when we either
    // encode or decode.
    //

    *bufferSize = maxOutBufferSize;

    //
    // _packedAcBuffer holds the quantized DCT coefficients prior
    // to Huffman encoding
    //

    if (maxLossyDctAcSize * numLossyDctChans > me->_packedAcBufferSize)
    {
        me->_packedAcBufferSize = maxLossyDctAcSize * numLossyDctChans;
        if (me->_packedAcBuffer != NULL) me->free_fn (me->_packedAcBuffer);
        me->_packedAcBuffer = me->alloc_fn (me->_packedAcBufferSize);
        if (!me->_packedAcBuffer) return EXR_ERR_OUT_OF_MEMORY;
        memset (me->_packedAcBuffer, 0, me->_packedAcBufferSize);
    }

    //
    // _packedDcBuffer holds one quantized DCT coef per 8x8 block
    //

    if (maxLossyDctDcSize * numLossyDctChans > me->_packedDcBufferSize)
    {
        me->_packedDcBufferSize = maxLossyDctDcSize * numLossyDctChans;
        if (me->_packedDcBuffer != NULL) me->free_fn (me->_packedDcBuffer);
        me->_packedDcBuffer = me->alloc_fn (me->_packedDcBufferSize);
        if (!me->_packedDcBuffer) return EXR_ERR_OUT_OF_MEMORY;
        memset (me->_packedDcBuffer, 0, me->_packedDcBufferSize);
    }

    if (rleBufferSize > me->_rleBufferSize)
    {
        me->_rleBufferSize = rleBufferSize;
        if (me->_rleBuffer != 0) me->free_fn (me->_rleBuffer);
        me->_rleBuffer = me->alloc_fn (rleBufferSize);
        if (!me->_rleBuffer) return EXR_ERR_OUT_OF_MEMORY;
        memset (me->_rleBuffer, 0, rleBufferSize);
    }

    //
    // The planar uncompressed buffer will hold float data for LOSSY_DCT
    // compressed values, and whatever the native type is for other
    // channels. We're going to use this to hold data in a planar
    // format, as opposed to the native interleaved format we take
    // into compress() and give back from uncompress().
    //
    // This also makes it easier to compress the UNKNOWN and RLE data
    // all in one swoop (for each compression scheme).
    //

    //
    // UNKNOWN data is going to be zlib compressed, which needs
    // a little extra headroom
    //

    if (planarUncBufferSize[UNKNOWN] > 0)
    {
        planarUncBufferSize[UNKNOWN] =
            exr_compress_max_buffer_size (planarUncBufferSize[UNKNOWN]);
    }

    for (int i = 0; i < NUM_COMPRESSOR_SCHEMES; ++i)
    {
        if (planarUncBufferSize[i] > me->_planarUncBufferSize[i])
        {
            me->_planarUncBufferSize[i] = planarUncBufferSize[i];
            if (me->_planarUncBuffer[i] != NULL)
                me->free_fn (me->_planarUncBuffer[i]);

            if (planarUncBufferSize[i] > SIZE_MAX)
            {
                return EXR_ERR_OUT_OF_MEMORY;
            }

            me->_planarUncBuffer[i] = me->alloc_fn (planarUncBufferSize[i]);
            if (!me->_planarUncBuffer[i]) return EXR_ERR_OUT_OF_MEMORY;
        }
    }

    return rv;
}

/**************************************/

exr_result_t
DwaCompressor_writeRelevantChannelRules (
    DwaCompressor* me, uint8_t** outPtr, uint64_t nAvail, uint64_t* nWritten)
{
    uint64_t nOut = sizeof (uint16_t);

    uint8_t* curp = *outPtr;

    uint16_t* ruleSize = (uint16_t*) curp;
    curp += sizeof (uint16_t);

    if (nAvail < (*nWritten + nOut)) return EXR_ERR_OUT_OF_MEMORY;

    for (size_t i = 0; i < me->_channelRuleCount; ++i)
    {
        for (int c = 0; c < me->_numChannels; ++c)
        {
            const exr_coding_channel_info_t* curc = me->_channelData[c].chan;
            const char* suffix = Classifier_find_suffix (curc->channel_name);

            if (Classifier_match (
                    &(me->_channelRules[i]),
                    suffix,
                    (exr_pixel_type_t) curc->data_type))
            {
                if (nAvail < (*nWritten + nOut +
                              Classifier_size (&(me->_channelRules[i]))))
                    return EXR_ERR_OUT_OF_MEMORY;

                nOut += Classifier_write (&(me->_channelRules[i]), &curp);
                break;
            }
        }
    }

    if (nOut > 65535) return EXR_ERR_OUT_OF_MEMORY;
    *ruleSize = one_from_native16 ((uint16_t) nOut);
    *nWritten += nOut;

    *outPtr = curp;
    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
DwaCompressor_readChannelRules (
    DwaCompressor*  me,
    const uint8_t** inPtr,
    uint64_t*       nAvail,
    uint64_t*       outRuleSize)
{
    exr_result_t   rv;
    const uint8_t* readPtr = *inPtr;

    *outRuleSize = 0;
    if (*nAvail > sizeof (uint16_t))
    {
        size_t ruleSize = one_to_native16 (*((const uint16_t*) readPtr));
        size_t nRules   = 0, dataSize;
        const uint8_t* tmpPtr;

        if (ruleSize < sizeof (uint16_t)) { return EXR_ERR_CORRUPT_CHUNK; }

        *outRuleSize = ruleSize;
        if (*nAvail < ruleSize) { return EXR_ERR_CORRUPT_CHUNK; }

        readPtr += sizeof (uint16_t);
        *inPtr += ruleSize;
        *nAvail -= ruleSize;

        ruleSize -= sizeof (uint16_t);
        // annoying, don't know how many there are yet...
        tmpPtr   = readPtr;
        dataSize = ruleSize;
        rv       = EXR_ERR_SUCCESS;
        while (rv == EXR_ERR_SUCCESS && dataSize > 0)
        {
            Classifier tmpc;
            memset (&tmpc, 0, sizeof (Classifier));
            rv = Classifier_read (me->alloc_fn, &tmpc, &tmpPtr, &dataSize);
            Classifier_destroy (me->free_fn, &tmpc);
            ++nRules;
        }

        // now (if we succeed) we can allocate and fill
        if (rv == EXR_ERR_SUCCESS)
        {
            me->_channelRuleCount = nRules;
            me->_channelRules     = me->alloc_fn (sizeof (Classifier) * nRules);

            dataSize = ruleSize;
            if (me->_channelRules)
            {
                memset (me->_channelRules, 0, sizeof (Classifier) * nRules);
                for (size_t i = 0; i < nRules; ++i)
                {
                    Classifier_read (
                        me->alloc_fn,
                        &(me->_channelRules[i]),
                        &readPtr,
                        &dataSize);
                }
            }
            else
                rv = EXR_ERR_OUT_OF_MEMORY;
        }
    }
    else
        rv = EXR_ERR_CORRUPT_CHUNK;
    return rv;
}

/**************************************/

exr_result_t
DwaCompressor_classifyChannels (DwaCompressor* me)
{
    CscPrefixMapItem* prefixMap;
    //
    // prefixMap used to map channel name prefixes to
    // potential CSC-able sets of channels.
    //

    me->_cscChannelSets =
        me->alloc_fn (sizeof (CscChannelSet) * (size_t) me->_numChannels);
    if (!me->_cscChannelSets) return EXR_ERR_OUT_OF_MEMORY;

    //
    // Try and figure out which channels should be
    // compressed by which means.
    //
    prefixMap =
        me->alloc_fn (sizeof (CscPrefixMapItem) * (size_t) me->_numChannels);
    if (!prefixMap) return EXR_ERR_OUT_OF_MEMORY;

    memset (
        prefixMap, 0, sizeof (CscPrefixMapItem) * (size_t) me->_numChannels);
    for (int c = 0; c < me->_numChannels; ++c)
    {
        const exr_coding_channel_info_t* curc = me->_channelData[c].chan;
        const char*       suffix = Classifier_find_suffix (curc->channel_name);
        CscPrefixMapItem* mapi   = CscPrefixMap_find (
            prefixMap,
            me->_numChannels,
            curc->channel_name,
            (size_t) (curc->channel_name - suffix));

        for (size_t i = 0; i < me->_channelRuleCount; ++i)
        {
            if (Classifier_match (
                    &(me->_channelRules[i]),
                    suffix,
                    (exr_pixel_type_t) curc->data_type))
            {
                me->_channelData[c].compression = me->_channelRules[i]._scheme;

                if (me->_channelRules[i]._cscIdx >= 0)
                    mapi->idx[me->_channelRules[i]._cscIdx] = c;
            }
        }
    }

    //
    // Finally, try and find RGB sets of channels which
    // can be CSC'ed to a Y'CbCr space prior to loss, for
    // better compression.
    //
    // Walk over our set of candidates, and see who has
    // all three channels defined (and has common sampling
    // patterns, etc).
    //

    for (int c = 0; c < me->_numChannels; ++c)
    {
        const exr_coding_channel_info_t *redc, *grnc, *bluc;
        CscChannelSet*                   cset;
        int                              red = prefixMap[c].idx[0];
        int                              grn = prefixMap[c].idx[1];
        int                              blu = prefixMap[c].idx[2];

        if (prefixMap[c].name == NULL) break;

        if ((red < 0) || (grn < 0) || (blu < 0)) continue;

        redc = me->_channelData[red].chan;
        grnc = me->_channelData[grn].chan;
        bluc = me->_channelData[blu].chan;

        if ((redc->x_samples != grnc->x_samples) ||
            (redc->x_samples != bluc->x_samples) ||
            (grnc->x_samples != bluc->x_samples) ||
            (redc->y_samples != grnc->y_samples) ||
            (redc->y_samples != bluc->y_samples) ||
            (grnc->y_samples != bluc->y_samples))
        {
            continue;
        }

        cset         = me->_cscChannelSets + me->_numCscChannelSets;
        cset->idx[0] = red;
        cset->idx[1] = grn;
        cset->idx[2] = blu;
        ++(me->_numCscChannelSets);
    }
    me->free_fn (prefixMap);

    return EXR_ERR_SUCCESS;
}

/**************************************/

exr_result_t
DwaCompressor_setupChannelData (DwaCompressor* me)
{
    uint8_t* planarUncBuffer[NUM_COMPRESSOR_SCHEMES];

    for (int i = 0; i < NUM_COMPRESSOR_SCHEMES; ++i)
    {
        planarUncBuffer[i] = 0;

        if (me->_planarUncBuffer[i])
            planarUncBuffer[i] = me->_planarUncBuffer[i];
    }

    for (int c = 0; c < me->_numChannels; ++c)
    {
        ChannelData*                     cd   = me->_channelData + c;
        const exr_coding_channel_info_t* curc = cd->chan;
        size_t                           uncSize;

        uncSize = (size_t) curc->width * (size_t) curc->height *
                  (size_t) curc->bytes_per_element;
        cd->planarUncSize = uncSize;

        cd->planarUncBuffer    = planarUncBuffer[cd->compression];
        cd->planarUncBufferEnd = cd->planarUncBuffer;

        cd->planarUncRle[0]    = cd->planarUncBuffer;
        cd->planarUncRleEnd[0] = cd->planarUncRle[0];

        for (int byte = 1; byte < curc->bytes_per_element; ++byte)
        {
            cd->planarUncRle[byte] =
                cd->planarUncRle[byte - 1] + curc->width * curc->height;

            cd->planarUncRleEnd[byte] = cd->planarUncRle[byte];
        }

        cd->planarUncType = (exr_pixel_type_t) curc->data_type;

        if (cd->compression == LOSSY_DCT)
        {
            cd->planarUncType = EXR_PIXEL_FLOAT;
        }
        else { planarUncBuffer[cd->compression] += uncSize; }
    }

    return EXR_ERR_SUCCESS;
}
