//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) DreamWorks Animation LLC and Contributors of the OpenEXR Project
//

#ifndef INCLUDED_IMF_DWA_COMRESSOR_H
#define INCLUDED_IMF_DWA_COMRESSOR_H

//------------------------------------------------------------------------------
//
// class DwaCompressor -- Store lossy RGB data by quantizing DCT components.
//
//------------------------------------------------------------------------------

#include "ImfCompressor.h"

#include "ImfZip.h"
#include "ImfChannelList.h"

#include <half.h>

#include <vector>
#include <cstdint>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class DwaCompressor: public Compressor
{
  public:

    enum AcCompression 
    {
        STATIC_HUFFMAN,
        DEFLATE,
    };


    DwaCompressor (const Header &hdr, 
                   int           maxScanLineSize,
                   int           numScanLines,    // ideally is a multiple of 8
                   AcCompression acCompression);

    virtual ~DwaCompressor ();

    DwaCompressor (const DwaCompressor& other) = delete;
    DwaCompressor& operator = (const DwaCompressor& other) = delete;
    DwaCompressor (DwaCompressor&& other) = delete;
    DwaCompressor& operator = (DwaCompressor&& other) = delete;
    
    virtual int numScanLines () const;

    virtual OPENEXR_IMF_NAMESPACE::Compressor::Format format () const;

    virtual int compress (const char *inPtr,
                          int         inSize,
                          int         minY,
                          const char *&outPtr);

    virtual int compressTile (const char              *inPtr,
                              int                     inSize,
                              IMATH_NAMESPACE::Box2i  range,
                              const char              *&outPtr);

    virtual int uncompress (const char *inPtr,
                            int         inSize,
                            int         minY,
                            const char *&outPtr);

    virtual int uncompressTile (const char             *inPtr,
                                int                    inSize,
                                IMATH_NAMESPACE::Box2i range,
                                const char             *&outPtr);

    static void initializeFuncs ();

  private:

    struct ChannelData;
    struct CscChannelSet;
    struct Classifier;
    
    class LossyDctDecoderBase;
    class LossyDctDecoder;
    class LossyDctDecoderCsc;

    class LossyDctEncoderBase;
    class LossyDctEncoder;
    class LossyDctEncoderCsc;

    enum CompressorScheme 
    {
        UNKNOWN = 0,
        LOSSY_DCT,
        RLE,
        
        NUM_COMPRESSOR_SCHEMES
    };

    //
    // Per-chunk compressed data sizes, one value per chunk
    //

    enum DataSizesSingle 
    {
        VERSION = 0,                  // Version number:
                                      //   0: classic
                                      //   1: adds "end of block" to the AC RLE

        UNKNOWN_UNCOMPRESSED_SIZE,    // Size of leftover data, uncompressed.
        UNKNOWN_COMPRESSED_SIZE,      // Size of leftover data, zlib compressed.

        AC_COMPRESSED_SIZE,           // AC RLE + Huffman size
        DC_COMPRESSED_SIZE,           // DC + Deflate size
        RLE_COMPRESSED_SIZE,          // RLE + Deflate data size
        RLE_UNCOMPRESSED_SIZE,        // RLE'd data size 
        RLE_RAW_SIZE,                 // Un-RLE'd data size

        AC_UNCOMPRESSED_COUNT,        // AC RLE number of elements
        DC_UNCOMPRESSED_COUNT,        // DC number of elements

        AC_COMPRESSION,               // AC compression strategy
        NUM_SIZES_SINGLE
    };

    AcCompression     _acCompression;

    int               _maxScanLineSize;
    int               _numScanLines;
    int               _min[2], _max[2];

    ChannelList                _channels;
    std::vector<ChannelData>   _channelData;
    std::vector<CscChannelSet> _cscSets;
    std::vector<Classifier>    _channelRules;

    char*             _packedAcBuffer;
    uint64_t          _packedAcBufferSize;
    char*             _packedDcBuffer;
    uint64_t          _packedDcBufferSize;
    char*             _rleBuffer;
    uint64_t          _rleBufferSize;
    char*             _outBuffer;
    uint64_t          _outBufferSize;
    char*             _planarUncBuffer[NUM_COMPRESSOR_SCHEMES];
    uint64_t          _planarUncBufferSize[NUM_COMPRESSOR_SCHEMES];

    Zip*  _zip;
    int   _zipLevel;
    float _dwaCompressionLevel;

    int compress (const char              *inPtr,
                  int                     inSize,
                  IMATH_NAMESPACE::Box2i  range,
                  const char              *&outPtr);

    int uncompress (const char             *inPtr,
                    int                    inSize,
                    IMATH_NAMESPACE::Box2i range,
                    const char             *&outPtr);

    void initializeBuffers (size_t&);
    void initializeDefaultChannelRules ();
    void initializeLegacyChannelRules ();

    void relevantChannelRules( std::vector<Classifier> &) const;

    //
    // Populate our cached version of the channel data with
    // data from the real channel list. We want to 
    // copy over attributes, determine compression schemes
    // releveant for the channel type, and find sets of
    // channels to be compressed from Y'CbCr data instead 
    // of R'G'B'.
    //

    void classifyChannels (ChannelList                  channels,
                           std::vector<ChannelData>    &chanData, 
                           std::vector<CscChannelSet>  &cscData);

    // 
    // Compute various buffer pointers for each channel
    //

    void setupChannelData (int minX, int minY, int maxX, int maxY);
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif 
