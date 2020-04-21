///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2009-2014 DreamWorks Animation LLC. 
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

//---------------------------------------------------
//
// class DwaCompressor -- Store lossy RGB data by quantizing
//                          DCT components.
//
// First, we try and figure out what compression strategy to take
// based in channel name. For RGB channels, we want a lossy method
// described below. But, if we have alpha, we should do something
// different (and probably using RLE). If we have depth, or velocity,
// or something else, just fall back to ZIP. The rules for deciding 
// which strategy to use are setup in initializeDefaultChannelRules().
// When writing a file, the relevant rules needed to decode are written
// into the start of the data block, making a self-contained file. 
// If initializeDefaultChannelRules() doesn't quite suite your naming
// conventions, you can adjust the rules without breaking decoder
// compatability.
//
// If we're going to lossy compress R, G, or B channels, it's easier
// to toss bits in a more perceptual uniform space. One could argue
// at length as to what constitutes perceptually uniform, expecially 
// when storing either scene/input/focal plane referred and output referred
// data. 
//
// We'll compromise. For values <= 1, we use a traditional power function
// (without any of that straight-line business at the bottom). For values > 1,
// we want something more like a log function, since power functions blow
// up. At 1, we want a smooth blend between the functions. So, we use a 
// piecewise function that does just that - see dwaLookups.cpp for 
// a little more detail.
//
// Also, if we find that we have R, G, and B channels from the same layer,
// we can get a bit more compression efficiency by transforming to a Y'CbCr
// space. We use the 709 transform, but with Cb,Cr = 0 for an input of 
// (0, 0, 0), instead of the traditional Cb,Cr = .5. Shifting the zero point
// makes no sense with large range data. Transforms are done to from 
// the perceptual space data, not the linear-light space data (R'G'B' ->
// (Y'CbCr, not RGB -> YCbCr).
//
// Next, we forward DCT the data. This is done with a floating
// point DCT, as we don't really have control over the src range. The 
// resulting values are dropped to half-float precision. 
//
// Now, we need to quantize. Quantization departs from the usual way 
// of dividing and rounding. Instead, we start with some floating 
// point "base-error" value. From this, we can derive quantization 
// error for each DCT component. Take the standard JPEG quantization
// tables and normalize them by the smallest value. Then, multiply
// the normalized quant tables by our base-error value. This gives
// a range of errors for each DCT component.
//
// For each DCT component, we want to find a quantized value that 
// is within +- the per-component error. Pick the quantized value
// that has the fewest bits set in its' binary representation. 
// Brute-forcing the search would make for extremly inefficient 
// compression. Fortunatly, we can precompute a table to assist 
// with this search. 
//
// For each 16-bit float value, there are at most 15 other values with
// fewer bits set. We can precompute these values in a compact form, since
// many source values have far fewer that 15 possible quantized values. 
// Now, instead of searching the entire range +- the component error,
// we can just search at most 15 quantization candidates. The search can
// be accelerated a bit more by sorting the candidates by the 
// number of bits set, in increasing order. Then, the search can stop
// once a candidate is found w/i the per-component quantization 
// error range.
//
// The quantization strategy has the side-benefit that there is no
// de-quantization step upon decode, so we don't bother recording
// the quantization table.
//
// Ok. So we now have quantized values. Time for entropy coding. We
// can use either static Huffman or zlib/DEFLATE. The static Huffman
// is more efficient at compacting data, but can have a greater 
// overhead, especially for smaller tile/strip sizes. 
//
// There is some additional fun, like ZIP compressing the DC components
// instead of Huffman/zlib, which helps make things slightly smaller.
//
// Compression level is controlled by setting an int/float/double attribute
// on the header named "dwaCompressionLevel". This is a thinly veiled name for 
// the "base-error" value mentioned above. The "base-error" is just
// dwaCompressionLevel / 100000. The default value of 45.0 is generally 
// pretty good at generating "visually lossless" values at reasonable
// data rates. Setting dwaCompressionLevel to 0 should result in no additional
// quantization at the quantization stage (though there may be 
// quantization in practice at the CSC/DCT steps). But if you really
// want lossless compression, there are pleanty of other choices 
// of compressors ;)
//
// When dealing with FLOAT source buffers, we first quantize the source
// to HALF and continue down as we would for HALF source.
//
//---------------------------------------------------


#include "ImfDwaCompressor.h"
#include "ImfDwaCompressorSimd.h"

#include "ImfChannelList.h"
#include "ImfStandardAttributes.h"
#include "ImfHeader.h"
#include "ImfHuf.h"
#include "ImfInt64.h"
#include "ImfIntAttribute.h"
#include "ImfIO.h"
#include "ImfMisc.h"
#include "ImfNamespace.h"
#include "ImfRle.h"
#include "ImfSimd.h"
#include "ImfSystemSpecific.h"
#include "ImfXdr.h"
#include "ImfZip.h"

#include "ImathFun.h"
#include "ImathBox.h"
#include "ImathVec.h"
#include "half.h"
#include "halfLimits.h"

#include "dwaLookups.h"

#include <vector>
#include <string>
#include <cctype>
#include <cassert>
#include <algorithm>

// Windows specific addition to prevent the indirect import of the redefined min/max macros
#if defined _WIN32 || defined _WIN64
	#ifdef NOMINMAX
		#undef NOMINMAX
	#endif
	#define NOMINMAX
#endif
#include <zlib.h>


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


namespace {

    //
    // Function pointer to dispatch to an approprate 
    // convertFloatToHalf64_* impl, based on runtime cpu checking.
    // Should be initialized in DwaCompressor::initializeFuncs()
    //

    void (*convertFloatToHalf64)(unsigned short*, float*) =
        convertFloatToHalf64_scalar;

    // 
    // Function pointer for dispatching a fromHalfZigZag_ impl
    //
    
    void (*fromHalfZigZag)(unsigned short*, float*) =
        fromHalfZigZag_scalar;

    //
    // Dispatch the inverse DCT on an 8x8 block, where the last
    // n rows can be all zeros. The n=0 case converts the full block.
    //
    void (*dctInverse8x8_0)(float*) = dctInverse8x8_scalar<0>;
    void (*dctInverse8x8_1)(float*) = dctInverse8x8_scalar<1>;
    void (*dctInverse8x8_2)(float*) = dctInverse8x8_scalar<2>;
    void (*dctInverse8x8_3)(float*) = dctInverse8x8_scalar<3>;
    void (*dctInverse8x8_4)(float*) = dctInverse8x8_scalar<4>;
    void (*dctInverse8x8_5)(float*) = dctInverse8x8_scalar<5>;
    void (*dctInverse8x8_6)(float*) = dctInverse8x8_scalar<6>;
    void (*dctInverse8x8_7)(float*) = dctInverse8x8_scalar<7>;
    
} // namespace


struct DwaCompressor::ChannelData
{
    std::string         name;
    CompressorScheme    compression;  
    int                 xSampling;
    int                 ySampling;
    PixelType           type;
    bool                pLinear;

    int                 width;
    int                 height;

    //
    // Incoming and outgoing data is scanline interleaved, and it's much
    // easier to operate on contiguous data.  Assuming the planare unc
    // buffer is to hold RLE data, we need to rearrange to make bytes
    // adjacent.
    //

    char               *planarUncBuffer;
    char               *planarUncBufferEnd;

    char               *planarUncRle[4];
    char               *planarUncRleEnd[4];

    PixelType           planarUncType;
    int                 planarUncSize;
};


struct DwaCompressor::CscChannelSet
{
    int idx[3];
};


struct DwaCompressor::Classifier
{
    Classifier (std::string suffix,
                CompressorScheme scheme,
                PixelType type,
                int cscIdx,
                bool caseInsensitive):
        _suffix(suffix),
        _scheme(scheme),
        _type(type),
        _cscIdx(cscIdx),
        _caseInsensitive(caseInsensitive)
    {
        if (caseInsensitive) 
            std::transform(_suffix.begin(), _suffix.end(), _suffix.begin(), tolower);
    }

    Classifier (const char *&ptr, int size)
    {
        if (size <= 0) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (truncated rule).");
            
        {
            char suffix[Name::SIZE];
            memset (suffix, 0, Name::SIZE);
            Xdr::read<CharPtrIO> (ptr, std::min(size, Name::SIZE-1), suffix);
            _suffix = std::string(suffix);
        }

        if (size < _suffix.length() + 1 + 2*Xdr::size<char>()) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (truncated rule).");

        char value;
        Xdr::read<CharPtrIO> (ptr, value);

        _cscIdx = (int)(value >> 4) - 1;
        if (_cscIdx < -1 || _cscIdx >= 3) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (corrupt cscIdx rule).");

        _scheme = (CompressorScheme)((value >> 2) & 3);
        if (_scheme < 0 || _scheme >= NUM_COMPRESSOR_SCHEMES) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (corrupt scheme rule).");

        _caseInsensitive = (value & 1 ? true : false);

        Xdr::read<CharPtrIO> (ptr, value);
        if (value < 0 || value >= NUM_PIXELTYPES) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (corrupt rule).");
        _type = (PixelType)value;
    }

    bool match (const std::string &suffix, const PixelType type) const
    {
        if (_type != type) return false;

        if (_caseInsensitive) 
        {
            std::string tmp(suffix);
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), tolower);
            return tmp == _suffix;
        }

        return suffix == _suffix;
    }

    size_t size () const 
    {
        // string length + \0
        size_t sizeBytes = _suffix.length() + 1;

        // 1 byte for scheme / cscIdx / caseInsensitive, and 1 byte for type
        sizeBytes += 2 * Xdr::size<char>();

        return sizeBytes;
    }

    void write (char *&ptr) const
    {
        Xdr::write<CharPtrIO> (ptr, _suffix.c_str());

        // Encode _cscIdx (-1-3) in the upper 4 bits,
        //        _scheme (0-2)  in the next 2 bits
        //        _caseInsen     in the bottom bit
        unsigned char value = 0;
        value |= ((unsigned char)(_cscIdx+1)      & 15) << 4;
        value |= ((unsigned char)_scheme          &  3) << 2;
        value |=  (unsigned char)_caseInsensitive &  1;

        Xdr::write<CharPtrIO> (ptr, value);
        Xdr::write<CharPtrIO> (ptr, (unsigned char)_type);
    }

    std::string      _suffix;
    CompressorScheme _scheme;
    PixelType        _type;
    int              _cscIdx;
    bool             _caseInsensitive;
};


//
// Base class for the LOSSY_DCT decoder classes
//

class DwaCompressor::LossyDctDecoderBase
{
  public:

    LossyDctDecoderBase
        (char *packedAc,
         char *packedDc,
         const unsigned short *toLinear,
         int width,
         int height);

    virtual ~LossyDctDecoderBase ();

    void execute();

    //
    // These return number of items, not bytes. Each item
    // is an unsigned short
    //

    int numAcValuesEncoded() const { return _packedAcCount; }
    int numDcValuesEncoded() const { return _packedDcCount; }

  protected:

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

    int unRleAc (unsigned short *&currAcComp,
                 unsigned short  *halfZigBlock); 


    //
    // if NATIVE and XDR are really the same values, we can
    // skip some processing and speed things along
    //

    bool                  _isNativeXdr;


    //
    // Counts of how many items have been packed into the
    // AC and DC buffers
    //

    int                   _packedAcCount;
    int                   _packedDcCount;


    //
    // AC and DC buffers to pack
    //

    char                 *_packedAc;
    char                 *_packedDc;


    // 
    // half -> half LUT to transform from nonlinear to linear
    //

    const unsigned short *_toLinear;


    //
    // image dimensions
    //

    int                   _width;
    int                   _height;


    //
    // Pointers to the start of each scanlines, to be filled on decode
    // Generally, these will be filled by the subclasses.
    //

    std::vector< std::vector<char *> > _rowPtrs;


    // 
    // The type of each data that _rowPtrs[i] is referring. Layout
    // is in the same order as _rowPtrs[].
    //

    std::vector<PixelType>             _type;
    std::vector<SimdAlignedBuffer64f>  _dctData;
};


//
// Used to decode a single channel of LOSSY_DCT data.
//

class DwaCompressor::LossyDctDecoder: public LossyDctDecoderBase
{
  public:

    //
    // toLinear is a half-float LUT to convert the encoded values 
    // back to linear light. If you want to skip this step, pass
    // in NULL here.
    //

    LossyDctDecoder
        (std::vector<char *> &rowPtrs,
         char *packedAc,
         char *packedDc,
         const unsigned short *toLinear,
         int width,
         int height,
         PixelType type)
    :
        LossyDctDecoderBase(packedAc, packedDc, toLinear, width, height)
    {
        _rowPtrs.push_back(rowPtrs);
        _type.push_back(type);
    }

    virtual ~LossyDctDecoder () {}
};


//
// Used to decode 3 channels of LOSSY_DCT data that
// are grouped together and color space converted.
//

class DwaCompressor::LossyDctDecoderCsc: public LossyDctDecoderBase
{
  public:

    //
    // toLinear is a half-float LUT to convert the encoded values 
    // back to linear light. If you want to skip this step, pass
    // in NULL here.
    //

    LossyDctDecoderCsc
        (std::vector<char *> &rowPtrsR,
         std::vector<char *> &rowPtrsG,
         std::vector<char *> &rowPtrsB,
         char *packedAc,
         char *packedDc,
         const unsigned short *toLinear,
         int width,
         int height,
         PixelType typeR,
         PixelType typeG,
         PixelType typeB)
    :
        LossyDctDecoderBase(packedAc, packedDc, toLinear, width, height)
    {
        _rowPtrs.push_back(rowPtrsR);
        _rowPtrs.push_back(rowPtrsG);
        _rowPtrs.push_back(rowPtrsB);
        _type.push_back(typeR);
        _type.push_back(typeG);
        _type.push_back(typeB);
    }

    virtual ~LossyDctDecoderCsc () {}
};


// 
// Base class for encoding using the lossy DCT scheme
//

class DwaCompressor::LossyDctEncoderBase
{
  public:

    LossyDctEncoderBase
        (float quantBaseError,
         char *packedAc,
         char *packedDc,
         const unsigned short *toNonlinear,
         int width,
         int height);

    virtual ~LossyDctEncoderBase ();

    void execute ();

    //
    // These return number of items, not bytes. Each item
    // is an unsigned short
    //

    int     numAcValuesEncoded () const {return _numAcComp;}
    int     numDcValuesEncoded () const {return _numDcComp;}

  protected:

    void    toZigZag (half *dst, half *src);
    int     countSetBits (unsigned short src);
    half    quantize (half src, float errorTolerance);
    void    rleAc (half *block, unsigned short *&acPtr);

    float                      _quantBaseError;

    int                        _width,
                               _height;
    const unsigned short      *_toNonlinear;

    int                        _numAcComp,
                               _numDcComp;

    std::vector< std::vector<const char *> > _rowPtrs;
    std::vector<PixelType>                   _type;
    std::vector<SimdAlignedBuffer64f>        _dctData;


    //
    // Pointers to the buffers where AC and DC
    // DCT components should be packed for 
    // lossless compression downstream
    //

    char                      *_packedAc;
    char                      *_packedDc;


    //
    // Our "quantization tables" - the example JPEG tables, 
    // normalized so that the smallest value in each is 1.0.
    // This gives us a relationship between error in DCT 
    // components
    //

    float                      _quantTableY[64];
    float                      _quantTableCbCr[64];
};



//
// Single channel lossy DCT encoder
//

class DwaCompressor::LossyDctEncoder: public LossyDctEncoderBase
{
  public:

    LossyDctEncoder
        (float quantBaseError,
         std::vector<const char *> &rowPtrs,
         char *packedAc,
         char *packedDc,
         const unsigned short *toNonlinear,
         int width,
         int height,
         PixelType type)
    :
        LossyDctEncoderBase
            (quantBaseError, packedAc, packedDc, toNonlinear, width, height)
    {
        _rowPtrs.push_back(rowPtrs);
        _type.push_back(type);
    }

    virtual ~LossyDctEncoder () {}
};
    

//
// RGB channel lossy DCT encoder
//

class DwaCompressor::LossyDctEncoderCsc: public LossyDctEncoderBase
{
  public:

    LossyDctEncoderCsc
        (float quantBaseError,
         std::vector<const char *> &rowPtrsR,
         std::vector<const char *> &rowPtrsG,
         std::vector<const char *> &rowPtrsB,
         char *packedAc,
         char *packedDc,
         const unsigned short *toNonlinear,
         int width,
         int height,
         PixelType typeR,
         PixelType typeG,
         PixelType typeB)
    :
        LossyDctEncoderBase
            (quantBaseError, packedAc, packedDc, toNonlinear, width, height)
    {
        _type.push_back(typeR);
        _type.push_back(typeG);
        _type.push_back(typeB);

        _rowPtrs.push_back(rowPtrsR);
        _rowPtrs.push_back(rowPtrsG);
        _rowPtrs.push_back(rowPtrsB);
    }

    virtual ~LossyDctEncoderCsc () {}
};


// ==============================================================
//
//                     LossyDctDecoderBase
//
// --------------------------------------------------------------

DwaCompressor::LossyDctDecoderBase::LossyDctDecoderBase
    (char *packedAc,
     char *packedDc,
     const unsigned short *toLinear,
     int width,
     int height)
:
    _isNativeXdr(false),
    _packedAcCount(0),
    _packedDcCount(0),
    _packedAc(packedAc),
    _packedDc(packedDc),
    _toLinear(toLinear),
    _width(width),
    _height(height)
{
    if (_toLinear == 0)
        _toLinear = get_dwaCompressorNoOp();

    _isNativeXdr = GLOBAL_SYSTEM_LITTLE_ENDIAN;
}


DwaCompressor::LossyDctDecoderBase::~LossyDctDecoderBase () {}


void
DwaCompressor::LossyDctDecoderBase::execute ()
{
    int numComp        = _rowPtrs.size();
    int lastNonZero    = 0;
    int numBlocksX     = (int) ceil ((float)_width  / 8.0f);
    int numBlocksY     = (int) ceil ((float)_height / 8.0f);
    int leftoverX      = _width  - (numBlocksX-1) * 8;
    int leftoverY      = _height - (numBlocksY-1) * 8;

    int numFullBlocksX = (int)floor ((float)_width / 8.0f);

    unsigned short tmpShortNative = 0;
    unsigned short tmpShortXdr    = 0;
    const char *tmpConstCharPtr   = 0;

    unsigned short                    *currAcComp = (unsigned short *)_packedAc;
    std::vector<unsigned short *>      currDcComp (_rowPtrs.size());
    std::vector<SimdAlignedBuffer64us> halfZigBlock (_rowPtrs.size());

    if (_type.size() != _rowPtrs.size())
        throw IEX_NAMESPACE::BaseExc ("Row pointers and types mismatch in count");

    if ((_rowPtrs.size() != 3) && (_rowPtrs.size() != 1))
        throw IEX_NAMESPACE::NoImplExc ("Only 1 and 3 channel encoding is supported");

    _dctData.resize(numComp);

    //
    // Allocate a temp aligned buffer to hold a rows worth of full 
    // 8x8 half-float blocks
    //

    unsigned char *rowBlockHandle = new unsigned char
        [numComp * numBlocksX * 64 * sizeof(unsigned short) + _SSE_ALIGNMENT];

    unsigned short *rowBlock[3];

    rowBlock[0] = (unsigned short*)rowBlockHandle;

    for (int i = 0; i < _SSE_ALIGNMENT; ++i)
    {
        if (((size_t)(rowBlockHandle + i) & _SSE_ALIGNMENT_MASK) == 0)
            rowBlock[0] = (unsigned short *)(rowBlockHandle + i);
    }

    for (int comp = 1; comp < numComp; ++comp)
        rowBlock[comp] = rowBlock[comp - 1] + numBlocksX * 64;
 
    //
    // Pack DC components together by common plane, so we can get 
    // a little more out of differencing them. We'll always have 
    // one component per block, so we can computed offsets.
    //

    currDcComp[0] = (unsigned short *)_packedDc;

    for (unsigned int comp = 1; comp < numComp; ++comp)
        currDcComp[comp] = currDcComp[comp - 1] + numBlocksX * numBlocksY;

    for (int blocky = 0; blocky < numBlocksY; ++blocky)
    {
        int maxY = 8;

        if (blocky == numBlocksY-1)
            maxY = leftoverY;

        int maxX = 8;

        for (int blockx = 0; blockx < numBlocksX; ++blockx)
        {
            if (blockx == numBlocksX-1)
                maxX = leftoverX;

            //
            // If we can detect that the block is constant values
            // (all components only have DC values, and all AC is 0),
            // we can do everything only on 1 value, instead of all
            // 64. 
            //
            // This won't really help for regular images, but it is
            // meant more for layers with large swaths of black 
            //

            bool blockIsConstant = true;

            for (unsigned int comp = 0; comp < numComp; ++comp)
            {

                //
                // DC component is stored separately
                //

                #ifdef IMF_HAVE_SSE2
                    {
                        __m128i *dst = (__m128i*)halfZigBlock[comp]._buffer;

                        dst[7] = _mm_setzero_si128();
                        dst[6] = _mm_setzero_si128();
                        dst[5] = _mm_setzero_si128();
                        dst[4] = _mm_setzero_si128();
                        dst[3] = _mm_setzero_si128();
                        dst[2] = _mm_setzero_si128();
                        dst[1] = _mm_setzero_si128();
                        dst[0] = _mm_insert_epi16
                            (_mm_setzero_si128(), *currDcComp[comp]++, 0);
                    }
                #else  /* IMF_HAVE_SSE2 */

                    memset (halfZigBlock[comp]._buffer, 0, 64 * 2);
                    halfZigBlock[comp]._buffer[0] = *currDcComp[comp]++;

                #endif /* IMF_HAVE_SSE2 */

                _packedDcCount++;
                
                //
                // UnRLE the AC. This will modify currAcComp
                //

                lastNonZero = unRleAc (currAcComp, halfZigBlock[comp]._buffer);

                //
                // Convert from XDR to NATIVE
                //

                if (!_isNativeXdr)
                {
                    for (int i = 0; i < 64; ++i)
                    {
                        tmpShortXdr      = halfZigBlock[comp]._buffer[i];
                        tmpConstCharPtr  = (const char *)&tmpShortXdr;

                        Xdr::read<CharPtrIO> (tmpConstCharPtr, tmpShortNative);

                        halfZigBlock[comp]._buffer[i] = tmpShortNative;
                    }
                }

                if (lastNonZero == 0)
                {
                    //
                    // DC only case - AC components are all 0   
                    //

                    half h;

                    h.setBits (halfZigBlock[comp]._buffer[0]);
                    _dctData[comp]._buffer[0] = (float)h;

                    dctInverse8x8DcOnly (_dctData[comp]._buffer);
                }
                else
                {
                    //
                    // We have some AC components that are non-zero. 
                    // Can't use the 'constant block' optimization
                    //

                    blockIsConstant = false;

                    //
                    // Un-Zig zag 
                    //

                    (*fromHalfZigZag)
                        (halfZigBlock[comp]._buffer, _dctData[comp]._buffer);

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
                        dctInverse8x8_7(_dctData[comp]._buffer);
                    else if (lastNonZero < 3)
                        dctInverse8x8_6(_dctData[comp]._buffer);
                    else if (lastNonZero < 9)
                        dctInverse8x8_5(_dctData[comp]._buffer);
                    else if (lastNonZero < 10)
                        dctInverse8x8_4(_dctData[comp]._buffer);
                    else if (lastNonZero < 20)
                        dctInverse8x8_3(_dctData[comp]._buffer);
                    else if (lastNonZero < 21)
                        dctInverse8x8_2(_dctData[comp]._buffer);
                    else if (lastNonZero < 35)
                        dctInverse8x8_1(_dctData[comp]._buffer);
                    else
                        dctInverse8x8_0(_dctData[comp]._buffer);
                }
            }

            //
            // Perform the CSC
            //

            if (numComp == 3)
            {
                if (!blockIsConstant)
                {
                    csc709Inverse64 (_dctData[0]._buffer, 
                                     _dctData[1]._buffer, 
                                     _dctData[2]._buffer);

                }
                else
                {
                    csc709Inverse (_dctData[0]._buffer[0], 
                                   _dctData[1]._buffer[0], 
                                   _dctData[2]._buffer[0]);
                }
            }

            //
            // Float -> Half conversion. 
            //
            // If the block has a constant value, just convert the first pixel.
            //

            for (unsigned int comp = 0; comp < numComp; ++comp)
            {
                if (!blockIsConstant)
                {
                    (*convertFloatToHalf64)
                        (&rowBlock[comp][blockx*64], _dctData[comp]._buffer);
                }
                else
                {
                    #ifdef IMF_HAVE_SSE2

                        __m128i *dst = (__m128i*)&rowBlock[comp][blockx*64];

                        dst[0] = _mm_set1_epi16
                            (((half)_dctData[comp]._buffer[0]).bits());

                        dst[1] = dst[0];
                        dst[2] = dst[0];
                        dst[3] = dst[0];
                        dst[4] = dst[0];
                        dst[5] = dst[0];
                        dst[6] = dst[0];
                        dst[7] = dst[0];

                    #else  /* IMF_HAVE_SSE2 */

                        unsigned short *dst = &rowBlock[comp][blockx*64];

                        dst[0] = ((half)_dctData[comp]._buffer[0]).bits();

                        for (int i = 1; i < 64; ++i)
                        {
                            dst[i] = dst[0];
                        }

                    #endif /* IMF_HAVE_SSE2 */
                } // blockIsConstant
            } // comp
        } // blockx

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

            bool fastPath = true;

            for (int y = 8 * blocky; y < 8 * blocky + maxY; ++y)
            {
                if ((size_t)_rowPtrs[comp][y] & _SSE_ALIGNMENT_MASK)
                    fastPath = false;
            }

            if (fastPath)
            {
                //
                // Handle all the full X blocks, in a fast path with sse2 and
                // aligned row pointers
                //

                for (int y=8*blocky; y<8*blocky+maxY; ++y)
                {
                    __m128i *dst = (__m128i *)_rowPtrs[comp][y];
                    __m128i *src = (__m128i *)&rowBlock[comp][(y & 0x7) * 8];


                    for (int blockx = 0; blockx < numFullBlocksX; ++blockx)
                    {
                        //
                        // These may need some twiddling.
                        // Run with multiples of 8
                        //

                        _mm_prefetch ((char *)(src + 16), _MM_HINT_NTA); 

                        unsigned short i0  = _mm_extract_epi16 (*src, 0);
                        unsigned short i1  = _mm_extract_epi16 (*src, 1);
                        unsigned short i2  = _mm_extract_epi16 (*src, 2);
                        unsigned short i3  = _mm_extract_epi16 (*src, 3);

                        unsigned short i4  = _mm_extract_epi16 (*src, 4);
                        unsigned short i5  = _mm_extract_epi16 (*src, 5);
                        unsigned short i6  = _mm_extract_epi16 (*src, 6);
                        unsigned short i7  = _mm_extract_epi16 (*src, 7);

                        i0 = _toLinear[i0];
                        i1 = _toLinear[i1];
                        i2 = _toLinear[i2];
                        i3 = _toLinear[i3];

                        i4 = _toLinear[i4];
                        i5 = _toLinear[i5];
                        i6 = _toLinear[i6];
                        i7 = _toLinear[i7];

                        *dst = _mm_insert_epi16 (_mm_setzero_si128(), i0, 0);
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
                    unsigned short *dst = (unsigned short *)_rowPtrs[comp][y];

                    for (int blockx = 0; blockx < numFullBlocksX; ++blockx)
                    {
                        unsigned short *src =
                            &rowBlock[comp][blockx * 64 + ((y & 0x7) * 8)];

                        dst[0] = _toLinear[src[0]];
                        dst[1] = _toLinear[src[1]];
                        dst[2] = _toLinear[src[2]];
                        dst[3] = _toLinear[src[3]];

                        dst[4] = _toLinear[src[4]];
                        dst[5] = _toLinear[src[5]];
                        dst[6] = _toLinear[src[6]];
                        dst[7] = _toLinear[src[7]];

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
                    unsigned short *src = (unsigned short *)
                        &rowBlock[comp][numFullBlocksX * 64 + ((y & 0x7) * 8)];

                    unsigned short *dst = (unsigned short *)_rowPtrs[comp][y];

                    dst += 8 * numFullBlocksX;

                    for (int x = 0; x < maxX; ++x)
                    {
                        *dst++ = _toLinear[*src++];
                    }
                }
            }
        } // comp
    } // blocky

    //
    // Walk over all the channels that are of type FLOAT.
    // Convert from HALF XDR back to FLOAT XDR.
    //

    for (unsigned int chan = 0; chan < numComp; ++chan)
    {

        if (_type[chan] != FLOAT)
            continue;

        std::vector<unsigned short> halfXdr (_width);

        for (int y=0; y<_height; ++y)
        {
            char *floatXdrPtr = _rowPtrs[chan][y];

            memcpy(&halfXdr[0], floatXdrPtr, _width*sizeof(unsigned short));

            const char *halfXdrPtr = (const char *)(&halfXdr[0]);

            for (int x=0; x<_width; ++x)
            {
                half tmpHalf;

                Xdr::read<CharPtrIO> (halfXdrPtr, tmpHalf);
                Xdr::write<CharPtrIO> (floatXdrPtr, (float)tmpHalf);

                // 
                // Xdr::write and Xdr::read will advance the ptrs
                //
            }
        }
    }

    delete[] rowBlockHandle;
}


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

int 
DwaCompressor::LossyDctDecoderBase::unRleAc
    (unsigned short *&currAcComp,
     unsigned short  *halfZigBlock) 
{
    //
    // Un-RLE the RLE'd blocks. If we find an item whose
    // high byte is 0xff, then insert the number of 0's
    // as indicated by the low byte.
    //
    // Otherwise, just copy the number verbaitm.
    //

    int lastNonZero          = 0;
    int dctComp              = 1; 

    //
    // Start with a zero'ed block, so we don't have to
    // write when we hit a run symbol
    //

    while (dctComp < 64)
    {
        if (*currAcComp == 0xff00)
        {
            // 
            // End of block
            //

            dctComp = 64;

        }
        else if ((*currAcComp) >> 8 == 0xff)
        {
            //
            // Run detected! Insert 0's.
            //
            // Since the block has been zeroed, just advance the ptr
            // 

            dctComp += (*currAcComp) & 0xff; 
        }
        else
        {
            // 
            // Not a run, just copy over the value
            //

            lastNonZero = dctComp;
            halfZigBlock[dctComp] = *currAcComp;

            dctComp++;
        }

        _packedAcCount++;
        currAcComp++;
    }

    return lastNonZero;
}


// ==============================================================
//
//                     LossyDctEncoderBase
//
// --------------------------------------------------------------

DwaCompressor::LossyDctEncoderBase::LossyDctEncoderBase
    (float quantBaseError,
     char *packedAc,
     char *packedDc,
     const unsigned short *toNonlinear,
     int width,
     int height)
:
    _quantBaseError(quantBaseError),
    _width(width),
    _height(height),
    _toNonlinear(toNonlinear),
    _numAcComp(0),
    _numDcComp(0),
    _packedAc(packedAc),
    _packedDc(packedDc)
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
    // for now, we'll just use statice quantization tables.
    //

    int jpegQuantTableY[] =
    {
        16,  11,  10,  16,   24,   40,   51,   61,
        12,  12,  14,  19,   26,   58,   60,   55,
        14,  13,  16,  24,   40,   57,   69,   56,
        14,  17,  22,  29,   51,   87,   80,   62,
        18,  22,  37,  56,   68,  109,  103,   77,
        24,  35,  55,  64,   81,  104,  113,   92,
        49,  64,  78,  87,  103,  121,  120,  101,
        72,  92,  95,  98,  112,  100,  103,   99
    };

    int jpegQuantTableYMin = 10;

    int jpegQuantTableCbCr[] =
    {
        17,  18,  24,  47,  99,  99,  99,  99,
        18,  21,  26,  66,  99,  99,  99,  99,
        24,  26,  56,  99,  99,  99,  99,  99,
        47,  66,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99
    };

    int jpegQuantTableCbCrMin = 17;

    for (int idx = 0; idx < 64; ++idx)
    {
        _quantTableY[idx] = static_cast<float> (jpegQuantTableY[idx]) /
                            static_cast<float> (jpegQuantTableYMin);

        _quantTableCbCr[idx] = static_cast<float> (jpegQuantTableCbCr[idx]) /
                               static_cast<float> (jpegQuantTableCbCrMin);
    }
    
    if (_quantBaseError < 0)
        quantBaseError = 0;
}


DwaCompressor::LossyDctEncoderBase::~LossyDctEncoderBase () 
{
}


//
// Given three channels of source data, encoding by first applying
// a color space conversion to a YCbCr space.  Otherwise, if we only
// have one channel, just encode it as is. 
//
// Other numbers of channels are somewhat unexpected at this point,
// and will throw an exception.
//

void
DwaCompressor::LossyDctEncoderBase::execute ()
{
    int  numBlocksX   = (int)ceil ((float)_width / 8.0f);
    int  numBlocksY   = (int)ceil ((float)_height/ 8.0f);

    half halfZigCoef[64]; 
    half halfCoef[64];

    std::vector<unsigned short *> currDcComp (_rowPtrs.size());
    unsigned short               *currAcComp = (unsigned short *)_packedAc;

    _dctData.resize (_rowPtrs.size());
    _numAcComp = 0;
    _numDcComp = 0;
 
    assert (_type.size() == _rowPtrs.size());
    assert ((_rowPtrs.size() == 3) || (_rowPtrs.size() == 1));

    // 
    // Allocate a temp half buffer to quantize into for
    // any FLOAT source channels.
    //

    int tmpHalfBufferElements = 0;

    for (unsigned int chan = 0; chan < _rowPtrs.size(); ++chan)
        if (_type[chan] == FLOAT)
            tmpHalfBufferElements += _width * _height;

    std::vector<unsigned short> tmpHalfBuffer (tmpHalfBufferElements);

    char *tmpHalfBufferPtr = 0;

    if (tmpHalfBufferElements)
        tmpHalfBufferPtr = (char *)&tmpHalfBuffer[0];

    //
    // Run over all the float scanlines, quantizing, 
    // and re-assigning _rowPtr[y]. We need to translate
    // FLOAT XDR to HALF XDR.
    //

    for (unsigned int chan = 0; chan < _rowPtrs.size(); ++chan)
    {
        if (_type[chan] != FLOAT)
            continue;
    
        for (int y = 0; y < _height; ++y)
        {
            float       src = 0;
            const char *srcXdr = _rowPtrs[chan][y];
            char       *dstXdr = tmpHalfBufferPtr;
           
            for (int x = 0; x < _width; ++x)
            {

                Xdr::read<CharPtrIO> (srcXdr, src);

                //
                // Clamp to half ranges, instead of just casting. This
                // avoids introducing Infs which end up getting zeroed later
                //
                src = std::max (
                    std::min ((float) std::numeric_limits<half>::max(), src),
                              (float)-std::numeric_limits<half>::max());

                Xdr::write<CharPtrIO> (dstXdr, ((half)src).bits());

                //
                // Xdr::read and Xdr::write will advance the ptr
                //
            }

            _rowPtrs[chan][y] = (const char *)tmpHalfBufferPtr;
            tmpHalfBufferPtr += _width * sizeof (unsigned short);
        }
    }

    //
    // Pack DC components together by common plane, so we can get 
    // a little more out of differencing them. We'll always have 
    // one component per block, so we can computed offsets.
    //

    currDcComp[0] = (unsigned short *)_packedDc;

    for (unsigned int chan = 1; chan < _rowPtrs.size(); ++chan)
        currDcComp[chan] = currDcComp[chan-1] + numBlocksX * numBlocksY;

    for (int blocky = 0; blocky < numBlocksY; ++blocky)
    {
        for (int blockx = 0; blockx < numBlocksX; ++blockx)
        {
            half           h;
            unsigned short tmpShortXdr, tmpShortNative;
            char          *tmpCharPtr;

            for (unsigned int chan = 0; chan < _rowPtrs.size(); ++chan)
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

                        if (vx >= _width)
                            vx = _width - (vx - (_width - 1));
                        
                        if (vx < 0) vx = _width-1;

                        if (vy >=_height)
                            vy = _height - (vy - (_height - 1));

                        if (vy < 0) vy = _height-1;
                    
                        tmpShortXdr =
                            ((const unsigned short *)(_rowPtrs[chan])[vy])[vx];

                        if (_toNonlinear)
                        {
                            h.setBits (_toNonlinear[tmpShortXdr]);
                        }
                        else
                        {
                            const char *tmpConstCharPtr =
                                (const char *)(&tmpShortXdr);

                            Xdr::read<CharPtrIO>
                                (tmpConstCharPtr, tmpShortNative);

                            h.setBits(tmpShortNative);
                        }

                        _dctData[chan]._buffer[y * 8 + x] = (float)h;
                    } // x
                } // y
            } // chan

            //
            // Color space conversion
            //

            if (_rowPtrs.size() == 3)
            {
                csc709Forward64 (_dctData[0]._buffer, 
                                 _dctData[1]._buffer, 
                                 _dctData[2]._buffer);
            }

            for (unsigned int chan = 0; chan < _rowPtrs.size(); ++chan)
            {
                //
                // Forward DCT
                //

                dctForward8x8(_dctData[chan]._buffer);

                //
                // Quantize to half, and zigzag
                //

                if (chan == 0)
                {
                    for (int i = 0; i < 64; ++i)
                    {
                        halfCoef[i] =
                            quantize ((half)_dctData[chan]._buffer[i],
                                      _quantBaseError*_quantTableY[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < 64; ++i)
                    {
                        halfCoef[i] =
                            quantize ((half)_dctData[chan]._buffer[i],
                                      _quantBaseError*_quantTableCbCr[i]);
                    }
                }

                toZigZag (halfZigCoef, halfCoef);
                
                //
                // Convert from NATIVE back to XDR, before we write out
                //

                for (int i = 0; i < 64; ++i)
                {
                    tmpCharPtr = (char *)&tmpShortXdr;
                    Xdr::write<CharPtrIO>(tmpCharPtr, halfZigCoef[i].bits());
                    halfZigCoef[i].setBits(tmpShortXdr);
                }

                //
                // Save the DC component separately, to be compressed on
                // its own.
                //

                *currDcComp[chan]++ = halfZigCoef[0].bits();
                _numDcComp++;
                
                //
                // Then RLE the AC components (which will record the count
                // of the resulting number of items)
                //

                rleAc (halfZigCoef, currAcComp);
            } // chan
        } // blockx
    } // blocky              
}


// 
// Reorder from zig-zag order to normal ordering
//

void 
DwaCompressor::LossyDctEncoderBase::toZigZag (half *dst, half *src) 
{
    const int remap[] =
    {
         0, 
         1,  8,
        16,  9,  2,
         3, 10, 17, 24,
        32, 25, 18, 11, 4,
         5, 12, 19, 26, 33, 40,
        48, 41, 34, 27, 20, 13, 6,
         7, 14, 21, 28, 35, 42, 49, 56,
            57, 50, 43, 36, 29, 22, 15,
                23, 30, 37, 44, 51, 58,
                    59, 52, 45, 38, 31,
                        39, 46, 53, 60,
                            61, 54, 47,
                                55, 62,
                                    63
    };

    for (int i=0; i<64; ++i)
        dst[i] = src[remap[i]];
}


//
// Precomputing the bit count runs faster than using
// the builtin instruction, at least in one case..
//
// Precomputing 8-bits is no slower than 16-bits,
// and saves a fair bit of overhead..
//

int
DwaCompressor::LossyDctEncoderBase::countSetBits (unsigned short src)
{
    static const unsigned short numBitsSet[256] =
    {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    };

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

half
DwaCompressor::LossyDctEncoderBase::quantize (half src, float errorTolerance)
{
    half            tmp;
    float           srcFloat      = (float)src;
    int             numSetBits    = countSetBits(src.bits());
    const unsigned short *closest = get_dwaClosest(src.bits());

    for (int targetNumSetBits = numSetBits - 1;
         targetNumSetBits >= 0;
         --targetNumSetBits)
    {
        tmp.setBits (*closest);

        if (fabs ((float)tmp - srcFloat) < errorTolerance)
            return tmp;

        closest++;
    }

    return src;
}


//
// RLE the zig-zag of the AC components + copy over 
// into another tmp buffer
//
// Try to do a simple RLE scheme to reduce run's of 0's. This
// differs from the jpeg EOB case, since EOB just indicates that
// the rest of the block is zero. In our case, we have lots of
// NaN symbols, which shouldn't be allowed to occur in DCT 
// coefficents - so we'll use them for encoding runs.
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
DwaCompressor::LossyDctEncoderBase::rleAc
    (half *block,
     unsigned short *&acPtr)
{
    int dctComp              = 1; 
    unsigned short rleSymbol = 0x0;

    while (dctComp < 64)
    {
        int runLen = 1;
    
        //
        // If we don't have a 0, output verbatim
        //

        if (block[dctComp].bits() != rleSymbol)
        {
            *acPtr++ =  block[dctComp].bits();
            _numAcComp++;

            dctComp += runLen;
            continue;
        }

        //
        // We're sitting on a 0, so see how big the run is.
        //

        while ((dctComp+runLen < 64) && 
               (block[dctComp+runLen].bits() == rleSymbol))
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
            runLen           = 1;
            *acPtr++ = block[dctComp].bits();
            _numAcComp++;

            //
            // Using 0xff00 for "end of block"
            //
        }
        else if (runLen + dctComp == 64)
        {
            //
            // Signal EOB
            //

            *acPtr++ = 0xff00;
            _numAcComp++;
        }
        else
        {
            // 
            // Signal normal run
            //

            *acPtr++   = 0xff00 | runLen;
            _numAcComp++;
        }

        //
        // Advance by runLen
        //

        dctComp += runLen;
    }
}


// ==============================================================
//
//                     DwaCompressor
//
// --------------------------------------------------------------

// 
// DwaCompressor()
//

DwaCompressor::DwaCompressor
    (const Header &hdr,
     int maxScanLineSize,
     int numScanLines,
     AcCompression acCompression)
:
    Compressor(hdr),
    _acCompression(acCompression),
    _maxScanLineSize(maxScanLineSize),
    _numScanLines(numScanLines),
    _channels(hdr.channels()),
    _packedAcBuffer(0),
    _packedAcBufferSize(0),
    _packedDcBuffer(0),
    _packedDcBufferSize(0),
    _rleBuffer(0),
    _rleBufferSize(0),
    _outBuffer(0),
    _outBufferSize(0),
    _zip(0),
    _dwaCompressionLevel(45.0)
{
    _min[0] = hdr.dataWindow().min.x;
    _min[1] = hdr.dataWindow().min.y;
    _max[0] = hdr.dataWindow().max.x;
    _max[1] = hdr.dataWindow().max.y;

    for (int i=0; i < NUM_COMPRESSOR_SCHEMES; ++i) 
    {
        _planarUncBuffer[i] = 0;
        _planarUncBufferSize[i] = 0;
    }
    
    //
    // Check the header for a quality attribute
    //

    if (hasDwaCompressionLevel (hdr))
        _dwaCompressionLevel = dwaCompressionLevel (hdr);
}


DwaCompressor::~DwaCompressor()
{
    delete[] _packedAcBuffer;
    delete[] _packedDcBuffer;
    delete[] _rleBuffer;
    delete[] _outBuffer;
    delete _zip;

    for (int i=0; i<NUM_COMPRESSOR_SCHEMES; ++i)
        delete[] _planarUncBuffer[i];
}


int
DwaCompressor::numScanLines() const
{
    return _numScanLines;
}


OPENEXR_IMF_NAMESPACE::Compressor::Format 
DwaCompressor::format() const
{
    if (GLOBAL_SYSTEM_LITTLE_ENDIAN)
        return NATIVE;
    else
        return XDR;
}


int
DwaCompressor::compress
    (const char *inPtr,
     int inSize,
     int minY,
     const char *&outPtr)
{
    return compress
        (inPtr,
         inSize, 
         IMATH_NAMESPACE::Box2i (IMATH_NAMESPACE::V2i (_min[0], minY),
                                 IMATH_NAMESPACE::V2i (_max[0], minY + numScanLines() - 1)),
         outPtr);
}


int
DwaCompressor::compressTile
    (const char             *inPtr,
     int                    inSize,
     IMATH_NAMESPACE::Box2i range,
     const char             *&outPtr)
{
    return compress (inPtr, inSize, range, outPtr);
}


int 
DwaCompressor::compress
    (const char             *inPtr,
     int                    inSize,
     IMATH_NAMESPACE::Box2i range,
     const char             *&outPtr)
{
    const char *inDataPtr   = inPtr;
    char       *packedAcEnd = 0;
    char       *packedDcEnd = 0; 
    int         fileVersion = 2;   // Starting with 2, we write the channel
                                   // classification rules into the file

    if (fileVersion < 2) 
        initializeLegacyChannelRules();
    else 
        initializeDefaultChannelRules();

    size_t outBufferSize = 0;
    initializeBuffers(outBufferSize);

    unsigned short          channelRuleSize = 0;
    std::vector<Classifier> channelRules;
    if (fileVersion >= 2) 
    {
        relevantChannelRules(channelRules);

        channelRuleSize = Xdr::size<unsigned short>();
        for (size_t i = 0; i < channelRules.size(); ++i) 
            channelRuleSize += channelRules[i].size();
    }

    //
    // Remember to allocate _outBuffer, if we haven't done so already.
    //

    outBufferSize += channelRuleSize;
    if (outBufferSize > _outBufferSize) 
    {
        _outBufferSize = outBufferSize;
        if (_outBuffer != 0)
            delete[] _outBuffer;       
        _outBuffer = new char[outBufferSize];
    }

    char *outDataPtr = &_outBuffer[NUM_SIZES_SINGLE * sizeof(OPENEXR_IMF_NAMESPACE::Int64) +
                                   channelRuleSize];

    //
    // We might not be dealing with any color data, in which
    // case the AC buffer size will be 0, and deferencing
    // a vector will not be a good thing to do.
    //

    if (_packedAcBuffer)
        packedAcEnd = _packedAcBuffer;

    if (_packedDcBuffer)
        packedDcEnd = _packedDcBuffer;

    #define OBIDX(x) (Int64 *)&_outBuffer[x * sizeof (Int64)]

    Int64 *version                 = OBIDX (VERSION);
    Int64 *unknownUncompressedSize = OBIDX (UNKNOWN_UNCOMPRESSED_SIZE);
    Int64 *unknownCompressedSize   = OBIDX (UNKNOWN_COMPRESSED_SIZE);
    Int64 *acCompressedSize        = OBIDX (AC_COMPRESSED_SIZE);
    Int64 *dcCompressedSize        = OBIDX (DC_COMPRESSED_SIZE);
    Int64 *rleCompressedSize       = OBIDX (RLE_COMPRESSED_SIZE);
    Int64 *rleUncompressedSize     = OBIDX (RLE_UNCOMPRESSED_SIZE);
    Int64 *rleRawSize              = OBIDX (RLE_RAW_SIZE);

    Int64 *totalAcUncompressedCount = OBIDX (AC_UNCOMPRESSED_COUNT);
    Int64 *totalDcUncompressedCount = OBIDX (DC_UNCOMPRESSED_COUNT);

    Int64 *acCompression            = OBIDX (AC_COMPRESSION);

    int minX   = range.min.x;
    int maxX   = std::min(range.max.x, _max[0]);
    int minY   = range.min.y;
    int maxY   = std::min(range.max.y, _max[1]);

    //
    // Zero all the numbers in the chunk header
    //

    memset (_outBuffer, 0, NUM_SIZES_SINGLE * sizeof (Int64));

    //
    // Setup the AC compression strategy and the version in the data block,
    // then write the relevant channel classification rules if needed
    //
    *version       = fileVersion;  
    *acCompression = _acCompression;

    setupChannelData (minX, minY, maxX, maxY);

    if (fileVersion >= 2) 
    {
        char *writePtr = &_outBuffer[NUM_SIZES_SINGLE * sizeof(OPENEXR_IMF_NAMESPACE::Int64)];
        Xdr::write<CharPtrIO> (writePtr, channelRuleSize);
        
        for (size_t i = 0; i < channelRules.size(); ++i) 
            channelRules[i].write(writePtr);
    }

    //
    // Determine the start of each row in the input buffer
    // Channels are interleaved by scanline
    //

    std::vector<bool> encodedChannels (_channelData.size());
    std::vector< std::vector<const char *> > rowPtrs (_channelData.size());

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
        encodedChannels[chan] = false;

    inDataPtr =  inPtr;

    for (int y = minY; y <= maxY; ++y)
    {
        for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
        {

            ChannelData *cd = &_channelData[chan];

            if (IMATH_NAMESPACE::modp(y, cd->ySampling) != 0)
                continue;

            rowPtrs[chan].push_back(inDataPtr);
            inDataPtr += cd->width * OPENEXR_IMF_NAMESPACE::pixelTypeSize(cd->type);
        }
    }

    inDataPtr = inPtr;

    // 
    // Make a pass over all our CSC sets and try to encode them first
    // 

    for (unsigned int csc = 0; csc < _cscSets.size(); ++csc)
    {

        LossyDctEncoderCsc encoder
            (_dwaCompressionLevel / 100000.f,
             rowPtrs[_cscSets[csc].idx[0]],
             rowPtrs[_cscSets[csc].idx[1]],
             rowPtrs[_cscSets[csc].idx[2]],
             packedAcEnd,
             packedDcEnd,
             get_dwaCompressorToNonlinear(),
             _channelData[_cscSets[csc].idx[0]].width,
             _channelData[_cscSets[csc].idx[0]].height,
             _channelData[_cscSets[csc].idx[0]].type,
             _channelData[_cscSets[csc].idx[1]].type,
             _channelData[_cscSets[csc].idx[2]].type);

        encoder.execute();

        *totalAcUncompressedCount  += encoder.numAcValuesEncoded();
        *totalDcUncompressedCount  += encoder.numDcValuesEncoded();

        packedAcEnd += encoder.numAcValuesEncoded() * sizeof(unsigned short);
        packedDcEnd += encoder.numDcValuesEncoded() * sizeof(unsigned short);

        encodedChannels[_cscSets[csc].idx[0]] = true;
        encodedChannels[_cscSets[csc].idx[1]] = true;
        encodedChannels[_cscSets[csc].idx[2]] = true;
    }

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
    {
        ChannelData *cd = &_channelData[chan];

        if (encodedChannels[chan])
            continue;

        switch (cd->compression)
        {
          case LOSSY_DCT:

            //
            // For LOSSY_DCT, treat this just like the CSC'd case,
            // but only operate on one channel
            //

            {
                const unsigned short *nonlinearLut = 0;

                if (!cd->pLinear)
                    nonlinearLut = get_dwaCompressorToNonlinear(); 

                LossyDctEncoder encoder
                    (_dwaCompressionLevel / 100000.f,
                     rowPtrs[chan],
                     packedAcEnd,
                     packedDcEnd,
                     nonlinearLut,
                     cd->width,
                     cd->height,
                     cd->type);

                encoder.execute();

                *totalAcUncompressedCount  += encoder.numAcValuesEncoded();
                *totalDcUncompressedCount  += encoder.numDcValuesEncoded();

                packedAcEnd +=
                    encoder.numAcValuesEncoded() * sizeof (unsigned short);

                packedDcEnd +=
                    encoder.numDcValuesEncoded() * sizeof (unsigned short);
            }

            break;

          case RLE:

            //
            // For RLE, bash the bytes up so that the first bytes of each
            // pixel are contingous, as are the second bytes, and so on.
            //

            for (unsigned int y = 0; y < rowPtrs[chan].size(); ++y)
            {
                const char *row = rowPtrs[chan][y];

                for (int x = 0; x < cd->width; ++x)
                {
                    for (int byte = 0;
                         byte < OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->type);
                         ++byte)
                    {
                            
                        *cd->planarUncRleEnd[byte]++ = *row++;
                    }
                }

                *rleRawSize += cd->width * OPENEXR_IMF_NAMESPACE::pixelTypeSize(cd->type);
            }

            break;

          case UNKNOWN:
           
            //
            // Otherwise, just copy data over verbatim
            //

            {
                int scanlineSize = cd->width * OPENEXR_IMF_NAMESPACE::pixelTypeSize(cd->type);

                for (unsigned int y = 0; y < rowPtrs[chan].size(); ++y)
                {
                    memcpy (cd->planarUncBufferEnd,
                            rowPtrs[chan][y],
                            scanlineSize);
    
                    cd->planarUncBufferEnd += scanlineSize;
                }

                *unknownUncompressedSize += cd->planarUncSize;
            }

            break;

          default:

            assert (false);
        }

        encodedChannels[chan] = true;
    }

    //
    // Pack the Unknown data into the output buffer first. Instead of
    // just copying it uncompressed, try zlib compression at least.
    //

    if (*unknownUncompressedSize > 0)
    {
        uLongf inSize  = (uLongf)(*unknownUncompressedSize);
        uLongf outSize = compressBound (inSize);

        if (Z_OK != ::compress2 ((Bytef *)outDataPtr,
                                 &outSize,
                                 (const Bytef *)_planarUncBuffer[UNKNOWN],
                                 inSize,
                                 9))
        {
            throw IEX_NAMESPACE::BaseExc ("Data compression (zlib) failed.");
        }

        outDataPtr += outSize;
        *unknownCompressedSize = outSize;
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
        switch (_acCompression)
        {
          case STATIC_HUFFMAN:

            *acCompressedSize = (int)
                hufCompress((unsigned short *)_packedAcBuffer,
                            (int)*totalAcUncompressedCount,
                            outDataPtr);                
            break;

          case DEFLATE:

            {
                uLongf destLen = compressBound (
                    (*totalAcUncompressedCount) * sizeof (unsigned short));

                if (Z_OK != ::compress2
                                ((Bytef *)outDataPtr,
                                 &destLen,
                                 (Bytef *)_packedAcBuffer, 
                                 (uLong)(*totalAcUncompressedCount
                                                * sizeof (unsigned short)),
                                 9))
                {
                    throw IEX_NAMESPACE::InputExc ("Data compression (zlib) failed.");
                }

                *acCompressedSize = destLen;        
            }

            break;

          default:
            
            assert (false);
        }

        outDataPtr += *acCompressedSize;
    }

    // 
    // Handle the DC components separately
    //

    if (*totalDcUncompressedCount > 0)
    {
        *dcCompressedSize = _zip->compress
            (_packedDcBuffer,
             (int)(*totalDcUncompressedCount) * sizeof (unsigned short),
             outDataPtr);

        outDataPtr += *dcCompressedSize;
    }

    // 
    // If we have RLE data, first RLE encode it and set the uncompressed
    // size. Then, deflate the results and set the compressed size.
    //    

    if (*rleRawSize > 0)
    {
        *rleUncompressedSize = rleCompress
            ((int)(*rleRawSize),
             _planarUncBuffer[RLE],
             (signed char *)_rleBuffer);

        uLongf dstLen = compressBound ((uLongf)*rleUncompressedSize);

        if (Z_OK != ::compress2
                        ((Bytef *)outDataPtr, 
                         &dstLen, 
                         (Bytef *)_rleBuffer, 
                         (uLong)(*rleUncompressedSize),
                         9))
        {
            throw IEX_NAMESPACE::BaseExc ("Error compressing RLE'd data.");
        }
        
       *rleCompressedSize = dstLen;
        outDataPtr       += *rleCompressedSize;
    }

    // 
    // Flip the counters to XDR format
    //         

    for (int i = 0; i < NUM_SIZES_SINGLE; ++i)
    {
        Int64  src = *(((Int64 *)_outBuffer) + i);
        char  *dst = (char *)(((Int64 *)_outBuffer) + i);

        Xdr::write<CharPtrIO> (dst, src);
    }

    //
    // We're done - compute the number of bytes we packed
    //

    outPtr = _outBuffer;

    return static_cast<int>(outDataPtr - _outBuffer + 1);
}


int
DwaCompressor::uncompress
    (const char *inPtr,
     int inSize,
     int minY,
     const char *&outPtr)
{
    return uncompress (inPtr,
                       inSize,
                       IMATH_NAMESPACE::Box2i (IMATH_NAMESPACE::V2i (_min[0], minY),
                       IMATH_NAMESPACE::V2i (_max[0], minY + numScanLines() - 1)),
                       outPtr);
}


int 
DwaCompressor::uncompressTile
    (const char *inPtr,
     int inSize,
     IMATH_NAMESPACE::Box2i range,
     const char *&outPtr)
{
    return uncompress (inPtr, inSize, range, outPtr);
}


int 
DwaCompressor::uncompress
    (const char *inPtr,
     int inSize,
     IMATH_NAMESPACE::Box2i range,
     const char *&outPtr)
{
    int minX = range.min.x;
    int maxX = std::min (range.max.x, _max[0]);
    int minY = range.min.y;
    int maxY = std::min (range.max.y, _max[1]);

    int headerSize = NUM_SIZES_SINGLE*sizeof(Int64);
    if (inSize < headerSize) 
    {
        throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                            "(truncated header).");
    }

    // 
    // Flip the counters from XDR to NATIVE
    //

    for (int i = 0; i < NUM_SIZES_SINGLE; ++i)
    {
        Int64      *dst =  (((Int64 *)inPtr) + i);
        const char *src = (char *)(((Int64 *)inPtr) + i);

        Xdr::read<CharPtrIO> (src, *dst);
    }

    //
    // Unwind all the counter info
    //

    const Int64 *inPtr64 = (const Int64*) inPtr;

    Int64 version                  = *(inPtr64 + VERSION);
    Int64 unknownUncompressedSize  = *(inPtr64 + UNKNOWN_UNCOMPRESSED_SIZE);
    Int64 unknownCompressedSize    = *(inPtr64 + UNKNOWN_COMPRESSED_SIZE);
    Int64 acCompressedSize         = *(inPtr64 + AC_COMPRESSED_SIZE);
    Int64 dcCompressedSize         = *(inPtr64 + DC_COMPRESSED_SIZE);
    Int64 rleCompressedSize        = *(inPtr64 + RLE_COMPRESSED_SIZE);
    Int64 rleUncompressedSize      = *(inPtr64 + RLE_UNCOMPRESSED_SIZE);
    Int64 rleRawSize               = *(inPtr64 + RLE_RAW_SIZE);
 
    Int64 totalAcUncompressedCount = *(inPtr64 + AC_UNCOMPRESSED_COUNT); 
    Int64 totalDcUncompressedCount = *(inPtr64 + DC_UNCOMPRESSED_COUNT); 

    Int64 acCompression            = *(inPtr64 + AC_COMPRESSION); 

    Int64 compressedSize           = unknownCompressedSize + 
                                     acCompressedSize +
                                     dcCompressedSize +
                                     rleCompressedSize;

    const char *dataPtr            = inPtr + NUM_SIZES_SINGLE * sizeof(Int64);

    /* Both the sum and individual sizes are checked in case of overflow. */
    if (inSize < (headerSize + compressedSize) ||
        inSize < unknownCompressedSize ||
        inSize < acCompressedSize ||
        inSize < dcCompressedSize ||
        inSize < rleCompressedSize)
    {
        throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                            "(truncated file).");
    }

    if ((SInt64)unknownUncompressedSize < 0  ||
        (SInt64)unknownCompressedSize < 0    ||
        (SInt64)acCompressedSize < 0         ||
        (SInt64)dcCompressedSize < 0         ||
        (SInt64)rleCompressedSize < 0        ||
        (SInt64)rleUncompressedSize < 0      ||
        (SInt64)rleRawSize < 0               ||
        (SInt64)totalAcUncompressedCount < 0 ||
        (SInt64)totalDcUncompressedCount < 0)
    {
        throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                            " (corrupt header).");
    }

    if (version < 2) 
        initializeLegacyChannelRules();
    else
    {
        unsigned short ruleSize = 0;
        Xdr::read<CharPtrIO>(dataPtr, ruleSize);

        if (ruleSize < 0) 
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (corrupt header file).");

        headerSize += ruleSize;
        if (inSize < headerSize + compressedSize)
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                " (truncated file).");

        _channelRules.clear();
        ruleSize -= Xdr::size<unsigned short> ();
        while (ruleSize > 0) 
        {
            Classifier rule(dataPtr, ruleSize);
            
            _channelRules.push_back(rule);
            ruleSize -= rule.size();
        }
    }


    size_t outBufferSize = 0;
    initializeBuffers(outBufferSize);

    //
    // Allocate _outBuffer, if we haven't done so already
    //

    if (_maxScanLineSize * numScanLines() > _outBufferSize) 
    {
        _outBufferSize = _maxScanLineSize * numScanLines();
        if (_outBuffer != 0)
            delete[] _outBuffer;
        _outBuffer = new char[_maxScanLineSize * numScanLines()];
    }


    char *outBufferEnd = _outBuffer;

       
    //
    // Find the start of the RLE packed AC components and
    // the DC components for each channel. This will be handy   
    // if you want to decode the channels in parallel later on.
    //

    char *packedAcBufferEnd = 0; 

    if (_packedAcBuffer)
        packedAcBufferEnd = _packedAcBuffer;

    char *packedDcBufferEnd = 0;

    if (_packedDcBuffer)
        packedDcBufferEnd = _packedDcBuffer;

    //
    // UNKNOWN data is packed first, followed by the 
    // Huffman-compressed AC, then the DC values, 
    // and then the zlib compressed RLE data.
    //
    
    const char *compressedUnknownBuf = dataPtr;

    const char *compressedAcBuf      = compressedUnknownBuf + 
                                  static_cast<ptrdiff_t>(unknownCompressedSize);
    const char *compressedDcBuf      = compressedAcBuf +
                                  static_cast<ptrdiff_t>(acCompressedSize);
    const char *compressedRleBuf     = compressedDcBuf + 
                                  static_cast<ptrdiff_t>(dcCompressedSize);

    // 
    // Sanity check that the version is something we expect. Right now, 
    // we can decode version 0, 1, and 2. v1 adds 'end of block' symbols
    // to the AC RLE. v2 adds channel classification rules at the 
    // start of the data block.
    //

    if (version > 2)
        throw IEX_NAMESPACE::InputExc ("Invalid version of compressed data block");    

    setupChannelData(minX, minY, maxX, maxY);

    // 
    // Uncompress the UNKNOWN data into _planarUncBuffer[UNKNOWN]
    //

    if (unknownCompressedSize > 0)
    {
        if (unknownUncompressedSize > _planarUncBufferSize[UNKNOWN]) 
        {
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                "(corrupt header).");
        }

        uLongf outSize = (uLongf)unknownUncompressedSize;

        if (Z_OK != ::uncompress
                        ((Bytef *)_planarUncBuffer[UNKNOWN],
                         &outSize,
                         (Bytef *)compressedUnknownBuf,
                         (uLong)unknownCompressedSize))
        {
            throw IEX_NAMESPACE::BaseExc("Error uncompressing UNKNOWN data.");
        }
    }

    // 
    // Uncompress the AC data into _packedAcBuffer
    //

    if (acCompressedSize > 0)
    {
        if (totalAcUncompressedCount*sizeof(unsigned short) > _packedAcBufferSize)
        {
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                "(corrupt header).");
        }

        //
        // Don't trust the user to get it right, look in the file.
        //

        switch (acCompression)
        {
          case STATIC_HUFFMAN:

            hufUncompress
                (compressedAcBuf, 
                 (int)acCompressedSize, 
                 (unsigned short *)_packedAcBuffer, 
                 (int)totalAcUncompressedCount); 

            break;

          case DEFLATE:
            {
                uLongf destLen =
                    (int)(totalAcUncompressedCount) * sizeof (unsigned short);

                if (Z_OK != ::uncompress
                                ((Bytef *)_packedAcBuffer,
                                 &destLen,
                                 (Bytef *)compressedAcBuf,
                                 (uLong)acCompressedSize))
                {
                    throw IEX_NAMESPACE::InputExc ("Data decompression (zlib) failed.");
                }

                if (totalAcUncompressedCount * sizeof (unsigned short) !=
                                destLen)
                {
                    throw IEX_NAMESPACE::InputExc ("AC data corrupt.");     
                }
            }
            break;

          default:

            throw IEX_NAMESPACE::NoImplExc ("Unknown AC Compression");
            break;
        }
    }

    //
    // Uncompress the DC data into _packedDcBuffer
    //

    if (dcCompressedSize > 0)
    {
        if (totalDcUncompressedCount*sizeof(unsigned short) > _packedDcBufferSize)
        {
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                "(corrupt header).");
        }

        if (_zip->uncompress
                    (compressedDcBuf, (int)dcCompressedSize, _packedDcBuffer)
            != (int)totalDcUncompressedCount * sizeof (unsigned short))
        {
            throw IEX_NAMESPACE::BaseExc("DC data corrupt.");
        }
    }

    //
    // Uncompress the RLE data into _rleBuffer, then unRLE the results
    // into _planarUncBuffer[RLE]
    //

    if (rleRawSize > 0)
    {
        if (rleUncompressedSize > _rleBufferSize ||
            rleRawSize > _planarUncBufferSize[RLE])
        {
            throw IEX_NAMESPACE::InputExc("Error uncompressing DWA data"
                                "(corrupt header).");
        }
 
        uLongf dstLen = (uLongf)rleUncompressedSize;

        if (Z_OK != ::uncompress
                        ((Bytef *)_rleBuffer,
                         &dstLen,
                         (Bytef *)compressedRleBuf,
                         (uLong)rleCompressedSize))
        {
            throw IEX_NAMESPACE::BaseExc("Error uncompressing RLE data.");
        }

        if (dstLen != rleUncompressedSize)
            throw IEX_NAMESPACE::BaseExc("RLE data corrupted");

        if (rleUncompress
                ((int)rleUncompressedSize, 
                 (int)rleRawSize,
                 (signed char *)_rleBuffer,
                 _planarUncBuffer[RLE]) != rleRawSize)
        {        
            throw IEX_NAMESPACE::BaseExc("RLE data corrupted");
        }
    }

    //
    // Determine the start of each row in the output buffer
    //

    std::vector<bool> decodedChannels (_channelData.size());
    std::vector< std::vector<char *> > rowPtrs (_channelData.size());

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
        decodedChannels[chan] = false;

    outBufferEnd = _outBuffer;

    for (int y = minY; y <= maxY; ++y)
    {
        for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
        {
            ChannelData *cd = &_channelData[chan];

            if (IMATH_NAMESPACE::modp (y, cd->ySampling) != 0)
                continue;

            rowPtrs[chan].push_back (outBufferEnd);
            outBufferEnd += cd->width * OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->type);
        }
    }

    //
    // Setup to decode each block of 3 channels that need to
    // be handled together
    //

    for (unsigned int csc = 0; csc < _cscSets.size(); ++csc)
    {
        int rChan = _cscSets[csc].idx[0];    
        int gChan = _cscSets[csc].idx[1];    
        int bChan = _cscSets[csc].idx[2];    


        LossyDctDecoderCsc decoder
            (rowPtrs[rChan],
             rowPtrs[gChan],
             rowPtrs[bChan],
             packedAcBufferEnd,
             packedDcBufferEnd,
             get_dwaCompressorToLinear(),
             _channelData[rChan].width,
             _channelData[rChan].height,
             _channelData[rChan].type,
             _channelData[gChan].type,
             _channelData[bChan].type);

        decoder.execute();

        packedAcBufferEnd +=
            decoder.numAcValuesEncoded() * sizeof (unsigned short);

        packedDcBufferEnd +=
            decoder.numDcValuesEncoded() * sizeof (unsigned short);

        decodedChannels[rChan] = true;
        decodedChannels[gChan] = true;
        decodedChannels[bChan] = true;
    }

    //
    // Setup to handle the remaining channels by themselves
    //

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
    {
        if (decodedChannels[chan])
            continue;

        ChannelData *cd = &_channelData[chan];
        int pixelSize = OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->type);

        switch (cd->compression)
        {
          case LOSSY_DCT:

            //
            // Setup a single-channel lossy DCT decoder pointing
            // at the output buffer
            //

            {
                const unsigned short *linearLut = 0;

                if (!cd->pLinear)
                    linearLut = get_dwaCompressorToLinear();

                LossyDctDecoder decoder
                    (rowPtrs[chan],
                     packedAcBufferEnd,
                     packedDcBufferEnd,
                     linearLut,
                     cd->width,
                     cd->height,
                     cd->type);

                decoder.execute();   

                packedAcBufferEnd += 
                    decoder.numAcValuesEncoded() * sizeof (unsigned short);

                packedDcBufferEnd += 
                    decoder.numDcValuesEncoded() * sizeof (unsigned short);
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

                for (int y = minY; y <= maxY; ++y)
                {
                    if (IMATH_NAMESPACE::modp (y, cd->ySampling) != 0)
                        continue;

                    char *dst = rowPtrs[chan][row];

                    if (pixelSize == 2)
                    {
                        interleaveByte2 (dst, 
                                         cd->planarUncRleEnd[0],
                                         cd->planarUncRleEnd[1],
                                         cd->width);
                                            
                        cd->planarUncRleEnd[0] += cd->width;
                        cd->planarUncRleEnd[1] += cd->width;
                    }
                    else
                    {
                        for (int x = 0; x < cd->width; ++x)
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
                int row             = 0;
                int dstScanlineSize = cd->width * OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->type);

                for (int y = minY; y <= maxY; ++y)
                {
                    if (IMATH_NAMESPACE::modp (y, cd->ySampling) != 0)
                        continue;

                    memcpy (rowPtrs[chan][row],
                            cd->planarUncBufferEnd,
                            dstScanlineSize);

                    cd->planarUncBufferEnd += dstScanlineSize;
                    row++;
                }
            }

            break;

          default:

            throw IEX_NAMESPACE::NoImplExc ("Unhandled compression scheme case");
            break;
        }

        decodedChannels[chan] = true;
    }

    //
    // Return a ptr to _outBuffer
    //

    outPtr = _outBuffer;
    return (int)(outBufferEnd - _outBuffer);
}


// static
void
DwaCompressor::initializeFuncs()
{
    convertFloatToHalf64 = convertFloatToHalf64_scalar;
    fromHalfZigZag       = fromHalfZigZag_scalar;

    CpuId cpuId;

    //
    // Setup HALF <-> FLOAT conversion implementations
    //

    if (cpuId.avx && cpuId.f16c)
    {
        convertFloatToHalf64 = convertFloatToHalf64_f16c;
        fromHalfZigZag       = fromHalfZigZag_f16c;
    } 

    //
    // Setup inverse DCT implementations
    //

    dctInverse8x8_0 = dctInverse8x8_scalar<0>;
    dctInverse8x8_1 = dctInverse8x8_scalar<1>;
    dctInverse8x8_2 = dctInverse8x8_scalar<2>;
    dctInverse8x8_3 = dctInverse8x8_scalar<3>;
    dctInverse8x8_4 = dctInverse8x8_scalar<4>;
    dctInverse8x8_5 = dctInverse8x8_scalar<5>;
    dctInverse8x8_6 = dctInverse8x8_scalar<6>;
    dctInverse8x8_7 = dctInverse8x8_scalar<7>;

    if (cpuId.avx) 
    {
        dctInverse8x8_0 = dctInverse8x8_avx<0>;
        dctInverse8x8_1 = dctInverse8x8_avx<1>;
        dctInverse8x8_2 = dctInverse8x8_avx<2>;
        dctInverse8x8_3 = dctInverse8x8_avx<3>;
        dctInverse8x8_4 = dctInverse8x8_avx<4>;
        dctInverse8x8_5 = dctInverse8x8_avx<5>;
        dctInverse8x8_6 = dctInverse8x8_avx<6>;
        dctInverse8x8_7 = dctInverse8x8_avx<7>;
    } 
    else if (cpuId.sse2) 
    {
        dctInverse8x8_0 = dctInverse8x8_sse2<0>;
        dctInverse8x8_1 = dctInverse8x8_sse2<1>;
        dctInverse8x8_2 = dctInverse8x8_sse2<2>;
        dctInverse8x8_3 = dctInverse8x8_sse2<3>;
        dctInverse8x8_4 = dctInverse8x8_sse2<4>;
        dctInverse8x8_5 = dctInverse8x8_sse2<5>;
        dctInverse8x8_6 = dctInverse8x8_sse2<6>;
        dctInverse8x8_7 = dctInverse8x8_sse2<7>;
    }
}


//
// Handle channel classification and buffer allocation once we know
// how to classify channels
//

void
DwaCompressor::initializeBuffers (size_t &outBufferSize)
{
    classifyChannels (_channels, _channelData, _cscSets);

    //
    // _outBuffer needs to be big enough to hold all our 
    // compressed data - which could vary depending on what sort
    // of channels we have. 
    //

    int maxOutBufferSize  = 0;
    int numLossyDctChans  = 0;
    int unknownBufferSize = 0;
    int rleBufferSize     = 0;

    int maxLossyDctAcSize = (int)ceil ((float)numScanLines() / 8.0f) * 
                            (int)ceil ((float)(_max[0] - _min[0] + 1) / 8.0f) *
                            63 * sizeof (unsigned short);

    int maxLossyDctDcSize = (int)ceil ((float)numScanLines() / 8.0f) * 
                            (int)ceil ((float)(_max[0] - _min[0] + 1) / 8.0f) *
                            sizeof (unsigned short);

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
    {
        switch (_channelData[chan].compression)
        {
          case LOSSY_DCT:

            //
            // This is the size of the number of packed
            // components, plus the requirements for
            // maximum Huffman encoding size (for STATIC_HUFFMAN)
            // or for zlib compression (for DEFLATE)
            //

            maxOutBufferSize += std::max(
                            (int)(2 * maxLossyDctAcSize + 65536),
                            (int)compressBound (maxLossyDctAcSize) );
            numLossyDctChans++;
            break;

          case RLE:
            {
                //
                // RLE, if gone horribly wrong, could double the size
                // of the source data.
                //

                int rleAmount = 2 * numScanLines() * (_max[0] - _min[0] + 1) *
                                OPENEXR_IMF_NAMESPACE::pixelTypeSize (_channelData[chan].type);

                rleBufferSize += rleAmount;
            }
            break;


          case UNKNOWN:

            unknownBufferSize += numScanLines() * (_max[0] - _min[0] + 1) *
                                 OPENEXR_IMF_NAMESPACE::pixelTypeSize (_channelData[chan].type);
            break;

          default:

            throw IEX_NAMESPACE::NoImplExc ("Unhandled compression scheme case");
            break;
        }
    }

    //
    // Also, since the results of the RLE are packed into 
    // the output buffer, we need the extra room there. But
    // we're going to zlib compress() the data we pack, 
    // which could take slightly more space
    //

    maxOutBufferSize += (int)compressBound ((uLongf)rleBufferSize);
    
    //
    // And the same goes for the UNKNOWN data
    //

    maxOutBufferSize += (int)compressBound ((uLongf)unknownBufferSize);

    //
    // Allocate a zip/deflate compressor big enought to hold the DC data
    // and include it's compressed results in the size requirements
    // for our output buffer
    //

    if (_zip == 0) 
        _zip = new Zip (maxLossyDctDcSize * numLossyDctChans);
    else if (_zip->maxRawSize() < maxLossyDctDcSize * numLossyDctChans)
    {
        delete _zip;
        _zip = new Zip (maxLossyDctDcSize * numLossyDctChans);
    }


    maxOutBufferSize += _zip->maxCompressedSize();

    //
    // We also need to reserve space at the head of the buffer to 
    // write out the size of our various packed and compressed data.
    //

    maxOutBufferSize += NUM_SIZES_SINGLE * sizeof (Int64); 
                    

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

    outBufferSize = maxOutBufferSize;


    //
    // _packedAcBuffer holds the quantized DCT coefficients prior
    // to Huffman encoding
    //

    if (maxLossyDctAcSize * numLossyDctChans > _packedAcBufferSize)
    {
        _packedAcBufferSize = maxLossyDctAcSize * numLossyDctChans;
        if (_packedAcBuffer != 0) 
            delete[] _packedAcBuffer;
        _packedAcBuffer = new char[_packedAcBufferSize];
    }

    //
    // _packedDcBuffer holds one quantized DCT coef per 8x8 block
    //

    if (maxLossyDctDcSize * numLossyDctChans > _packedDcBufferSize)
    {
        _packedDcBufferSize = maxLossyDctDcSize * numLossyDctChans;
        if (_packedDcBuffer != 0) 
            delete[] _packedDcBuffer;
        _packedDcBuffer     = new char[_packedDcBufferSize];
    }

    if (rleBufferSize > _rleBufferSize) 
    {
        _rleBufferSize = rleBufferSize;
        if (_rleBuffer != 0) 
            delete[] _rleBuffer;
        _rleBuffer = new char[rleBufferSize];
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

    int planarUncBufferSize[NUM_COMPRESSOR_SCHEMES];
    for (int i=0; i<NUM_COMPRESSOR_SCHEMES; ++i)
        planarUncBufferSize[i] = 0;

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
    {
        switch (_channelData[chan].compression)
        {
          case LOSSY_DCT:
            break;

          case RLE:
            planarUncBufferSize[RLE] +=
                     numScanLines() * (_max[0] - _min[0] + 1) *
                     OPENEXR_IMF_NAMESPACE::pixelTypeSize (_channelData[chan].type);
            break;

          case UNKNOWN: 
            planarUncBufferSize[UNKNOWN] +=
                     numScanLines() * (_max[0] - _min[0] + 1) *
                     OPENEXR_IMF_NAMESPACE::pixelTypeSize (_channelData[chan].type);
            break;

          default:
            throw IEX_NAMESPACE::NoImplExc ("Unhandled compression scheme case");
            break;
        }
    }

    //
    // UNKNOWN data is going to be zlib compressed, which needs 
    // a little extra headroom
    //

    if (planarUncBufferSize[UNKNOWN] > 0)
    {
        planarUncBufferSize[UNKNOWN] = 
            compressBound ((uLongf)planarUncBufferSize[UNKNOWN]);
    }

    for (int i = 0; i < NUM_COMPRESSOR_SCHEMES; ++i)
    {
        if (planarUncBufferSize[i] > _planarUncBufferSize[i]) 
        {
            _planarUncBufferSize[i] = planarUncBufferSize[i];
            if (_planarUncBuffer[i] != 0) 
                delete[] _planarUncBuffer[i];
            _planarUncBuffer[i] = new char[planarUncBufferSize[i]];
        }
    }
}


//
// Setup channel classification rules to use when writing files
//

void
DwaCompressor::initializeDefaultChannelRules ()
{
    _channelRules.clear();

    _channelRules.push_back (Classifier ("R",     LOSSY_DCT, HALF,   0, false));
    _channelRules.push_back (Classifier ("R",     LOSSY_DCT, FLOAT,  0, false));
    _channelRules.push_back (Classifier ("G",     LOSSY_DCT, HALF,   1, false));
    _channelRules.push_back (Classifier ("G",     LOSSY_DCT, FLOAT,  1, false));
    _channelRules.push_back (Classifier ("B",     LOSSY_DCT, HALF,   2, false));
    _channelRules.push_back (Classifier ("B",     LOSSY_DCT, FLOAT,  2, false));

    _channelRules.push_back (Classifier ("Y",     LOSSY_DCT, HALF,  -1, false));
    _channelRules.push_back (Classifier ("Y",     LOSSY_DCT, FLOAT, -1, false));
    _channelRules.push_back (Classifier ("BY",    LOSSY_DCT, HALF,  -1, false));
    _channelRules.push_back (Classifier ("BY",    LOSSY_DCT, FLOAT, -1, false));
    _channelRules.push_back (Classifier ("RY",    LOSSY_DCT, HALF,  -1, false));
    _channelRules.push_back (Classifier ("RY",    LOSSY_DCT, FLOAT, -1, false));

    _channelRules.push_back (Classifier ("A",     RLE,       UINT,  -1, false));
    _channelRules.push_back (Classifier ("A",     RLE,       HALF,  -1, false));
    _channelRules.push_back (Classifier ("A",     RLE,       FLOAT, -1, false));
}


//
// Setup channel classification rules when reading files with VERSION < 2
//

void
DwaCompressor::initializeLegacyChannelRules ()
{
    _channelRules.clear();

    _channelRules.push_back (Classifier ("r",     LOSSY_DCT, HALF,   0, true));
    _channelRules.push_back (Classifier ("r",     LOSSY_DCT, FLOAT,  0, true));
    _channelRules.push_back (Classifier ("red",   LOSSY_DCT, HALF,   0, true));
    _channelRules.push_back (Classifier ("red",   LOSSY_DCT, FLOAT,  0, true));
    _channelRules.push_back (Classifier ("g",     LOSSY_DCT, HALF,   1, true));
    _channelRules.push_back (Classifier ("g",     LOSSY_DCT, FLOAT,  1, true));
    _channelRules.push_back (Classifier ("grn",   LOSSY_DCT, HALF,   1, true));
    _channelRules.push_back (Classifier ("grn",   LOSSY_DCT, FLOAT,  1, true));
    _channelRules.push_back (Classifier ("green", LOSSY_DCT, HALF,   1, true));
    _channelRules.push_back (Classifier ("green", LOSSY_DCT, FLOAT,  1, true));
    _channelRules.push_back (Classifier ("b",     LOSSY_DCT, HALF,   2, true));
    _channelRules.push_back (Classifier ("b",     LOSSY_DCT, FLOAT,  2, true));
    _channelRules.push_back (Classifier ("blu",   LOSSY_DCT, HALF,   2, true));
    _channelRules.push_back (Classifier ("blu",   LOSSY_DCT, FLOAT,  2, true));
    _channelRules.push_back (Classifier ("blue",  LOSSY_DCT, HALF,   2, true));
    _channelRules.push_back (Classifier ("blue",  LOSSY_DCT, FLOAT,  2, true));

    _channelRules.push_back (Classifier ("y",     LOSSY_DCT, HALF,  -1, true));
    _channelRules.push_back (Classifier ("y",     LOSSY_DCT, FLOAT, -1, true));
    _channelRules.push_back (Classifier ("by",    LOSSY_DCT, HALF,  -1, true));
    _channelRules.push_back (Classifier ("by",    LOSSY_DCT, FLOAT, -1, true));
    _channelRules.push_back (Classifier ("ry",    LOSSY_DCT, HALF,  -1, true));
    _channelRules.push_back (Classifier ("ry",    LOSSY_DCT, FLOAT, -1, true));
    _channelRules.push_back (Classifier ("a",     RLE,       UINT,  -1, true));
    _channelRules.push_back (Classifier ("a",     RLE,       HALF,  -1, true));
    _channelRules.push_back (Classifier ("a",     RLE,       FLOAT, -1, true));
}


// 
// Given a set of rules and ChannelData, figure out which rules apply
//

void
DwaCompressor::relevantChannelRules (std::vector<Classifier> &rules) const 
{
    rules.clear();

    std::vector<std::string> suffixes;
    
    for (size_t cd = 0; cd < _channelData.size(); ++cd) 
    {
        std::string suffix  = _channelData[cd].name;
        size_t      lastDot = suffix.find_last_of ('.');

        if (lastDot != std::string::npos)
            suffix = suffix.substr (lastDot+1, std::string::npos);

        suffixes.push_back(suffix);
    }

    
    for (size_t i = 0; i < _channelRules.size(); ++i) 
    {
        for (size_t cd = 0; cd < _channelData.size(); ++cd) 
        {
            if (_channelRules[i].match (suffixes[cd], _channelData[cd].type ))
            {
                rules.push_back (_channelRules[i]);
                break;
            }
        }       
    }
}


//
// Take our initial list of channels, and cache the contents.
//
// Determine approprate compression schemes for each channel,
// and figure out which sets should potentially be CSC'ed 
// prior to lossy compression.
//

void
DwaCompressor::classifyChannels
    (ChannelList channels,
     std::vector<ChannelData> &chanData,
     std::vector<CscChannelSet> &cscData)
{
    //
    // prefixMap used to map channel name prefixes to 
    // potential CSC-able sets of channels.
    //

    std::map<std::string, DwaCompressor::CscChannelSet> prefixMap;
    std::vector<DwaCompressor::CscChannelSet>           tmpCscSet;

    unsigned int numChan = 0;

    for (ChannelList::Iterator c = channels.begin(); c != channels.end(); ++c)
        numChan++;
    
    if (numChan)
        chanData.resize (numChan);

    //
    // Cache the relevant data from the channel structs.
    //

    unsigned int offset = 0;

    for (ChannelList::Iterator c = channels.begin(); c != channels.end(); ++c)
    {
        chanData[offset].name        = std::string (c.name());
        chanData[offset].compression = UNKNOWN;
        chanData[offset].xSampling   = c.channel().xSampling;
        chanData[offset].ySampling   = c.channel().ySampling;
        chanData[offset].type        = c.channel().type;
        chanData[offset].pLinear     = c.channel().pLinear;

        offset++;
    }

    //
    // Try and figure out which channels should be
    // compressed by which means.
    //

    for (offset = 0; offset<numChan; ++offset)
    {
        std::string prefix  = "";
        std::string suffix  = chanData[offset].name;
        size_t      lastDot = suffix.find_last_of ('.');

        if (lastDot != std::string::npos)
        {
            prefix = suffix.substr (0,         lastDot);
            suffix = suffix.substr (lastDot+1, std::string::npos);
        } 

        //
        // Make sure we have an entry in our CSC set map 
        //

        std::map<std::string, DwaCompressor::CscChannelSet>::iterator 
            theSet = prefixMap.find (prefix);

        if (theSet == prefixMap.end())
        {
            DwaCompressor::CscChannelSet tmpSet;

            tmpSet.idx[0] = 
            tmpSet.idx[1] = 
            tmpSet.idx[2] = -1;

            prefixMap[prefix] = tmpSet;
        }

        // 
        // Check the suffix against the list of classifications
        // we defined previously. If the _cscIdx is not negative,
        // it indicates that we should be part of a CSC group.
        //

        for (std::vector<Classifier>::iterator i = _channelRules.begin();
             i != _channelRules.end();
             ++i)
        {
            if ( i->match(suffix, chanData[offset].type) )
            {
                chanData[offset].compression = i->_scheme;

                if ( i->_cscIdx >= 0)
                    prefixMap[prefix].idx[i->_cscIdx] = offset;
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

    for (std::map<std::string, DwaCompressor::CscChannelSet>::iterator 
         theItem = prefixMap.begin(); theItem != prefixMap.end();
         ++theItem)
    {
        int red = (*theItem).second.idx[0];
        int grn = (*theItem).second.idx[1];
        int blu = (*theItem).second.idx[2];

        if ((red < 0) || (grn < 0) || (blu < 0))
            continue;

        if ((chanData[red].xSampling != chanData[grn].xSampling) ||
            (chanData[red].xSampling != chanData[blu].xSampling) ||
            (chanData[grn].xSampling != chanData[blu].xSampling) ||
            (chanData[red].ySampling != chanData[grn].ySampling) ||
            (chanData[red].ySampling != chanData[blu].ySampling) ||
            (chanData[grn].ySampling != chanData[blu].ySampling))
        {
            continue;
        }
        
        tmpCscSet.push_back ((*theItem).second);
    }
    
    size_t numCsc = tmpCscSet.size();

    if (numCsc)
        cscData.resize(numCsc);

    for (offset = 0; offset < numCsc; ++offset)
        cscData[offset] = tmpCscSet[offset];
}



//
// Setup some buffer pointers, determine channel sizes, things
// like that.
//

void
DwaCompressor::setupChannelData (int minX, int minY, int maxX, int maxY)
{
    char *planarUncBuffer[NUM_COMPRESSOR_SCHEMES];

    for (int i=0; i<NUM_COMPRESSOR_SCHEMES; ++i)
    {
        planarUncBuffer[i] = 0;

        if (_planarUncBuffer[i])
            planarUncBuffer[i] =  _planarUncBuffer[i];
    }

    for (unsigned int chan = 0; chan < _channelData.size(); ++chan)
    {
        ChannelData *cd = &_channelData[chan];

        cd->width  = OPENEXR_IMF_NAMESPACE::numSamples (cd->xSampling, minX, maxX);
        cd->height = OPENEXR_IMF_NAMESPACE::numSamples (cd->ySampling, minY, maxY);
                                
        cd->planarUncSize =
            cd->width * cd->height * OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->type);
                                  
        cd->planarUncBuffer    = planarUncBuffer[cd->compression];
        cd->planarUncBufferEnd = cd->planarUncBuffer;

        cd->planarUncRle[0]    = cd->planarUncBuffer;
        cd->planarUncRleEnd[0] = cd->planarUncRle[0];

        for (int byte = 1; byte < OPENEXR_IMF_NAMESPACE::pixelTypeSize(cd->type); ++byte)
        {
            cd->planarUncRle[byte] = 
                         cd->planarUncRle[byte-1] + cd->width * cd->height;

            cd->planarUncRleEnd[byte] =
                         cd->planarUncRle[byte];
        }

        cd->planarUncType = cd->type;

        if (cd->compression == LOSSY_DCT)
        {
            cd->planarUncType = FLOAT;
        }
        else
        {
            planarUncBuffer[cd->compression] +=
                cd->width * cd->height * OPENEXR_IMF_NAMESPACE::pixelTypeSize (cd->planarUncType);
        }
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
