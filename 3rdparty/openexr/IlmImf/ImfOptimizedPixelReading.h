///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012, Autodesk, Inc.
// 
// All rights reserved.
//
// Implementation of IIF-specific file format and speed optimizations 
// provided by Innobec Technologies inc on behalf of Autodesk.
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
// *       Neither the name of Industrial Light & Magic nor the names of
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

#pragma once

#ifndef INCLUDED_IMF_OPTIMIZED_PIXEL_READING_H
#define INCLUDED_IMF_OPTIMIZED_PIXEL_READING_H

#include "ImfSimd.h"
#include "ImfSystemSpecific.h"
#include <iostream>
#include "ImfChannelList.h"
#include "ImfFrameBuffer.h"
#include "ImfStringVectorAttribute.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class OptimizationMode
{
public:


    bool _optimizable;
    int _ySampling;
    OptimizationMode() : _optimizable(false) {}
    
};


#if IMF_HAVE_SSE2


//------------------------------------------------------------------------
// Test for SSE pointer alignemnt
//------------------------------------------------------------------------
EXR_FORCEINLINE
bool
isPointerSSEAligned (const void* EXR_RESTRICT pPointer)
{
    unsigned long trailingBits = ((unsigned long)pPointer) & 15;
    return trailingBits == 0;
}

//------------------------------------------------------------------------
// Load SSE from address into register
//------------------------------------------------------------------------
template<bool IS_ALIGNED>
EXR_FORCEINLINE
__m128i loadSSE (__m128i*& loadAddress)
{
    // throw exception :: this is not accepted
    return _mm_loadu_si128 (loadAddress);
}

template<>
EXR_FORCEINLINE
__m128i loadSSE<false> (__m128i*& loadAddress)
{
    return _mm_loadu_si128 (loadAddress);
}

template<>
EXR_FORCEINLINE
__m128i loadSSE<true> (__m128i*& loadAddress)
{
    return _mm_load_si128 (loadAddress);
}

//------------------------------------------------------------------------
// Store SSE from register into address
//------------------------------------------------------------------------
template<bool IS_ALIGNED>
EXR_FORCEINLINE
void storeSSE (__m128i*& storeAddress, __m128i& dataToStore)
{

}

template<>
EXR_FORCEINLINE
void
storeSSE<false> (__m128i*& storeAddress, __m128i& dataToStore)
{
    _mm_storeu_si128 (storeAddress, dataToStore);
}

template<>
EXR_FORCEINLINE
void
storeSSE<true> (__m128i*& storeAddress, __m128i& dataToStore)
{
    _mm_stream_si128 (storeAddress, dataToStore);
}



//------------------------------------------------------------------------
//
// Write to RGBA
//
//------------------------------------------------------------------------

//
// Using SSE intrinsics
//
template<bool READ_PTR_ALIGNED, bool WRITE_PTR_ALIGNED>
EXR_FORCEINLINE 
void writeToRGBASSETemplate 
    (__m128i*& readPtrSSERed,
     __m128i*& readPtrSSEGreen,
     __m128i*& readPtrSSEBlue,
     __m128i*& readPtrSSEAlpha,
     __m128i*& writePtrSSE,
     const size_t& lPixelsToCopySSE)
{
    for (size_t i = 0; i < lPixelsToCopySSE; ++i)
    {
        __m128i redRegister   = loadSSE<READ_PTR_ALIGNED> (readPtrSSERed);
        __m128i greenRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSEGreen);
        __m128i blueRegister  = loadSSE<READ_PTR_ALIGNED> (readPtrSSEBlue);
        __m128i alphaRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSEAlpha);

        __m128i redGreenRegister  = _mm_unpacklo_epi16 (redRegister,
                                                        greenRegister);
        __m128i blueAlphaRegister = _mm_unpacklo_epi16 (blueRegister,
                                                        alphaRegister);

        __m128i pixel12Register   = _mm_unpacklo_epi32 (redGreenRegister,
                                                        blueAlphaRegister);
        __m128i pixel34Register   = _mm_unpackhi_epi32 (redGreenRegister,
                                                        blueAlphaRegister);

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel12Register);
        ++writePtrSSE;

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel34Register);
        ++writePtrSSE;

        redGreenRegister  = _mm_unpackhi_epi16 (redRegister, greenRegister);
        blueAlphaRegister = _mm_unpackhi_epi16 (blueRegister, alphaRegister);

        pixel12Register   = _mm_unpacklo_epi32 (redGreenRegister,
                                                blueAlphaRegister);
        pixel34Register   = _mm_unpackhi_epi32 (redGreenRegister,
                                                blueAlphaRegister);

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel12Register);
        ++writePtrSSE;
        
        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel34Register);
        ++writePtrSSE;

        ++readPtrSSEAlpha;
        ++readPtrSSEBlue;
        ++readPtrSSEGreen;
        ++readPtrSSERed;
    }
}

//
// Not using SSE intrinsics.  This is still faster than the alternative
// because we have multiple read pointers and therefore we are able to
// take advantage of data locality for write operations.
//
EXR_FORCEINLINE 
void writeToRGBANormal (unsigned short*& readPtrRed,
                        unsigned short*& readPtrGreen,
                        unsigned short*& readPtrBlue,
                        unsigned short*& readPtrAlpha,
                        unsigned short*& writePtr,
                        const size_t& lPixelsToCopy)
{
    for (size_t i = 0; i < lPixelsToCopy; ++i)
    {
        *(writePtr++) = *(readPtrRed++);
        *(writePtr++) = *(readPtrGreen++);
        *(writePtr++) = *(readPtrBlue++);
        *(writePtr++) = *(readPtrAlpha++);
    }
}

//
// Determine which (template) version to use by checking whether pointers
// are aligned
//
EXR_FORCEINLINE 
void optimizedWriteToRGBA (unsigned short*& readPtrRed,
                           unsigned short*& readPtrGreen,
                           unsigned short*& readPtrBlue,
                           unsigned short*& readPtrAlpha,
                           unsigned short*& writePtr,
                           const size_t& pixelsToCopySSE,
                           const size_t& pixelsToCopyNormal)
{
    bool readPtrAreAligned = true;

    readPtrAreAligned &= isPointerSSEAligned(readPtrRed);
    readPtrAreAligned &= isPointerSSEAligned(readPtrGreen);
    readPtrAreAligned &= isPointerSSEAligned(readPtrBlue);
    readPtrAreAligned &= isPointerSSEAligned(readPtrAlpha);

    bool writePtrIsAligned = isPointerSSEAligned(writePtr);

    if (!readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBASSETemplate<false, false> ((__m128i*&)readPtrRed,
                                              (__m128i*&)readPtrGreen,
                                              (__m128i*&)readPtrBlue,
                                              (__m128i*&)readPtrAlpha,
                                              (__m128i*&)writePtr,
                                              pixelsToCopySSE);
    }
    else if (!readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBASSETemplate<false, true> ((__m128i*&)readPtrRed,
                                             (__m128i*&)readPtrGreen,
                                             (__m128i*&)readPtrBlue,
                                             (__m128i*&)readPtrAlpha,
                                             (__m128i*&)writePtr,
                                             pixelsToCopySSE);
    }
    else if (readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBASSETemplate<true, false> ((__m128i*&)readPtrRed,
                                             (__m128i*&)readPtrGreen,
                                             (__m128i*&)readPtrBlue,
                                             (__m128i*&)readPtrAlpha,
                                             (__m128i*&)writePtr,
                                             pixelsToCopySSE);
    }
    else if(readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBASSETemplate<true, true> ((__m128i*&)readPtrRed,
                                            (__m128i*&)readPtrGreen,
                                            (__m128i*&)readPtrBlue,
                                            (__m128i*&)readPtrAlpha,
                                            (__m128i*&)writePtr,
                                            pixelsToCopySSE);
    }

    writeToRGBANormal (readPtrRed, readPtrGreen, readPtrBlue, readPtrAlpha,
                       writePtr, pixelsToCopyNormal);
}



//------------------------------------------------------------------------
//
// Write to RGBA Fill A
//
//------------------------------------------------------------------------

//
// Using SSE intrinsics
//
template<bool READ_PTR_ALIGNED, bool WRITE_PTR_ALIGNED>
EXR_FORCEINLINE 
void
writeToRGBAFillASSETemplate (__m128i*& readPtrSSERed,
                             __m128i*& readPtrSSEGreen,
                             __m128i*& readPtrSSEBlue,
                             const unsigned short& alphaFillValue,
                             __m128i*& writePtrSSE,
                             const size_t& pixelsToCopySSE)
{
    const __m128i dummyAlphaRegister = _mm_set_epi16 (alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue,
                                                      alphaFillValue);

    for (size_t pixelCounter = 0; pixelCounter < pixelsToCopySSE; ++pixelCounter)
    {
        __m128i redRegister   = loadSSE<READ_PTR_ALIGNED> (readPtrSSERed);
        __m128i greenRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSEGreen);
        __m128i blueRegister  = loadSSE<READ_PTR_ALIGNED> (readPtrSSEBlue);

        __m128i redGreenRegister  = _mm_unpacklo_epi16 (redRegister,
                                                        greenRegister);
        __m128i blueAlphaRegister = _mm_unpacklo_epi16 (blueRegister,
                                                        dummyAlphaRegister);

        __m128i pixel12Register   = _mm_unpacklo_epi32 (redGreenRegister,
                                                        blueAlphaRegister);
        __m128i pixel34Register   = _mm_unpackhi_epi32 (redGreenRegister,
                                                        blueAlphaRegister);

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel12Register);
        ++writePtrSSE;

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel34Register);
        ++writePtrSSE;

        redGreenRegister  = _mm_unpackhi_epi16 (redRegister,
                                                greenRegister);
        blueAlphaRegister = _mm_unpackhi_epi16 (blueRegister,
                                                dummyAlphaRegister);

        pixel12Register   = _mm_unpacklo_epi32 (redGreenRegister,
                                                blueAlphaRegister);
        pixel34Register   = _mm_unpackhi_epi32 (redGreenRegister,
                                                blueAlphaRegister);

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel12Register);
        ++writePtrSSE;

        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, pixel34Register);
        ++writePtrSSE;

        ++readPtrSSEBlue;
        ++readPtrSSEGreen;
        ++readPtrSSERed;
    }
}

//
// Not using SSE intrinsics.  This is still faster than the alternative
// because we have multiple read pointers and therefore we are able to
// take advantage of data locality for write operations.
//
EXR_FORCEINLINE
void
writeToRGBAFillANormal (unsigned short*& readPtrRed,
                        unsigned short*& readPtrGreen,
                        unsigned short*& readPtrBlue,
                        const unsigned short& alphaFillValue,
                        unsigned short*& writePtr,
                        const size_t& pixelsToCopy)
{
    for (size_t i = 0; i < pixelsToCopy; ++i)
    {
        *(writePtr++) = *(readPtrRed++);
        *(writePtr++) = *(readPtrGreen++);
        *(writePtr++) = *(readPtrBlue++);
        *(writePtr++) = alphaFillValue;
    }
}

//
// Determine which (template) version to use by checking whether pointers
// are aligned.
//
EXR_FORCEINLINE 
void
optimizedWriteToRGBAFillA (unsigned short*& readPtrRed,
                           unsigned short*& readPtrGreen,
                           unsigned short*& readPtrBlue,
                           const unsigned short& alphaFillValue,
                           unsigned short*& writePtr,
                           const size_t& pixelsToCopySSE,
                           const size_t& pixelsToCopyNormal)
{
    bool readPtrAreAligned = true;

    readPtrAreAligned &= isPointerSSEAligned (readPtrRed);
    readPtrAreAligned &= isPointerSSEAligned (readPtrGreen);
    readPtrAreAligned &= isPointerSSEAligned (readPtrBlue);

    bool writePtrIsAligned = isPointerSSEAligned (writePtr);

    if (!readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBAFillASSETemplate<false, false> ((__m128i*&)readPtrRed,
                                                   (__m128i*&)readPtrGreen,
                                                   (__m128i*&)readPtrBlue,
                                                   alphaFillValue,
                                                   (__m128i*&)writePtr,
                                                   pixelsToCopySSE);
    }
    else if (!readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBAFillASSETemplate<false, true> ((__m128i*&)readPtrRed,
                                                  (__m128i*&)readPtrGreen,
                                                  (__m128i*&)readPtrBlue,
                                                  alphaFillValue,
                                                  (__m128i*&)writePtr,
                                                  pixelsToCopySSE);
    }
    else if (readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBAFillASSETemplate<true, false> ((__m128i*&)readPtrRed,
                                                  (__m128i*&)readPtrGreen,
                                                  (__m128i*&)readPtrBlue,
                                                  alphaFillValue,
                                                  (__m128i*&)writePtr,
                                                  pixelsToCopySSE);
    }
    else if (readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBAFillASSETemplate<true, true> ((__m128i*&)readPtrRed,
                                                 (__m128i*&)readPtrGreen,
                                                 (__m128i*&)readPtrBlue,
                                                 alphaFillValue,
                                                 (__m128i*&)writePtr,
                                                 pixelsToCopySSE);
    }

    writeToRGBAFillANormal (readPtrRed,
                            readPtrGreen, readPtrBlue, alphaFillValue,
                            writePtr, pixelsToCopyNormal);
}



//------------------------------------------------------------------------
//
// Write to RGB
//
//------------------------------------------------------------------------

//
// Using SSE intrinsics
//
template<bool READ_PTR_ALIGNED, bool WRITE_PTR_ALIGNED>
EXR_FORCEINLINE 
void
writeToRGBSSETemplate (__m128i*& readPtrSSERed,
                       __m128i*& readPtrSSEGreen,
                       __m128i*& readPtrSSEBlue,
                       __m128i*& writePtrSSE,
                       const size_t& pixelsToCopySSE)
{

    for (size_t pixelCounter = 0; pixelCounter < pixelsToCopySSE; ++pixelCounter)
    {
        //
        // Need to shuffle and unpack pointers to obtain my first register
        // We must save 8 pixels at a time, so we must have the following three registers at the end:
        // 1) R1 G1 B1 R2 G2 B2 R3 G3
        // 2) B3 R4 G4 B4 R5 G5 B5 R6
        // 3) G6 B6 R7 G7 B7 R8 G8 B8
        //
        __m128i redRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSERed);
        __m128i greenRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSEGreen);
        __m128i blueRegister = loadSSE<READ_PTR_ALIGNED> (readPtrSSEBlue);

        //
        // First register: R1 G1 B1 R2 G2 B2 R3 G3
        // Construct 2 registers and then unpack them to obtain our final result:
        //
        __m128i redGreenRegister  = _mm_unpacklo_epi16 (redRegister,
                                                        greenRegister);
        __m128i redBlueRegister   = _mm_unpacklo_epi16 (redRegister,
                                                        blueRegister);
        __m128i greenBlueRegister = _mm_unpacklo_epi16 (greenRegister,
                                                        blueRegister);

        // Left Part (R1 G1 B1 R2)
        __m128i quarterRight = _mm_shufflelo_epi16 (redBlueRegister,
                                                    _MM_SHUFFLE(3,0,2,1));
        __m128i halfLeft     = _mm_unpacklo_epi32 (redGreenRegister,
                                                   quarterRight);

        // Right Part (G2 B2 R3 G3)
        __m128i quarterLeft  = _mm_shuffle_epi32 (greenBlueRegister,
                                                 _MM_SHUFFLE(3,2,0,1));
        quarterRight         = _mm_shuffle_epi32 (redGreenRegister,
                                                 _MM_SHUFFLE(3,0,1,2));
        __m128i halfRight    = _mm_unpacklo_epi32 (quarterLeft, quarterRight);

        __m128i fullRegister = _mm_unpacklo_epi64 (halfLeft, halfRight);
        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, fullRegister);
        ++writePtrSSE;

        //
        // Second register: B3 R4 G4 B4 R5 G5 B5 R6
        //

        // Left Part (B3, R4, G4, B4)
        quarterLeft  = _mm_shufflehi_epi16 (redBlueRegister,
                                            _MM_SHUFFLE(0, 3, 2, 1));
        quarterRight = _mm_shufflehi_epi16 (greenBlueRegister,
                                            _MM_SHUFFLE(1, 0, 3, 2));
        halfLeft     = _mm_unpackhi_epi32 (quarterLeft, quarterRight);

        // Update the registers
        redGreenRegister  = _mm_unpackhi_epi16 (redRegister, greenRegister);
        redBlueRegister   = _mm_unpackhi_epi16 (redRegister, blueRegister);
        greenBlueRegister = _mm_unpackhi_epi16 (greenRegister, blueRegister);

        // Right Part (R5 G5 B5 R6)
        quarterRight = _mm_shufflelo_epi16 (redBlueRegister,
                                            _MM_SHUFFLE(3,0,2,1));
        halfRight    = _mm_unpacklo_epi32 (redGreenRegister, quarterRight);

        fullRegister = _mm_unpacklo_epi64 (halfLeft, halfRight);
        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, fullRegister);
        ++writePtrSSE;

        //
        // Third register: G6 B6 R7 G7 B7 R8 G8 B8
        //

        // Left part (G6 B6 R7 G7)
        quarterLeft  = _mm_shuffle_epi32 (greenBlueRegister,
                                          _MM_SHUFFLE(3,2,0,1));
        quarterRight = _mm_shuffle_epi32 (redGreenRegister,
                                          _MM_SHUFFLE(3,0,1,2));
        halfLeft     = _mm_unpacklo_epi32 (quarterLeft, quarterRight);

        // Right part (B7 R8 G8 B8)
        quarterLeft  = _mm_shufflehi_epi16 (redBlueRegister,
                                            _MM_SHUFFLE(0, 3, 2, 1));
        quarterRight = _mm_shufflehi_epi16 (greenBlueRegister,
                                            _MM_SHUFFLE(1, 0, 3, 2));
        halfRight    = _mm_unpackhi_epi32 (quarterLeft, quarterRight);

        fullRegister = _mm_unpacklo_epi64 (halfLeft, halfRight);
        storeSSE<WRITE_PTR_ALIGNED> (writePtrSSE, fullRegister);
        ++writePtrSSE;

        //
        // Increment read pointers
        //
        ++readPtrSSEBlue;
        ++readPtrSSEGreen;
        ++readPtrSSERed;
    }
}

//
// Not using SSE intrinsics.  This is still faster than the alternative
// because we have multiple read pointers and therefore we are able to
// take advantage of data locality for write operations.
//
EXR_FORCEINLINE 
void
writeToRGBNormal (unsigned short*& readPtrRed,
                  unsigned short*& readPtrGreen,
                  unsigned short*& readPtrBlue,
                  unsigned short*& writePtr,
                  const size_t& pixelsToCopy)
{
    for (size_t i = 0; i < pixelsToCopy; ++i)
    {
        *(writePtr++) = *(readPtrRed++);
        *(writePtr++) = *(readPtrGreen++);
        *(writePtr++) = *(readPtrBlue++);
    }
}

//
// Determine which (template) version to use by checking whether pointers
// are aligned
//
EXR_FORCEINLINE 
void optimizedWriteToRGB (unsigned short*& readPtrRed,
                          unsigned short*& readPtrGreen,
                          unsigned short*& readPtrBlue,
                          unsigned short*& writePtr,
                          const size_t& pixelsToCopySSE,
                          const size_t& pixelsToCopyNormal)
{
    bool readPtrAreAligned = true;

    readPtrAreAligned &= isPointerSSEAligned(readPtrRed);
    readPtrAreAligned &= isPointerSSEAligned(readPtrGreen);
    readPtrAreAligned &= isPointerSSEAligned(readPtrBlue);

    bool writePtrIsAligned = isPointerSSEAligned(writePtr);

    if (!readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBSSETemplate<false, false> ((__m128i*&)readPtrRed,
                                             (__m128i*&)readPtrGreen,
                                             (__m128i*&)readPtrBlue,
                                             (__m128i*&)writePtr,
                                             pixelsToCopySSE);
    }
    else if (!readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBSSETemplate<false, true> ((__m128i*&)readPtrRed,
                                            (__m128i*&)readPtrGreen,
                                            (__m128i*&)readPtrBlue,
                                            (__m128i*&)writePtr,
                                            pixelsToCopySSE);
    }
    else if (readPtrAreAligned && !writePtrIsAligned)
    {
        writeToRGBSSETemplate<true, false> ((__m128i*&)readPtrRed,
                                            (__m128i*&)readPtrGreen,
                                            (__m128i*&)readPtrBlue,
                                            (__m128i*&)writePtr,
                                            pixelsToCopySSE);
    }
    else if (readPtrAreAligned && writePtrIsAligned)
    {
        writeToRGBSSETemplate<true, true> ((__m128i*&)readPtrRed,
                                           (__m128i*&)readPtrGreen,
                                           (__m128i*&)readPtrBlue,
                                           (__m128i*&)writePtr,
                                           pixelsToCopySSE);
    }


    writeToRGBNormal (readPtrRed, readPtrGreen, readPtrBlue,
                      writePtr, pixelsToCopyNormal);
}




#else // ! defined IMF_HAVE_SSE2

#endif // defined IMF_HAVE_SSE2


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
