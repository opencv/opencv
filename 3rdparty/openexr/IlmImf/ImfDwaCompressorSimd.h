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

#ifndef IMF_DWACOMPRESSORSIMD_H_HAS_BEEN_INCLUDED
#define IMF_DWACOMPRESSORSIMD_H_HAS_BEEN_INCLUDED

//
// Various SSE accelerated functions, used by Imf::DwaCompressor. 
// These have been separated into a separate .h file, as the fast
// paths are done with template specialization.
//
// Unless otherwise noted, all pointers are assumed to be 32-byte 
// aligned. Unaligned pointers may risk seg-faulting.
//

#include "ImfNamespace.h"
#include "ImfSimd.h"
#include "ImfSystemSpecific.h"
#include "OpenEXRConfig.h"

#include <half.h>
#include <assert.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

#define _SSE_ALIGNMENT        32
#define _SSE_ALIGNMENT_MASK 0x0F
#define _AVX_ALIGNMENT_MASK 0x1F

//
// Test if we should enable GCC inline asm paths for AVX
//

#ifdef OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX 

    #define IMF_HAVE_GCC_INLINEASM

    #ifdef __LP64__
        #define IMF_HAVE_GCC_INLINEASM_64
    #endif /* __LP64__ */

#endif /* OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX */

//
// A simple 64-element array, aligned properly for SIMD access. 
//

template <class T>
class SimdAlignedBuffer64
{
    public:

        SimdAlignedBuffer64(): _buffer (0), _handle (0)           
        {
            alloc();
        }

        SimdAlignedBuffer64(const SimdAlignedBuffer64 &rhs): _handle(0)
        {
            alloc();
            memcpy (_buffer, rhs._buffer, 64 * sizeof (T));
        }

        ~SimdAlignedBuffer64 ()
        {
            EXRFreeAligned (_handle);
            _handle = 0;
            _buffer = 0;
        }

        void alloc()
        {
            //
            // Try EXRAllocAligned first - but it might fallback to
            // unaligned allocs. If so, overalloc.
            //

            _handle = (char *) EXRAllocAligned
                (64 * sizeof(T), _SSE_ALIGNMENT);

            if (((size_t)_handle & (_SSE_ALIGNMENT - 1)) == 0)
            {
                _buffer = (T *)_handle;
                return;
            }

            EXRFreeAligned(_handle);
            _handle = (char *) EXRAllocAligned
                (64 * sizeof(T) + _SSE_ALIGNMENT, _SSE_ALIGNMENT);

            char *aligned = _handle;

            while ((size_t)aligned & (_SSE_ALIGNMENT - 1))
                aligned++;

            _buffer = (T *)aligned;    
        }

        T     *_buffer;

    private:

        char  *_handle;
};

typedef SimdAlignedBuffer64<float>          SimdAlignedBuffer64f;
typedef SimdAlignedBuffer64<unsigned short> SimdAlignedBuffer64us;

namespace {

//
// Color space conversion, Inverse 709 CSC, Y'CbCr -> R'G'B'
//

void
csc709Inverse (float &comp0, float &comp1, float &comp2)
{
    float src[3];

    src[0] = comp0;
    src[1] = comp1;
    src[2] = comp2;

    comp0 = src[0]                    + 1.5747f * src[2];
    comp1 = src[0] - 0.1873f * src[1] - 0.4682f * src[2];
    comp2 = src[0] + 1.8556f * src[1];
}

#ifndef IMF_HAVE_SSE2


//
// Scalar color space conversion, based on 709 primiary chromaticies.
// No scaling or offsets, just the matrix
//

void
csc709Inverse64 (float *comp0, float *comp1, float *comp2)
{
    for (int i = 0; i < 64; ++i)
        csc709Inverse (comp0[i], comp1[i], comp2[i]);
}

#else /* IMF_HAVE_SSE2 */

//
// SSE2 color space conversion
//

void
csc709Inverse64 (float *comp0, float *comp1, float *comp2)
{
    __m128 c0 = { 1.5747f,  1.5747f,  1.5747f,  1.5747f};
    __m128 c1 = { 1.8556f,  1.8556f,  1.8556f,  1.8556f};
    __m128 c2 = {-0.1873f, -0.1873f, -0.1873f, -0.1873f};
    __m128 c3 = {-0.4682f, -0.4682f, -0.4682f, -0.4682f}; 

    __m128 *r = (__m128 *)comp0;
    __m128 *g = (__m128 *)comp1;
    __m128 *b = (__m128 *)comp2;
    __m128 src[3];

    #define CSC_INVERSE_709_SSE2_LOOP(i)                       \
            src[0] = r[i];                                     \
            src[1] = g[i];                                     \
            src[2] = b[i];                                     \
                                                               \
            r[i] = _mm_add_ps (r[i], _mm_mul_ps (src[2], c0)); \
                                                               \
            g[i]   = _mm_mul_ps (g[i], c2);                    \
            src[2] = _mm_mul_ps (src[2], c3);                  \
            g[i]   = _mm_add_ps (g[i], src[0]);                \
            g[i]   = _mm_add_ps (g[i], src[2]);                \
                                                               \
            b[i] = _mm_mul_ps (c1,   src[1]);                  \
            b[i] = _mm_add_ps (b[i], src[0]);

    CSC_INVERSE_709_SSE2_LOOP (0)
    CSC_INVERSE_709_SSE2_LOOP (1)
    CSC_INVERSE_709_SSE2_LOOP (2)
    CSC_INVERSE_709_SSE2_LOOP (3)

    CSC_INVERSE_709_SSE2_LOOP (4)
    CSC_INVERSE_709_SSE2_LOOP (5)
    CSC_INVERSE_709_SSE2_LOOP (6)
    CSC_INVERSE_709_SSE2_LOOP (7)

    CSC_INVERSE_709_SSE2_LOOP (8)
    CSC_INVERSE_709_SSE2_LOOP (9)
    CSC_INVERSE_709_SSE2_LOOP (10)
    CSC_INVERSE_709_SSE2_LOOP (11)

    CSC_INVERSE_709_SSE2_LOOP (12)
    CSC_INVERSE_709_SSE2_LOOP (13)
    CSC_INVERSE_709_SSE2_LOOP (14)
    CSC_INVERSE_709_SSE2_LOOP (15)
}

#endif /* IMF_HAVE_SSE2 */


//
// Color space conversion, Forward 709 CSC, R'G'B' -> Y'CbCr
//
// Simple FPU color space conversion. Based on the 709
// primary chromaticies, with no scaling or offsets.
//

void
csc709Forward64 (float *comp0, float *comp1, float *comp2)
{
    float src[3];

    for (int i = 0; i<64; ++i)
    {
        src[0] = comp0[i];    
        src[1] = comp1[i]; 
        src[2] = comp2[i];     

        comp0[i] =  0.2126f * src[0] + 0.7152f * src[1] + 0.0722f * src[2];
        comp1[i] = -0.1146f * src[0] - 0.3854f * src[1] + 0.5000f * src[2];
        comp2[i] =  0.5000f * src[0] - 0.4542f * src[1] - 0.0458f * src[2];
    }
}


//
// Byte interleaving of 2 byte arrays:
//    src0 = AAAA 
//    src1 = BBBB
//    dst  = ABABABAB
//
// numBytes is the size of each of the source buffers
//

#ifndef IMF_HAVE_SSE2 

// 
// Scalar default implementation 
//

void
interleaveByte2 (char *dst, char *src0, char *src1, int numBytes)
{
    for (int x = 0; x < numBytes; ++x)
    {
        dst[2 * x]     = src0[x];
        dst[2 * x + 1] = src1[x];
    }
}

#else  /* IMF_HAVE_SSE2 */

// 
// SSE2 byte interleaving
//

void
interleaveByte2 (char *dst, char *src0, char *src1, int numBytes)
{
    int dstAlignment  = (size_t)dst  % 16;
    int src0Alignment = (size_t)src0 % 16;
    int src1Alignment = (size_t)src1 % 16;

    __m128i *dst_epi8  = (__m128i*)dst;
    __m128i *src0_epi8 = (__m128i*)src0;
    __m128i *src1_epi8 = (__m128i*)src1;
    int sseWidth  =  numBytes / 16;

    if ((!dstAlignment) && (!src0Alignment) && (!src1Alignment))
    {
        __m128i tmp0, tmp1;

        //
        // Aligned loads and stores
        //

        for (int x = 0; x < sseWidth; ++x)
        {
            tmp0 = src0_epi8[x];
            tmp1 = src1_epi8[x];

            _mm_stream_si128 (&dst_epi8[2 * x],
                              _mm_unpacklo_epi8 (tmp0, tmp1));

            _mm_stream_si128 (&dst_epi8[2 * x + 1],
                              _mm_unpackhi_epi8 (tmp0, tmp1));
        }

        //
        // Then do run the leftovers one at a time
        //

        for (int x = 16 * sseWidth; x < numBytes; ++x)
        {
            dst[2 * x]     = src0[x];
            dst[2 * x + 1] = src1[x];
        }
    }
    else if ((!dstAlignment) && (src0Alignment == 8) && (src1Alignment == 8))
    {
        //
        // Aligned stores, but catch up a few values so we can 
        // use aligned loads
        //
    
        for (int x = 0; x < 8; ++x)
        {
            dst[2 * x]     = src0[x];
            dst[2 * x + 1] = src1[x];
        }

        dst_epi8  = (__m128i*)&dst[16];
        src0_epi8 = (__m128i*)&src0[8];
        src1_epi8 = (__m128i*)&src1[8];
        sseWidth  =  (numBytes - 8) / 16;

        for (int x=0; x<sseWidth; ++x)
        {
            _mm_stream_si128 (&dst_epi8[2 * x],
                              _mm_unpacklo_epi8 (src0_epi8[x], src1_epi8[x]));

            _mm_stream_si128 (&dst_epi8[2 * x + 1],
                              _mm_unpackhi_epi8 (src0_epi8[x], src1_epi8[x]));
        }

        //
        // Then do run the leftovers one at a time
        //

        for (int x = 16 * sseWidth + 8; x < numBytes; ++x)
        {
            dst[2 * x]     = src0[x];
            dst[2 * x + 1] = src1[x];
        }
    }
    else
    {
        //
        // Unaligned everything
        //

        for (int x = 0; x < sseWidth; ++x)
        {
            __m128i tmpSrc0_epi8 = _mm_loadu_si128 (&src0_epi8[x]);
            __m128i tmpSrc1_epi8 = _mm_loadu_si128 (&src1_epi8[x]);

            _mm_storeu_si128 (&dst_epi8[2 * x],
                              _mm_unpacklo_epi8 (tmpSrc0_epi8, tmpSrc1_epi8));

            _mm_storeu_si128 (&dst_epi8[2 * x + 1],
                              _mm_unpackhi_epi8 (tmpSrc0_epi8, tmpSrc1_epi8));
        }

        //
        // Then do run the leftovers one at a time
        //

        for (int x = 16 * sseWidth; x < numBytes; ++x)
        {
            dst[2 * x]     = src0[x];
            dst[2 * x + 1] = src1[x];
        }
    }
}

#endif /* IMF_HAVE_SSE2 */


//
// Float -> half float conversion
//
// To enable F16C based conversion, we can't rely on compile-time
// detection, hence the multiple defined versions. Pick one based
// on runtime cpuid detection.
//

//
// Default boring conversion
//

void 
convertFloatToHalf64_scalar (unsigned short *dst, float *src)
{
    for (int i=0; i<64; ++i)
        dst[i] = ((half)src[i]).bits();
}


//
// F16C conversion - Assumes aligned src and dst
//

void
convertFloatToHalf64_f16c (unsigned short *dst, float *src)
{
    //
    // Ordinarly, I'd avoid using inline asm and prefer intrinsics. 
    // However, in order to get the intrinsics, we need to tell 
    // the compiler to generate VEX instructions.
    //
    // (On the GCC side, -mf16c goes ahead and activates -mavc,
    //  resulting in VEX code. Without -mf16c, no intrinsics..)
    //
    // Now, it's quite likely that we'll find ourselves in situations
    // where we want to build *without* VEX, in order to maintain
    // maximum compatability. But to get there with intrinsics,
    // we'd need to break out code into a separate file. Bleh.
    // I'll take the asm.
    //

    #if defined IMF_HAVE_GCC_INLINEASM
        __asm__
           ("vmovaps       (%0),     %%ymm0         \n"
            "vmovaps   0x20(%0),     %%ymm1         \n"
            "vmovaps   0x40(%0),     %%ymm2         \n"
            "vmovaps   0x60(%0),     %%ymm3         \n"
            "vcvtps2ph $0,           %%ymm0, %%xmm0 \n"
            "vcvtps2ph $0,           %%ymm1, %%xmm1 \n"
            "vcvtps2ph $0,           %%ymm2, %%xmm2 \n"
            "vcvtps2ph $0,           %%ymm3, %%xmm3 \n"
            "vmovdqa   %%xmm0,       0x00(%1)       \n"
            "vmovdqa   %%xmm1,       0x10(%1)       \n"
            "vmovdqa   %%xmm2,       0x20(%1)       \n"
            "vmovdqa   %%xmm3,       0x30(%1)       \n"
            "vmovaps   0x80(%0),     %%ymm0         \n"
            "vmovaps   0xa0(%0),     %%ymm1         \n"
            "vmovaps   0xc0(%0),     %%ymm2         \n"
            "vmovaps   0xe0(%0),     %%ymm3         \n"
            "vcvtps2ph $0,           %%ymm0, %%xmm0 \n"
            "vcvtps2ph $0,           %%ymm1, %%xmm1 \n"
            "vcvtps2ph $0,           %%ymm2, %%xmm2 \n"
            "vcvtps2ph $0,           %%ymm3, %%xmm3 \n"
            "vmovdqa   %%xmm0,       0x40(%1)       \n"
            "vmovdqa   %%xmm1,       0x50(%1)       \n"
            "vmovdqa   %%xmm2,       0x60(%1)       \n"
            "vmovdqa   %%xmm3,       0x70(%1)       \n"
        #ifndef __AVX__
            "vzeroupper                             \n"
        #endif /* __AVX__ */
            : /* Output  */                
            : /* Input   */ "r"(src), "r"(dst)
        #ifndef __AVX__
            : /* Clobber */ "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
        #else
            : /* Clobber */ "%ymm0", "%ymm1", "%ymm2", "%ymm3", "memory"
        #endif /* __AVX__ */
           );
    #else
        convertFloatToHalf64_scalar (dst, src);
    #endif /* IMF_HAVE_GCC_INLINEASM */
}


//
// Convert an 8x8 block of HALF from zig-zag order to
// FLOAT in normal order. The order we want is:
//
//          src                           dst 
//  0  1  2  3  4  5  6  7       0  1  5  6 14 15 27 28
//  8  9 10 11 12 13 14 15       2  4  7 13 16 26 29 42
// 16 17 18 19 20 21 22 23       3  8 12 17 25 30 41 43
// 24 25 26 27 28 29 30 31       9 11 18 24 31 40 44 53  
// 32 33 34 35 36 37 38 39      10 19 23 32 39 45 52 54
// 40 41 42 43 44 45 46 47      20 22 33 38 46 51 55 60
// 48 49 50 51 52 53 54 55      21 34 37 47 50 56 59 61
// 56 57 58 59 60 61 62 63      35 36 48 49 57 58 62 63
//

void
fromHalfZigZag_scalar (unsigned short *src, float *dst)
{
    half *srcHalf = (half *)src;

    dst[0] = (float)srcHalf[0];
    dst[1] = (float)srcHalf[1];
    dst[2] = (float)srcHalf[5];
    dst[3] = (float)srcHalf[6];
    dst[4] = (float)srcHalf[14];
    dst[5] = (float)srcHalf[15];
    dst[6] = (float)srcHalf[27];
    dst[7] = (float)srcHalf[28];
    dst[8] = (float)srcHalf[2];
    dst[9] = (float)srcHalf[4];

    dst[10] = (float)srcHalf[7];
    dst[11] = (float)srcHalf[13];
    dst[12] = (float)srcHalf[16];
    dst[13] = (float)srcHalf[26];
    dst[14] = (float)srcHalf[29];
    dst[15] = (float)srcHalf[42];
    dst[16] = (float)srcHalf[3];
    dst[17] = (float)srcHalf[8];
    dst[18] = (float)srcHalf[12];
    dst[19] = (float)srcHalf[17];

    dst[20] = (float)srcHalf[25];
    dst[21] = (float)srcHalf[30];
    dst[22] = (float)srcHalf[41];
    dst[23] = (float)srcHalf[43];
    dst[24] = (float)srcHalf[9];
    dst[25] = (float)srcHalf[11];
    dst[26] = (float)srcHalf[18];
    dst[27] = (float)srcHalf[24];
    dst[28] = (float)srcHalf[31];
    dst[29] = (float)srcHalf[40];

    dst[30] = (float)srcHalf[44];
    dst[31] = (float)srcHalf[53];
    dst[32] = (float)srcHalf[10];
    dst[33] = (float)srcHalf[19];
    dst[34] = (float)srcHalf[23];
    dst[35] = (float)srcHalf[32];
    dst[36] = (float)srcHalf[39];
    dst[37] = (float)srcHalf[45];
    dst[38] = (float)srcHalf[52];
    dst[39] = (float)srcHalf[54];

    dst[40] = (float)srcHalf[20];
    dst[41] = (float)srcHalf[22];
    dst[42] = (float)srcHalf[33];
    dst[43] = (float)srcHalf[38];
    dst[44] = (float)srcHalf[46];
    dst[45] = (float)srcHalf[51];
    dst[46] = (float)srcHalf[55];
    dst[47] = (float)srcHalf[60];
    dst[48] = (float)srcHalf[21];
    dst[49] = (float)srcHalf[34];

    dst[50] = (float)srcHalf[37];
    dst[51] = (float)srcHalf[47];
    dst[52] = (float)srcHalf[50];
    dst[53] = (float)srcHalf[56];
    dst[54] = (float)srcHalf[59];
    dst[55] = (float)srcHalf[61];
    dst[56] = (float)srcHalf[35];
    dst[57] = (float)srcHalf[36];
    dst[58] = (float)srcHalf[48];
    dst[59] = (float)srcHalf[49];

    dst[60] = (float)srcHalf[57];
    dst[61] = (float)srcHalf[58];
    dst[62] = (float)srcHalf[62];
    dst[63] = (float)srcHalf[63];
}


//
// If we can form the correct ordering in xmm registers,
// we can use F16C to convert from HALF -> FLOAT. However,
// making the correct order isn't trivial. 
// 
// We want to re-order a source 8x8 matrix from:
//
//  0  1  2  3  4  5  6  7       0  1  5  6 14 15 27 28
//  8  9 10 11 12 13 14 15       2  4  7 13 16 26 29 42
// 16 17 18 19 20 21 22 23       3  8 12 17 25 30 41 43
// 24 25 26 27 28 29 30 31       9 11 18 24 31 40 44 53   (A)
// 32 33 34 35 36 37 38 39  --> 10 19 23 32 39 45 52 54
// 40 41 42 43 44 45 46 47      20 22 33 38 46 51 55 60
// 48 49 50 51 52 53 54 55      21 34 37 47 50 56 59 61
// 56 57 58 59 60 61 62 63      35 36 48 49 57 58 62 63
//
// Which looks like a mess, right? 
//
// Now, check out the NE/SW diagonals of (A). Along those lines, 
// we have runs of contiguous values! If we rewrite (A) a bit, we get:
//
//  0
//  1  2
//  5  4  3
//  6  7  8  9
// 14 13 12 11 10
// 15 16 17 18 19 20
// 27 26 25 24 23 22 21            (B)
// 28 29 30 31 32 33 34 35
//    42 41 40 39 38 37 36
//       43 44 45 46 47 48
//          53 52 51 50 49
//             54 55 56 57
//                60 59 58
//                   61 62
//                      63
//
// In this ordering, the columns are the rows (A). If we can 'transpose' 
// (B), we'll achieve our goal. But we want this to fit nicely into 
// xmm registers and still be able to load large runs efficiently.  
// Also, notice that the odd rows are in ascending order, while 
// the even rows are in descending order. 
//
// If we 'fold' the bottom half up into the top, we can preserve ordered
// runs accross rows, and still keep all the correct values in columns. 
// After transposing, we'll need to rotate things back into place. 
// This gives us:
//
//  0 | 42   41   40   39   38   37   36
//  1    2 | 43   44   45   46   47   48
//  5    4    3 | 53   52   51   50   49
//  6    7    8    9 | 54   55   56   57      (C)
// 14   13   12   11   10 | 60   59   58
// 15   16   17   18   19   20 | 61   62
// 27   26   25   24   23   22   21 | 61
// 28   29   30   31   32   33   34   35
//
// But hang on. We still have the backwards descending rows to deal with.
// Lets reverse the even rows so that all values are in ascending order
//
//  36   37  38   39   40   41   42 | 0
//  1    2 | 43   44   45   46   47   48
//  49   50  51   52   53 |  3    4    5  
//  6    7    8    9 | 54   55   56   57      (D)
// 58   59   60 | 10   11   12   13   14  
// 15   16   17   18   19   20 | 61   62
// 61 | 21   22   23   24   25   26   27 
// 28   29   30   31   32   33   34   35
//
// If we can form (D),  we will then:
//   1) Reverse the even rows
//   2) Transpose
//   3) Rotate the rows 
//
// and we'll have (A).
//

void 
fromHalfZigZag_f16c (unsigned short *src, float *dst)
{
    #if defined IMF_HAVE_GCC_INLINEASM_64
        __asm__

           /* x3 <- 0                    
            * x8 <- [ 0- 7]              
            * x6 <- [56-63]              
            * x9 <- [21-28]              
            * x7 <- [28-35]              
            * x3 <- [ 6- 9] (lower half) */
          
          ("vpxor   %%xmm3,  %%xmm3, %%xmm3   \n"
           "vmovdqa    (%0), %%xmm8           \n"
           "vmovdqa 112(%0), %%xmm6           \n"
           "vmovdqu  42(%0), %%xmm9           \n"
           "vmovdqu  56(%0), %%xmm7           \n"
           "vmovq    12(%0), %%xmm3           \n"

           /* Setup rows 0-2 of A in xmm0-xmm2 
            * x1 <- x8 >> 16 (1 value)     
            * x2 <- x8 << 32 (2 values)    
            * x0 <- alignr([35-42], x8, 2) 
            * x1 <- blend(x1, [41-48])     
            * x2 <- blend(x2, [49-56])     */

           "vpsrldq      $2, %%xmm8, %%xmm1   \n"      
           "vpslldq      $4, %%xmm8, %%xmm2   \n"      
           "vpalignr     $2, 70(%0), %%xmm8, %%xmm0 \n"
           "vpblendw  $0xfc, 82(%0), %%xmm1, %%xmm1 \n"
           "vpblendw  $0x1f, 98(%0), %%xmm2, %%xmm2 \n"
     
           /* Setup rows 4-6 of A in xmm4-xmm6 
            * x4 <- x6 >> 32 (2 values)   
            * x5 <- x6 << 16 (1 value)    
            * x6 <- alignr(x6,x9,14)      
            * x4 <- blend(x4, [ 7-14])    
            * x5 <- blend(x5, [15-22])    */

           "vpsrldq      $4, %%xmm6, %%xmm4         \n"
           "vpslldq      $2, %%xmm6, %%xmm5         \n"
           "vpalignr    $14, %%xmm6, %%xmm9, %%xmm6 \n"
           "vpblendw  $0xf8, 14(%0), %%xmm4, %%xmm4 \n"
           "vpblendw  $0x3f, 30(%0), %%xmm5, %%xmm5 \n"

           /* Load the upper half of row 3 into xmm3 
            * x3 <- [54-57] (upper half) */

           "vpinsrq      $1, 108(%0), %%xmm3, %%xmm3\n"

           /* Reverse the even rows. We're not using PSHUFB as
            * that requires loading an extra constant all the time,
            * and we're alreadly pretty memory bound.
            */

           "vpshuflw $0x1b, %%xmm0, %%xmm0          \n" 
           "vpshuflw $0x1b, %%xmm2, %%xmm2          \n" 
           "vpshuflw $0x1b, %%xmm4, %%xmm4          \n" 
           "vpshuflw $0x1b, %%xmm6, %%xmm6          \n" 

           "vpshufhw $0x1b, %%xmm0, %%xmm0          \n" 
           "vpshufhw $0x1b, %%xmm2, %%xmm2          \n" 
           "vpshufhw $0x1b, %%xmm4, %%xmm4          \n" 
           "vpshufhw $0x1b, %%xmm6, %%xmm6          \n" 

           "vpshufd $0x4e, %%xmm0, %%xmm0          \n" 
           "vpshufd $0x4e, %%xmm2, %%xmm2          \n" 
           "vpshufd $0x4e, %%xmm4, %%xmm4          \n" 
           "vpshufd $0x4e, %%xmm6, %%xmm6          \n" 

           /* Transpose xmm0-xmm7 into xmm8-xmm15 */

           "vpunpcklwd %%xmm1, %%xmm0, %%xmm8       \n"
           "vpunpcklwd %%xmm3, %%xmm2, %%xmm9       \n"
           "vpunpcklwd %%xmm5, %%xmm4, %%xmm10      \n"
           "vpunpcklwd %%xmm7, %%xmm6, %%xmm11      \n"
           "vpunpckhwd %%xmm1, %%xmm0, %%xmm12      \n"
           "vpunpckhwd %%xmm3, %%xmm2, %%xmm13      \n"
           "vpunpckhwd %%xmm5, %%xmm4, %%xmm14      \n"
           "vpunpckhwd %%xmm7, %%xmm6, %%xmm15      \n"
     
           "vpunpckldq  %%xmm9,  %%xmm8, %%xmm0     \n"
           "vpunpckldq %%xmm11, %%xmm10, %%xmm1     \n"
           "vpunpckhdq  %%xmm9,  %%xmm8, %%xmm2     \n"
           "vpunpckhdq %%xmm11, %%xmm10, %%xmm3     \n"
           "vpunpckldq %%xmm13, %%xmm12, %%xmm4     \n"
           "vpunpckldq %%xmm15, %%xmm14, %%xmm5     \n"
           "vpunpckhdq %%xmm13, %%xmm12, %%xmm6     \n"
           "vpunpckhdq %%xmm15, %%xmm14, %%xmm7     \n"
     
           "vpunpcklqdq %%xmm1,  %%xmm0, %%xmm8     \n"
           "vpunpckhqdq %%xmm1,  %%xmm0, %%xmm9     \n"
           "vpunpcklqdq %%xmm3,  %%xmm2, %%xmm10    \n"
           "vpunpckhqdq %%xmm3,  %%xmm2, %%xmm11    \n"
           "vpunpcklqdq %%xmm4,  %%xmm5, %%xmm12    \n"
           "vpunpckhqdq %%xmm5,  %%xmm4, %%xmm13    \n"
           "vpunpcklqdq %%xmm7,  %%xmm6, %%xmm14    \n"
           "vpunpckhqdq %%xmm7,  %%xmm6, %%xmm15    \n"

           /* Rotate the rows to get the correct final order. 
            * Rotating xmm12 isn't needed, as we can handle
            * the rotation in the PUNPCKLQDQ above. Rotating
            * xmm8 isn't needed as it's already in the right order           
            */

           "vpalignr  $2,  %%xmm9,  %%xmm9,  %%xmm9 \n"
           "vpalignr  $4, %%xmm10, %%xmm10, %%xmm10 \n"
           "vpalignr  $6, %%xmm11, %%xmm11, %%xmm11 \n"
           "vpalignr $10, %%xmm13, %%xmm13, %%xmm13 \n"
           "vpalignr $12, %%xmm14, %%xmm14, %%xmm14 \n"
           "vpalignr $14, %%xmm15, %%xmm15, %%xmm15 \n"

            /* Convert from half -> float */

           "vcvtph2ps  %%xmm8, %%ymm8            \n"  
           "vcvtph2ps  %%xmm9, %%ymm9            \n"
           "vcvtph2ps %%xmm10, %%ymm10           \n"
           "vcvtph2ps %%xmm11, %%ymm11           \n"
           "vcvtph2ps %%xmm12, %%ymm12           \n"
           "vcvtph2ps %%xmm13, %%ymm13           \n"
           "vcvtph2ps %%xmm14, %%ymm14           \n"
           "vcvtph2ps %%xmm15, %%ymm15           \n"
           
           /* Move float values to dst */

           "vmovaps    %%ymm8,    (%1)           \n"
           "vmovaps    %%ymm9,  32(%1)           \n"
           "vmovaps   %%ymm10,  64(%1)           \n" 
           "vmovaps   %%ymm11,  96(%1)           \n" 
           "vmovaps   %%ymm12, 128(%1)           \n" 
           "vmovaps   %%ymm13, 160(%1)           \n" 
           "vmovaps   %%ymm14, 192(%1)           \n" 
           "vmovaps   %%ymm15, 224(%1)           \n"
        #ifndef __AVX__
            "vzeroupper                          \n"
        #endif /* __AVX__ */
            : /* Output  */                
            : /* Input   */ "r"(src), "r"(dst)
            : /* Clobber */ "memory",
        #ifndef __AVX__
                            "%xmm0",  "%xmm1",  "%xmm2",  "%xmm3", 
                            "%xmm4",  "%xmm5",  "%xmm6",  "%xmm7",
                            "%xmm8",  "%xmm9",  "%xmm10", "%xmm11",
                            "%xmm12", "%xmm13", "%xmm14", "%xmm15"
        #else
                            "%ymm0",  "%ymm1",  "%ymm2",  "%ymm3", 
                            "%ymm4",  "%ymm5",  "%ymm6",  "%ymm7",
                            "%ymm8",  "%ymm9",  "%ymm10", "%ymm11",
                            "%ymm12", "%ymm13", "%ymm14", "%ymm15"
        #endif /* __AVX__ */
        );

    #else
        fromHalfZigZag_scalar(src, dst);
    #endif /* defined IMF_HAVE_GCC_INLINEASM_64 */
}


//
// Inverse 8x8 DCT, only inverting the DC. This assumes that
// all AC frequencies are 0.
//

#ifndef IMF_HAVE_SSE2

void 
dctInverse8x8DcOnly (float *data)
{
    float val = data[0] * 3.535536e-01f * 3.535536e-01f;

    for (int i = 0; i < 64; ++i)
        data[i] = val;
}

#else  /* IMF_HAVE_SSE2 */

void
dctInverse8x8DcOnly (float *data)
{
    __m128 src = _mm_set1_ps (data[0] * 3.535536e-01f * 3.535536e-01f);
    __m128 *dst = (__m128 *)data;

    for (int i = 0; i < 16; ++i)
        dst[i] = src;
}

#endif /* IMF_HAVE_SSE2 */


//
// Full 8x8 Inverse DCT:
//
// Simple inverse DCT on an 8x8 block, with scalar ops only.
//  Operates on data in-place.
//
// This is based on the iDCT formuation (y = frequency domain,
//                                       x = spatial domain)
//
//    [x0]    [        ][y0]    [        ][y1] 
//    [x1] =  [  M1    ][y2]  + [  M2    ][y3] 
//    [x2]    [        ][y4]    [        ][y5] 
//    [x3]    [        ][y6]    [        ][y7]
//
//    [x7]    [        ][y0]    [        ][y1] 
//    [x6] =  [  M1    ][y2]  - [  M2    ][y3] 
//    [x5]    [        ][y4]    [        ][y5] 
//    [x4]    [        ][y6]    [        ][y7]
//
// where M1:             M2:
//
//   [a  c  a   f]     [b  d  e  g]
//   [a  f -a  -c]     [d -g -b -e]
//   [a -f -a   c]     [e -b  g  d]
//   [a -c  a  -f]     [g -e  d -b]
//
// and the constants are as defined below..
//
// If you know how many of the lower rows are zero, that can
// be passed in to help speed things up. If you don't know, 
// just set zeroedRows=0.
//

//
// Default implementation
//

template <int zeroedRows>
void
dctInverse8x8_scalar (float *data)
{
    const float a = .5f * cosf (3.14159f / 4.0f);
    const float b = .5f * cosf (3.14159f / 16.0f);
    const float c = .5f * cosf (3.14159f / 8.0f);
    const float d = .5f * cosf (3.f*3.14159f / 16.0f);
    const float e = .5f * cosf (5.f*3.14159f / 16.0f);
    const float f = .5f * cosf (3.f*3.14159f / 8.0f);
    const float g = .5f * cosf (7.f*3.14159f / 16.0f);

    float alpha[4], beta[4], theta[4], gamma[4];

    float *rowPtr = NULL;

    //
    // First pass - row wise.
    //
    // This looks less-compact than the description above in
    // an attempt to fold together common sub-expressions.
    //

    for (int row = 0; row < 8 - zeroedRows; ++row)
    {
        rowPtr = data + row * 8;

        alpha[0] = c * rowPtr[2]; 
        alpha[1] = f * rowPtr[2]; 
        alpha[2] = c * rowPtr[6]; 
        alpha[3] = f * rowPtr[6]; 

        beta[0] = b * rowPtr[1] + d * rowPtr[3] + e * rowPtr[5] + g * rowPtr[7];
        beta[1] = d * rowPtr[1] - g * rowPtr[3] - b * rowPtr[5] - e * rowPtr[7];
        beta[2] = e * rowPtr[1] - b * rowPtr[3] + g * rowPtr[5] + d * rowPtr[7];
        beta[3] = g * rowPtr[1] - e * rowPtr[3] + d * rowPtr[5] - b * rowPtr[7];

        theta[0] = a * (rowPtr[0] + rowPtr[4]);
        theta[3] = a * (rowPtr[0] - rowPtr[4]);

        theta[1] = alpha[0] + alpha[3]; 
        theta[2] = alpha[1] - alpha[2]; 


        gamma[0] = theta[0] + theta[1];
        gamma[1] = theta[3] + theta[2];
        gamma[2] = theta[3] - theta[2];
        gamma[3] = theta[0] - theta[1];


        rowPtr[0] = gamma[0] + beta[0];
        rowPtr[1] = gamma[1] + beta[1];
        rowPtr[2] = gamma[2] + beta[2];
        rowPtr[3] = gamma[3] + beta[3];

        rowPtr[4] = gamma[3] - beta[3];
        rowPtr[5] = gamma[2] - beta[2];
        rowPtr[6] = gamma[1] - beta[1];
        rowPtr[7] = gamma[0] - beta[0];
    }

    //
    // Second pass - column wise.
    //

    for (int column = 0; column < 8; ++column)
    {
        alpha[0] = c * data[16+column]; 
        alpha[1] = f * data[16+column]; 
        alpha[2] = c * data[48+column]; 
        alpha[3] = f * data[48+column]; 

        beta[0] = b * data[8+column]  + d * data[24+column] +
                  e * data[40+column] + g * data[56+column];

        beta[1] = d * data[8+column]  - g * data[24+column] -
                  b * data[40+column] - e * data[56+column];

        beta[2] = e * data[8+column]  - b * data[24+column] + 
                  g * data[40+column] + d * data[56+column];

        beta[3] = g * data[8+column]  - e * data[24+column] + 
                  d * data[40+column] - b * data[56+column];

        theta[0] = a * (data[column] + data[32+column]);
        theta[3] = a * (data[column] - data[32+column]);

        theta[1] = alpha[0] + alpha[3]; 
        theta[2] = alpha[1] - alpha[2]; 

        gamma[0] = theta[0] + theta[1];
        gamma[1] = theta[3] + theta[2];
        gamma[2] = theta[3] - theta[2];
        gamma[3] = theta[0] - theta[1];

        data[     column] = gamma[0] + beta[0];
        data[ 8 + column] = gamma[1] + beta[1];
        data[16 + column] = gamma[2] + beta[2];
        data[24 + column] = gamma[3] + beta[3];

        data[32 + column] = gamma[3] - beta[3];
        data[40 + column] = gamma[2] - beta[2];
        data[48 + column] = gamma[1] - beta[1];
        data[56 + column] = gamma[0] - beta[0];
    }
}


//
// SSE2 Implementation
//

template <int zeroedRows>
void
dctInverse8x8_sse2 (float *data)
{
    #ifdef IMF_HAVE_SSE2
        __m128 a  = {3.535536e-01f,3.535536e-01f,3.535536e-01f,3.535536e-01f};
        __m128 b  = {4.903927e-01f,4.903927e-01f,4.903927e-01f,4.903927e-01f};
        __m128 c  = {4.619398e-01f,4.619398e-01f,4.619398e-01f,4.619398e-01f};
        __m128 d  = {4.157349e-01f,4.157349e-01f,4.157349e-01f,4.157349e-01f};
        __m128 e  = {2.777855e-01f,2.777855e-01f,2.777855e-01f,2.777855e-01f};
        __m128 f  = {1.913422e-01f,1.913422e-01f,1.913422e-01f,1.913422e-01f};
        __m128 g  = {9.754573e-02f,9.754573e-02f,9.754573e-02f,9.754573e-02f};

        __m128 c0 = {3.535536e-01f, 3.535536e-01f, 3.535536e-01f, 3.535536e-01f};
        __m128 c1 = {4.619398e-01f, 1.913422e-01f,-1.913422e-01f,-4.619398e-01f};
        __m128 c2 = {3.535536e-01f,-3.535536e-01f,-3.535536e-01f, 3.535536e-01f};
        __m128 c3 = {1.913422e-01f,-4.619398e-01f, 4.619398e-01f,-1.913422e-01f};

        __m128 c4 = {4.903927e-01f, 4.157349e-01f, 2.777855e-01f, 9.754573e-02f};
        __m128 c5 = {4.157349e-01f,-9.754573e-02f,-4.903927e-01f,-2.777855e-01f};
        __m128 c6 = {2.777855e-01f,-4.903927e-01f, 9.754573e-02f, 4.157349e-01f};
        __m128 c7 = {9.754573e-02f,-2.777855e-01f, 4.157349e-01f,-4.903927e-01f};

        __m128 *srcVec = (__m128 *)data;
        __m128 x[8], evenSum, oddSum;
        __m128 in[8], alpha[4], beta[4], theta[4], gamma[4];
        
        //
        // Rows -   
        //
        //  Treat this just like matrix-vector multiplication. The
        //  trick is to note that:
        //
        //    [M00 M01 M02 M03][v0]   [(v0 M00) + (v1 M01) + (v2 M02) + (v3 M03)]
        //    [M10 M11 M12 M13][v1] = [(v0 M10) + (v1 M11) + (v2 M12) + (v3 M13)]
        //    [M20 M21 M22 M23][v2]   [(v0 M20) + (v1 M21) + (v2 M22) + (v3 M23)]
        //    [M30 M31 M32 M33][v3]   [(v0 M30) + (v1 M31) + (v2 M32) + (v3 M33)]
        //
        // Then, we can fill a register with v_i and multiply by the i-th column
        // of M, accumulating across all i-s. 
        //
        // The kids refer to the populating of a register with a single value
        // "broadcasting", and it can be done with a shuffle instruction. It
        // seems to be the slowest part of the whole ordeal.
        //
        // Our matrix columns are stored above in c0-c7. c0-3 make up M1, and
        // c4-7 are from M2.
        //

        #define DCT_INVERSE_8x8_SS2_ROW_LOOP(i)                             \
            /*                                                              \
             * Broadcast the components of the row                          \
             */                                                             \
                                                                            \
            x[0] = _mm_shuffle_ps (srcVec[2 * i],                           \
                                   srcVec[2 * i],                           \
                                   _MM_SHUFFLE (0, 0, 0, 0));               \
                                                                            \
            x[1] = _mm_shuffle_ps (srcVec[2 * i],                           \
                                   srcVec[2 * i],                           \
                                   _MM_SHUFFLE (1, 1, 1, 1));               \
                                                                            \
            x[2] = _mm_shuffle_ps (srcVec[2 * i],                           \
                                   srcVec[2 * i],                           \
                                   _MM_SHUFFLE (2, 2, 2, 2));               \
                                                                            \
            x[3] = _mm_shuffle_ps (srcVec[2 * i],                           \
                                   srcVec[2 * i],                           \
                                   _MM_SHUFFLE (3, 3, 3, 3));               \
                                                                            \
            x[4] = _mm_shuffle_ps (srcVec[2 * i + 1],                       \
                                   srcVec[2 * i + 1],                       \
                                   _MM_SHUFFLE (0, 0, 0, 0));               \
                                                                            \
            x[5] = _mm_shuffle_ps (srcVec[2 * i + 1],                       \
                                   srcVec[2 * i + 1],                       \
                                   _MM_SHUFFLE (1, 1, 1, 1));               \
                                                                            \
            x[6] = _mm_shuffle_ps (srcVec[2 * i + 1],                       \
                                   srcVec[2 * i + 1],                       \
                                   _MM_SHUFFLE (2, 2, 2, 2));               \
                                                                            \
            x[7] = _mm_shuffle_ps (srcVec[2 * i + 1],                       \
                                   srcVec[2 * i + 1],                       \
                                   _MM_SHUFFLE (3, 3, 3, 3));               \
            /*                                                              \
             * Multiply the components by each column of the matrix         \
             */                                                             \
                                                                            \
            x[0] = _mm_mul_ps (x[0], c0);                                   \
            x[2] = _mm_mul_ps (x[2], c1);                                   \
            x[4] = _mm_mul_ps (x[4], c2);                                   \
            x[6] = _mm_mul_ps (x[6], c3);                                   \
                                                                            \
            x[1] = _mm_mul_ps (x[1], c4);                                   \
            x[3] = _mm_mul_ps (x[3], c5);                                   \
            x[5] = _mm_mul_ps (x[5], c6);                                   \
            x[7] = _mm_mul_ps (x[7], c7);                                   \
                                                                            \
            /*                                                              \
             * Add across                                                   \
             */                                                             \
                                                                            \
            evenSum = _mm_setzero_ps();                                     \
            evenSum = _mm_add_ps (evenSum, x[0]);                           \
            evenSum = _mm_add_ps (evenSum, x[2]);                           \
            evenSum = _mm_add_ps (evenSum, x[4]);                           \
            evenSum = _mm_add_ps (evenSum, x[6]);                           \
                                                                            \
            oddSum = _mm_setzero_ps();                                      \
            oddSum = _mm_add_ps (oddSum, x[1]);                             \
            oddSum = _mm_add_ps (oddSum, x[3]);                             \
            oddSum = _mm_add_ps (oddSum, x[5]);                             \
            oddSum = _mm_add_ps (oddSum, x[7]);                             \
                                                                            \
            /*                                                              \
             * Final Sum:                                                   \
             *    out [0, 1, 2, 3] = evenSum + oddSum                       \
             *    out [7, 6, 5, 4] = evenSum - oddSum                       \
             */                                                             \
                                                                            \
            srcVec[2 * i]     = _mm_add_ps (evenSum, oddSum);               \
            srcVec[2 * i + 1] = _mm_sub_ps (evenSum, oddSum);               \
            srcVec[2 * i + 1] = _mm_shuffle_ps (srcVec[2 * i + 1],          \
                                                srcVec[2 * i + 1],          \
                                                _MM_SHUFFLE (0, 1, 2, 3));

        switch (zeroedRows)
        {
          case 0:
          default:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (3)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (4)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (5)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (6)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (7)
            break;

          case 1:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (3)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (4)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (5)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (6)
            break;

          case 2:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (3)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (4)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (5)
            break;

          case 3:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (3)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (4)
            break;

          case 4:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (3)
            break;

          case 5:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (2)
            break;

          case 6:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            DCT_INVERSE_8x8_SS2_ROW_LOOP (1)
            break;

          case 7:
            DCT_INVERSE_8x8_SS2_ROW_LOOP (0)
            break;
        }

        //
        // Columns -
        //
        // This is slightly more straightforward, if less readable. Here
        // we just operate on 4 columns at a time, in two batches.
        //
        // The slight mess is to try and cache sub-expressions, which
        // we ignore in the row-wise pass.
        //

        for (int col = 0; col < 2; ++col)
        {

            for (int i = 0; i < 8; ++i)
                in[i] = srcVec[2 * i + col];

            alpha[0] = _mm_mul_ps (c, in[2]);
            alpha[1] = _mm_mul_ps (f, in[2]);
            alpha[2] = _mm_mul_ps (c, in[6]);
            alpha[3] = _mm_mul_ps (f, in[6]);

            beta[0] = _mm_add_ps (_mm_add_ps (_mm_mul_ps (in[1], b),
                                                          _mm_mul_ps (in[3], d)),
                                              _mm_add_ps (_mm_mul_ps (in[5], e),
                                                          _mm_mul_ps (in[7], g)));

            beta[1] = _mm_sub_ps (_mm_sub_ps (_mm_mul_ps (in[1], d),
                                                          _mm_mul_ps (in[3], g)),
                                              _mm_add_ps (_mm_mul_ps (in[5], b),
                                                          _mm_mul_ps (in[7], e)));

            beta[2] = _mm_add_ps (_mm_sub_ps (_mm_mul_ps (in[1], e),
                                                          _mm_mul_ps (in[3], b)),
                                              _mm_add_ps (_mm_mul_ps (in[5], g),
                                                          _mm_mul_ps (in[7], d)));

            beta[3] = _mm_add_ps (_mm_sub_ps (_mm_mul_ps (in[1], g),
                                                          _mm_mul_ps (in[3], e)),
                                              _mm_sub_ps (_mm_mul_ps (in[5], d),
                                                          _mm_mul_ps (in[7], b)));

            theta[0] = _mm_mul_ps (a, _mm_add_ps (in[0], in[4]));
            theta[3] = _mm_mul_ps (a, _mm_sub_ps (in[0], in[4]));

            theta[1] = _mm_add_ps (alpha[0], alpha[3]);
            theta[2] = _mm_sub_ps (alpha[1], alpha[2]);

            gamma[0] = _mm_add_ps (theta[0], theta[1]);
            gamma[1] = _mm_add_ps (theta[3], theta[2]);
            gamma[2] = _mm_sub_ps (theta[3], theta[2]);
            gamma[3] = _mm_sub_ps (theta[0], theta[1]);

            srcVec[  col] = _mm_add_ps (gamma[0], beta[0]);
            srcVec[2+col] = _mm_add_ps (gamma[1], beta[1]);
            srcVec[4+col] = _mm_add_ps (gamma[2], beta[2]);
            srcVec[6+col] = _mm_add_ps (gamma[3], beta[3]);

            srcVec[ 8+col] = _mm_sub_ps (gamma[3], beta[3]);
            srcVec[10+col] = _mm_sub_ps (gamma[2], beta[2]);
            srcVec[12+col] = _mm_sub_ps (gamma[1], beta[1]);
            srcVec[14+col] = _mm_sub_ps (gamma[0], beta[0]);
        }

    #else /* IMF_HAVE_SSE2 */

        dctInverse8x8_scalar<zeroedRows> (data);

    #endif /* IMF_HAVE_SSE2 */
}


//
// AVX Implementation
//

#define STR(A) #A

#define IDCT_AVX_SETUP_2_ROWS(_DST0,  _DST1,  _TMP0,  _TMP1, \
                              _OFF00, _OFF01, _OFF10, _OFF11) \
    "vmovaps                 " STR(_OFF00) "(%0),  %%xmm" STR(_TMP0) "  \n" \
    "vmovaps                 " STR(_OFF01) "(%0),  %%xmm" STR(_TMP1) "  \n" \
    "                                                                                \n" \
    "vinsertf128  $1, " STR(_OFF10) "(%0), %%ymm" STR(_TMP0) ", %%ymm" STR(_TMP0) "  \n" \
    "vinsertf128  $1, " STR(_OFF11) "(%0), %%ymm" STR(_TMP1) ", %%ymm" STR(_TMP1) "  \n" \
    "                                                                                \n" \
    "vunpcklpd      %%ymm" STR(_TMP1) ",  %%ymm" STR(_TMP0) ",  %%ymm" STR(_DST0) "  \n" \
    "vunpckhpd      %%ymm" STR(_TMP1) ",  %%ymm" STR(_TMP0) ",  %%ymm" STR(_DST1) "  \n" \
    "                                                                                \n" \
    "vunpcklps      %%ymm" STR(_DST1) ",  %%ymm" STR(_DST0) ",  %%ymm" STR(_TMP0) "  \n" \
    "vunpckhps      %%ymm" STR(_DST1) ",  %%ymm" STR(_DST0) ",  %%ymm" STR(_TMP1) "  \n" \
    "                                                                                \n" \
    "vunpcklpd      %%ymm" STR(_TMP1) ",  %%ymm" STR(_TMP0) ",  %%ymm" STR(_DST0) "  \n" \
    "vunpckhpd      %%ymm" STR(_TMP1) ",  %%ymm" STR(_TMP0) ",  %%ymm" STR(_DST1) "  \n" 

#define IDCT_AVX_MMULT_ROWS(_SRC)                       \
    /* Broadcast the source values into y12-y15 */      \
    "vpermilps $0x00, " STR(_SRC) ", %%ymm12       \n"  \
    "vpermilps $0x55, " STR(_SRC) ", %%ymm13       \n"  \
    "vpermilps $0xaa, " STR(_SRC) ", %%ymm14       \n"  \
    "vpermilps $0xff, " STR(_SRC) ", %%ymm15       \n"  \
                                                        \
    /* Multiple coefs and the broadcasted values */     \
    "vmulps    %%ymm12,  %%ymm8, %%ymm12     \n"        \
    "vmulps    %%ymm13,  %%ymm9, %%ymm13     \n"        \
    "vmulps    %%ymm14, %%ymm10, %%ymm14     \n"        \
    "vmulps    %%ymm15, %%ymm11, %%ymm15     \n"        \
                                                        \
    /* Accumulate the result back into the source */    \
    "vaddps    %%ymm13, %%ymm12, %%ymm12      \n"       \
    "vaddps    %%ymm15, %%ymm14, %%ymm14      \n"       \
    "vaddps    %%ymm14, %%ymm12, " STR(_SRC) "\n"     

#define IDCT_AVX_EO_TO_ROW_HALVES(_EVEN, _ODD, _FRONT, _BACK)      \
    "vsubps   " STR(_ODD) "," STR(_EVEN) "," STR(_BACK)  "\n"  \
    "vaddps   " STR(_ODD) "," STR(_EVEN) "," STR(_FRONT) "\n"  \
    /* Reverse the back half                                */ \
    "vpermilps $0x1b," STR(_BACK) "," STR(_BACK) "\n"  

/* In order to allow for path paths when we know certain rows
 * of the 8x8 block are zero, most of the body of the DCT is
 * in the following macro. Statements are wrapped in a ROWn()
 * macro, where n is the lowest row in the 8x8 block in which
 * they depend.
 *
 * This should work for the cases where we have 2-8 full rows.
 * the 1-row case is special, and we'll handle it seperately.  
 */
#define IDCT_AVX_BODY \
    /* ==============================================               
     *               Row 1D DCT                                     
     * ----------------------------------------------
     */                                                           \
                                                                  \
    /* Setup for the row-oriented 1D DCT. Assuming that (%0) holds 
     * the row-major 8x8 block, load ymm0-3 with the even columns
     * and ymm4-7 with the odd columns. The lower half of the ymm
     * holds one row, while the upper half holds the next row.
     *
     * If our source is:
     *    a0 a1 a2 a3   a4 a5 a6 a7
     *    b0 b1 b2 b3   b4 b5 b6 b7
     *
     * We'll be forming:
     *    a0 a2 a4 a6   b0 b2 b4 b6
     *    a1 a3 a5 a7   b1 b3 b5 b7
     */                                                              \
    ROW0( IDCT_AVX_SETUP_2_ROWS(0, 4, 14, 15,    0,  16,  32,  48) ) \
    ROW2( IDCT_AVX_SETUP_2_ROWS(1, 5, 12, 13,   64,  80,  96, 112) ) \
    ROW4( IDCT_AVX_SETUP_2_ROWS(2, 6, 10, 11,  128, 144, 160, 176) ) \
    ROW6( IDCT_AVX_SETUP_2_ROWS(3, 7,  8,  9,  192, 208, 224, 240) ) \
                                                                     \
    /* Multiple the even columns (ymm0-3) by the matrix M1
     * storing the results back in ymm0-3
     *
     * Assume that (%1) holds the matrix in column major order
     */                                                              \
    "vbroadcastf128   (%1),  %%ymm8         \n"                      \
    "vbroadcastf128 16(%1),  %%ymm9         \n"                      \
    "vbroadcastf128 32(%1), %%ymm10         \n"                      \
    "vbroadcastf128 48(%1), %%ymm11         \n"                      \
                                                                     \
    ROW0( IDCT_AVX_MMULT_ROWS(%%ymm0) )                              \
    ROW2( IDCT_AVX_MMULT_ROWS(%%ymm1) )                              \
    ROW4( IDCT_AVX_MMULT_ROWS(%%ymm2) )                              \
    ROW6( IDCT_AVX_MMULT_ROWS(%%ymm3) )                              \
                                                                     \
    /* Repeat, but with the odd columns (ymm4-7) and the 
     * matrix M2
     */                                                              \
    "vbroadcastf128  64(%1),  %%ymm8         \n"                     \
    "vbroadcastf128  80(%1),  %%ymm9         \n"                     \
    "vbroadcastf128  96(%1), %%ymm10         \n"                     \
    "vbroadcastf128 112(%1), %%ymm11         \n"                     \
                                                                     \
    ROW0( IDCT_AVX_MMULT_ROWS(%%ymm4) )                              \
    ROW2( IDCT_AVX_MMULT_ROWS(%%ymm5) )                              \
    ROW4( IDCT_AVX_MMULT_ROWS(%%ymm6) )                              \
    ROW6( IDCT_AVX_MMULT_ROWS(%%ymm7) )                              \
                                                                     \
    /* Sum the M1 (ymm0-3) and M2 (ymm4-7) results to get the 
     * front halves of the results, and difference to get the 
     * back halves. The front halfs end up in ymm0-3, the back
     * halves end up in ymm12-15. 
     */                                                                \
    ROW0( IDCT_AVX_EO_TO_ROW_HALVES(%%ymm0, %%ymm4, %%ymm0, %%ymm12) ) \
    ROW2( IDCT_AVX_EO_TO_ROW_HALVES(%%ymm1, %%ymm5, %%ymm1, %%ymm13) ) \
    ROW4( IDCT_AVX_EO_TO_ROW_HALVES(%%ymm2, %%ymm6, %%ymm2, %%ymm14) ) \
    ROW6( IDCT_AVX_EO_TO_ROW_HALVES(%%ymm3, %%ymm7, %%ymm3, %%ymm15) ) \
                                                                       \
    /* Reassemble the rows halves into ymm0-7  */                      \
    ROW7( "vperm2f128 $0x13, %%ymm3, %%ymm15, %%ymm7   \n" )           \
    ROW6( "vperm2f128 $0x02, %%ymm3, %%ymm15, %%ymm6   \n" )           \
    ROW5( "vperm2f128 $0x13, %%ymm2, %%ymm14, %%ymm5   \n" )           \
    ROW4( "vperm2f128 $0x02, %%ymm2, %%ymm14, %%ymm4   \n" )           \
    ROW3( "vperm2f128 $0x13, %%ymm1, %%ymm13, %%ymm3   \n" )           \
    ROW2( "vperm2f128 $0x02, %%ymm1, %%ymm13, %%ymm2   \n" )           \
    ROW1( "vperm2f128 $0x13, %%ymm0, %%ymm12, %%ymm1   \n" )           \
    ROW0( "vperm2f128 $0x02, %%ymm0, %%ymm12, %%ymm0   \n" )           \
                                                                       \
                                                                       \
    /* ==============================================
     *                Column 1D DCT 
     * ----------------------------------------------
     */                                                                \
                                                                       \
    /* Rows should be in ymm0-7, and M2 columns should still be 
     * preserved in ymm8-11.  M2 has 4 unique values (and +- 
     * versions of each), and all (positive) values appear in 
     * the first column (and row), which is in ymm8.
     *
     * For the column-wise DCT, we need to:
     *   1) Broadcast each element a row of M2 into 4 vectors
     *   2) Multiple the odd rows (ymm1,3,5,7) by the broadcasts.
     *   3) Accumulate into ymm12-15 for the odd outputs.
     *
     * Instead of doing 16 broadcasts for each element in M2, 
     * do 4, filling y8-11 with:
     *
     *     ymm8:  [ b  b  b  b  | b  b  b  b ]
     *     ymm9:  [ d  d  d  d  | d  d  d  d ]
     *     ymm10: [ e  e  e  e  | e  e  e  e ]
     *     ymm11: [ g  g  g  g  | g  g  g  g ]
     * 
     * And deal with the negative values by subtracting during accum.
     */                                                                \
    "vpermilps        $0xff,  %%ymm8, %%ymm11  \n"                     \
    "vpermilps        $0xaa,  %%ymm8, %%ymm10  \n"                     \
    "vpermilps        $0x55,  %%ymm8, %%ymm9   \n"                     \
    "vpermilps        $0x00,  %%ymm8, %%ymm8   \n"                     \
                                                                       \
    /* This one is easy, since we have ymm12-15 open for scratch   
     *    ymm12 = b ymm1 + d ymm3 + e ymm5 + g ymm7 
     */                                                                \
    ROW1( "vmulps    %%ymm1,  %%ymm8, %%ymm12    \n" )                 \
    ROW3( "vmulps    %%ymm3,  %%ymm9, %%ymm13    \n" )                 \
    ROW5( "vmulps    %%ymm5, %%ymm10, %%ymm14    \n" )                 \
    ROW7( "vmulps    %%ymm7, %%ymm11, %%ymm15    \n" )                 \
                                                                       \
    ROW3( "vaddps   %%ymm12, %%ymm13, %%ymm12    \n" )                 \
    ROW7( "vaddps   %%ymm14, %%ymm15, %%ymm14    \n" )                 \
    ROW5( "vaddps   %%ymm12, %%ymm14, %%ymm12    \n" )                 \
                                                                       \
    /* Tricker, since only y13-15 are open for scratch   
     *    ymm13 = d ymm1 - g ymm3 - b ymm5 - e ymm7 
     */                                                                \
    ROW1( "vmulps    %%ymm1,   %%ymm9, %%ymm13   \n" )                 \
    ROW3( "vmulps    %%ymm3,  %%ymm11, %%ymm14   \n" )                 \
    ROW5( "vmulps    %%ymm5,   %%ymm8, %%ymm15   \n" )                 \
                                                                       \
    ROW5( "vaddps    %%ymm14, %%ymm15, %%ymm14   \n" )                 \
    ROW3( "vsubps    %%ymm14, %%ymm13, %%ymm13   \n" )                 \
                                                                       \
    ROW7( "vmulps    %%ymm7,  %%ymm10, %%ymm15   \n" )                 \
    ROW7( "vsubps    %%ymm15, %%ymm13, %%ymm13   \n" )                 \
                                                                       \
    /* Tricker still, as only y14-15 are open for scratch   
     *    ymm14 = e ymm1 - b ymm3 + g ymm5 + d ymm7 
     */                                                                \
    ROW1( "vmulps     %%ymm1, %%ymm10,  %%ymm14  \n" )                 \
    ROW3( "vmulps     %%ymm3,  %%ymm8,  %%ymm15  \n" )                 \
                                                                       \
    ROW3( "vsubps    %%ymm15, %%ymm14, %%ymm14   \n" )                 \
                                                                       \
    ROW5( "vmulps     %%ymm5, %%ymm11, %%ymm15   \n" )                 \
    ROW5( "vaddps    %%ymm15, %%ymm14, %%ymm14   \n" )                 \
                                                                       \
    ROW7( "vmulps    %%ymm7,   %%ymm9, %%ymm15   \n" )                 \
    ROW7( "vaddps    %%ymm15, %%ymm14, %%ymm14   \n" )                 \
                                                                       \
                                                                       \
    /* Easy, as we can blow away ymm1,3,5,7 for scratch
     *    ymm15 = g ymm1 - e ymm3 + d ymm5 - b ymm7 
     */                                                                \
    ROW1( "vmulps    %%ymm1, %%ymm11, %%ymm15    \n" )                 \
    ROW3( "vmulps    %%ymm3, %%ymm10,  %%ymm3    \n" )                 \
    ROW5( "vmulps    %%ymm5,  %%ymm9,  %%ymm5    \n" )                 \
    ROW7( "vmulps    %%ymm7,  %%ymm8,  %%ymm7    \n" )                 \
                                                                       \
    ROW5( "vaddps   %%ymm15,  %%ymm5, %%ymm15    \n" )                 \
    ROW7( "vaddps    %%ymm3,  %%ymm7,  %%ymm3    \n" )                 \
    ROW3( "vsubps    %%ymm3, %%ymm15, %%ymm15    \n" )                 \
                                                                       \
                                                                       \
    /* Load coefs for M1. Because we're going to broadcast
     * coefs, we don't need to load the actual structure from
     * M1. Instead, just load enough that we can broadcast.
     * There are only 6 unique values in M1, but they're in +-
     * pairs, leaving only 3 unique coefs if we add and subtract 
     * properly.
     *
     * Fill      ymm1 with coef[2] = [ a  a  c  f | a  a  c  f ]
     * Broadcast ymm5 with           [ f  f  f  f | f  f  f  f ]
     * Broadcast ymm3 with           [ c  c  c  c | c  c  c  c ]
     * Broadcast ymm1 with           [ a  a  a  a | a  a  a  a ]
     */                                                                \
    "vbroadcastf128   8(%1),  %%ymm1          \n"                      \
    "vpermilps        $0xff,  %%ymm1, %%ymm5  \n"                      \
    "vpermilps        $0xaa,  %%ymm1, %%ymm3  \n"                      \
    "vpermilps        $0x00,  %%ymm1, %%ymm1  \n"                      \
                                                                       \
    /* If we expand E = [M1] [x0 x2 x4 x6]^t, we get the following 
     * common expressions:
     *
     *   E_0 = ymm8  = (a ymm0 + a ymm4) + (c ymm2 + f ymm6) 
     *   E_3 = ymm11 = (a ymm0 + a ymm4) - (c ymm2 + f ymm6)
     * 
     *   E_1 = ymm9  = (a ymm0 - a ymm4) + (f ymm2 - c ymm6)
     *   E_2 = ymm10 = (a ymm0 - a ymm4) - (f ymm2 - c ymm6)
     *
     * Afterwards, ymm8-11 will hold the even outputs.
     */                                                                \
                                                                       \
    /*  ymm11 = (a ymm0 + a ymm4),   ymm1 = (a ymm0 - a ymm4) */       \
    ROW0( "vmulps    %%ymm1,  %%ymm0, %%ymm11   \n" )                  \
    ROW4( "vmulps    %%ymm1,  %%ymm4,  %%ymm4   \n" )                  \
    ROW0( "vmovaps   %%ymm11, %%ymm1            \n" )                  \
    ROW4( "vaddps    %%ymm4, %%ymm11, %%ymm11   \n" )                  \
    ROW4( "vsubps    %%ymm4,  %%ymm1,  %%ymm1   \n" )                  \
                                                                       \
    /* ymm7 = (c ymm2 + f ymm6) */                                     \
    ROW2( "vmulps    %%ymm3, %%ymm2,  %%ymm7    \n" )                  \
    ROW6( "vmulps    %%ymm5, %%ymm6,  %%ymm9    \n" )                  \
    ROW6( "vaddps    %%ymm9, %%ymm7,  %%ymm7    \n" )                  \
                                                                       \
    /* E_0 = ymm8  = (a ymm0 + a ymm4) + (c ymm2 + f ymm6) 
     * E_3 = ymm11 = (a ymm0 + a ymm4) - (c ymm2 + f ymm6) 
     */                                                                \
    ROW0( "vmovaps   %%ymm11, %%ymm8            \n" )                  \
    ROW2( "vaddps     %%ymm7, %%ymm8,  %%ymm8   \n" )                  \
    ROW2( "vsubps     %%ymm7, %%ymm11, %%ymm11  \n" )                  \
                                                                       \
    /* ymm7 = (f ymm2 - c ymm6) */                                     \
    ROW2( "vmulps     %%ymm5,  %%ymm2, %%ymm7   \n" )                  \
    ROW6( "vmulps     %%ymm3,  %%ymm6, %%ymm9   \n" )                  \
    ROW6( "vsubps     %%ymm9,  %%ymm7, %%ymm7   \n" )                  \
                                                                       \
    /* E_1 = ymm9  = (a ymm0 - a ymm4) + (f ymm2 - c ymm6) 
     * E_2 = ymm10 = (a ymm0 - a ymm4) - (f ymm2 - c ymm6)
     */                                                                \
    ROW0( "vmovaps   %%ymm1,  %%ymm9            \n" )                  \
    ROW0( "vmovaps   %%ymm1, %%ymm10            \n" )                  \
    ROW2( "vaddps    %%ymm7,  %%ymm1,  %%ymm9   \n" )                  \
    ROW2( "vsubps    %%ymm7,  %%ymm1,  %%ymm10  \n" )                  \
                                                                       \
    /* Add the even (ymm8-11) and the odds (ymm12-15), 
     * placing the results into ymm0-7 
     */                                                                \
    "vaddps   %%ymm12,  %%ymm8, %%ymm0       \n"                       \
    "vaddps   %%ymm13,  %%ymm9, %%ymm1       \n"                       \
    "vaddps   %%ymm14, %%ymm10, %%ymm2       \n"                       \
    "vaddps   %%ymm15, %%ymm11, %%ymm3       \n"                       \
                                                                       \
    "vsubps   %%ymm12,  %%ymm8, %%ymm7       \n"                       \
    "vsubps   %%ymm13,  %%ymm9, %%ymm6       \n"                       \
    "vsubps   %%ymm14, %%ymm10, %%ymm5       \n"                       \
    "vsubps   %%ymm15, %%ymm11, %%ymm4       \n"                       \
                                                                       \
    /* Copy out the results from ymm0-7  */                            \
    "vmovaps   %%ymm0,    (%0)                   \n"                   \
    "vmovaps   %%ymm1,  32(%0)                   \n"                   \
    "vmovaps   %%ymm2,  64(%0)                   \n"                   \
    "vmovaps   %%ymm3,  96(%0)                   \n"                   \
    "vmovaps   %%ymm4, 128(%0)                   \n"                   \
    "vmovaps   %%ymm5, 160(%0)                   \n"                   \
    "vmovaps   %%ymm6, 192(%0)                   \n"                   \
    "vmovaps   %%ymm7, 224(%0)                   \n"            

/* Output, input, and clobber (OIC) sections of the inline asm */
#define IDCT_AVX_OIC(_IN0)                          \
        : /* Output  */                            \
        : /* Input   */ "r"(_IN0), "r"(sAvxCoef)      \
        : /* Clobber */ "memory",                  \
                        "%xmm0",  "%xmm1",  "%xmm2",  "%xmm3", \
                        "%xmm4",  "%xmm5",  "%xmm6",  "%xmm7", \
                        "%xmm8",  "%xmm9",  "%xmm10", "%xmm11",\
                        "%xmm12", "%xmm13", "%xmm14", "%xmm15" 

/* Include vzeroupper for non-AVX builds                */
#ifndef __AVX__ 
    #define IDCT_AVX_ASM(_IN0)   \
        __asm__(                 \
            IDCT_AVX_BODY        \
            "vzeroupper      \n" \
            IDCT_AVX_OIC(_IN0)   \
        );                       
#else /* __AVX__ */
    #define IDCT_AVX_ASM(_IN0)   \
        __asm__(                 \
            IDCT_AVX_BODY        \
            IDCT_AVX_OIC(_IN0)   \
        );                       
#endif /* __AVX__ */

template <int zeroedRows>
void
dctInverse8x8_avx (float *data)
{
    #if defined IMF_HAVE_GCC_INLINEASM_64

    /* The column-major version of M1, followed by the 
     * column-major version of M2:
     *   
     *          [ a  c  a  f ]          [ b  d  e  g ]
     *   M1  =  [ a  f -a -c ]    M2 =  [ d -g -b -e ]
     *          [ a -f -a  c ]          [ e -b  g  d ]
     *          [ a -c  a -f ]          [ g -e  d -b ]
     */   
    const float sAvxCoef[32]  __attribute__((aligned(32))) = {
        3.535536e-01,  3.535536e-01,  3.535536e-01,  3.535536e-01, /* a  a  a  a */
        4.619398e-01,  1.913422e-01, -1.913422e-01, -4.619398e-01, /* c  f -f -c */
        3.535536e-01, -3.535536e-01, -3.535536e-01,  3.535536e-01, /* a -a -a  a */
        1.913422e-01, -4.619398e-01,  4.619398e-01, -1.913422e-01, /* f -c  c -f */

        4.903927e-01,  4.157349e-01,  2.777855e-01,  9.754573e-02, /* b  d  e  g */
        4.157349e-01, -9.754573e-02, -4.903927e-01, -2.777855e-01, /* d -g -b -e */
        2.777855e-01, -4.903927e-01,  9.754573e-02,  4.157349e-01, /* e -b  g  d */
        9.754573e-02, -2.777855e-01,  4.157349e-01, -4.903927e-01  /* g -e  d -b */
    };

        #define ROW0(_X) _X
        #define ROW1(_X) _X
        #define ROW2(_X) _X
        #define ROW3(_X) _X 
        #define ROW4(_X) _X
        #define ROW5(_X) _X 
        #define ROW6(_X) _X
        #define ROW7(_X) _X 

        if (zeroedRows == 0) {

            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 1) {

            #undef  ROW7
            #define ROW7(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 2) {

            #undef  ROW6
            #define ROW6(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 3) {

            #undef  ROW5
            #define ROW5(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 4) {

            #undef  ROW4
            #define ROW4(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 5) {

            #undef  ROW3
            #define ROW3(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 6) {

            #undef  ROW2
            #define ROW2(_X)
            IDCT_AVX_ASM(data)

        } else if (zeroedRows == 7) {

            __asm__(  

                /* ==============================================
                 *                Row 1D DCT 
                 * ----------------------------------------------
                 */ 
                IDCT_AVX_SETUP_2_ROWS(0, 4, 14, 15,    0,  16,  32,  48) 

                "vbroadcastf128   (%1),  %%ymm8         \n"
                "vbroadcastf128 16(%1),  %%ymm9         \n"
                "vbroadcastf128 32(%1), %%ymm10         \n"
                "vbroadcastf128 48(%1), %%ymm11         \n"

                /* Stash a vector of [a a a a | a a a a] away  in ymm2 */
                "vinsertf128 $1,  %%xmm8,  %%ymm8,  %%ymm2 \n"

                IDCT_AVX_MMULT_ROWS(%%ymm0) 

                "vbroadcastf128  64(%1),  %%ymm8         \n"
                "vbroadcastf128  80(%1),  %%ymm9         \n"
                "vbroadcastf128  96(%1), %%ymm10         \n"
                "vbroadcastf128 112(%1), %%ymm11         \n"

                IDCT_AVX_MMULT_ROWS(%%ymm4) 

                IDCT_AVX_EO_TO_ROW_HALVES(%%ymm0, %%ymm4, %%ymm0, %%ymm12) 

                "vperm2f128 $0x02, %%ymm0, %%ymm12, %%ymm0   \n" 

                /* ==============================================
                 *                Column 1D DCT 
                 * ----------------------------------------------
                 */ 

                /* DC only, so multiple by a and we're done */
                "vmulps   %%ymm2, %%ymm0, %%ymm0  \n"

                /* Copy out results  */
                "vmovaps %%ymm0,    (%0)          \n"
                "vmovaps %%ymm0,  32(%0)          \n"
                "vmovaps %%ymm0,  64(%0)          \n"
                "vmovaps %%ymm0,  96(%0)          \n"
                "vmovaps %%ymm0, 128(%0)          \n"
                "vmovaps %%ymm0, 160(%0)          \n"
                "vmovaps %%ymm0, 192(%0)          \n"
                "vmovaps %%ymm0, 224(%0)          \n"

                #ifndef __AVX__
                    "vzeroupper                   \n" 
                #endif /* __AVX__ */
                IDCT_AVX_OIC(data)
            );
        } else {
            assert(false); // Invalid template instance parameter
        }
    #else  /* IMF_HAVE_GCC_INLINEASM_64 */

        dctInverse8x8_scalar<zeroedRows>(data);

    #endif /*  IMF_HAVE_GCC_INLINEASM_64 */
}


//
// Full 8x8 Forward DCT:
//
// Base forward 8x8 DCT implementation. Works on the data in-place
//
// The implementation describedin Pennebaker + Mitchell,
//  section 4.3.2, and illustrated in figure 4-7
//
// The basic idea is that the 1D DCT math reduces to:
//
//   2*out_0            = c_4 [(s_07 + s_34) + (s_12 + s_56)]
//   2*out_4            = c_4 [(s_07 + s_34) - (s_12 + s_56)]
//
//   {2*out_2, 2*out_6} = rot_6 ((d_12 - d_56), (s_07 - s_34))
//
//   {2*out_3, 2*out_5} = rot_-3 (d_07 - c_4 (s_12 - s_56),
//                                d_34 - c_4 (d_12 + d_56))
//
//   {2*out_1, 2*out_7} = rot_-1 (d_07 + c_4 (s_12 - s_56),
//                               -d_34 - c_4 (d_12 + d_56))
//
// where:
//
//    c_i  = cos(i*pi/16)
//    s_i  = sin(i*pi/16)
//
//    s_ij = in_i + in_j
//    d_ij = in_i - in_j
//
//    rot_i(x, y) = {c_i*x + s_i*y, -s_i*x + c_i*y} 
//
// We'll run the DCT in two passes. First, run the 1D DCT on 
// the rows, in-place. Then, run over the columns in-place, 
// and be done with it.
//

#ifndef IMF_HAVE_SSE2

//
// Default implementation
//

void 
dctForward8x8 (float *data)
{
    float A0, A1, A2, A3, A4, A5, A6, A7;
    float K0, K1, rot_x, rot_y;

    float *srcPtr = data;
    float *dstPtr = data;

    const float c1 = cosf (3.14159f * 1.0f / 16.0f);
    const float c2 = cosf (3.14159f * 2.0f / 16.0f);
    const float c3 = cosf (3.14159f * 3.0f / 16.0f);
    const float c4 = cosf (3.14159f * 4.0f / 16.0f);
    const float c5 = cosf (3.14159f * 5.0f / 16.0f);
    const float c6 = cosf (3.14159f * 6.0f / 16.0f);
    const float c7 = cosf (3.14159f * 7.0f / 16.0f);

    const float c1Half = .5f * c1; 
    const float c2Half = .5f * c2;
    const float c3Half = .5f * c3;
    const float c5Half = .5f * c5;
    const float c6Half = .5f * c6;
    const float c7Half = .5f * c7;

    //
    // First pass - do a 1D DCT over the rows and write the 
    //              results back in place
    //

    for (int row=0; row<8; ++row)
    {
        float *srcRowPtr = srcPtr + 8 * row;
        float *dstRowPtr = dstPtr + 8 * row;

        A0 = srcRowPtr[0] + srcRowPtr[7];
        A1 = srcRowPtr[1] + srcRowPtr[2];
        A2 = srcRowPtr[1] - srcRowPtr[2];
        A3 = srcRowPtr[3] + srcRowPtr[4];
        A4 = srcRowPtr[3] - srcRowPtr[4];
        A5 = srcRowPtr[5] + srcRowPtr[6];
        A6 = srcRowPtr[5] - srcRowPtr[6];
        A7 = srcRowPtr[0] - srcRowPtr[7];      

        K0 = c4 * (A0 + A3); 
        K1 = c4 * (A1 + A5); 

        dstRowPtr[0] = .5f * (K0 + K1);
        dstRowPtr[4] = .5f * (K0 - K1);

        //
        // (2*dst2, 2*dst6) = rot 6 (d12 - d56,  s07 - s34)
        //

        rot_x = A2 - A6;
        rot_y = A0 - A3;

        dstRowPtr[2] =  c6Half * rot_x + c2Half * rot_y;
        dstRowPtr[6] =  c6Half * rot_y - c2Half * rot_x;

        //
        // K0, K1 are active until after dst[1],dst[7]
        //  as well as dst[3], dst[5] are computed.
        //

        K0 = c4 * (A1 - A5);      
        K1 = -1 * c4 * (A2 + A6); 

        //
        // Two ways to do a rotation:
        //
        //  rot i (x, y) = 
        //           X =  c_i*x + s_i*y
        //           Y = -s_i*x + c_i*y
        //
        //        OR
        //
        //           X = c_i*(x+y) + (s_i-c_i)*y
        //           Y = c_i*y     - (s_i+c_i)*x
        //
        // the first case has 4 multiplies, but fewer constants,
        // while the 2nd case has fewer multiplies but takes more space.

        //
        // (2*dst3, 2*dst5) = rot -3 ( d07 - K0,  d34 + K1 )
        //

        rot_x = A7 - K0;
        rot_y = A4 + K1;

        dstRowPtr[3] = c3Half * rot_x - c5Half * rot_y;
        dstRowPtr[5] = c5Half * rot_x + c3Half * rot_y;

        //
        // (2*dst1, 2*dst7) = rot -1 ( d07 + K0,  K1  - d34 )
        //

        rot_x = A7 + K0;
        rot_y = K1 - A4;

        //
        // A: 4, 7 are inactive. All A's are inactive
        //

        dstRowPtr[1] = c1Half * rot_x - c7Half * rot_y;
        dstRowPtr[7] = c7Half * rot_x + c1Half * rot_y;
    }

    //
    // Second pass - do the same, but on the columns
    //

    for (int column = 0; column < 8; ++column)
    {

        A0 = srcPtr[     column] + srcPtr[56 + column];
        A7 = srcPtr[     column] - srcPtr[56 + column];

        A1 = srcPtr[ 8 + column] + srcPtr[16 + column];
        A2 = srcPtr[ 8 + column] - srcPtr[16 + column];

        A3 = srcPtr[24 + column] + srcPtr[32 + column];
        A4 = srcPtr[24 + column] - srcPtr[32 + column];

        A5 = srcPtr[40 + column] + srcPtr[48 + column];
        A6 = srcPtr[40 + column] - srcPtr[48 + column];

        K0 = c4 * (A0 + A3); 
        K1 = c4 * (A1 + A5); 

        dstPtr[   column] = .5f * (K0 + K1);
        dstPtr[32+column] = .5f * (K0 - K1);

        //
        // (2*dst2, 2*dst6) = rot 6 ( d12 - d56,  s07 - s34 )
        //

        rot_x = A2 - A6;
        rot_y = A0 - A3;

        dstPtr[16+column] = .5f * (c6 * rot_x + c2 * rot_y);
        dstPtr[48+column] = .5f * (c6 * rot_y - c2 * rot_x);

        //
        // K0, K1 are active until after dst[1],dst[7]
        //  as well as dst[3], dst[5] are computed.
        //

        K0 = c4 * (A1 - A5);      
        K1 = -1 * c4 * (A2 + A6); 

        //
        // (2*dst3, 2*dst5) = rot -3 ( d07 - K0,  d34 + K1 )
        //

        rot_x = A7 - K0;
        rot_y = A4 + K1;

        dstPtr[24+column] = .5f * (c3 * rot_x - c5 * rot_y);
        dstPtr[40+column] = .5f * (c5 * rot_x + c3 * rot_y);

        //
        // (2*dst1, 2*dst7) = rot -1 ( d07 + K0,  K1  - d34 )
        //

        rot_x = A7 + K0;
        rot_y = K1 - A4;

        dstPtr[ 8+column] = .5f * (c1 * rot_x - c7 * rot_y);
        dstPtr[56+column] = .5f * (c7 * rot_x + c1 * rot_y);
    }
}

#else  /* IMF_HAVE_SSE2 */

//
// SSE2 implementation
//
// Here, we're always doing a column-wise operation
// plus transposes. This might be faster to do differently
// between rows-wise and column-wise
//

void 
dctForward8x8 (float *data)
{
    __m128 *srcVec = (__m128 *)data;
    __m128  a0Vec, a1Vec, a2Vec, a3Vec, a4Vec, a5Vec, a6Vec, a7Vec;
    __m128  k0Vec, k1Vec, rotXVec, rotYVec;
    __m128  transTmp[4], transTmp2[4];

    __m128  c4Vec     = { .70710678f,  .70710678f,  .70710678f,  .70710678f};
    __m128  c4NegVec  = {-.70710678f, -.70710678f, -.70710678f, -.70710678f};

    __m128  c1HalfVec = {.490392640f, .490392640f, .490392640f, .490392640f}; 
    __m128  c2HalfVec = {.461939770f, .461939770f, .461939770f, .461939770f};
    __m128  c3HalfVec = {.415734810f, .415734810f, .415734810f, .415734810f}; 
    __m128  c5HalfVec = {.277785120f, .277785120f, .277785120f, .277785120f}; 
    __m128  c6HalfVec = {.191341720f, .191341720f, .191341720f, .191341720f};
    __m128  c7HalfVec = {.097545161f, .097545161f, .097545161f, .097545161f}; 

    __m128  halfVec   = {.5f, .5f, .5f, .5f};

    for (int iter = 0; iter < 2; ++iter)
    {
        //
        //  Operate on 4 columns at a time. The
        //    offsets into our row-major array are:
        //                  0:  0      1
        //                  1:  2      3
        //                  2:  4      5
        //                  3:  6      7
        //                  4:  8      9
        //                  5: 10     11
        //                  6: 12     13
        //                  7: 14     15
        //

        for (int pass=0; pass<2; ++pass)
        {
            a0Vec = _mm_add_ps (srcVec[ 0 + pass], srcVec[14 + pass]);
            a1Vec = _mm_add_ps (srcVec[ 2 + pass], srcVec[ 4 + pass]);
            a3Vec = _mm_add_ps (srcVec[ 6 + pass], srcVec[ 8 + pass]);
            a5Vec = _mm_add_ps (srcVec[10 + pass], srcVec[12 + pass]);
 
            a7Vec = _mm_sub_ps (srcVec[ 0 + pass], srcVec[14 + pass]);
            a2Vec = _mm_sub_ps (srcVec[ 2 + pass], srcVec[ 4 + pass]);
            a4Vec = _mm_sub_ps (srcVec[ 6 + pass], srcVec[ 8 + pass]);
            a6Vec = _mm_sub_ps (srcVec[10 + pass], srcVec[12 + pass]);

            //
            // First stage; Compute out_0 and out_4
            //

            k0Vec = _mm_add_ps (a0Vec, a3Vec);
            k1Vec = _mm_add_ps (a1Vec, a5Vec);

            k0Vec = _mm_mul_ps (c4Vec, k0Vec);
            k1Vec = _mm_mul_ps (c4Vec, k1Vec);

            srcVec[0 + pass] = _mm_add_ps (k0Vec, k1Vec);
            srcVec[8 + pass] = _mm_sub_ps (k0Vec, k1Vec);

            srcVec[0 + pass] = _mm_mul_ps (srcVec[0 + pass], halfVec );
            srcVec[8 + pass] = _mm_mul_ps (srcVec[8 + pass], halfVec );


            //
            // Second stage; Compute out_2 and out_6
            //
            
            k0Vec = _mm_sub_ps (a2Vec, a6Vec);
            k1Vec = _mm_sub_ps (a0Vec, a3Vec);

            srcVec[ 4 + pass] = _mm_add_ps (_mm_mul_ps (c6HalfVec, k0Vec),
                                            _mm_mul_ps (c2HalfVec, k1Vec));

            srcVec[12 + pass] = _mm_sub_ps (_mm_mul_ps (c6HalfVec, k1Vec), 
                                            _mm_mul_ps (c2HalfVec, k0Vec));

            //
            // Precompute K0 and K1 for the remaining stages
            //

            k0Vec = _mm_mul_ps (_mm_sub_ps (a1Vec, a5Vec), c4Vec);
            k1Vec = _mm_mul_ps (_mm_add_ps (a2Vec, a6Vec), c4NegVec); 

            //
            // Third Stage, compute out_3 and out_5
            //

            rotXVec = _mm_sub_ps (a7Vec, k0Vec);
            rotYVec = _mm_add_ps (a4Vec, k1Vec);

            srcVec[ 6 + pass] = _mm_sub_ps (_mm_mul_ps (c3HalfVec, rotXVec),
                                            _mm_mul_ps (c5HalfVec, rotYVec));

            srcVec[10 + pass] = _mm_add_ps (_mm_mul_ps (c5HalfVec, rotXVec),
                                            _mm_mul_ps (c3HalfVec, rotYVec));

            //
            // Fourth Stage, compute out_1 and out_7
            //

            rotXVec = _mm_add_ps (a7Vec, k0Vec);
            rotYVec = _mm_sub_ps (k1Vec, a4Vec);

            srcVec[ 2 + pass] = _mm_sub_ps (_mm_mul_ps (c1HalfVec, rotXVec),
                                            _mm_mul_ps (c7HalfVec, rotYVec));

            srcVec[14 + pass] = _mm_add_ps (_mm_mul_ps (c7HalfVec, rotXVec), 
                                            _mm_mul_ps (c1HalfVec, rotYVec));
        }

        //
        // Transpose the matrix, in 4x4 blocks. So, if we have our
        // 8x8 matrix divied into 4x4 blocks:
        //
        //         M0 | M1         M0t | M2t
        //        ----+---   -->  -----+------
        //         M2 | M3         M1t | M3t
        //

        //
        // M0t, done in place, the first half.
        //

        transTmp[0] = _mm_shuffle_ps (srcVec[0], srcVec[2], 0x44);
        transTmp[1] = _mm_shuffle_ps (srcVec[4], srcVec[6], 0x44);
        transTmp[3] = _mm_shuffle_ps (srcVec[4], srcVec[6], 0xEE);
        transTmp[2] = _mm_shuffle_ps (srcVec[0], srcVec[2], 0xEE);

        //
        // M3t, also done in place, the first half.
        //

        transTmp2[0] = _mm_shuffle_ps (srcVec[ 9], srcVec[11], 0x44);
        transTmp2[1] = _mm_shuffle_ps (srcVec[13], srcVec[15], 0x44);
        transTmp2[2] = _mm_shuffle_ps (srcVec[ 9], srcVec[11], 0xEE);
        transTmp2[3] = _mm_shuffle_ps (srcVec[13], srcVec[15], 0xEE);

        //
        // M0t, the second half.
        //

        srcVec[0] = _mm_shuffle_ps (transTmp[0], transTmp[1], 0x88);
        srcVec[4] = _mm_shuffle_ps (transTmp[2], transTmp[3], 0x88);
        srcVec[2] = _mm_shuffle_ps (transTmp[0], transTmp[1], 0xDD);
        srcVec[6] = _mm_shuffle_ps (transTmp[2], transTmp[3], 0xDD);

        //
        // M3t, the second half.
        //

        srcVec[ 9] = _mm_shuffle_ps (transTmp2[0], transTmp2[1], 0x88);
        srcVec[13] = _mm_shuffle_ps (transTmp2[2], transTmp2[3], 0x88);
        srcVec[11] = _mm_shuffle_ps (transTmp2[0], transTmp2[1], 0xDD);
        srcVec[15] = _mm_shuffle_ps (transTmp2[2], transTmp2[3], 0xDD);

        //
        // M1 and M2 need to be done at the same time, because we're
        //  swapping. 
        //
        // First, the first half of M1t
        //

        transTmp[0] = _mm_shuffle_ps (srcVec[1], srcVec[3], 0x44);
        transTmp[1] = _mm_shuffle_ps (srcVec[5], srcVec[7], 0x44);
        transTmp[2] = _mm_shuffle_ps (srcVec[1], srcVec[3], 0xEE);
        transTmp[3] = _mm_shuffle_ps (srcVec[5], srcVec[7], 0xEE);

        //
        // And the first half of M2t
        //

        transTmp2[0] = _mm_shuffle_ps (srcVec[ 8], srcVec[10], 0x44);
        transTmp2[1] = _mm_shuffle_ps (srcVec[12], srcVec[14], 0x44);
        transTmp2[2] = _mm_shuffle_ps (srcVec[ 8], srcVec[10], 0xEE);
        transTmp2[3] = _mm_shuffle_ps (srcVec[12], srcVec[14], 0xEE);

        //
        // Second half of M1t
        //

        srcVec[ 8] = _mm_shuffle_ps (transTmp[0], transTmp[1], 0x88);
        srcVec[12] = _mm_shuffle_ps (transTmp[2], transTmp[3], 0x88);
        srcVec[10] = _mm_shuffle_ps (transTmp[0], transTmp[1], 0xDD);
        srcVec[14] = _mm_shuffle_ps (transTmp[2], transTmp[3], 0xDD);

        //
        // Second half of M2
        //

        srcVec[1] = _mm_shuffle_ps (transTmp2[0], transTmp2[1], 0x88);
        srcVec[5] = _mm_shuffle_ps (transTmp2[2], transTmp2[3], 0x88);
        srcVec[3] = _mm_shuffle_ps (transTmp2[0], transTmp2[1], 0xDD);
        srcVec[7] = _mm_shuffle_ps (transTmp2[2], transTmp2[3], 0xDD);
    }
}

#endif /* IMF_HAVE_SSE2 */

} // anonymous namespace

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
