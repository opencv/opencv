///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
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

#ifndef INCLUDED_IMF_COMPILER_SPECIFIC_H
#define INCLUDED_IMF_COMPILER_SPECIFIC_H

#include "ImfNamespace.h"
#include "ImfSimd.h"
#include <stdlib.h>
#include "ImfExport.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


static unsigned long  systemEndianCheckValue   = 0x12345678;
static unsigned long* systemEndianCheckPointer = &systemEndianCheckValue;

// EXR files are little endian - check processor architecture is too
// (optimisation currently not supported for big endian machines)
static bool GLOBAL_SYSTEM_LITTLE_ENDIAN =
        (*(unsigned char*)systemEndianCheckPointer == 0x78 ? true : false);


#ifdef IMF_HAVE_SSE2

#if defined(__GNUC__)
// Causes issues on certain gcc versions
//#define EXR_FORCEINLINE inline __attribute__((always_inline))
#define EXR_FORCEINLINE inline
#define EXR_RESTRICT __restrict

static void* EXRAllocAligned(size_t size, size_t alignment)
{
    // GNUC is used for things like mingw to (cross-)compile for windows
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(__ANDROID__)
    return memalign(alignment, size);
#else
    void* ptr = 0;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}


static void EXRFreeAligned(void* ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#elif defined _MSC_VER

#define EXR_FORCEINLINE __forceinline
#define EXR_RESTRICT __restrict

static void* EXRAllocAligned(size_t size, size_t alignment)
{
    return _aligned_malloc(size, alignment);
}


static void EXRFreeAligned(void* ptr)
{
    _aligned_free(ptr);
}

#elif defined (__INTEL_COMPILER) || \
        defined(__ICL) || \
        defined(__ICC) || \
        defined(__ECC)

#define EXR_FORCEINLINE inline
#define EXR_RESTRICT restrict

static void* EXRAllocAligned(size_t size, size_t alignment)
{
    return _mm_malloc(size, alignment);
}


static void EXRFreeAligned(void* ptr)
{
    _mm_free(ptr);
}

#else

// generic compiler
#define EXR_FORCEINLINE inline
#define EXR_RESTRICT

static void* EXRAllocAligned(size_t size, size_t alignment)
{
    return malloc(size);
}


static void EXRFreeAligned(void* ptr)
{
    free(ptr);
}

#endif // compiler switch


#else // IMF_HAVE_SSE2


#define EXR_FORCEINLINE inline
#define EXR_RESTRICT

static void* EXRAllocAligned(size_t size, size_t alignment)
{
    return malloc(size);
}


static void EXRFreeAligned(void* ptr)
{
    free(ptr);
}


#endif  // IMF_HAVE_SSE2

// 
// Simple CPUID based runtime detection of various capabilities
//
class IMF_EXPORT CpuId
{
    public:
        CpuId();

        bool sse2;
        bool sse3;
        bool ssse3;
        bool sse4_1;
        bool sse4_2;
        bool avx;
        bool f16c;
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif //include guard
