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

#include "ImfSimd.h"
#include "ImfSystemSpecific.h"
#include "ImfNamespace.h"
#include "OpenEXRConfig.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {
#if defined(IMF_HAVE_SSE2) &&  defined(__GNUC__)

    // Helper functions for gcc + SSE enabled
    void cpuid(int n, int &eax, int &ebx, int &ecx, int &edx)
    {
        __asm__ __volatile__ (
            "cpuid"
            : /* Output  */ "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) 
            : /* Input   */ "a"(n)
            : /* Clobber */);
    }

#else // IMF_HAVE_SSE2 && __GNUC__

    // Helper functions for generic compiler - all disabled
    void cpuid(int n, int &eax, int &ebx, int &ecx, int &edx)
    {
        eax = ebx = ecx = edx = 0;
    }

#endif // IMF_HAVE_SSE2 && __GNUC__


#ifdef OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX

    void xgetbv(int n, int &eax, int &edx)
    {
        __asm__ __volatile__ (
            "xgetbv"
            : /* Output  */ "=a"(eax), "=d"(edx) 
            : /* Input   */ "c"(n)
            : /* Clobber */);
    }

#else //  OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX

    void xgetbv(int n, int &eax, int &edx)
    {
        eax = edx = 0;
    }

#endif //  OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX

} // namespace 

CpuId::CpuId():
    sse2(false), 
    sse3(false), 
    ssse3(false),
    sse4_1(false), 
    sse4_2(false), 
    avx(false), 
    f16c(false)
{
    bool osxsave = false;
    int  max     = 0;
    int  eax, ebx, ecx, edx;

    cpuid(0, max, ebx, ecx, edx);
    if (max > 0)
    {
        cpuid(1, eax, ebx, ecx, edx);
        sse2    = ( edx & (1<<26) );
        sse3    = ( ecx & (1<< 0) );
        ssse3   = ( ecx & (1<< 9) );
        sse4_1  = ( ecx & (1<<19) );
        sse4_2  = ( ecx & (1<<20) );
        osxsave = ( ecx & (1<<27) );
        avx     = ( ecx & (1<<28) );
        f16c    = ( ecx & (1<<29) );

        if (!osxsave)
        {
            avx = f16c = false;
        }
        else
        {
            xgetbv(0, eax, edx);
            // eax bit 1 - SSE managed, bit 2 - AVX managed
            if ((eax & 6) != 6)
            {
                avx = f16c = false;
            }
        }
    }
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
