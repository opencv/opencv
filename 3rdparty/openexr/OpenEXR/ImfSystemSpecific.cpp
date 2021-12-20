//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) DreamWorks Animation LLC and Contributors of the OpenEXR Project
//

#include "ImfSimd.h"
#include "ImfSystemSpecific.h"
#include "ImfNamespace.h"
#include "OpenEXRConfig.h"
#include "OpenEXRConfigInternal.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {
#if defined(IMF_HAVE_SSE2) &&  defined(__GNUC__) && !defined(__e2k__)

    // Helper functions for gcc + SSE enabled
    void cpuid(int n, int &eax, int &ebx, int &ecx, int &edx)
    {
        __asm__ __volatile__ (
            "cpuid"
            : /* Output  */ "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : /* Input   */ "a"(n)
            : /* Clobber */);
    }

#else // IMF_HAVE_SSE2 && __GNUC__ && !__e2k__

    // Helper functions for generic compiler - all disabled
    void cpuid(int n, int &eax, int &ebx, int &ecx, int &edx)
    {
        eax = ebx = ecx = edx = 0;
    }

#endif // IMF_HAVE_SSE2 && __GNUC__ && !__e2k__


#ifdef IMF_HAVE_GCC_INLINEASM_X86

    void xgetbv(int n, int &eax, int &edx)
    {
        __asm__ __volatile__ (
            "xgetbv"
            : /* Output  */ "=a"(eax), "=d"(edx)
            : /* Input   */ "c"(n)
            : /* Clobber */);
    }

#else //  IMF_HAVE_GCC_INLINEASM_X86

    void xgetbv(int n, int &eax, int &edx)
    {
        eax = edx = 0;
    }

#endif //  IMF_HAVE_GCC_INLINEASM_X86

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
#if defined(__e2k__) // e2k - MCST Elbrus 2000 architecture
    // Use IMF_HAVE definitions to determine e2k CPU features
#   if defined(IMF_HAVE_SSE2)
        sse2 = true;
#   endif
#   if defined(IMF_HAVE_SSE3)
        sse3 = true;
#   endif
#   if defined(IMF_HAVE_SSSE3)
        ssse3 = true;
#   endif
#   if defined(IMF_HAVE_SSE4_1)
        sse4_1 = true;
#   endif
#   if defined(IMF_HAVE_SSE4_2)
        sse4_2 = true;
#   endif
#   if defined(IMF_HAVE_AVX)
        avx = true;
#   endif
#   if defined(IMF_HAVE_F16C)
        f16c = true;
#   endif
#else // x86/x86_64
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
#endif
}

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
