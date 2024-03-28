/* x86_features.c - x86 feature check
 *
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Author:
 *  Jim Kukunas
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "../../zbuild.h"
#include "x86_features.h"

#ifdef _MSC_VER
#  include <intrin.h>
#else
// Newer versions of GCC and clang come with cpuid.h
#  include <cpuid.h>
#endif

#include <string.h>

static inline void cpuid(int info, unsigned* eax, unsigned* ebx, unsigned* ecx, unsigned* edx) {
#ifdef _MSC_VER
    unsigned int registers[4];
    __cpuid((int *)registers, info);

    *eax = registers[0];
    *ebx = registers[1];
    *ecx = registers[2];
    *edx = registers[3];
#else
    __cpuid(info, *eax, *ebx, *ecx, *edx);
#endif
}

static inline void cpuidex(int info, int subinfo, unsigned* eax, unsigned* ebx, unsigned* ecx, unsigned* edx) {
#ifdef _MSC_VER
    unsigned int registers[4];
    __cpuidex((int *)registers, info, subinfo);

    *eax = registers[0];
    *ebx = registers[1];
    *ecx = registers[2];
    *edx = registers[3];
#else
    __cpuid_count(info, subinfo, *eax, *ebx, *ecx, *edx);
#endif
}

static inline uint64_t xgetbv(unsigned int xcr) {
#ifdef _MSC_VER
    return _xgetbv(xcr);
#else
    uint32_t eax, edx;
    __asm__ ( ".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return (uint64_t)(edx) << 32 | eax;
#endif
}

void Z_INTERNAL x86_check_features(struct x86_cpu_features *features) {
    unsigned eax, ebx, ecx, edx;
    unsigned maxbasic;

    cpuid(0, &maxbasic, &ebx, &ecx, &edx);
    cpuid(1 /*CPU_PROCINFO_AND_FEATUREBITS*/, &eax, &ebx, &ecx, &edx);

    features->has_sse2 = edx & 0x4000000;
    features->has_ssse3 = ecx & 0x200;
    features->has_sse42 = ecx & 0x100000;
    features->has_pclmulqdq = ecx & 0x2;

    if (ecx & 0x08000000) {
        uint64_t xfeature = xgetbv(0);

        features->has_os_save_ymm = ((xfeature & 0x06) == 0x06);
        features->has_os_save_zmm = ((xfeature & 0xe6) == 0xe6);
    }

    if (maxbasic >= 7) {
        cpuidex(7, 0, &eax, &ebx, &ecx, &edx);

        // check BMI1 bit
        // Reference: https://software.intel.com/sites/default/files/article/405250/how-to-detect-new-instruction-support-in-the-4th-generation-intel-core-processor-family.pdf
        features->has_vpclmulqdq = ecx & 0x400;

        // check AVX2 bit if the OS supports saving YMM registers
        if (features->has_os_save_ymm) {
            features->has_avx2 = ebx & 0x20;
        }

        // check AVX512 bits if the OS supports saving ZMM registers
        if (features->has_os_save_zmm) {
            features->has_avx512 = ebx & 0x00010000;
            features->has_avx512vnni = ecx & 0x800;
        }
    }
}
