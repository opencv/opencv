/*
 * Copyright (C) 2025, Samsung Electronics Co., Ltd.
 *                     Author:  Filip Wasil
 * Copyright (C) 2026, D. R. Commander.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations when running on a
 * 64-bit Arm architecture.
 */

#include "../jsimdint.h"

#if (!defined(__riscv_v) || __riscv_v < 1000000 || __riscv_v >= 2000000) && \
    (defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO))

#if defined(__linux__)
#include <asm/hwcap.h>
#include <asm/hwprobe.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#define COMPAT_HWCAP_ISA_V  (1 << ('V' - 'A'))
#endif
#include <sys/auxv.h>

extern int has_compliant_vsetvli(void);

static int is_rvv_1_0_available(void)
{
#if defined(__linux__)
  struct riscv_hwprobe pair = { RISCV_HWPROBE_KEY_IMA_EXT_0, 0 };

  if (syscall(__NR_riscv_hwprobe, &pair, 1, 0, 0, 0) >= 0)
    return (pair.value & RISCV_HWPROBE_IMA_V);
  else
  /* At this point, we have already checked AT_HWCAP, so we know we have some
   * version of RVV at our disposal.  If this version of the kernel is failing
   * on syscall(__NR_riscv_hwprobe, ...), then we will check the RVV version by
   * looking at the vsetvli behavior.
   */
#endif
  return has_compliant_vsetvli();
}

#endif


HIDDEN unsigned int
jpeg_simd_cpu_support(void)
{
#if (!defined(__riscv_v) || __riscv_v < 1000000 || __riscv_v >= 2000000) && \
    defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO)
  unsigned long cpufeatures = 0;
#endif
  unsigned int simd_support = 0;

#if defined(__riscv_v) && __riscv_v >= 1000000 && __riscv_v < 2000000
  simd_support |= JSIMD_RVV;
#elif defined(HAVE_GETAUXVAL)
  cpufeatures = getauxval(AT_HWCAP);
  if (cpufeatures & COMPAT_HWCAP_ISA_V && is_rvv_1_0_available())
    simd_support |= JSIMD_RVV;
#elif defined(HAVE_ELF_AUX_INFO)
  elf_aux_info(AT_HWCAP, &cpufeatures, sizeof(cpufeatures));
  if (cpufeatures & COMPAT_HWCAP_ISA_V && is_rvv_1_0_available())
    simd_support |= JSIMD_RVV;
#endif

  return simd_support;
}
