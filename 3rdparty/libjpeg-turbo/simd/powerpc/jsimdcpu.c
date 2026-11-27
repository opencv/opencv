/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2015-2016, 2025-2026, D. R. Commander.
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
 * PowerPC architecture.
 */

#ifdef __amigaos4__
/* This must be defined first as it re-defines GLOBAL otherwise */
#include <proto/exec.h>
#endif

#include "../jsimdint.h"

#include <ctype.h>

#if !defined(__ALTIVEC__)
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO)
#if defined(__FreeBSD__)
#include <machine/cpu.h>
#endif
#include <sys/auxv.h>
#elif defined(__OpenBSD__)
#include <sys/param.h>
#include <sys/sysctl.h>
#include <machine/cpu.h>
#endif
#endif

#if !defined(__ALTIVEC__) && !defined(HAVE_GETAUXVAL) && \
    (defined(__linux__) || defined(ANDROID) || defined(__ANDROID__))

#define SOMEWHAT_SANE_PROC_CPUINFO_SIZE_LIMIT  (1024 * 1024)

LOCAL(boolean)
check_feature(char *buffer, char *feature)
{
  char *p;

  if (*feature == 0)
    return FALSE;
  if (strncmp(buffer, "cpu", 3) != 0)
    return FALSE;
  buffer += 3;
  while (isspace(*buffer))
    buffer++;

  /* Check if 'feature' is present in the buffer as a separate word */
  while ((p = strstr(buffer, feature))) {
    if (p > buffer && !isspace(*(p - 1))) {
      buffer++;
      continue;
    }
    p += strlen(feature);
    if (*p != 0 && !isspace(*p)) {
      buffer++;
      continue;
    }
    return TRUE;
  }
  return FALSE;
}


LOCAL(boolean)
parse_proc_cpuinfo(int bufsize, unsigned int *simd_support)
{
  char *buffer = (char *)malloc(bufsize);
  FILE *fd;

  *simd_support = 0;

  if (!buffer)
    return FALSE;

  fd = fopen("/proc/cpuinfo", "r");
  if (fd) {
    while (fgets(buffer, bufsize, fd)) {
      if (!strchr(buffer, '\n') && !feof(fd)) {
        /* "impossible" happened - insufficient size of the buffer! */
        fclose(fd);
        free(buffer);
        return FALSE;
      }
      if (check_feature(buffer, "altivec"))
        *simd_support |= JSIMD_ALTIVEC;
    }
    fclose(fd);
  }
  free(buffer);
  return TRUE;
}

#endif


HIDDEN unsigned int
jpeg_simd_cpu_support(void)
{
#if !defined(__ALTIVEC__)
#if defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO)
  unsigned long cpufeatures = 0;
#elif defined(__linux__) || defined(ANDROID) || defined(__ANDROID__)
  int bufsize = 1024; /* an initial guess for the line buffer size limit */
#elif defined(__amigaos4__)
  uint32 altivec = 0;
#elif defined(__APPLE__)
  int mib[2] = { CTL_HW, HW_VECTORUNIT };
  int altivec;
  size_t len = sizeof(altivec);
#elif defined(__OpenBSD__)
  int mib[2] = { CTL_MACHDEP, CPU_ALTIVEC };
  int altivec;
  size_t len = sizeof(altivec);
#endif
#endif
  unsigned int simd_support = 0;

#if defined(__ALTIVEC__)
  simd_support |= JSIMD_ALTIVEC;
#elif defined(HAVE_GETAUXVAL)
  cpufeatures = getauxval(AT_HWCAP);
  if (cpufeatures & PPC_FEATURE_HAS_ALTIVEC)
    simd_support |= JSIMD_ALTIVEC;
#elif defined(__linux__) || defined(ANDROID) || defined(__ANDROID__)
  while (!parse_proc_cpuinfo(bufsize, &simd_support)) {
    bufsize *= 2;
    if (bufsize > SOMEWHAT_SANE_PROC_CPUINFO_SIZE_LIMIT)
      break;
  }
#elif defined(__amigaos4__)
  IExec->GetCPUInfoTags(GCIT_VectorUnit, &altivec, TAG_DONE);
  if (altivec == VECTORTYPE_ALTIVEC)
    simd_support |= JSIMD_ALTIVEC;
#elif defined(__APPLE__) || \
      (defined(__OpenBSD__) && !defined(HAVE_ELF_AUX_INFO))
  if (sysctl(mib, 2, &altivec, &len, NULL, 0) == 0 && altivec != 0)
    simd_support |= JSIMD_ALTIVEC;
#elif defined(HAVE_ELF_AUX_INFO)
  elf_aux_info(AT_HWCAP, &cpufeatures, sizeof(cpufeatures));
  if (cpufeatures & PPC_FEATURE_HAS_ALTIVEC)
    simd_support |= JSIMD_ALTIVEC;
#endif

  return simd_support;
}
