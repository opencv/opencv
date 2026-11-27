/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2016, 2025, D. R. Commander.
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
 * 64-bit MIPS architecture.
 */

#include "../jsimdint.h"

#include <ctype.h>

#if !defined(__mips_loongson_vector_rev) && defined(__linux__)

#define SOMEWHAT_SANE_PROC_CPUINFO_SIZE_LIMIT  (1024 * 1024)

LOCAL(boolean)
check_feature(char *buffer, char *feature)
{
  char *p;

  if (*feature == 0)
    return FALSE;
  if (strncmp(buffer, "ASEs implemented", 16) != 0)
    return FALSE;
  buffer += 16;
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
      if (check_feature(buffer, "loongson-mmi"))
        *simd_support |= JSIMD_MMI;
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
#if !defined(__mips_loongson_vector_rev) && defined(__linux__)
  int bufsize = 1024; /* an initial guess for the line buffer size limit */
#endif
  unsigned int simd_support = 0;

#if defined(__mips_loongson_vector_rev)
  simd_support |= JSIMD_MMI;
#elif defined(__linux__)
  while (!parse_proc_cpuinfo(bufsize, &simd_support)) {
    bufsize *= 2;
    if (bufsize > SOMEWHAT_SANE_PROC_CPUINFO_SIZE_LIMIT)
      break;
  }
#endif

  return simd_support;
}
