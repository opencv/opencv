/* Optimized slide_hash for POWER processors
 * Copyright (C) 2019-2020 IBM Corporation
 * Author: Matheus Castanho <msc@linux.ibm.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef POWER8_VSX

#define SLIDE_PPC slide_hash_power8
#include "slide_ppc_tpl.h"

#endif /* POWER8_VSX */
