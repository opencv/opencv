/* Optimized slide_hash for PowerPC processors with VMX instructions
 * Copyright (C) 2017-2021 Mika T. Lindqvist <postmaster@raasu.org>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifdef PPC_VMX

#define SLIDE_PPC slide_hash_vmx
#include "slide_ppc_tpl.h"

#endif /* PPC_VMX */
