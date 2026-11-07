/*
 * Compute the CRC32 using a parallelized folding approach with the PCLMULQDQ
 * instruction.
 *
 * A white paper describing this algorithm can be found at:
 *     doc/crc-pclmulqdq.pdf
 *
 * Copyright (C) 2013 Intel Corporation. All rights reserved.
 * Copyright (C) 2016 Marian Beermann (support for initial value)
 * Authors:
 *     Wajdi Feghali   <wajdi.k.feghali@intel.com>
 *     Jim Guilford    <james.guilford@intel.com>
 *     Vinodh Gopal    <vinodh.gopal@intel.com>
 *     Erdinc Ozturk   <erdinc.ozturk@intel.com>
 *     Jim Kukunas     <james.t.kukunas@linux.intel.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifdef X86_PCLMULQDQ_CRC

#define CRC32_FOLD_COPY  crc32_fold_pclmulqdq_copy
#define CRC32_FOLD       crc32_fold_pclmulqdq
#define CRC32_FOLD_RESET crc32_fold_pclmulqdq_reset
#define CRC32_FOLD_FINAL crc32_fold_pclmulqdq_final
#define CRC32            crc32_pclmulqdq

#include "crc32_pclmulqdq_tpl.h"

#endif
