/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision matrix/matrix multiply operation (SGEMM).

--*/

//
// Define the single precision parameters.
//

#define    LFgemmElementShift 2
#define    LFgemmElementSize (1 << LFgemmElementShift)
#define    LFgemmYmmElementCount   (32/LFgemmElementSize)

#include "FgemmKernelCommon.h"

//
// Define the typed instructions for single precision.
//

FGEMM_TYPED_INSTRUCTION(xvfadd, xvfadd.s)
FGEMM_TYPED_INSTRUCTION(xvfmadd, xvfmadd.s)
FGEMM_TYPED_INSTRUCTION(xvldrepl, xvldrepl.w)
FGEMM_TYPED_INSTRUCTION(xvfmul, xvfmul.s)
