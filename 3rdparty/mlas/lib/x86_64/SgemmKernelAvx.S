/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelAvx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"
#include "FgemmKernelAvxCommon.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelAvxFunction MlasGemmFloatKernelAvx

        .end
