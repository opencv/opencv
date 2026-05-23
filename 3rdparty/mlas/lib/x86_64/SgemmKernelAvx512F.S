/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelAvx512F.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX512F instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"
#include "FgemmKernelAvx512FCommon.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelAvx512FFunction MlasGemmFloatKernelAvx512F

        .end
