/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelFma3.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses AVX fused multiply/add instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"
#include "FgemmKernelFma3Common.h"

        .intel_syntax noprefix

        .text

//
// Generate the GEMM kernel.
//

FgemmKernelFma3Function MlasGemmFloatKernelFma3

        .end
