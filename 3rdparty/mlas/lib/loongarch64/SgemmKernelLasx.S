/*++

Copyright (C) 2023 Loongson Technology Corporation Limited. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelLasx.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses LASX instructions.

--*/

#include "asmmacro.h"
#include "SgemmKernelCommon.h"
#include "FgemmKernelLasxCommon.h"


        .text

//
// Generate the GEMM kernel.
//

FgemmKernelLasxFunction MlasGemmFloatKernelLasx

        .end
