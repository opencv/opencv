/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu_neon_fp16.h

Abstract:

    This module contains Gelu helper functions .

--*/

#pragma once

#include "fp16_common.h"
#include "erf_neon_fp16.h"

void
MLASCALL
MlasNeonGeluFP16Kernel(
    const MLAS_FP16* input,
    MLAS_FP16* output,
    MLAS_FP16* temp,
    size_t count,
    MLAS_GELU_ALGORITHM algo
);
