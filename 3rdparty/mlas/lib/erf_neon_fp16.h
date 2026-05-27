/*++

Copyright 2025 FUJITSU LIMITED
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

   erf_neon_fp16.h

Abstract:

    This module contains the procedure prototypes for the ERF NEON FP16 intrinsics.

--*/

#pragma once

#include <arm_neon.h>

#include "mlasi.h"
#include "fp16_common.h"
#include "softmax_kernel_neon.h"
#include <cstring>

void MlasNeonErfFP16Kernel(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N);
