/*++

Copyright (c) Microsoft Corporation. All rights reserved.

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

        .equ    .LFgemmElementShift, 2
        .equ    .LFgemmElementSize, 1 << .LFgemmElementShift

#include "FgemmKernelCommon.h"

//
// Define the typed instructions for single precision.
//

FGEMM_TYPED_INSTRUCTION(addpf, addps)
FGEMM_TYPED_INSTRUCTION(movsf, movss)
FGEMM_TYPED_INSTRUCTION(movupf, movups)

FGEMM_TYPED_INSTRUCTION(vaddpf, vaddps)
FGEMM_TYPED_INSTRUCTION(vbroadcastsf, vbroadcastss)
FGEMM_TYPED_INSTRUCTION(vfmadd213pf, vfmadd213ps)
FGEMM_TYPED_INSTRUCTION(vfmadd231pf, vfmadd231ps)
FGEMM_TYPED_INSTRUCTION(vmaskmovpf, vmaskmovps)
FGEMM_TYPED_INSTRUCTION(vmovapf, vmovaps)
FGEMM_TYPED_INSTRUCTION(vmovsf, vmovss)
FGEMM_TYPED_INSTRUCTION(vmovupf, vmovups)
FGEMM_TYPED_INSTRUCTION(vmulpf, vmulps)
FGEMM_TYPED_INSTRUCTION(vxorpf, vxorps)

        .macro vfmadd231pf_bcst DestReg, SrcReg, Address

        vfmadd231ps \DestReg\(), \SrcReg\(), \Address\(){1to16}

        .endm
