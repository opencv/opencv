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

        .intel_syntax noprefix

//
// Stack frame layout for the SGEMM kernel.
//

        .equ    .LSgemmKernelFrame_SavedEdi, 0
        .equ    .LSgemmKernelFrame_SavedEsi, 4
        .equ    .LSgemmKernelFrame_SavedEbx, 8
        .equ    .LSgemmKernelFrame_SavedEbp, 12
        .equ    .LSgemmKernelFrame_ReturnAddress, 16
        .equ    .LSgemmKernelFrame_MatrixA, 20
        .equ    .LSgemmKernelFrame_MatrixB, 24
        .equ    .LSgemmKernelFrame_MatrixC, 28
        .equ    .LSgemmKernelFrame_CountK, 32
        .equ    .LSgemmKernelFrame_CountM, 36
        .equ    .LSgemmKernelFrame_CountN, 40
        .equ    .LSgemmKernelFrame_lda, 44
        .equ    .LSgemmKernelFrame_ldc, 48
        .equ    .LSgemmKernelFrame_alpha, 52
        .equ    .LSgemmKernelFrame_ZeroMode, 56

        .text

/*++

Macro Description:

    This macro multiplies and accumulates for a 16xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy16 RowCount, VectorOffset, BroadcastOffset

.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm1,ymm3,YMMWORD PTR [edx+\VectorOffset\()]
        vaddps  ymm4,ymm1,ymm4
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+\VectorOffset\()+32]
        vaddps  ymm5,ymm3,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [edx+\VectorOffset\()]
        vmovaps ymm1,YMMWORD PTR [edx+\VectorOffset\()+32]
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm4,ymm2,ymm4
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm5,ymm2,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+\BroadcastOffset\()]
        vmulps  ymm2,ymm3,ymm0
        vaddps  ymm6,ymm2,ymm6
        vmulps  ymm2,ymm3,ymm1
        vaddps  ymm7,ymm2,ymm7
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for a 8xN block of the output matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxBy8 RowCount, VectorOffset, BroadcastOffset

.if \RowCount\() == 1
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,YMMWORD PTR [edx+\VectorOffset\()]
        vaddps  ymm5,ymm3,ymm5
.else
        vmovaps ymm0,YMMWORD PTR [edx+\VectorOffset\()]
        vbroadcastss ymm3,DWORD PTR [ecx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm5,ymm3,ymm5
        vbroadcastss ymm3,DWORD PTR [ecx+ebx+\BroadcastOffset\()]
        vmulps  ymm3,ymm3,ymm0
        vaddps  ymm7,ymm3,ymm7
.endif

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    ebx - Supplies the number of bytes to the next row of matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    edi - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    ymm4-ymm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvxLoop ComputeBlock, RowCount

        sub     edi,4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0
        \ComputeBlock\() \RowCount\(), 16*4, 4
        sub     edx,-32*4                   # advance matrix B by 32 columns
        \ComputeBlock\() \RowCount\(), 0, 8
        \ComputeBlock\() \RowCount\(), 16*4, 12
        sub     edx,-32*4                   # advance matrix B by 32 columns
        add     ecx,4*4                     # advance matrix A by 4 columns
        sub     edi,4
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     edi,4                       # correct for over-subtract above
        jz      .LOutputBlock\@

.LComputeBlockBy1Loop\@:
        \ComputeBlock\() \RowCount\(), 0, 0
        add     edx,16*4                    # advance matrix B by 16 columns
        add     ecx,4                       # advance matrix A by 1 column
        dec     edi
        jne     .LComputeBlockBy1Loop\@

.LOutputBlock\@:

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha - Supplies the scalar multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY MlasGemmFloatKernelAvx

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,.LSgemmKernelFrame_MatrixB[esp]
        mov     esi,.LSgemmKernelFrame_MatrixC[esp]
        mov     ebp,.LSgemmKernelFrame_CountN[esp]

//
// Process 2 rows of the matrices.
//

        cmp     DWORD PTR .LSgemmKernelFrame_CountM[esp],2
        jb      .LProcessCountMLessThan2
        mov     BYTE PTR .LSgemmKernelFrame_CountM[esp],2
        mov     eax,.LSgemmKernelFrame_ldc[esp]
        mov     ebx,.LSgemmKernelFrame_lda[esp]
        shl     eax,2                       # convert ldc to bytes
        shl     ebx,2                       # convert lda to bytes
        cmp     ebp,8
        jbe     .LProcessRemainingCountN2

.LProcessNextColumnLoop16x2:
        mov     edi,.LSgemmKernelFrame_CountK[esp]
        mov     ecx,.LSgemmKernelFrame_MatrixA[esp]
        vxorps  xmm4,xmm4,xmm4              # clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        vxorps  xmm6,xmm6,xmm6
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop ComputeBlockAvxBy16, 2
        vbroadcastss ymm2,DWORD PTR .LSgemmKernelFrame_alpha[esp]
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        vmulps  ymm6,ymm6,ymm2
        vmulps  ymm7,ymm7,ymm2
        sub     ebp,16
        jb      .LOutputMasked16x2Block
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateOutput16x2
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax+32]

.LSkipAccumulateOutput16x2:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm6
        vmovups YMMWORD PTR [esi+eax+32],ymm7
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,8
        ja      .LProcessNextColumnLoop16x2
        test    ebp,ebp
        jz      .LExitKernel

.LProcessRemainingCountN2:
        mov     edi,.LSgemmKernelFrame_CountK[esp]
        mov     ecx,.LSgemmKernelFrame_MatrixA[esp]
        vxorps  xmm5,xmm5,xmm5              # clear block accumulators
        vxorps  xmm7,xmm7,xmm7
        ComputeBlockAvxLoop ComputeBlockAvxBy8, 2
        vbroadcastss ymm2,DWORD PTR .LSgemmKernelFrame_alpha[esp]
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        vmulps  ymm7,ymm7,ymm2
        cmp     ebp,8
        jb      .LOutputMasked8x2Block
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateOutput8x2
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]
        vaddps  ymm7,ymm7,YMMWORD PTR [esi+eax]

.LSkipAccumulateOutput8x2:
        vmovups YMMWORD PTR [esi],ymm5
        vmovups YMMWORD PTR [esi+eax],ymm7

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        movzx   eax,BYTE PTR .LSgemmKernelFrame_CountM[esp]
        vzeroupper
        pop     edi
        pop     esi
        pop     ebx
        pop     ebp
        ret

.LOutputMasked16x2Block:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateMasked16x2Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm6,ymm6,YMMWORD PTR [esi+eax]

.LSkipAccumulateMasked16x2Block:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+eax],ymm6
        add     esi,8*4                     # advance matrix C by 8 columns
        add     ebp,8                       # correct for over-subtract above

.LOutputMasked8x2Block:
        neg     ebp
        LoadGlobalOffsetTable bx
        mov     ebx,DWORD PTR C_UNDERSCORE(MlasMaskMoveTableAvx)@GOT[ebx]
        vmovdqu ymm0,YMMWORD PTR [ebx+ebp*4+8*4]
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateMasked8x2Block
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vmaskmovps ymm6,ymm0,YMMWORD PTR [esi+eax]
        vaddps  ymm5,ymm5,ymm4
        vaddps  ymm7,ymm7,ymm6

.LSkipAccumulateMasked8x2Block:
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        vmaskmovps YMMWORD PTR [esi+eax],ymm0,ymm7
        jmp     .LExitKernel

//
// Process 1 row of the matrices.
//

.LProcessCountMLessThan2:
        mov     BYTE PTR .LSgemmKernelFrame_CountM[esp],1
        mov     ebx,.LSgemmKernelFrame_MatrixA[esp]
        vbroadcastss ymm2,DWORD PTR .LSgemmKernelFrame_alpha[esp]
        cmp     ebp,8
        jbe     .LProcessRemainingCountN1

.LProcessNextColumnLoop16x1:
        mov     edi,.LSgemmKernelFrame_CountK[esp]
        mov     ecx,ebx                     # reload matrix A
        vxorps  xmm4,xmm4,xmm4              # clear block accumulators
        vxorps  xmm5,xmm5,xmm5
        ComputeBlockAvxLoop ComputeBlockAvxBy16, 1
        vmulps  ymm4,ymm4,ymm2              # multiply by alpha
        vmulps  ymm5,ymm5,ymm2
        sub     ebp,16
        jb      .LOutputMasked16x1Block
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulate16x1Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,YMMWORD PTR [esi+32]

.LSkipAccumulate16x1Block:
        vmovups YMMWORD PTR [esi],ymm4
        vmovups YMMWORD PTR [esi+32],ymm5
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,8
        ja      .LProcessNextColumnLoop16x1
        test    ebp,ebp
        jz      .LExitKernel

.LProcessRemainingCountN1:
        mov     edi,.LSgemmKernelFrame_CountK[esp]
        mov     ecx,ebx                     # reload matrix A
        vxorps  xmm5,xmm5,xmm5              # clear block accumulators
        ComputeBlockAvxLoop ComputeBlockAvxBy8, 1
        vmulps  ymm5,ymm5,ymm2              # multiply by alpha
        cmp     ebp,8
        jb      .LOutputMasked8x1Block
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulate8x1Block
        vaddps  ymm5,ymm5,YMMWORD PTR [esi]

.LSkipAccumulate8x1Block:
        vmovups YMMWORD PTR [esi],ymm5
        jmp     .LExitKernel

.LOutputMasked16x1Block:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateMasked16x1Block
        vaddps  ymm4,ymm4,YMMWORD PTR [esi]

.LSkipAccumulateMasked16x1Block:
        vmovups YMMWORD PTR [esi],ymm4
        add     esi,8*4                     # advance matrix C by 8 columns
        add     ebp,8                       # correct for over-subtract above

.LOutputMasked8x1Block:
        neg     ebp
        LoadGlobalOffsetTable bx
        mov     ebx,DWORD PTR C_UNDERSCORE(MlasMaskMoveTableAvx)@GOT[ebx]
        vmovdqu ymm0,YMMWORD PTR [ebx+ebp*4+8*4]
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateMasked8x1Block
        vmaskmovps ymm4,ymm0,YMMWORD PTR [esi]
        vaddps  ymm5,ymm5,ymm4

.LSkipAccumulateMasked8x1Block:
        vmaskmovps YMMWORD PTR [esi],ymm0,ymm5
        jmp     .LExitKernel

        .end
