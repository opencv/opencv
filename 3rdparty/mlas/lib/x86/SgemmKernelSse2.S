/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelSse2.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

    This implementation uses SSE2 instructions.

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

    This macro multiplies and accumulates for a Nx1 block of the output matrix.

Arguments:

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    Shuffle - Supplies the shuffle mask to extract the element from matrix A.

Implicit Arguments:

    ebx - Supplies the length in bytes of a row from matrix A.

    ecx - Supplies the address into the matrix A data.

    edx - Supplies the address into the matrix B data.

    xmm2 - Supplies up to four elements loaded from matrix A.

    xmm4-xmm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseBy4 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm4,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+32]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+48]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy3 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm5,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+32]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy2 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm6,xmm0
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()+16]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

        .endm

        .macro ComputeBlockSseBy1 VectorOffset, Shuffle

        pshufd  xmm3,xmm1,\Shuffle\()
        movaps  xmm0,XMMWORD PTR [edx+\VectorOffset\()]
        mulps   xmm0,xmm3
        addps   xmm7,xmm0

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

    xmm4-xmm7 - Supplies the block accumulators.

--*/

        .macro ComputeBlockSseLoop RowCount

        sub     edi,4
        jb      .LProcessRemainingBlocks\@

.LComputeBlockBy4Loop\@:
        movups  xmm1,XMMWORD PTR [ecx]
        ComputeBlockSseBy\RowCount\() 0, 0x00
        ComputeBlockSseBy\RowCount\() 16*4, 0x55
        sub     edx,-32*4                   # advance matrix B by 32 columns
        ComputeBlockSseBy\RowCount\() 0, 0xAA
        ComputeBlockSseBy\RowCount\() 16*4, 0xFF
        sub     edx,-32*4                   # advance matrix B by 32 columns
        add     ecx,4*4                     # advance matrix A by 4 columns
        sub     edi,4
        jae     .LComputeBlockBy4Loop\@

.LProcessRemainingBlocks\@:
        add     edi,4                       # correct for over-subtract above
        jz      .LOutputBlock\@

.LComputeBlockBy1Loop\@:
        movss   xmm1,DWORD PTR [ecx]
        ComputeBlockSseBy\RowCount\() 0, 0x00
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

    CountK - Supplies the number of columns from matrix A and the number of
        rows from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha - Supplies the scalar multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY MlasGemmFloatKernelSse

        push    ebp
        push    ebx
        push    esi
        push    edi
        mov     edx,.LSgemmKernelFrame_MatrixB[esp]
        mov     esi,.LSgemmKernelFrame_MatrixC[esp]
        mov     ebp,.LSgemmKernelFrame_CountN[esp]

//
// Process 1 row of the matrices.
//

        mov     eax,.LSgemmKernelFrame_CountK[esp]
        mov     ebx,.LSgemmKernelFrame_MatrixA[esp]
        cmp     ebp,12
        jbe     .LProcessRemainingCountN

.LProcessNextColumnLoop16x1:
        mov     edi,eax                     # reload CountK
        mov     ecx,ebx                     # reload matrix A
        xorps   xmm4,xmm4                   # clear block accumulators
        xorps   xmm5,xmm5
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        ComputeBlockSseLoop 4
        movss   xmm2,DWORD PTR .LSgemmKernelFrame_alpha[esp]
        shufps  xmm2,xmm2,0
        mulps   xmm4,xmm2                   # multiply by alpha
        mulps   xmm5,xmm2
        mulps   xmm6,xmm2
        mulps   xmm7,xmm2
        sub     ebp,16
        jb      .LOutputMasked16x1Block
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateOutput16x1
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        movups  xmm2,XMMWORD PTR [esi+32]
        movups  xmm3,XMMWORD PTR [esi+48]
        addps   xmm4,xmm0
        addps   xmm5,xmm1
        addps   xmm6,xmm2
        addps   xmm7,xmm3

.LSkipAccumulateOutput16x1:
        movups  XMMWORD PTR [esi],xmm4
        movups  XMMWORD PTR [esi+16],xmm5
        movups  XMMWORD PTR [esi+32],xmm6
        movups  XMMWORD PTR [esi+48],xmm7
        add     esi,16*4                    # advance matrix C by 16 columns
        cmp     ebp,12
        ja      .LProcessNextColumnLoop16x1
        test    ebp,ebp
        jnz     .LProcessRemainingCountN

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,1                       # return 1 row handled
        pop     edi
        pop     esi
        pop     ebx
        pop     ebp
        ret

//
// Process the remaining 1 to 12 columns of the matrices.
//

.LProcessRemainingCountN:
        mov     edi,eax                     # reload CountK
        mov     ecx,ebx                     # reload matrix A
        movss   xmm4,DWORD PTR .LSgemmKernelFrame_alpha[esp]
        shufps  xmm4,xmm4,0
        xorps   xmm5,xmm5                   # clear block accumulators
        xorps   xmm6,xmm6
        xorps   xmm7,xmm7
        cmp     ebp,4
        jbe     .LProcessRemainingCountN4OrLess
        cmp     ebp,8
        jbe     .LProcessRemainingCountN8OrLess

.LProcessRemainingCountN12OrLess:
        ComputeBlockSseLoop 3
        mulps   xmm5,xmm4                   # multiply by alpha
        mulps   xmm6,xmm4
        mulps   xmm7,xmm4
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateLeadingN12OrLess
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        addps   xmm5,xmm0
        addps   xmm6,xmm1

.LSkipAccumulateLeadingN12OrLess:
        movups  XMMWORD PTR [esi],xmm5
        movups  XMMWORD PTR [esi+16],xmm6
        add     esi,8*4                     # advance matrix C by 8 columns
        jmp     .LOutputTrailingBlock

.LProcessRemainingCountN8OrLess:
        ComputeBlockSseLoop 2
        mulps   xmm6,xmm4                   # multiply by alpha
        mulps   xmm7,xmm4
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateLeadingN8OrLess
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm6,xmm0

.LSkipAccumulateLeadingN8OrLess:
        movups  XMMWORD PTR [esi],xmm6
        add     esi,4*4                     # advance matrix C by 4 columns
        jmp     .LOutputTrailingBlock

.LProcessRemainingCountN4OrLess:
        ComputeBlockSseLoop 1
        mulps   xmm7,xmm4                   # multiply by alpha
        jmp     .LOutputTrailingBlock

.LOutputMasked16x1Block:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateLeading16x1Block
        movups  xmm0,XMMWORD PTR [esi]
        movups  xmm1,XMMWORD PTR [esi+16]
        movups  xmm2,XMMWORD PTR [esi+32]
        addps   xmm4,xmm0
        addps   xmm5,xmm1
        addps   xmm6,xmm2

.LSkipAccumulateLeading16x1Block:
        movups  XMMWORD PTR [esi],xmm4
        movups  XMMWORD PTR [esi+16],xmm5
        movups  XMMWORD PTR [esi+32],xmm6
        add     esi,12*4                    # advance matrix C by 12 columns

.LOutputTrailingBlock:
        test    ebp,3
        jz      .LOutputTrailingBlock4Elements
        test    ebp,2
        jz      .LOutputTrailingBlock1Element

.LOutputTrailingBlock2Elements:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateTrailingBlock2Elements
        movsd   xmm0,QWORD PTR [esi]
        addps   xmm7,xmm0

.LSkipAccumulateTrailingBlock2Elements:
        movsd   QWORD PTR [esi],xmm7
        test    ebp,1
        jz      .LExitKernel
        shufps  xmm7,xmm7,0xAA              # shuffle third float down
        add     esi,2*4                     # advance matrix C by 2 columns

.LOutputTrailingBlock1Element:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateTrailingBlock1Element
        movss   xmm0,DWORD PTR [esi]
        addss   xmm7,xmm0

.LSkipAccumulateTrailingBlock1Element:
        movss   DWORD PTR [esi],xmm7
        jmp     .LExitKernel

.LOutputTrailingBlock4Elements:
        cmp     BYTE PTR .LSgemmKernelFrame_ZeroMode[esp],0
        jnz     .LSkipAccumulateTrailingBlock4Elements
        movups  xmm0,XMMWORD PTR [esi]
        addps   xmm7,xmm0

.LSkipAccumulateTrailingBlock4Elements:
        movups  XMMWORD PTR [esi],xmm7
        jmp     .LExitKernel

        .end
