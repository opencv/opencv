/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelNeon.s

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "asmmacro.h"

        .text

//
// ClearRowAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearRowAccumulators Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        movi    v\Vec1Reg\().16b,#0
        movi    v\Vec2Reg\().16b,#0
.if \Columns\() > 8
        movi    v\Vec3Reg\().16b,#0
        movi    v\Vec4Reg\().16b,#0
.endif

        .endm

//
// ClearBlockAccumulators
//
// Generates the code to clear the accumulators for a single row of the output
// block.
//

        .macro  ClearBlockAccumulators Columns, Rows

        ClearRowAccumulators \Columns\(),16,17,18,19
.if \Rows\() >= 2
        ClearRowAccumulators \Columns\(),20,21,22,23
.endif
.if \Rows\() >= 4
        ClearRowAccumulators \Columns\(),24,25,26,27
        ClearRowAccumulators \Columns\(),28,29,30,31
.endif

        .endm

//
// LoadMatrixAElementsBy4
// LoadMatrixAElementsBy1
//
// Generates the code to load 1 or 4 elements from matrix A.
//

        .macro  LoadMatrixAElementsBy4 Rows

        ldr     q8,[x0],#16
.if \Rows\() >= 2
        ldr     q9,[x10],#16
.endif
.if \Rows\() >= 4
        ldr     q10,[x11],#16
        ldr     q11,[x12],#16
.endif

        .endm

        .macro  LoadMatrixAElementsBy1 Rows

        ldr     s8,[x0],#4
.if \Rows\() >= 2
        ldr     s9,[x10],#4
.endif
.if \Rows\() >= 4
        ldr     s10,[x11],#4
        ldr     s11,[x12],#4
.endif

        .endm

//
// MultiplyAccumulateRow
//
// Generates the code to multiply and accumulate a single row of the output
// block.
//

        .macro  MultiplyAccumulateRow Columns, MatrixAReg, Broadcast, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

        fmla    v\Vec1Reg\().4s,v4.4s,\MatrixAReg\().s[\Broadcast\()]
        fmla    v\Vec2Reg\().4s,v5.4s,\MatrixAReg\().s[\Broadcast\()]
.if \Columns\() > 8
        fmla    v\Vec3Reg\().4s,v6.4s,\MatrixAReg\().s[\Broadcast\()]
        fmla    v\Vec4Reg\().4s,v7.4s,\MatrixAReg\().s[\Broadcast\()]
.endif

        .endm

//
// MultiplyAccumulateBlock
//
// Generates the code to multiply and accumulate into the output block.
//

        .macro  MultiplyAccumulateBlock Columns, Rows, Broadcast

        MultiplyAccumulateRow \Columns\(),v8,\Broadcast\(),16,17,18,19
.if \Rows\() >= 2
        MultiplyAccumulateRow \Columns\(),v9,\Broadcast\(),20,21,22,23
.endif
.if \Rows\() >= 4
        MultiplyAccumulateRow \Columns\(),v10,\Broadcast\(),24,25,26,27
        MultiplyAccumulateRow \Columns\(),v11,\Broadcast\(),28,29,30,31
.endif

        .endm

//
// ComputeBlockLoop
//
// Generates the code to loop over K entries of the input matrices to produce
// the output block.
//

        .macro  ComputeBlockLoop Mode, Columns, Rows

        ClearBlockAccumulators \Columns\(),\Rows\()

.if \Rows\() >= 2
        add     x10,x0,x6,lsl #2            // compute matrix A plus 1 row
.endif
.if \Rows\() >= 4
        add     x11,x10,x6,lsl #2           // compute matrix A plus 2 rows
        add     x12,x11,x6,lsl #2           // compute matrix A plus 3 rows
.endif

        sub     x9,x3,#4                    // decrement block count to process
        tbnz    x9,#63,.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop:
        LoadMatrixAElementsBy4 \Rows\()
        ldp     q4,q5,[x1],#64*4
.if \Columns\() > 8
        ldp     q6,q7,[x1,#-56*4]
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),0
        ldp     q4,q5,[x1,#-48*4]
.if \Columns\() > 8
        ldp     q6,q7,[x1,#-40*4]
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),1
        ldp     q4,q5,[x1,#-32*4]
.if \Columns\() > 8
        ldp     q6,q7,[x1,#-24*4]
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),2
        ldp     q4,q5,[x1,#-16*4]
.if \Columns\() > 8
        ldp     q6,q7,[x1,#-8*4]
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),3
        sub     x9,x9,#4
        tbz     x9,#63,.L\Mode\().Compute\Columns\().x\Rows\().BlockBy4Loop

.L\Mode\().ProcessRemaining\Columns\().x\Rows\().Blocks:
        add     x9,x9,#4                    // correct for over-subtract above
        cbz     x9,.L\Mode\().Output\Columns\().x\Rows\().Block

.L\Mode\().Compute\Columns\().x\Rows\().BlockBy1Loop:
        LoadMatrixAElementsBy1 \Rows\()
        ldp     q4,q5,[x1],#16*4
.if \Columns\() > 8
        ldp     q6,q7,[x1,#-8*4]
.endif
        MultiplyAccumulateBlock \Columns\(),\Rows\(),0
        sub     x9,x9,#1
        cbnz    x9,.L\Mode\().Compute\Columns\().x\Rows\().BlockBy1Loop

.L\Mode\().Output\Columns\().x\Rows\().Block:

        .endm

//
// MultiplyAlphaRow
//
// Generates the code to multiply a single row of the output block by the alpha
// value.
//

        .macro  MultiplyAlphaRow Columns, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.if \Columns\() <= 4
        fmul    v\Vec1Reg\().4s,v\Vec1Reg\().4s,v0.s[0]
.elif \Columns\() <= 8
        fmul    v\Vec1Reg\().4s,v\Vec1Reg\().4s,v0.s[0]
        fmul    v\Vec2Reg\().4s,v\Vec2Reg\().4s,v0.s[0]
.elif \Columns\() <= 12
        fmul    v\Vec1Reg\().4s,v\Vec1Reg\().4s,v0.s[0]
        fmul    v\Vec2Reg\().4s,v\Vec2Reg\().4s,v0.s[0]
        fmul    v\Vec3Reg\().4s,v\Vec3Reg\().4s,v0.s[0]
.else
        fmul    v\Vec1Reg\().4s,v\Vec1Reg\().4s,v0.s[0]
        fmul    v\Vec2Reg\().4s,v\Vec2Reg\().4s,v0.s[0]
        fmul    v\Vec3Reg\().4s,v\Vec3Reg\().4s,v0.s[0]
        fmul    v\Vec4Reg\().4s,v\Vec4Reg\().4s,v0.s[0]
.endif

        .endm

//
// MultiplyAlphaBlock
//
// Generates the code to multiply the output block by the alpha value.
//

        .macro  MultiplyAlphaBlock Columns, Rows

        MultiplyAlphaRow \Columns\(),16,17,18,19
.if \Rows\() >= 2
        MultiplyAlphaRow \Columns\(),20,21,22,23
.endif
.if \Rows\() >= 4
        MultiplyAlphaRow \Columns\(),24,25,26,27
        MultiplyAlphaRow \Columns\(),28,29,30,31
.endif

        .endm

//
// OutputRow1Element
// OutputRow2Element
// OutputRow4Element
// OutputRow8Element
// OutputRow16Element
//
// Generates the code to store elements to the output block.
//

        .macro  OutputRow1Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
        ld1     {v4.s}[0],[\AddrReg\()]
        fmla    v4.2s,v\Vec1Reg\().2s,v0.s[0]
        st1     {v4.s}[0],[\AddrReg\()]         // post-increment not needed for last element
.else
        st1     {v\Vec1Reg\().s}[0],[\AddrReg\()]// post-increment not needed for last element
.endif

        .endm

        .macro  OutputRow2Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
        ld1     {v4.2s},[\AddrReg\()]
        fmla    v4.2s,v\Vec1Reg\().2s,v0.s[0]
        st1     {v4.2s},[\AddrReg\()],#2*4
.else
        st1     {v\Vec1Reg\().2s},[\AddrReg\()],#2*4
.endif
        dup     v\Vec1Reg\().4s,v\Vec1Reg\().s[2] // shift remaining elements down

        .endm

        .macro  OutputRow4Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
        ld1     {v4.4s},[\AddrReg\()]
        fmla    v4.4s,v\Vec1Reg\().4s,v0.s[0]
        st1     {v4.4s},[\AddrReg\()],#4*4
.else
        st1     {v\Vec1Reg\().4s},[\AddrReg\()],#4*4
.endif
        mov     v\Vec1Reg\().16b,v\Vec2Reg\().16b // shift remaining elements down

        .endm

        .macro  OutputRow8Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
        ldp     q4,q5,[\AddrReg\()]
        fmla    v4.4s,v\Vec1Reg\().4s,v0.s[0]
        fmla    v5.4s,v\Vec2Reg\().4s,v0.s[0]
        stp     q4,q5,[\AddrReg\()],#8*4
.else
        stp     q\Vec1Reg\(),q\Vec2Reg\(),[\AddrReg\()],#8*4
.endif
        mov     v\Vec1Reg\().16b,v\Vec3Reg\().16b // shift remaining elements down
        mov     v\Vec2Reg\().16b,v\Vec4Reg\().16b

        .endm

        .macro  OutputRow16Element Mode, AddrReg, Vec1Reg, Vec2Reg, Vec3Reg, Vec4Reg

.ifeqs "\Mode\()","Add"
        ldp     q4,q5,[\AddrReg\()]
        ldp     q6,q7,[\AddrReg\(),#8*4]
        fmla    v4.4s,v\Vec1Reg\().4s,v0.s[0]
        fmla    v5.4s,v\Vec2Reg\().4s,v0.s[0]
        fmla    v6.4s,v\Vec3Reg\().4s,v0.s[0]
        fmla    v7.4s,v\Vec4Reg\().4s,v0.s[0]
        stp     q4,q5,[\AddrReg\()],#16*4
        stp     q6,q7,[\AddrReg\(),#-8*4]
.else
        stp     q\Vec1Reg\(),q\Vec2Reg\(),[\AddrReg\()],#16*4
        stp     q\Vec3Reg\(),q\Vec4Reg\(),[\AddrReg\(),#-8*4]
.endif

        .endm

//
// OutputBlock
//
// Generates the code to store the output block.
//

        .macro  OutputBlock Mode, Columns, Rows

        OutputRow\Columns\()Element \Mode\(),x2,16,17,18,19
.if \Rows\() >= 2
        OutputRow\Columns\()Element \Mode\(),x13,20,21,22,23
.endif
.if \Rows\() >= 4
        OutputRow\Columns\()Element \Mode\(),x14,24,25,26,27
        OutputRow\Columns\()Element \Mode\(),x15,28,29,30,31
.endif

        .endm

//
// ProcessRows
//
// Generates the code to process a compute and store the output block for a
// fixed number of rows.
//

        .macro  ProcessRows Mode, Rows

        mov     x4,#\Rows\()                   // return number of rows handled
        cmp     x5,#8
        ble     .L\Mode\().ProcessRemainingCountN\Rows\()

.L\Mode\().ProcessNextColumnLoop16x\Rows\():
        ComputeBlockLoop \Mode\(),16,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 16,\Rows\()
.endif
        sub     x5,x5,#16
        tbnz    x5,#63,.L\Mode\().OutputMasked16x\Rows\().Block
        OutputBlock \Mode\(),16,\Rows\()
        mov     x0,x8                       // reload matrix A
        cmp     x5,#8
        bgt     .L\Mode\().ProcessNextColumnLoop16x\Rows\()
        cbz     x5,.L\Mode\().ExitKernel

.L\Mode\().ProcessRemainingCountN\Rows\():
        ComputeBlockLoop \Mode\(),8,\Rows\()
.ifeqs "\Mode\()","Zero"
        MultiplyAlphaBlock 8,\Rows\()
.endif

.L\Mode\().OutputMasked16x\Rows\().Block:
        tbz     x5,#3,.L\Mode\().OutputRemaining7x\Rows\().Block
        OutputBlock \Mode\(),8,\Rows\()

.L\Mode\().OutputRemaining7x\Rows\().Block:
        tbz     x5,#2,.L\Mode\().OutputRemaining3x\Rows\().Block
        OutputBlock \Mode\(),4,\Rows\()

.L\Mode\().OutputRemaining3x\Rows\().Block:
        tbz     x5,#1,.L\Mode\().OutputRemaining1x\Rows\().Block
        OutputBlock \Mode\(),2,\Rows\()

.L\Mode\().OutputRemaining1x\Rows\().Block:
        tbz     x5,#0,.L\Mode\().ExitKernel
        OutputBlock \Mode\(),1,\Rows\()

        .endm

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (x0) - Supplies the address of matrix A.

    B (x1) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (x2) - Supplies the address of matrix C.

    CountK (x3) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (x4) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (x5) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda (x6) - Supplies the first dimension of matrix A.

    ldc (x7) - Supplies the first dimension of matrix C.

    Alpha (s0) - Supplies the scalar multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/

        .macro  SgemmKernelNeonFunction Mode

        FUNCTION_ENTRY MlasSgemmKernel\Mode\()

        stp     d8,d9,[sp,#-32]!
        stp     d10,d11,[sp,#16]

        add     x13,x2,x7,lsl #2            // compute matrix C plus 1 row
        add     x14,x13,x7,lsl #2           // compute matrix C plus 2 rows
        add     x15,x14,x7,lsl #2           // compute matrix C plus 3 rows
        mov     x8,x0                       // save matrix A

//
// Process 4 rows of the matrices.
//

        cmp     x4,#4
        blt     .L\Mode\().ProcessCountMLessThan4
        ProcessRows \Mode\(),4

//
// Restore non-volatile registers and return.
//

.L\Mode\().ExitKernel:
        mov     x0,x4
        ldp     d10,d11,[sp,#16]
        ldp     d8,d9,[sp],#32
        ret

//
// Process 2 rows of the matrices.
//

.L\Mode\().ProcessCountMLessThan4:
        cmp     x4,#2
        blt     .L\Mode\().ProcessCountMLessThan2
        ProcessRows \Mode\(),2
        b       .L\Mode\().ExitKernel

//
// Process 1 row of the matrices.
//

.L\Mode\().ProcessCountMLessThan2:
        ProcessRows \Mode\(),1
        b       .L\Mode\().ExitKernel

        .endm

        SgemmKernelNeonFunction Zero
        SgemmKernelNeonFunction Add

        .end
