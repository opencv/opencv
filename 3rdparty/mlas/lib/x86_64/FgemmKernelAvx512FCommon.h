/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    FgemmKernelAvx512FCommon.h

Abstract:

    This module implements the kernels for the floating point matrix/matrix
    multiply operation (SGEMM and DGEMM).

    This implementation uses AVX512F instructions.

--*/

/*++

Macro Description:

    This macro multiplies and accumulates for 2 ZMMWORDs by N rows of the output
    matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy2 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
        prefetcht0 [rsi+r12+\VectorOffset\()+\PrefetchOffset\()]
.endif
.if \RowCount\() == 1
        vbroadcastsf zmm3,[rdi+\BroadcastOffset\()]
        vfmadd231pf zmm4,zmm3,ZMMWORD PTR [rsi+\VectorOffset\()]
        vfmadd231pf zmm5,zmm3,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
.else
        vmovapf zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        vmovapf zmm1,ZMMWORD PTR [rsi+r12+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vbroadcastsf zmm3,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf zmm4,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf zmm5,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 2, "vbroadcastsf zmm3,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf zmm6,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf zmm7,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 3, "vbroadcastsf zmm3,[rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf zmm8,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf zmm9,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 4, "vbroadcastsf zmm3,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf zmm10,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf zmm11,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 5, "vbroadcastsf zmm3,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf zmm12,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf zmm13,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 6, "vbroadcastsf zmm3,[rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf zmm14,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf zmm15,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r13+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm16,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm17,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r13+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm18,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm19,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r13+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm20,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm21,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r14+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm22,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm23,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r14+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm24,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm25,zmm3,zmm1"
        EmitIfCountGE \RowCount\(), 12, "vbroadcastsf zmm3,[r14+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm26,zmm3,zmm0"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf zmm27,zmm3,zmm1"
.endif

        .endm

/*++

Macro Description:

    This macro multiplies and accumulates for 1 ZMMWORD by N rows of the output
    matrix.

Arguments:

    RowCount - Supplies the number of rows to process.

    VectorOffset - Supplies the byte offset from matrix B to fetch elements.

    BroadcastOffset - Supplies the byte offset from matrix A to fetch elements.

    PrefetchOffset - Optionally supplies the byte offset from matrix B to
        prefetch elements.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rbx - Supplies the address into the matrix A data plus 3 rows.

    r13 - Supplies the address into the matrix A data plus 6 rows.

    r14 - Supplies the address into the matrix A data plus 9 rows.

    rsi - Supplies the address into the matrix B data.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FBy1 RowCount, VectorOffset, BroadcastOffset, PrefetchOffset

.ifnb \PrefetchOffset\()
        prefetcht0 [rsi+\VectorOffset\()+\PrefetchOffset\()]
.endif
        vmovapf zmm0,ZMMWORD PTR [rsi+\VectorOffset\()]
        EmitIfCountGE \RowCount\(), 1, "vfmadd231pf_bcst zmm5,zmm0,[rdi+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd231pf_bcst zmm7,zmm0,[rdi+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd231pf_bcst zmm9,zmm0,[rdi+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd231pf_bcst zmm11,zmm0,[rbx+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd231pf_bcst zmm13,zmm0,[rbx+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd231pf_bcst zmm15,zmm0,[rbx+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm17,zmm0,[r13+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm19,zmm0,[r13+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm21,zmm0,[r13+r10*2+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm23,zmm0,[r14+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm25,zmm0,[r14+r10+\BroadcastOffset\()]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd231pf_bcst zmm27,zmm0,[r14+r10*2+\BroadcastOffset\()]"

        .endm

/*++

Macro Description:

    This macro generates code to execute the block compute macro multiple
    times and advancing the matrix A and matrix B data pointers.

Arguments:

    ComputeBlock - Supplies the macro to compute a single block.

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rdi - Supplies the address into the matrix A data.

    rsi - Supplies the address into the matrix B data.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    zmm4-zmm27 - Supplies the block accumulators.

--*/

        .macro ComputeBlockAvx512FLoop ComputeBlock, RowCount

.if \RowCount\() > 3
        lea     rbx,[r10*2+r10]
.if \RowCount\() == 12
        lea     r13,[rdi+rbx*2]             # compute matrix A plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix A plus 9 rows
.endif
        add     rbx,rdi                     # compute matrix A plus 3 rows
.endif
        ComputeBlockLoop \ComputeBlock\(), \RowCount\(), \RowCount\() > 3
.if \RowCount\() > 3
        lea     rbx,[rax*2+rax]
.if \RowCount\() == 12
        lea     r13,[rdx+rbx*2]             # compute matrix C plus 6 rows
        lea     r14,[r13+rbx]               # compute matrix C plus 9 rows
.endif
        add     rbx,rdx                     # compute matrix C plus 3 rows
.endif

        .endm

/*++

Macro Description:

    This macro generates code to compute matrix multiplication for a fixed set
    of rows.

Arguments:

    RowCount - Supplies the number of rows to process.

Implicit Arguments:

    rdi - Supplies the address of matrix A.

    rsi - Supplies the address of matrix B.

    r11 - Supplies the address of matrix A.

    r9 - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    rdx - Supplies the address of matrix C.

    rcx - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    r10 - Supplies the length in bytes of a row from matrix A.

    rax - Supplies the length in bytes of a row from matrix C.

    r15 - Stores the ZeroMode argument from the stack frame.

--*/

        .macro ProcessCountM RowCount

        cmp     r9,.LFgemmZmmElementCount
        jbe     .LProcessRemainingCountN\@

.LProcessNextColumnLoop2xN\@:
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm16,zmm4"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm17,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm18,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm20,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm22,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm24,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm26,zmm4"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm27,zmm5"
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy2, \RowCount\()
        add     rsi,r12                     # advance matrix B by 64*CountK bytes
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha2xNBlock\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf zmm4,zmm31,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf zmm6,zmm31,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf zmm8,zmm31,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf zmm10,zmm31,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf zmm12,zmm31,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf zmm14,zmm31,ZMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm16,zmm31,ZMMWORD PTR [r13]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm18,zmm31,ZMMWORD PTR [r13+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm20,zmm31,ZMMWORD PTR [r13+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm22,zmm31,ZMMWORD PTR [r14]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm24,zmm31,ZMMWORD PTR [r14+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm26,zmm31,ZMMWORD PTR [r14+rax*2]"
        jmp     .LStore2xNBlock\@

.LMultiplyAlpha2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf zmm4,zmm4,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulpf zmm6,zmm6,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulpf zmm8,zmm8,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulpf zmm10,zmm10,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulpf zmm12,zmm12,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulpf zmm14,zmm14,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm16,zmm16,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm18,zmm18,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm20,zmm20,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm22,zmm22,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm24,zmm24,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm26,zmm26,zmm31"

.LStore2xNBlock\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf ZMMWORD PTR [rdx],zmm4"
        EmitIfCountGE \RowCount\(), 2, "vmovupf ZMMWORD PTR [rdx+rax],zmm6"
        EmitIfCountGE \RowCount\(), 3, "vmovupf ZMMWORD PTR [rdx+rax*2],zmm8"
        EmitIfCountGE \RowCount\(), 4, "vmovupf ZMMWORD PTR [rbx],zmm10"
        EmitIfCountGE \RowCount\(), 5, "vmovupf ZMMWORD PTR [rbx+rax],zmm12"
        EmitIfCountGE \RowCount\(), 6, "vmovupf ZMMWORD PTR [rbx+rax*2],zmm14"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13],zmm16"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax],zmm18"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax*2],zmm20"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14],zmm22"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax],zmm24"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax*2],zmm26"
        add     rdx,64                      # advance matrix C by ZMMWORD
.if \RowCount\() > 3
        add     rbx,64                      # advance matrix C plus 3 rows by ZMMWORD
.if \RowCount\() == 12
        add     r13,64                      # advance matrix C plus 6 rows by ZMMWORD
        add     r14,64                      # advance matrix C plus 9 rows by ZMMWORD
.endif
.endif
        sub     r9,.LFgemmZmmElementCount

.LOutput1xNBlock\@:
        sub     r9,.LFgemmZmmElementCount
        jae     .LOutput1xNBlockWithMask\@
        lea     rcx,[r9+.LFgemmZmmElementCount]
                                            # correct for over-subtract above
        mov     ebp,1
        shl     ebp,cl
        dec     ebp
        kmovw   k1,ebp                      # update mask for remaining columns
        xor     r9,r9                       # no more columns remaining

.LOutput1xNBlockWithMask\@:
        test    r15b,r15b                   # ZeroMode?
        jnz     .LMultiplyAlpha1xNBlockWithMask\@
        EmitIfCountGE \RowCount\(), 1, "vfmadd213pf zmm5{k1},zmm31,ZMMWORD PTR [rdx]"
        EmitIfCountGE \RowCount\(), 2, "vfmadd213pf zmm7{k1},zmm31,ZMMWORD PTR [rdx+rax]"
        EmitIfCountGE \RowCount\(), 3, "vfmadd213pf zmm9{k1},zmm31,ZMMWORD PTR [rdx+rax*2]"
        EmitIfCountGE \RowCount\(), 4, "vfmadd213pf zmm11{k1},zmm31,ZMMWORD PTR [rbx]"
        EmitIfCountGE \RowCount\(), 5, "vfmadd213pf zmm13{k1},zmm31,ZMMWORD PTR [rbx+rax]"
        EmitIfCountGE \RowCount\(), 6, "vfmadd213pf zmm15{k1},zmm31,ZMMWORD PTR [rbx+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm17{k1},zmm31,ZMMWORD PTR [r13]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm19{k1},zmm31,ZMMWORD PTR [r13+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm21{k1},zmm31,ZMMWORD PTR [r13+rax*2]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm23{k1},zmm31,ZMMWORD PTR [r14]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm25{k1},zmm31,ZMMWORD PTR [r14+rax]"
        EmitIfCountGE \RowCount\(), 12, "vfmadd213pf zmm27{k1},zmm31,ZMMWORD PTR [r14+rax*2]"
        jmp     .LStore1xNBlockWithMask\@

.LMultiplyAlpha1xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmulpf zmm5,zmm5,zmm31"
        EmitIfCountGE \RowCount\(), 2, "vmulpf zmm7,zmm7,zmm31"
        EmitIfCountGE \RowCount\(), 3, "vmulpf zmm9,zmm9,zmm31"
        EmitIfCountGE \RowCount\(), 4, "vmulpf zmm11,zmm11,zmm31"
        EmitIfCountGE \RowCount\(), 5, "vmulpf zmm13,zmm13,zmm31"
        EmitIfCountGE \RowCount\(), 6, "vmulpf zmm15,zmm15,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm17,zmm17,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm19,zmm19,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm21,zmm21,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm23,zmm23,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm25,zmm25,zmm31"
        EmitIfCountGE \RowCount\(), 12, "vmulpf zmm27,zmm27,zmm31"

.LStore1xNBlockWithMask\@:
        EmitIfCountGE \RowCount\(), 1, "vmovupf ZMMWORD PTR [rdx]{k1},zmm5"
        EmitIfCountGE \RowCount\(), 2, "vmovupf ZMMWORD PTR [rdx+rax]{k1},zmm7"
        EmitIfCountGE \RowCount\(), 3, "vmovupf ZMMWORD PTR [rdx+rax*2]{k1},zmm9"
        EmitIfCountGE \RowCount\(), 4, "vmovupf ZMMWORD PTR [rbx]{k1},zmm11"
        EmitIfCountGE \RowCount\(), 5, "vmovupf ZMMWORD PTR [rbx+rax]{k1},zmm13"
        EmitIfCountGE \RowCount\(), 6, "vmovupf ZMMWORD PTR [rbx+rax*2]{k1},zmm15"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13]{k1},zmm17"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax]{k1},zmm19"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r13+rax*2]{k1},zmm21"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14]{k1},zmm23"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax]{k1},zmm25"
        EmitIfCountGE \RowCount\(), 12, "vmovupf ZMMWORD PTR [r14+rax*2]{k1},zmm27"
        add     rdx,64                      # advance matrix C by ZMMWORD
        mov     rdi,r11                     # reload matrix A
        vzeroall
        cmp     r9,.LFgemmZmmElementCount
        ja      .LProcessNextColumnLoop2xN\@
        test    r9,r9
        jz      .LExitKernel

.LProcessRemainingCountN\@:
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm17,zmm5"
                                            # clear upper block accumulators
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm19,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm21,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm23,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm25,zmm5"
        EmitIfCountGE \RowCount\(), 12, "vmovapf zmm27,zmm5"
        ComputeBlockAvx512FLoop ComputeBlockAvx512FBy1, \RowCount\()
        jmp     .LOutput1xNBlock\@

        .endm

/*++

Macro Description:

    This macro generates the inner kernel to compute matrix multiplication.

Arguments:

    FunctionName - Supplies the name for the generated function.

--*/

        .macro FgemmKernelAvx512FFunction FunctionName

/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A (rdi) - Supplies the address of matrix A.

    B (rsi) - Supplies the address of matrix B. The matrix data has been packed
        using MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C (rdx) - Supplies the address of matrix C.

    CountK (rcx) - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountM (r8) - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN (r9) - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    Alpha (xmm0) - Supplies the scalar alpha multiplier (see GEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/

        FUNCTION_ENTRY \FunctionName\()

        push    rbp
        push    rbx
        push    r15
        mov     .LFgemmKernelFrame_SavedR12[rsp],r12
        mov     .LFgemmKernelFrame_SavedR13[rsp],r13
        mov     .LFgemmKernelFrame_SavedR14[rsp],r14
        mov     r11,rdi
        mov     r10,.LFgemmKernelFrame_lda[rsp]
        shl     r10,.LFgemmElementShift     # convert lda to bytes
        mov     rax,.LFgemmKernelFrame_ldc[rsp]
        shl     rax,.LFgemmElementShift     # convert ldc to bytes
        mov     r12,rcx
        shl     r12,6                       # compute 64*CountK bytes
        mov     ebp,-1
        kmovw   k1,ebp                      # update mask to write all columns
        movzx   r15,BYTE PTR .LFgemmKernelFrame_ZeroMode[rsp]
        vbroadcastsf zmm31,xmm0
        vzeroall

//
// Process CountM rows of the matrices.
//

        cmp     r8,12
        jb      .LProcessCountMLessThan12
        mov     r8d,12                      # return 12 rows handled
        ProcessCountM 12

.LProcessCountMLessThan12:
        cmp     r8,5
        ja      .LProcessCountM6
        je      .LProcessCountM5
        cmp     r8,3
        ja      .LProcessCountM4
        je      .LProcessCountM3
        cmp     r8,1
        je      .LProcessCountM1

.LProcessCountM2:
        ProcessCountM 2

.LProcessCountM4:
        ProcessCountM 4

.LProcessCountM6:
        mov     r8d,6                       # return 6 rows handled
        ProcessCountM 6

//
// Restore non-volatile registers and return.
//

.LExitKernel:
        mov     eax,r8d
        mov     r12,.LFgemmKernelFrame_SavedR12[rsp]
        mov     r13,.LFgemmKernelFrame_SavedR13[rsp]
        mov     r14,.LFgemmKernelFrame_SavedR14[rsp]
        pop     r15
        pop     rbx
        pop     rbp
        ret

.LProcessCountM1:
        ProcessCountM 1

.LProcessCountM3:
        ProcessCountM 3

.LProcessCountM5:
        ProcessCountM 5

        .endm
