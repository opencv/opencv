/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    asmmacro.h

Abstract:

    This module implements common macros for the assembly modules.

--*/

#if defined(__APPLE__)
#define C_UNDERSCORE(symbol) _##symbol
#else
#define C_UNDERSCORE(symbol) symbol
#endif

/*++

Macro Description:

    This macro emits the assembler directives to annotate a new function.

Arguments:

    FunctionName - Supplies the name of the function.

--*/

        .macro FUNCTION_ENTRY FunctionName

        .p2align 4
#if defined(__APPLE__)
        .globl  _\FunctionName\()
_\FunctionName\():
#else
        .globl  \FunctionName\()
        .type   \FunctionName\(),@function
\FunctionName\():
#endif

        .endm

/*++

Macro Description:

    This macro emits the code to load the global offset table address into the
    supplied register.

Arguments:

    TargetReg - Specifies the target register.

--*/

        .macro  LoadGlobalOffsetTable, TargetReg

//
// The LLVM integrated assembler doesn't support the Intel syntax for OFFSET:
//
//      add     ebx,OFFSET _GLOBAL_OFFSET_TABLE_
//
// Workaround this by temporarily switching to AT&T syntax.
//

        .att_syntax

        calll   __x86.get_pc_thunk.\TargetReg\()
        addl    $_GLOBAL_OFFSET_TABLE_,%e\TargetReg\()

        .intel_syntax noprefix

        .endm
