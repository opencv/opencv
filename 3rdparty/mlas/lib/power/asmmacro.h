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
