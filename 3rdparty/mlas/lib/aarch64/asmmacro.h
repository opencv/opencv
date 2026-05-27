/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    asmmacro.h

Abstract:

    This module implements common macros for the assembly modules.

--*/

/*++

Macro Description:

    This macro emits the assembler directives to annotate a new function.

Arguments:

    FunctionName - Supplies the name of the function.

--*/

        .macro FUNCTION_ENTRY FunctionName

        .p2align 2
#if defined(__APPLE__)
        .globl  _\FunctionName\()
_\FunctionName\():
#else
        .globl  \FunctionName\()
        .type   \FunctionName\(),%function
\FunctionName\():
#endif

        .endm

/*++

Macro Description:

    This macro conditionally emits the statement if Count is greater than or
    equal to Value.

Arguments:

    Count - Supplies the variable used in the comparison.

    Value - Supplies the static used in the comparison.

    Statement - Supplies the statement to conditionally emit.

--*/

        .macro EmitIfCountGE Count1, Value1, Statement

.if (\Count1\() >= \Value1\())
        \Statement\()
.endif

        .endm

/*++

Macro Description:

    This macro conditionally emits the statement if Count1 is greater than or
    equal to Value1 and Count2 is greater than or equal to Value2.

Arguments:

    Count1 - Supplies the variable used in the comparison.

    Value1 - Supplies the static used in the comparison.

    Count2 - Supplies the variable used in the comparison.

    Value2 - Supplies the static used in the comparison.

    Statement - Supplies the statement to conditionally emit.

--*/

        .macro EmitIfCount2GE Count1, Value1, Count2, Value2, Statement

.if (\Count1\() >= \Value1\()) && (\Count2\() >= \Value2\())
        \Statement\()
.endif

        .endm
