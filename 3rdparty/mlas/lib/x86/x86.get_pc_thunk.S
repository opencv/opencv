/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    x86.get_pc_thunk.S

Abstract:

    This module implements __x86.get_pc_thunk.* to avoid external dependency.

--*/

        .intel_syntax noprefix

/*++

Routine Description:

    The routine loads its return address -- which is the address of the
    instruction that immediately follows -- into the ebx register.

--*/

        .p2align 4
        .weak  __x86.get_pc_thunk.bx
        .type   __x86.get_pc_thunk.bx,@function
__x86.get_pc_thunk.bx:
        mov ebx, [esp]
        ret
