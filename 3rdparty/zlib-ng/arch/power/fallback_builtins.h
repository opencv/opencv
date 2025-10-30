/* Helper functions to work around issues with clang builtins
 * Copyright (C) 2021 IBM Corporation
 *
 * Authors:
 *   Daniel Black <daniel@linux.vnet.ibm.com>
 *   Rogerio Alves <rogealve@br.ibm.com>
 *   Tulio Magno Quites Machado Filho <tuliom@linux.ibm.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef POWER_BUILTINS_H
#define POWER_BUILTINS_H

/*
 * These stubs fix clang incompatibilities with GCC builtins.
 */

#ifndef __builtin_crypto_vpmsumw
#define __builtin_crypto_vpmsumw __builtin_crypto_vpmsumb
#endif
#ifndef __builtin_crypto_vpmsumd
#define __builtin_crypto_vpmsumd __builtin_crypto_vpmsumb
#endif

static inline __vector unsigned long long __attribute__((overloadable))
vec_ld(int __a, const __vector unsigned long long* __b) {
    return (__vector unsigned long long)__builtin_altivec_lvx(__a, __b);
}

#endif
