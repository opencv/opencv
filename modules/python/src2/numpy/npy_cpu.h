/*
 * This set (target) cpu specific macros:
 *      - Possible values:
 *              NPY_CPU_X86
 *              NPY_CPU_AMD64
 *              NPY_CPU_PPC
 *              NPY_CPU_PPC64
 *              NPY_CPU_PPC64LE
 *              NPY_CPU_SPARC
 *              NPY_CPU_S390
 *              NPY_CPU_IA64
 *              NPY_CPU_HPPA
 *              NPY_CPU_ALPHA
 *              NPY_CPU_ARMEL
 *              NPY_CPU_ARMEB
 *              NPY_CPU_SH_LE
 *              NPY_CPU_SH_BE
 */
#ifndef _NPY_CPUARCH_H_
#define _NPY_CPUARCH_H_

#include "numpyconfig.h"
#include <string.h> /* for memcpy */

#if defined( __i386__ ) || defined(i386) || defined(_M_IX86)
    /*
     * __i386__ is defined by gcc and Intel compiler on Linux,
     * _M_IX86 by VS compiler,
     * i386 by Sun compilers on opensolaris at least
     */
    #define NPY_CPU_X86
#elif defined(__x86_64__) || defined(__amd64__) || defined(__x86_64) || defined(_M_AMD64)
    /*
     * both __x86_64__ and __amd64__ are defined by gcc
     * __x86_64 defined by sun compiler on opensolaris at least
     * _M_AMD64 defined by MS compiler
     */
    #define NPY_CPU_AMD64
#elif defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC)
    /*
     * __ppc__ is defined by gcc, I remember having seen __powerpc__ once,
     * but can't find it ATM
     * _ARCH_PPC is used by at least gcc on AIX
     */
    #define NPY_CPU_PPC
#elif defined(__ppc64le__)
    #define NPY_CPU_PPC64LE
#elif defined(__ppc64__)
    #define NPY_CPU_PPC64
#elif defined(__sparc__) || defined(__sparc)
    /* __sparc__ is defined by gcc and Forte (e.g. Sun) compilers */
    #define NPY_CPU_SPARC
#elif defined(__s390__)
    #define NPY_CPU_S390
#elif defined(__ia64)
    #define NPY_CPU_IA64
#elif defined(__hppa)
    #define NPY_CPU_HPPA
#elif defined(__alpha__)
    #define NPY_CPU_ALPHA
#elif defined(__arm__) && defined(__ARMEL__)
    #define NPY_CPU_ARMEL
#elif defined(__arm__) && defined(__ARMEB__)
    #define NPY_CPU_ARMEB
#elif defined(__sh__) && defined(__LITTLE_ENDIAN__)
    #define NPY_CPU_SH_LE
#elif defined(__sh__) && defined(__BIG_ENDIAN__)
    #define NPY_CPU_SH_BE
#elif defined(__MIPSEL__)
    #define NPY_CPU_MIPSEL
#elif defined(__MIPSEB__)
    #define NPY_CPU_MIPSEB
#elif defined(__or1k__)
    #define NPY_CPU_OR1K
#elif defined(__aarch64__)
    #define NPY_CPU_AARCH64
#elif defined(__mc68000__)
    #define NPY_CPU_M68K
#else
    #error Unknown CPU, please report this to numpy maintainers with \
    information about your platform (OS, CPU and compiler)
#endif

#define NPY_COPY_PYOBJECT_PTR(dst, src) memcpy(dst, src, sizeof(PyObject *))

#if (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
#define NPY_CPU_HAVE_UNALIGNED_ACCESS 1
#else
#define NPY_CPU_HAVE_UNALIGNED_ACCESS 0
#endif

#endif
