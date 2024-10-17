// archdetect.c -- Detect compiler architecture and raise preprocessor error
//                 containing a simple arch identifier.
// Copyright (C) 2019 Hans Kristian Rosbach
// Licensed under the Zlib license, see LICENSE.md for details

// x86_64
#if defined(__x86_64__) || defined(_M_X64)
    #error archfound x86_64

// x86
#elif defined(__i386) || defined(_M_IX86)
    #error archfound i686

// ARM
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
    #error archfound aarch64
#elif defined(__arm__) || defined(__arm) || defined(_M_ARM) || defined(__TARGET_ARCH_ARM)
    #if defined(__ARM64_ARCH_8__) || defined(__ARMv8__) || defined(__ARMv8_A__)
        #error archfound armv8
    #elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__)
        #error archfound armv7
    #elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6ZK__) || defined(__ARM_ARCH_6M__)
        #error archfound armv6
    #elif defined(__ARM_ARCH_5T__) || defined(__ARM_ARCH_5TE__) || defined(__ARM_ARCH_5TEJ__)
        #error archfound armv5
    #elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARCH_5E__)
        #error archfound armv4
    #elif defined(__ARM_ARCH_3__) || defined(__TARGET_ARCH_3M__)
        #error archfound armv3
    #elif defined(__ARM_ARCH_2__)
        #error archfound armv2
    #endif

// PowerPC
#elif defined(__powerpc__) || defined(_ppc__) || defined(__PPC__)
    #if defined(__64BIT__) || defined(__powerpc64__) || defined(__ppc64__)
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            #error archfound powerpc64le
        #else
            #error archfound powerpc64
        #endif
    #else
        #error archfound powerpc
    #endif

// --------------- Less common architectures alphabetically below ---------------

// ALPHA
#elif defined(__alpha__) || defined(__alpha)
    #error archfound alpha

// Blackfin
#elif defined(__BFIN__)
    #error archfound blackfin

// Itanium
#elif defined(__ia64) || defined(_M_IA64)
    #error archfound ia64

// MIPS
#elif defined(__mips__) || defined(__mips)
    #error archfound mips

// Motorola 68000-series
#elif defined(__m68k__)
    #error archfound m68k

// SuperH
#elif defined(__sh__)
    #error archfound sh

// SPARC
#elif defined(__sparc__) || defined(__sparc)
    #if defined(__sparcv9) || defined(__sparc_v9__)
        #error archfound sparc9
    #elif defined(__sparcv8) || defined(__sparc_v8__)
        #error archfound sparc8
    #endif

// SystemZ
#elif defined(__370__)
    #error archfound s370
#elif defined(__s390__)
    #error archfound s390
#elif defined(__s390x) || defined(__zarch__)
    #error archfound s390x

// PARISC
#elif defined(__hppa__)
    #error archfound parisc

// RS-6000
#elif defined(__THW_RS6000)
    #error archfound rs6000

// RISC-V
#elif defined(__riscv)
    #if __riscv_xlen == 64
        #error archfound riscv64
    #elif __riscv_xlen == 32
        #error archfound riscv32
    #endif

// LOONGARCH
#elif defined(__loongarch_lp64)
    #error archfound loongarch64

// Emscripten (WebAssembly)
#elif defined(__EMSCRIPTEN__)
    #error archfound wasm32

// return 'unrecognized' if we do not know what architecture this is
#else
    #error archfound unrecognized
#endif
