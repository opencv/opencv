# detect-intrinsics.cmake -- Detect compiler intrinsics support
# Licensed under the Zlib license, see LICENSE.md for details

macro(check_acle_compiler_flag)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            check_c_compiler_flag("-march=armv8-a+crc" HAVE_MARCH_ARMV8_CRC)
            if(HAVE_MARCH_ARMV8_CRC)
                set(ACLEFLAG "-march=armv8-a+crc" CACHE INTERNAL "Compiler option to enable ACLE support")
            else()
                check_c_compiler_flag("-march=armv8-a+crc+simd" HAVE_MARCH_ARMV8_CRC_SIMD)
                if(HAVE_MARCH_ARMV8_CRC_SIMD)
                    set(ACLEFLAG "-march=armv8-a+crc+simd" CACHE INTERNAL "Compiler option to enable ACLE support")
                endif()
            endif()
        endif()
    endif()
    # Check whether compiler supports ARMv8 CRC intrinsics
    set(CMAKE_REQUIRED_FLAGS "${ACLEFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#if defined(_MSC_VER)
        #include <intrin.h>
        #else
        #include <arm_acle.h>
        #endif
        unsigned int f(unsigned int a, unsigned int b) {
            return __crc32w(a, b);
        }
        int main(void) { return 0; }"
        HAVE_ACLE_FLAG
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_armv6_compiler_flag)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            check_c_compiler_flag("-march=armv6" HAVE_MARCH_ARMV6)
            if(HAVE_MARCH_ARMV6)
                set(ARMV6FLAG "-march=armv6" CACHE INTERNAL "Compiler option to enable ARMv6 support")
            endif()
        endif()
    endif()
    # Check whether compiler supports ARMv6 inline asm
    set(CMAKE_REQUIRED_FLAGS "${ARMV6FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "unsigned int f(unsigned int a, unsigned int b) {
            unsigned int c;
            __asm__ __volatile__ ( \"uqsub16 %0, %1, %2\" : \"=r\" (c) : \"r\" (a), \"r\" (b) );
            return (int)c;
        }
        int main(void) { return f(1,2); }"
        HAVE_ARMV6_INLINE_ASM
    )
    # Check whether compiler supports ARMv6 intrinsics
    check_c_source_compiles(
        "#if defined(_MSC_VER)
        #include <intrin.h>
        #else
        #include <arm_acle.h>
        #endif
        unsigned int f(unsigned int a, unsigned int b) {
        #if defined(_MSC_VER)
            return _arm_uqsub16(a, b);
        #else
            return __uqsub16(a, b);
        #endif
        }
        int main(void) { return f(1,2); }"
        HAVE_ARMV6_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_avx512_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE)
                set(AVX512FLAG "-mavx512f -mavx512dq -mavx512bw -mavx512vl")
            else()
                set(AVX512FLAG "/arch:AVX512")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            # For CPUs that can benefit from AVX512, it seems GCC generates suboptimal
            # instruction scheduling unless you specify a reasonable -mtune= target
            set(AVX512FLAG "-mavx512f -mavx512dq -mavx512bw -mavx512vl")
            if(NOT MSVC)
                check_c_compiler_flag("-mtune=cascadelake" HAVE_CASCADE_LAKE)
                if(HAVE_CASCADE_LAKE)
                    set(AVX512FLAG "${AVX512FLAG} -mtune=cascadelake")
                else()
                    set(AVX512FLAG "${AVX512FLAG} -mtune=skylake-avx512")
                endif()
                unset(HAVE_CASCADE_LAKE)
            endif()
        elseif(MSVC)
            set(AVX512FLAG "/arch:AVX512")
        endif()
    endif()
    # Check whether compiler supports AVX512 intrinsics
    set(CMAKE_REQUIRED_FLAGS "${AVX512FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <immintrin.h>
        __m512i f(__m512i y) {
          __m512i x = _mm512_set1_epi8(2);
          return _mm512_sub_epi8(x, y);
        }
        int main(void) { return 0; }"
        HAVE_AVX512_INTRIN
    )
endmacro()

macro(check_avx512vnni_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
                set(AVX512VNNIFLAG "-mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni")
            else()
                set(AVX512VNNIFLAG "/arch:AVX512")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(AVX512VNNIFLAG "-mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni")
            if(NOT MSVC)
                check_c_compiler_flag("-mtune=cascadelake" HAVE_CASCADE_LAKE)
                if(HAVE_CASCADE_LAKE)
                    set(AVX512VNNIFLAG "${AVX512VNNIFLAG} -mtune=cascadelake")
                else()
                    set(AVX512VNNIFLAG "${AVX512VNNIFLAG} -mtune=skylake-avx512")
                endif()
                unset(HAVE_CASCADE_LAKE)
            endif()
        elseif(MSVC)
            set(AVX512VNNIFLAG "/arch:AVX512")
        endif()
    endif()
    # Check whether compiler supports AVX512vnni intrinsics
    set(CMAKE_REQUIRED_FLAGS "${AVX512VNNIFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <immintrin.h>
        __m512i f(__m512i x, __m512i y) {
            __m512i z = _mm512_setzero_epi32();
            return _mm512_dpbusd_epi32(z, x, y);
        }
        int main(void) { return 0; }"
        HAVE_AVX512VNNI_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_avx2_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE)
                set(AVX2FLAG "-mavx2")
            else()
                set(AVX2FLAG "/arch:AVX2")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(AVX2FLAG "-mavx2")
        elseif(MSVC)
            set(AVX2FLAG "/arch:AVX2")
        endif()
    endif()
    # Check whether compiler supports AVX2 intrinics
    set(CMAKE_REQUIRED_FLAGS "${AVX2FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <immintrin.h>
        __m256i f(__m256i x) {
            const __m256i y = _mm256_set1_epi16(1);
            return _mm256_subs_epu16(x, y);
        }
        int main(void) { return 0; }"
        HAVE_AVX2_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_neon_compiler_flag)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            if("${ARCH}" MATCHES "aarch64")
                set(NEONFLAG "-march=armv8-a+simd")
            else()
                set(NEONFLAG "-mfpu=neon")
            endif()
        endif()
    endif()
    # Check whether compiler supports NEON flag
    set(CMAKE_REQUIRED_FLAGS "${NEONFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#if defined(_M_ARM64) || defined(_M_ARM64EC)
        #  include <arm64_neon.h>
        #else
        #  include <arm_neon.h>
        #endif
        int main() { return 0; }"
        NEON_AVAILABLE FAIL_REGEX "not supported")
    # Check whether compiler native flag is enough for NEON support
    # Some GCC versions don't enable FPU (vector unit) when using -march=native
    if(NEON_AVAILABLE AND NATIVEFLAG AND (NOT "${ARCH}" MATCHES "aarch64"))
        check_c_source_compiles(
            "#include <arm_neon.h>
            uint8x16_t f(uint8x16_t x, uint8x16_t y) {
                return vaddq_u8(x, y);
            }
            int main(int argc, char* argv[]) {
                uint8x16_t a = vdupq_n_u8(argc);
                uint8x16_t b = vdupq_n_u8(argc);
                uint8x16_t result = f(a, b);
                return result[0];
            }"
            ARM_NEON_SUPPORT_NATIVE
        )
        if(NOT ARM_NEON_SUPPORT_NATIVE)
            set(CMAKE_REQUIRED_FLAGS "${NATIVEFLAG} -mfpu=neon ${ZNOLTOFLAG}")
            check_c_source_compiles(
                "#include <arm_neon.h>
                uint8x16_t f(uint8x16_t x, uint8x16_t y) {
                    return vaddq_u8(x, y);
                }
                int main(int argc, char* argv[]) {
                    uint8x16_t a = vdupq_n_u8(argc);
                    uint8x16_t b = vdupq_n_u8(argc);
                    uint8x16_t result = f(a, b);
                    return result[0];
                }"
                ARM_NEON_SUPPORT_NATIVE_MFPU
            )
            if(ARM_NEON_SUPPORT_NATIVE_MFPU)
                set(NEONFLAG "-mfpu=neon")
            else()
                # Remove local NEON_AVAILABLE variable and overwrite the cache
                unset(NEON_AVAILABLE)
                set(NEON_AVAILABLE "" CACHE INTERNAL "NEON support available" FORCE)
            endif()
        endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_neon_ld4_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            if("${ARCH}" MATCHES "aarch64")
                set(NEONFLAG "-march=armv8-a+simd")
            else()
                set(NEONFLAG "-mfpu=neon")
            endif()
        endif()
    endif()
    # Check whether compiler supports loading 4 neon vecs into a register range
    set(CMAKE_REQUIRED_FLAGS "${NEONFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
        #  include <arm64_neon.h>
        #else
        #  include <arm_neon.h>
        #endif
        int32x4x4_t f(int var[16]) { return vld1q_s32_x4(var); }
        int main(void) { return 0; }"
        NEON_HAS_LD4)
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_pclmulqdq_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
            set(PCLMULFLAG "-mpclmul")
        endif()
    endif()
    # Check whether compiler supports PCLMULQDQ intrinsics
    if(NOT (APPLE AND "${ARCH}" MATCHES "i386"))
        # The pclmul code currently crashes on Mac in 32bit mode. Avoid for now.
        set(CMAKE_REQUIRED_FLAGS "${PCLMULFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
        check_c_source_compiles(
            "#include <immintrin.h>
            #include <wmmintrin.h>
            __m128i f(__m128i a, __m128i b) { return _mm_clmulepi64_si128(a, b, 0x10); }
            int main(void) { return 0; }"
            HAVE_PCLMULQDQ_INTRIN
        )
        set(CMAKE_REQUIRED_FLAGS)
    else()
        set(HAVE_PCLMULQDQ_INTRIN OFF)
    endif()
endmacro()

macro(check_vpclmulqdq_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
            set(VPCLMULFLAG "-mvpclmulqdq -mavx512f")
        endif()
    endif()
    # Check whether compiler supports VPCLMULQDQ intrinsics
    if(NOT (APPLE AND "${ARCH}" MATCHES "i386"))
        set(CMAKE_REQUIRED_FLAGS "${VPCLMULFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
        check_c_source_compiles(
            "#include <immintrin.h>
            #include <wmmintrin.h>
            __m512i f(__m512i a) {
                __m512i b = _mm512_setzero_si512();
                return _mm512_clmulepi64_epi128(a, b, 0x10);
            }
            int main(void) { return 0; }"
            HAVE_VPCLMULQDQ_INTRIN
        )
        set(CMAKE_REQUIRED_FLAGS)
    else()
        set(HAVE_VPCLMULQDQ_INTRIN OFF)
    endif()
endmacro()

macro(check_ppc_intrinsics)
    # Check if compiler supports AltiVec
    set(CMAKE_REQUIRED_FLAGS "-maltivec ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <altivec.h>
        int main(void)
        {
            vector int a = vec_splats(0);
            vector int b = vec_splats(0);
            a = vec_add(a, b);
            return 0;
        }"
        HAVE_ALTIVEC
        )
    set(CMAKE_REQUIRED_FLAGS)

    if(HAVE_ALTIVEC)
        set(PPCFLAGS "-maltivec")
    endif()

    set(CMAKE_REQUIRED_FLAGS "-maltivec -mno-vsx ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <altivec.h>
        int main(void)
        {
            vector int a = vec_splats(0);
            vector int b = vec_splats(0);
            a = vec_add(a, b);
            return 0;
        }"
        HAVE_NOVSX
        )
    set(CMAKE_REQUIRED_FLAGS)

    if(HAVE_NOVSX)
        set(PPCFLAGS "${PPCFLAGS} -mno-vsx")
    endif()

    # Check if we have what we need for AltiVec optimizations
    set(CMAKE_REQUIRED_FLAGS "${PPCFLAGS} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <sys/auxv.h>
        #ifdef __FreeBSD__
        #include <machine/cpu.h>
        #endif
        int main() {
        #ifdef __FreeBSD__
            unsigned long hwcap;
            elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
            return (hwcap & PPC_FEATURE_HAS_ALTIVEC);
        #else
            return (getauxval(AT_HWCAP) & PPC_FEATURE_HAS_ALTIVEC);
        #endif
        }"
        HAVE_VMX
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_power8_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(POWER8FLAG "-mcpu=power8")
        endif()
    endif()
    # Check if we have what we need for POWER8 optimizations
    set(CMAKE_REQUIRED_FLAGS "${POWER8FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <sys/auxv.h>
        #ifdef __FreeBSD__
        #include <machine/cpu.h>
        #endif
        int main() {
        #ifdef __FreeBSD__
            unsigned long hwcap;
            elf_aux_info(AT_HWCAP2, &hwcap, sizeof(hwcap));
            return (hwcap & PPC_FEATURE2_ARCH_2_07);
        #else
            return (getauxval(AT_HWCAP2) & PPC_FEATURE2_ARCH_2_07);
        #endif
        }"
        HAVE_POWER8_INTRIN
    )
    if(NOT HAVE_POWER8_INTRIN AND HAVE_LINUX_AUXVEC_H)
        check_c_source_compiles(
            "#include <sys/auxv.h>
            #include <linux/auxvec.h>
            int main() {
                return (getauxval(AT_HWCAP2) & PPC_FEATURE2_ARCH_2_07);
            }"
            HAVE_POWER8_INTRIN2
        )
        if(HAVE_POWER8_INTRIN2)
            set(POWER8_NEED_AUXVEC_H 1)
            set(HAVE_POWER8_INTRIN ${HAVE_POWER8_INTRIN2} CACHE INTERNAL "Have POWER8 intrinsics" FORCE)
            unset(HAVE_POWER8_INTRIN2 CACHE)
        endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_rvv_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(RISCVFLAG "-march=rv64gcv")
        endif()
    endif()
    # Check whether compiler supports RVV
    set(CMAKE_REQUIRED_FLAGS "${RISCVFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <riscv_vector.h>
        int main() {
            return 0;
        }"
        HAVE_RVV_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_s390_intrinsics)
    check_c_source_compiles(
        "#include <sys/auxv.h>
        #ifndef HWCAP_S390_VXRS
        #define HWCAP_S390_VXRS HWCAP_S390_VX
        #endif
        int main() {
            return (getauxval(AT_HWCAP) & HWCAP_S390_VXRS);
        }"
        HAVE_S390_INTRIN
    )
endmacro()

macro(check_power9_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(POWER9FLAG "-mcpu=power9")
        endif()
    endif()
    # Check if we have what we need for POWER9 optimizations
    set(CMAKE_REQUIRED_FLAGS "${POWER9FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <sys/auxv.h>
        #ifdef __FreeBSD__
        #include <machine/cpu.h>
        #endif
        int main() {
        #ifdef __FreeBSD__
            unsigned long hwcap;
            elf_aux_info(AT_HWCAP2, &hwcap, sizeof(hwcap));
            return (hwcap & PPC_FEATURE2_ARCH_3_00);
        #else
            return (getauxval(AT_HWCAP2) & PPC_FEATURE2_ARCH_3_00);
        #endif
        }"
        HAVE_POWER9_INTRIN
    )
    if(NOT HAVE_POWER9_INTRIN AND HAVE_LINUX_AUXVEC_H)
        check_c_source_compiles(
            "#include <sys/auxv.h>
            #include <linux/auxvec.h>
            int main() {
                return (getauxval(AT_HWCAP2) & PPC_FEATURE2_ARCH_3_00);
            }"
            HAVE_POWER9_INTRIN2
        )
        if(HAVE_POWER9_INTRIN2)
            set(POWER9_NEED_AUXVEC_H 1)
            set(HAVE_POWER9_INTRIN ${HAVE_POWER9_INTRIN2} CACHE INTERNAL "Have POWER9 intrinsics" FORCE)
            unset(HAVE_POWER9_INTRIN2 CACHE)
        endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_sse2_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE)
                set(SSE2FLAG "-msse2")
            else()
                set(SSE2FLAG "/arch:SSE2")
            endif()
        elseif(MSVC)
            if(NOT "${ARCH}" MATCHES "x86_64")
                set(SSE2FLAG "/arch:SSE2")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(SSE2FLAG "-msse2")
        endif()
    endif()
    # Check whether compiler supports SSE2 intrinsics
    set(CMAKE_REQUIRED_FLAGS "${SSE2FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <immintrin.h>
        __m128i f(__m128i x, __m128i y) { return _mm_sad_epu8(x, y); }
        int main(void) { return 0; }"
        HAVE_SSE2_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_ssse3_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE)
                set(SSSE3FLAG "-mssse3")
            else()
                set(SSSE3FLAG "/arch:SSSE3")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(SSSE3FLAG "-mssse3")
        endif()
    endif()
    # Check whether compiler supports SSSE3 intrinsics
    set(CMAKE_REQUIRED_FLAGS "${SSSE3FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <immintrin.h>
        __m128i f(__m128i u) {
          __m128i v = _mm_set1_epi32(1);
          return _mm_hadd_epi32(u, v);
        }
        int main(void) { return 0; }"
        HAVE_SSSE3_INTRIN
    )
endmacro()

macro(check_sse42_intrinsics)
    if(NOT NATIVEFLAG)
        if(CMAKE_C_COMPILER_ID MATCHES "Intel")
            if(CMAKE_HOST_UNIX OR APPLE)
                set(SSE42FLAG "-msse4.2")
            else()
                set(SSE42FLAG "/arch:SSE4.2")
            endif()
        elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(SSE42FLAG "-msse4.2")
        endif()
    endif()
    # Check whether compiler supports SSE4.2 intrinsics
    set(CMAKE_REQUIRED_FLAGS "${SSE42FLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <nmmintrin.h>
        unsigned int f(unsigned int a, unsigned int b) { return _mm_crc32_u32(a, b); }
        int main(void) { return 0; }"
        HAVE_SSE42_INTRIN
    )
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_vgfma_intrinsics)
    if(NOT NATIVEFLAG)
        set(VGFMAFLAG "-march=z13")
        if(CMAKE_C_COMPILER_ID MATCHES "GNU")
            set(VGFMAFLAG "${VGFMAFLAG} -mzarch")
        endif()
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(VGFMAFLAG "${VGFMAFLAG} -fzvector")
        endif()
    endif()
    # Check whether compiler supports "VECTOR GALOIS FIELD MULTIPLY SUM AND ACCUMULATE" intrinsic
    set(CMAKE_REQUIRED_FLAGS "${VGFMAFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#include <vecintrin.h>
        int main(void) {
            unsigned long long a __attribute__((vector_size(16))) = { 0 };
            unsigned long long b __attribute__((vector_size(16))) = { 0 };
            unsigned char c __attribute__((vector_size(16))) = { 0 };
            c = vec_gfmsum_accum_128(a, b, c);
            return c[0];
        }"
        HAVE_VGFMA_INTRIN FAIL_REGEX "not supported")
    set(CMAKE_REQUIRED_FLAGS)
endmacro()

macro(check_xsave_intrinsics)
    if(NOT NATIVEFLAG AND NOT MSVC AND NOT CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(XSAVEFLAG "-mxsave")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${XSAVEFLAG} ${NATIVEFLAG} ${ZNOLTOFLAG}")
    check_c_source_compiles(
        "#ifdef _MSC_VER
        #  include <intrin.h>
        #elif __GNUC__ == 8 && __GNUC_MINOR__ > 1
        #  include <xsaveintrin.h>
        #else
        #  include <immintrin.h>
        #endif
        unsigned int f(unsigned int a) { return (int) _xgetbv(a); }
        int main(void) { return 0; }"
        HAVE_XSAVE_INTRIN FAIL_REGEX "not supported")
    set(CMAKE_REQUIRED_FLAGS)
endmacro()
