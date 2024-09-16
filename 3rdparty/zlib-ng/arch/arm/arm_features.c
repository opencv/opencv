#include "zbuild.h"
#include "arm_features.h"

#if defined(__linux__) && defined(HAVE_SYS_AUXV_H)
#  include <sys/auxv.h>
#  ifdef ARM_ASM_HWCAP
#    include <asm/hwcap.h>
#  endif
#elif defined(__FreeBSD__) && defined(__aarch64__)
#  include <machine/armreg.h>
#  ifndef ID_AA64ISAR0_CRC32_VAL
#    define ID_AA64ISAR0_CRC32_VAL ID_AA64ISAR0_CRC32
#  endif
#elif defined(__OpenBSD__) && defined(__aarch64__)
#  include <machine/armreg.h>
#  include <machine/cpu.h>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#elif defined(__APPLE__)
#  if !defined(_DARWIN_C_SOURCE)
#    define _DARWIN_C_SOURCE /* enable types aliases (eg u_int) */
#  endif
#  include <sys/sysctl.h>
#elif defined(_WIN32)
#  include <windows.h>
#endif

static int arm_has_crc32() {
#if defined(__linux__) && defined(ARM_AUXV_HAS_CRC32)
#  ifdef HWCAP_CRC32
    return (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0 ? 1 : 0;
#  else
    return (getauxval(AT_HWCAP2) & HWCAP2_CRC32) != 0 ? 1 : 0;
#  endif
#elif defined(__FreeBSD__) && defined(__aarch64__)
    return getenv("QEMU_EMULATING") == NULL
      && ID_AA64ISAR0_CRC32_VAL(READ_SPECIALREG(id_aa64isar0_el1)) >= ID_AA64ISAR0_CRC32_BASE;
#elif defined(__OpenBSD__) && defined(__aarch64__)
    int hascrc32 = 0;
    int isar0_mib[] = { CTL_MACHDEP, CPU_ID_AA64ISAR0 };
    uint64_t isar0 = 0;
    size_t len = sizeof(isar0);
    if (sysctl(isar0_mib, 2, &isar0, &len, NULL, 0) != -1) {
      if (ID_AA64ISAR0_CRC32(isar0) >= ID_AA64ISAR0_CRC32_BASE)
          hascrc32 = 1;
    }
    return hascrc32;
#elif defined(__APPLE__)
    int hascrc32;
    size_t size = sizeof(hascrc32);
    return sysctlbyname("hw.optional.armv8_crc32", &hascrc32, &size, NULL, 0) == 0
      && hascrc32 == 1;
#elif defined(_WIN32)
    return IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE);
#elif defined(ARM_NOCHECK_ACLE)
    return 1;
#else
    return 0;
#endif
}

/* AArch64 has neon. */
#if !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)
static inline int arm_has_neon() {
#if defined(__linux__) && defined(ARM_AUXV_HAS_NEON)
#  ifdef HWCAP_ARM_NEON
    return (getauxval(AT_HWCAP) & HWCAP_ARM_NEON) != 0 ? 1 : 0;
#  else
    return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0 ? 1 : 0;
#  endif
#elif defined(__APPLE__)
    int hasneon;
    size_t size = sizeof(hasneon);
    return sysctlbyname("hw.optional.neon", &hasneon, &size, NULL, 0) == 0
      && hasneon == 1;
#elif defined(_M_ARM) && defined(WINAPI_FAMILY_PARTITION)
#  if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PHONE_APP)
    return 1; /* Always supported */
#  endif
#endif

#if defined(ARM_NOCHECK_NEON)
    return 1;
#else
    return 0;
#endif
}
#endif

/* AArch64 does not have ARMv6 SIMD. */
#if !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)
static inline int arm_has_simd() {
#if defined(__linux__) && defined(HAVE_SYS_AUXV_H)
    const char *platform = (const char *)getauxval(AT_PLATFORM);
    return strncmp(platform, "v6l", 3) == 0
        || strncmp(platform, "v7l", 3) == 0
        || strncmp(platform, "v8l", 3) == 0;
#elif defined(ARM_NOCHECK_SIMD)
    return 1;
#else
    return 0;
#endif
}
#endif

void Z_INTERNAL arm_check_features(struct arm_cpu_features *features) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
    features->has_simd = 0; /* never available */
    features->has_neon = 1; /* always available */
#else
    features->has_simd = arm_has_simd();
    features->has_neon = arm_has_neon();
#endif
    features->has_crc32 = arm_has_crc32();
}
