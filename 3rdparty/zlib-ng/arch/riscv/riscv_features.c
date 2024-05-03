#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/utsname.h>

#include "../../zbuild.h"
#include "riscv_features.h"

#define ISA_V_HWCAP (1 << ('v' - 'a'))

int Z_INTERNAL is_kernel_version_greater_or_equal_to_6_5() {
    struct utsname buffer;
    uname(&buffer);

    int major, minor;
    if (sscanf(buffer.release, "%d.%d", &major, &minor) != 2) {
        // Something bad with uname()
        return 0;
    }

    if (major > 6 || major == 6 && minor >= 5)
        return 1;
    return 0;
}

void Z_INTERNAL riscv_check_features_compile_time(struct riscv_cpu_features *features) {
#if defined(__riscv_v) && defined(__linux__)
    features->has_rvv = 1;
#else
    features->has_rvv = 0;
#endif
}

void Z_INTERNAL riscv_check_features_runtime(struct riscv_cpu_features *features) {
    unsigned long hw_cap = getauxval(AT_HWCAP);
    features->has_rvv = hw_cap & ISA_V_HWCAP;
}

void Z_INTERNAL riscv_check_features(struct riscv_cpu_features *features) {
    if (is_kernel_version_greater_or_equal_to_6_5())
        riscv_check_features_runtime(features);
    else
        riscv_check_features_compile_time(features);
}
