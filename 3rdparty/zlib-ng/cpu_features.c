/* cpu_features.c -- CPU architecture feature check
 * Copyright (C) 2017 Hans Kristian Rosbach
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "cpu_features.h"
#include <string.h>

Z_INTERNAL void cpu_check_features(struct cpu_features *features) {
    memset(features, 0, sizeof(struct cpu_features));
#if defined(X86_FEATURES)
    x86_check_features(&features->x86);
#elif defined(ARM_FEATURES)
    arm_check_features(&features->arm);
#elif defined(PPC_FEATURES) || defined(POWER_FEATURES)
    power_check_features(&features->power);
#elif defined(S390_FEATURES)
    s390_check_features(&features->s390);
#elif defined(RISCV_FEATURES)
    riscv_check_features(&features->riscv);
#endif
}
