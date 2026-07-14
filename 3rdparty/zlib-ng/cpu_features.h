/* cpu_features.h -- CPU architecture feature check
 * Copyright (C) 2017 Hans Kristian Rosbach
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef CPU_FEATURES_H_
#define CPU_FEATURES_H_

#ifndef DISABLE_RUNTIME_CPU_DETECTION

#if defined(X86_FEATURES)
#  include "arch/x86/x86_features.h"
#elif defined(ARM_FEATURES)
#  include "arch/arm/arm_features.h"
#elif defined(PPC_FEATURES) || defined(POWER_FEATURES)
#  include "arch/power/power_features.h"
#elif defined(S390_FEATURES)
#  include "arch/s390/s390_features.h"
#elif defined(RISCV_FEATURES)
#  include "arch/riscv/riscv_features.h"
#endif

struct cpu_features {
#if defined(X86_FEATURES)
    struct x86_cpu_features x86;
#elif defined(ARM_FEATURES)
    struct arm_cpu_features arm;
#elif defined(PPC_FEATURES) || defined(POWER_FEATURES)
    struct power_cpu_features power;
#elif defined(S390_FEATURES)
    struct s390_cpu_features s390;
#elif defined(RISCV_FEATURES)
    struct riscv_cpu_features riscv;
#else
    char empty;
#endif
};

void cpu_check_features(struct cpu_features *features);

#endif

#endif
