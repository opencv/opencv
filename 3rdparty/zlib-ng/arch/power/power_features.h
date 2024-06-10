/* power_features.h -- check for POWER CPU features
 * Copyright (C) 2020 Matheus Castanho <msc@linux.ibm.com>, IBM
 * Copyright (C) 2021 Mika T. Lindqvist <postmaster@raasu.org>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef POWER_H_
#define POWER_H_

struct power_cpu_features {
    int has_altivec;
    int has_arch_2_07;
    int has_arch_3_00;
};

void Z_INTERNAL power_check_features(struct power_cpu_features *features);

#endif /* POWER_H_ */
