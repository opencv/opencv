/* riscv_features.h -- check for riscv features.
 *
 * Copyright (C) 2023 SiFive, Inc. All rights reserved.
 * Contributed by Alex Chiang <alex.chiang@sifive.com>
 *
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef RISCV_FEATURES_H_
#define RISCV_FEATURES_H_

struct riscv_cpu_features {
    int has_rvv;
};

void Z_INTERNAL riscv_check_features(struct riscv_cpu_features *features);

#endif /* RISCV_FEATURES_H_ */
