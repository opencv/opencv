/* x86_features.h -- check for CPU features
* Copyright (C) 2013 Intel Corporation Jim Kukunas
* For conditions of distribution and use, see copyright notice in zlib.h
*/

#ifndef X86_FEATURES_H_
#define X86_FEATURES_H_

struct x86_cpu_features {
    int has_avx2;
    int has_avx512;
    int has_avx512vnni;
    int has_sse2;
    int has_ssse3;
    int has_sse42;
    int has_pclmulqdq;
    int has_vpclmulqdq;
    int has_os_save_ymm;
    int has_os_save_zmm;
};

void Z_INTERNAL x86_check_features(struct x86_cpu_features *features);

#endif /* CPU_H_ */
