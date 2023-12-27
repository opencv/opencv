/* arm_features.h -- check for ARM features.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ARM_H_
#define ARM_H_

struct arm_cpu_features {
    int has_simd;
    int has_neon;
    int has_crc32;
};

void Z_INTERNAL arm_check_features(struct arm_cpu_features *features);

#endif /* ARM_H_ */
