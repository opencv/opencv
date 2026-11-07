/* s390_features.h -- check for s390 features.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef S390_FEATURES_H_
#define S390_FEATURES_H_

struct s390_cpu_features {
    int has_vx;
};

void Z_INTERNAL s390_check_features(struct s390_cpu_features *features);

#endif
