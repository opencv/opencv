// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_HAL_MSA_MACROS_H
#define OPENCV_CORE_HAL_MSA_MACROS_H

#ifdef __mips_msa
#include "msa.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Define 64 bits vector types */
typedef signed char v8i8 __attribute__ ((vector_size(8), aligned(8)));
typedef unsigned char v8u8 __attribute__ ((vector_size(8), aligned(8)));
typedef short v4i16 __attribute__ ((vector_size(8), aligned(8)));
typedef unsigned short v4u16 __attribute__ ((vector_size(8), aligned(8)));
typedef int v2i32 __attribute__ ((vector_size(8), aligned(8)));
typedef unsigned int v2u32 __attribute__ ((vector_size(8), aligned(8)));
typedef long long v1i64 __attribute__ ((vector_size(8), aligned(8)));
typedef unsigned long long v1u64 __attribute__ ((vector_size(8), aligned(8)));
typedef float v2f32 __attribute__ ((vector_size(8), aligned(8)));
typedef double v1f64 __attribute__ ((vector_size(8), aligned(8)));


/* Load values from the given memory a 64-bit vector. */
#define msa_ld1_s8(__a)  (*((v8i8*)(__a)))
#define msa_ld1_s16(__a) (*((v4i16*)(__a)))
#define msa_ld1_s32(__a) (*((v2i32*)(__a)))
#define msa_ld1_s64(__a) (*((v1i64*)(__a)))
#define msa_ld1_u8(__a)  (*((v8u8*)(__a)))
#define msa_ld1_u16(__a) (*((v4u16*)(__a)))
#define msa_ld1_u32(__a) (*((v2u32*)(__a)))
#define msa_ld1_u64(__a) (*((v1u64*)(__a)))
#define msa_ld1_f32(__a) (*((v2f32*)(__a)))
#define msa_ld1_f64(__a) (*((v1f64*)(__a)))

/* Load values from the given memory address to a 128-bit vector */
#define msa_ld1q_s8(__a)  ((v16i8)__builtin_msa_ld_b(__a, 0))
#define msa_ld1q_s16(__a) ((v8i16)__builtin_msa_ld_h(__a, 0))
#define msa_ld1q_s32(__a) ((v4i32)__builtin_msa_ld_w(__a, 0))
#define msa_ld1q_s64(__a) ((v2i64)__builtin_msa_ld_d(__a, 0))
#define msa_ld1q_u8(__a)  ((v16u8)__builtin_msa_ld_b(__a, 0))
#define msa_ld1q_u16(__a) ((v8u16)__builtin_msa_ld_h(__a, 0))
#define msa_ld1q_u32(__a) ((v4u32)__builtin_msa_ld_w(__a, 0))
#define msa_ld1q_u64(__a) ((v2u64)__builtin_msa_ld_d(__a, 0))
#define msa_ld1q_f32(__a) ((v4f32)__builtin_msa_ld_w(__a, 0))
#define msa_ld1q_f64(__a) ((v2f64)__builtin_msa_ld_d(__a, 0))

/* Store 64bits vector elements values to the given memory address. */
#define msa_st1_s8(__a, __b)  (*((v8i8*)(__a)) = __b)
#define msa_st1_s16(__a, __b) (*((v4i16*)(__a)) = __b)
#define msa_st1_s32(__a, __b) (*((v2i32*)(__a)) = __b)
#define msa_st1_s64(__a, __b) (*((v1i64*)(__a)) = __b)
#define msa_st1_u8(__a, __b)  (*((v8u8*)(__a)) = __b)
#define msa_st1_u16(__a, __b) (*((v4u16*)(__a)) = __b)
#define msa_st1_u32(__a, __b) (*((v2u32*)(__a)) = __b)
#define msa_st1_u64(__a, __b) (*((v1u64*)(__a)) = __b)
#define msa_st1_f32(__a, __b) (*((v2f32*)(__a)) = __b)
#define msa_st1_f64(__a, __b) (*((v1f64*)(__a)) = __b)

/* Store the values of elements in the 128 bits vector __a to the given memory address __a. */
#define msa_st1q_s8(__a, __b)  (__builtin_msa_st_b((v16i8)(__b), __a, 0))
#define msa_st1q_s16(__a, __b) (__builtin_msa_st_h((v8i16)(__b), __a, 0))
#define msa_st1q_s32(__a, __b) (__builtin_msa_st_w((v4i32)(__b), __a, 0))
#define msa_st1q_s64(__a, __b) (__builtin_msa_st_d((v2i64)(__b), __a, 0))
#define msa_st1q_u8(__a, __b)  (__builtin_msa_st_b((v16i8)(__b), __a, 0))
#define msa_st1q_u16(__a, __b) (__builtin_msa_st_h((v8i16)(__b), __a, 0))
#define msa_st1q_u32(__a, __b) (__builtin_msa_st_w((v4i32)(__b), __a, 0))
#define msa_st1q_u64(__a, __b) (__builtin_msa_st_d((v2i64)(__b), __a, 0))
#define msa_st1q_f32(__a, __b) (__builtin_msa_st_w((v4i32)(__b), __a, 0))
#define msa_st1q_f64(__a, __b) (__builtin_msa_st_d((v2i64)(__b), __a, 0))

/* Store the value of the element with the index __c in vector __a to the given memory address __a. */
#define msa_st1_lane_s8(__a, __b, __c)   (*((int8_t*)(__a)) = __b[__c])
#define msa_st1_lane_s16(__a, __b, __c)  (*((int16_t*)(__a)) = __b[__c])
#define msa_st1_lane_s32(__a, __b, __c)  (*((int32_t*)(__a)) = __b[__c])
#define msa_st1_lane_s64(__a, __b, __c)  (*((int64_t*)(__a)) = __b[__c])
#define msa_st1_lane_u8(__a, __b, __c)   (*((uint8_t*)(__a)) = __b[__c])
#define msa_st1_lane_u16(__a, __b, __c)  (*((uint16_t*)(__a)) = __b[__c])
#define msa_st1_lane_u32(__a, __b, __c)  (*((uint32_t*)(__a)) = __b[__c])
#define msa_st1_lane_u64(__a, __b, __c)  (*((uint64_t*)(__a)) = __b[__c])
#define msa_st1_lane_f32(__a, __b, __c)  (*((float*)(__a)) = __b[__c])
#define msa_st1_lane_f64(__a, __b, __c)  (*((double*)(__a)) = __b[__c])
#define msa_st1q_lane_s8(__a, __b, __c)  (*((int8_t*)(__a)) = (int8_t)__builtin_msa_copy_s_b(__b, __c))
#define msa_st1q_lane_s16(__a, __b, __c) (*((int16_t*)(__a)) = (int16_t)__builtin_msa_copy_s_h(__b, __c))
#define msa_st1q_lane_s32(__a, __b, __c) (*((int32_t*)(__a)) = __builtin_msa_copy_s_w(__b, __c))
#define msa_st1q_lane_s64(__a, __b, __c) (*((int64_t*)(__a)) = __builtin_msa_copy_s_d(__b, __c))
#define msa_st1q_lane_u8(__a, __b, __c)  (*((uint8_t*)(__a)) = (uint8_t)__builtin_msa_copy_u_b((v16i8)(__b), __c))
#define msa_st1q_lane_u16(__a, __b, __c) (*((uint16_t*)(__a)) = (uint16_t)__builtin_msa_copy_u_h((v8i16)(__b), __c))
#define msa_st1q_lane_u32(__a, __b, __c) (*((uint32_t*)(__a)) = __builtin_msa_copy_u_w((v4i32)(__b), __c))
#define msa_st1q_lane_u64(__a, __b, __c) (*((uint64_t*)(__a)) = __builtin_msa_copy_u_d((v2i64)(__b), __c))
#define msa_st1q_lane_f32(__a, __b, __c) (*((float*)(__a)) = __b[__c])
#define msa_st1q_lane_f64(__a, __b, __c) (*((double*)(__a)) = __b[__c])

/* Duplicate elements for 64-bit doubleword vectors */
#define msa_dup_n_s8(__a)  ((v8i8)__builtin_msa_copy_s_d((v2i64)__builtin_msa_fill_b((int32_t)(__a)), 0))
#define msa_dup_n_s16(__a) ((v4i16)__builtin_msa_copy_s_d((v2i64)__builtin_msa_fill_h((int32_t)(__a)), 0))
#define msa_dup_n_s32(__a) ((v2i32){__a, __a})
#define msa_dup_n_s64(__a) ((v1i64){__a})
#define msa_dup_n_u8(__a)  ((v8u8)__builtin_msa_copy_u_d((v2i64)__builtin_msa_fill_b((int32_t)(__a)), 0))
#define msa_dup_n_u16(__a) ((v4u16)__builtin_msa_copy_u_d((v2i64)__builtin_msa_fill_h((int32_t)(__a)), 0))
#define msa_dup_n_u32(__a) ((v2u32){__a, __a})
#define msa_dup_n_u64(__a) ((v1u64){__a})
#define msa_dup_n_f32(__a) ((v2f32){__a, __a})
#define msa_dup_n_f64(__a) ((v1f64){__a})

/* Duplicate elements for 128-bit quadword vectors */
#define msa_dupq_n_s8(__a)  (__builtin_msa_fill_b((int32_t)(__a)))
#define msa_dupq_n_s16(__a) (__builtin_msa_fill_h((int32_t)(__a)))
#define msa_dupq_n_s32(__a) (__builtin_msa_fill_w((int32_t)(__a)))
#define msa_dupq_n_s64(__a) (__builtin_msa_fill_d((int64_t)(__a)))
#define msa_dupq_n_u8(__a)  ((v16u8)__builtin_msa_fill_b((int32_t)(__a)))
#define msa_dupq_n_u16(__a) ((v8u16)__builtin_msa_fill_h((int32_t)(__a)))
#define msa_dupq_n_u32(__a) ((v4u32)__builtin_msa_fill_w((int32_t)(__a)))
#define msa_dupq_n_u64(__a) ((v2u64)__builtin_msa_fill_d((int64_t)(__a)))
#define msa_dupq_n_f32(__a) ((v4f32){__a, __a, __a, __a})
#define msa_dupq_n_f64(__a) ((v2f64){__a, __a})
#define msa_dupq_lane_s8(__a, __b)  (__builtin_msa_splat_b(__a, __b))
#define msa_dupq_lane_s16(__a, __b) (__builtin_msa_splat_h(__a, __b))
#define msa_dupq_lane_s32(__a, __b) (__builtin_msa_splat_w(__a, __b))
#define msa_dupq_lane_s64(__a, __b) (__builtin_msa_splat_d(__a, __b))
#define msa_dupq_lane_u8(__a, __b)  ((v16u8)__builtin_msa_splat_b((v16i8)(__a), __b))
#define msa_dupq_lane_u16(__a, __b) ((v8u16)__builtin_msa_splat_h((v8i16)(__a), __b))
#define msa_dupq_lane_u32(__a, __b) ((v4u32)__builtin_msa_splat_w((v4i32)(__a), __b))
#define msa_dupq_lane_u64(__a, __b) ((v2u64)__builtin_msa_splat_d((v2i64)(__a), __b))

/* Create a 64 bits vector */
#define msa_create_s8(__a)  ((v8i8)((uint64_t)(__a)))
#define msa_create_s16(__a) ((v4i16)((uint64_t)(__a)))
#define msa_create_s32(__a) ((v2i32)((uint64_t)(__a)))
#define msa_create_s64(__a) ((v1i64)((uint64_t)(__a)))
#define msa_create_u8(__a)  ((v8u8)((uint64_t)(__a)))
#define msa_create_u16(__a) ((v4u16)((uint64_t)(__a)))
#define msa_create_u32(__a) ((v2u32)((uint64_t)(__a)))
#define msa_create_u64(__a) ((v1u64)((uint64_t)(__a)))
#define msa_create_f32(__a) ((v2f32)((uint64_t)(__a)))
#define msa_create_f64(__a) ((v1f64)((uint64_t)(__a)))

/* Sign extends or zero extends each element in a 64 bits vector to twice its original length, and places the results in a 128 bits vector. */
/*Transform v8i8 to v8i16*/
#define msa_movl_s8(__a) \
((v8i16){(__a)[0], (__a)[1], (__a)[2], (__a)[3], \
         (__a)[4], (__a)[5], (__a)[6], (__a)[7]})

/*Transform v8u8 to v8u16*/
#define msa_movl_u8(__a) \
((v8u16){(__a)[0], (__a)[1], (__a)[2], (__a)[3], \
         (__a)[4], (__a)[5], (__a)[6], (__a)[7]})

/*Transform v4i16 to v8i16*/
#define msa_movl_s16(__a) ((v4i32){(__a)[0], (__a)[1], (__a)[2], (__a)[3]})

/*Transform v2i32 to v4i32*/
#define msa_movl_s32(__a) ((v2i64){(__a)[0], (__a)[1]})

/*Transform v4u16 to v8u16*/
#define msa_movl_u16(__a) ((v4u32){(__a)[0], (__a)[1], (__a)[2], (__a)[3]})

/*Transform v2u32 to v4u32*/
#define msa_movl_u32(__a) ((v2u64){(__a)[0], (__a)[1]})

/* Copies the least significant half of each element of a 128 bits vector into the corresponding elements of a 64 bits vector. */
#define msa_movn_s16(__a) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)(__a)); \
  (v8i8)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_movn_s32(__a) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)(__a)); \
  (v4i16)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_movn_s64(__a) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)(__a)); \
  (v2i32)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_movn_u16(__a) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)(__a)); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_movn_u32(__a) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)(__a)); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_movn_u64(__a) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)(__a)); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

/* qmovn */
#define msa_qmovn_s16(__a) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_sat_s_h((v8i16)(__a), 7)); \
  (v8i8)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_qmovn_s32(__a) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_sat_s_w((v4i32)(__a), 15)); \
  (v4i16)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_qmovn_s64(__a) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_sat_s_d((v2i64)(__a), 31)); \
  (v2i32)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_qmovn_u16(__a) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_sat_u_h((v8u16)(__a), 7)); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_qmovn_u32(__a) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_sat_u_w((v4u32)(__a), 15)); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_qmovn_u64(__a) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_sat_u_d((v2u64)(__a), 31)); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

/* qmovun */
#define msa_qmovun_s16(__a) \
({ \
  v8i16 __d = __builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__a)); \
  v16i8 __e = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_sat_u_h((v8u16)__d, 7)); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qmovun_s32(__a) \
({ \
  v4i32 __d = __builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__a)); \
  v8i16 __e = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_sat_u_w((v4u32)__d, 15)); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qmovun_s64(__a) \
({ \
  v2i64 __d = __builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__a)); \
  v4i32 __e = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_sat_u_d((v2u64)__d, 31)); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

/* Right shift elements in a 128 bits vector by an immediate value, and places the results in a 64 bits vector. */
#define msa_shrn_n_s16(__a, __b) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_srai_h((v8i16)(__a), (int)(__b))); \
  (v8i8)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_shrn_n_s32(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_srai_w((v4i32)(__a), (int)(__b))); \
  (v4i16)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_shrn_n_s64(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_srai_d((v2i64)(__a), (int)(__b))); \
  (v2i32)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_shrn_n_u16(__a, __b) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_srli_h((v8i16)(__a), (int)(__b))); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_shrn_n_u32(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_srli_w((v4i32)(__a), (int)(__b))); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_shrn_n_u64(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_srli_d((v2i64)(__a), (int)(__b))); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

/* Right shift elements in a 128 bits vector by an immediate value, and places the results in a 64 bits vector. */
#define msa_rshrn_n_s16(__a, __b) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_srari_h((v8i16)(__a), (int)__b)); \
  (v8i8)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_rshrn_n_s32(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_srari_w((v4i32)(__a), (int)__b)); \
  (v4i16)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_rshrn_n_s64(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_srari_d((v2i64)(__a), (int)__b)); \
  (v2i32)__builtin_msa_copy_s_d((v2i64)__d, 0); \
})

#define msa_rshrn_n_u16(__a, __b) \
({ \
  v16i8 __d = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_srlri_h((v8i16)(__a), (int)__b)); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_rshrn_n_u32(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_srlri_w((v4i32)(__a), (int)__b)); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

#define msa_rshrn_n_u64(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_srlri_d((v2i64)(__a), (int)__b)); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__d, 0); \
})

/* Right shift elements in a 128 bits vector by an immediate value, saturate the results and them in a 64 bits vector. */
#define msa_qrshrn_n_s16(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_sat_s_h(__builtin_msa_srari_h((v8i16)(__a), (int)(__b)), 7); \
  v16i8 __e = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__d); \
  (v8i8)__builtin_msa_copy_s_d((v2i64)__e, 0); \
})

#define msa_qrshrn_n_s32(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_sat_s_w(__builtin_msa_srari_w((v4i32)(__a), (int)(__b)), 15); \
  v8i16 __e = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__d); \
  (v4i16)__builtin_msa_copy_s_d((v2i64)__e, 0); \
})

#define msa_qrshrn_n_s64(__a, __b) \
({ \
  v2i64 __d = __builtin_msa_sat_s_d(__builtin_msa_srari_d((v2i64)(__a), (int)(__b)), 31); \
  v4i32 __e = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__d); \
  (v2i32)__builtin_msa_copy_s_d((v2i64)__e, 0); \
})

#define msa_qrshrn_n_u16(__a, __b) \
({ \
  v8u16 __d = __builtin_msa_sat_u_h((v8u16)__builtin_msa_srlri_h((v8i16)(__a), (int)(__b)), 7); \
  v16i8 __e = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__d); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qrshrn_n_u32(__a, __b) \
({ \
  v4u32 __d = __builtin_msa_sat_u_w((v4u32)__builtin_msa_srlri_w((v4i32)(__a), (int)(__b)), 15); \
  v8i16 __e = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__d); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qrshrn_n_u64(__a, __b) \
({ \
  v2u64 __d = __builtin_msa_sat_u_d((v2u64)__builtin_msa_srlri_d((v2i64)(__a), (int)(__b)), 31); \
  v4i32 __e = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__d); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

/* Right shift elements in a 128 bits vector by an immediate value, saturate the results and them in a 64 bits vector.
   Input is signed and output is unsigned. */
#define msa_qrshrun_n_s16(__a, __b) \
({ \
  v8i16 __d = __builtin_msa_srlri_h(__builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__a)), (int)(__b)); \
  v16i8 __e = __builtin_msa_pckev_b(__builtin_msa_fill_b(0), (v16i8)__builtin_msa_sat_u_h((v8u16)__d, 7)); \
  (v8u8)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qrshrun_n_s32(__a, __b) \
({ \
  v4i32 __d = __builtin_msa_srlri_w(__builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__a)), (int)(__b)); \
  v8i16 __e = __builtin_msa_pckev_h(__builtin_msa_fill_h(0), (v8i16)__builtin_msa_sat_u_w((v4u32)__d, 15)); \
  (v4u16)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

#define msa_qrshrun_n_s64(__a, __b) \
({ \
  v2i64 __d = __builtin_msa_srlri_d(__builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__a)), (int)(__b)); \
  v4i32 __e = __builtin_msa_pckev_w(__builtin_msa_fill_w(0), (v4i32)__builtin_msa_sat_u_d((v2u64)__d, 31)); \
  (v2u32)__builtin_msa_copy_u_d((v2i64)__e, 0); \
})

/* pack */
#define msa_pack_s16(__a, __b) (__builtin_msa_pckev_b((v16i8)(__b), (v16i8)(__a)))
#define msa_pack_s32(__a, __b) (__builtin_msa_pckev_h((v8i16)(__b), (v8i16)(__a)))
#define msa_pack_s64(__a, __b) (__builtin_msa_pckev_w((v4i32)(__b), (v4i32)(__a)))
#define msa_pack_u16(__a, __b) ((v16u8)__builtin_msa_pckev_b((v16i8)(__b), (v16i8)(__a)))
#define msa_pack_u32(__a, __b) ((v8u16)__builtin_msa_pckev_h((v8i16)(__b), (v8i16)(__a)))
#define msa_pack_u64(__a, __b) ((v4u32)__builtin_msa_pckev_w((v4i32)(__b), (v4i32)(__a)))

/* qpack */
#define msa_qpack_s16(__a, __b) \
(__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_s_h((v8i16)(__b), 7), (v16i8)__builtin_msa_sat_s_h((v8i16)(__a), 7)))
#define msa_qpack_s32(__a, __b) \
(__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_s_w((v4i32)(__b), 15), (v8i16)__builtin_msa_sat_s_w((v4i32)(__a), 15)))
#define msa_qpack_s64(__a, __b) \
(__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_s_d((v2i64)(__b), 31), (v4i32)__builtin_msa_sat_s_d((v2i64)(__a), 31)))
#define msa_qpack_u16(__a, __b) \
((v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_u_h((v8u16)(__b), 7), (v16i8)__builtin_msa_sat_u_h((v8u16)(__a), 7)))
#define msa_qpack_u32(__a, __b) \
((v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_u_w((v4u32)(__b), 15), (v8i16)__builtin_msa_sat_u_w((v4u32)(__a), 15)))
#define msa_qpack_u64(__a, __b) \
((v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_u_d((v2u64)(__b), 31), (v4i32)__builtin_msa_sat_u_d((v2u64)(__a), 31)))

/* qpacku */
#define msa_qpacku_s16(__a, __b) \
((v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_u_h((v8u16)(__builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__b))), 7), \
                              (v16i8)__builtin_msa_sat_u_h((v8u16)(__builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__a))), 7)))
#define msa_qpacku_s32(__a, __b) \
((v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_u_w((v4u32)(__builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__b))), 15), \
                              (v8i16)__builtin_msa_sat_u_w((v4u32)(__builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__a))), 15)))
#define msa_qpacku_s64(__a, __b) \
((v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_u_d((v2u64)(__builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__b))), 31), \
                              (v4i32)__builtin_msa_sat_u_d((v2u64)(__builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__a))), 31)))

/* packr */
#define msa_packr_s16(__a, __b, __c) \
(__builtin_msa_pckev_b((v16i8)__builtin_msa_srai_h((v8i16)(__b), (int)(__c)), (v16i8)__builtin_msa_srai_h((v8i16)(__a), (int)(__c))))
#define msa_packr_s32(__a, __b, __c) \
(__builtin_msa_pckev_h((v8i16)__builtin_msa_srai_w((v4i32)(__b), (int)(__c)), (v8i16)__builtin_msa_srai_w((v4i32)(__a), (int)(__c))))
#define msa_packr_s64(__a, __b, __c) \
(__builtin_msa_pckev_w((v4i32)__builtin_msa_srai_d((v2i64)(__b), (int)(__c)), (v4i32)__builtin_msa_srai_d((v2i64)(__a), (int)(__c))))
#define msa_packr_u16(__a, __b, __c) \
((v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_srli_h((v8i16)(__b), (int)(__c)), (v16i8)__builtin_msa_srli_h((v8i16)(__a), (int)(__c))))
#define msa_packr_u32(__a, __b, __c) \
((v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_srli_w((v4i32)(__b), (int)(__c)), (v8i16)__builtin_msa_srli_w((v4i32)(__a), (int)(__c))))
#define msa_packr_u64(__a, __b, __c) \
((v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_srli_d((v2i64)(__b), (int)(__c)), (v4i32)__builtin_msa_srli_d((v2i64)(__a), (int)(__c))))

/* rpackr */
#define msa_rpackr_s16(__a, __b, __c) \
(__builtin_msa_pckev_b((v16i8)__builtin_msa_srari_h((v8i16)(__b), (int)(__c)), (v16i8)__builtin_msa_srari_h((v8i16)(__a), (int)(__c))))
#define msa_rpackr_s32(__a, __b, __c) \
(__builtin_msa_pckev_h((v8i16)__builtin_msa_srari_w((v4i32)(__b), (int)(__c)), (v8i16)__builtin_msa_srari_w((v4i32)(__a), (int)(__c))))
#define msa_rpackr_s64(__a, __b, __c) \
(__builtin_msa_pckev_w((v4i32)__builtin_msa_srari_d((v2i64)(__b), (int)(__c)), (v4i32)__builtin_msa_srari_d((v2i64)(__a), (int)(__c))))
#define msa_rpackr_u16(__a, __b, __c) \
((v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_srlri_h((v8i16)(__b), (int)(__c)), (v16i8)__builtin_msa_srlri_h((v8i16)(__a), (int)(__c))))
#define msa_rpackr_u32(__a, __b, __c) \
((v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_srlri_w((v4i32)(__b), (int)(__c)), (v8i16)__builtin_msa_srlri_w((v4i32)(__a), (int)(__c))))
#define msa_rpackr_u64(__a, __b, __c) \
((v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_srlri_d((v2i64)(__b), (int)(__c)), (v4i32)__builtin_msa_srlri_d((v2i64)(__a), (int)(__c))))

/* qrpackr */
#define msa_qrpackr_s16(__a, __b, __c) \
(__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_s_h(__builtin_msa_srari_h((v8i16)(__b), (int)(__c)), 7), \
                       (v16i8)__builtin_msa_sat_s_h(__builtin_msa_srari_h((v8i16)(__a), (int)(__c)), 7)))
#define msa_qrpackr_s32(__a, __b, __c) \
(__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_s_w(__builtin_msa_srari_w((v4i32)(__b), (int)(__c)), 15), \
                       (v8i16)__builtin_msa_sat_s_w(__builtin_msa_srari_w((v4i32)(__a), (int)(__c)), 15)))
#define msa_qrpackr_s64(__a, __b, __c) \
(__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_s_d(__builtin_msa_srari_d((v2i64)(__b), (int)(__c)), 31), \
                       (v4i32)__builtin_msa_sat_s_d(__builtin_msa_srari_d((v2i64)(__a), (int)(__c)), 31)))
#define msa_qrpackr_u16(__a, __b, __c) \
((v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_u_h((v8u16)__builtin_msa_srlri_h((v8i16)(__b), (int)(__c)), 7), \
                              (v16i8)__builtin_msa_sat_u_h((v8u16)__builtin_msa_srlri_h((v8i16)(__a), (int)(__c)), 7)))
#define msa_qrpackr_u32(__a, __b, __c) \
((v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_u_w((v4u32)__builtin_msa_srlri_w((v4i32)(__b), (int)(__c)), 15), \
                              (v8i16)__builtin_msa_sat_u_w((v4u32)__builtin_msa_srlri_w((v4i32)(__a), (int)(__c)), 15)))
#define msa_qrpackr_u64(__a, __b, __c) \
((v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_u_d((v2u64)__builtin_msa_srlri_d((v2i64)(__b), (int)(__c)), 31), \
                              (v4i32)__builtin_msa_sat_u_d((v2u64)__builtin_msa_srlri_d((v2i64)(__a), (int)(__c)), 31)))

/* qrpackru */
#define msa_qrpackru_s16(__a, __b, __c) \
({ \
  v8i16 __d = __builtin_msa_srlri_h(__builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__a)), (int)(__c)); \
  v8i16 __e = __builtin_msa_srlri_h(__builtin_msa_max_s_h(__builtin_msa_fill_h(0), (v8i16)(__b)), (int)(__c)); \
  (v16u8)__builtin_msa_pckev_b((v16i8)__builtin_msa_sat_u_h((v8u16)__e, 7), (v16i8)__builtin_msa_sat_u_h((v8u16)__d, 7)); \
})

#define msa_qrpackru_s32(__a, __b, __c) \
({ \
  v4i32 __d = __builtin_msa_srlri_w(__builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__a)), (int)(__c)); \
  v4i32 __e = __builtin_msa_srlri_w(__builtin_msa_max_s_w(__builtin_msa_fill_w(0), (v4i32)(__b)), (int)(__c)); \
  (v8u16)__builtin_msa_pckev_h((v8i16)__builtin_msa_sat_u_w((v4u32)__e, 15), (v8i16)__builtin_msa_sat_u_w((v4u32)__d, 15)); \
})

#define msa_qrpackru_s64(__a, __b, __c) \
({ \
  v2i64 __d = __builtin_msa_srlri_d(__builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__a)), (int)(__c)); \
  v2i64 __e = __builtin_msa_srlri_d(__builtin_msa_max_s_d(__builtin_msa_fill_d(0), (v2i64)(__b)), (int)(__c)); \
  (v4u32)__builtin_msa_pckev_w((v4i32)__builtin_msa_sat_u_d((v2u64)__e, 31), (v4i32)__builtin_msa_sat_u_d((v2u64)__d, 31)); \
})

/* Minimum values between corresponding elements in the two vectors are written to the returned vector. */
#define msa_minq_s8(__a, __b)  (__builtin_msa_min_s_b(__a, __b))
#define msa_minq_s16(__a, __b) (__builtin_msa_min_s_h(__a, __b))
#define msa_minq_s32(__a, __b) (__builtin_msa_min_s_w(__a, __b))
#define msa_minq_s64(__a, __b) (__builtin_msa_min_s_d(__a, __b))
#define msa_minq_u8(__a, __b)  ((v16u8)__builtin_msa_min_u_b(__a, __b))
#define msa_minq_u16(__a, __b) ((v8u16)__builtin_msa_min_u_h(__a, __b))
#define msa_minq_u32(__a, __b) ((v4u32)__builtin_msa_min_u_w(__a, __b))
#define msa_minq_u64(__a, __b) ((v2u64)__builtin_msa_min_u_d(__a, __b))
#define msa_minq_f32(__a, __b) (__builtin_msa_fmin_w(__a, __b))
#define msa_minq_f64(__a, __b) (__builtin_msa_fmin_d(__a, __b))

/* Maximum values between corresponding elements in the two vectors are written to the returned vector. */
#define msa_maxq_s8(__a, __b)  (__builtin_msa_max_s_b(__a, __b))
#define msa_maxq_s16(__a, __b) (__builtin_msa_max_s_h(__a, __b))
#define msa_maxq_s32(__a, __b) (__builtin_msa_max_s_w(__a, __b))
#define msa_maxq_s64(__a, __b) (__builtin_msa_max_s_d(__a, __b))
#define msa_maxq_u8(__a, __b)  ((v16u8)__builtin_msa_max_u_b(__a, __b))
#define msa_maxq_u16(__a, __b) ((v8u16)__builtin_msa_max_u_h(__a, __b))
#define msa_maxq_u32(__a, __b) ((v4u32)__builtin_msa_max_u_w(__a, __b))
#define msa_maxq_u64(__a, __b) ((v2u64)__builtin_msa_max_u_d(__a, __b))
#define msa_maxq_f32(__a, __b) (__builtin_msa_fmax_w(__a, __b))
#define msa_maxq_f64(__a, __b) (__builtin_msa_fmax_d(__a, __b))

/* Vector type reinterpretion */
#define MSA_TPV_REINTERPRET(_Tpv, Vec) ((_Tpv)(Vec))

/* Add the odd elements in vector __a with the even elements in vector __b to double width elements in the returned vector. */
/* v8i16 msa_hadd_s16 ((v16i8)__a, (v16i8)__b) */
#define msa_hadd_s16(__a, __b) (__builtin_msa_hadd_s_h((v16i8)(__a), (v16i8)(__b)))
/* v4i32 msa_hadd_s32 ((v8i16)__a, (v8i16)__b) */
#define msa_hadd_s32(__a, __b) (__builtin_msa_hadd_s_w((v8i16)(__a), (v8i16)(__b)))
/* v2i64 msa_hadd_s64 ((v4i32)__a, (v4i32)__b) */
#define msa_hadd_s64(__a, __b) (__builtin_msa_hadd_s_d((v4i32)(__a), (v4i32)(__b)))

/* Copy even elements in __a to the left half and even elements in __b to the right half and return the result vector. */
#define msa_pckev_s8(__a, __b)  (__builtin_msa_pckev_b((v16i8)(__a), (v16i8)(__b)))
#define msa_pckev_s16(__a, __b) (__builtin_msa_pckev_h((v8i16)(__a), (v8i16)(__b)))
#define msa_pckev_s32(__a, __b) (__builtin_msa_pckev_w((v4i32)(__a), (v4i32)(__b)))
#define msa_pckev_s64(__a, __b) (__builtin_msa_pckev_d((v2i64)(__a), (v2i64)(__b)))

/* Copy even elements in __a to the left half and even elements in __b to the right half and return the result vector. */
#define msa_pckod_s8(__a, __b)  (__builtin_msa_pckod_b((v16i8)(__a), (v16i8)(__b)))
#define msa_pckod_s16(__a, __b) (__builtin_msa_pckod_h((v8i16)(__a), (v8i16)(__b)))
#define msa_pckod_s32(__a, __b) (__builtin_msa_pckod_w((v4i32)(__a), (v4i32)(__b)))
#define msa_pckod_s64(__a, __b) (__builtin_msa_pckod_d((v2i64)(__a), (v2i64)(__b)))

#ifdef _MIPSEB
#define LANE_IMM0_1(x)  (0b1 - ((x) & 0b1))
#define LANE_IMM0_3(x)  (0b11 - ((x) & 0b11))
#define LANE_IMM0_7(x)  (0b111 - ((x) & 0b111))
#define LANE_IMM0_15(x) (0b1111 - ((x) & 0b1111))
#else
#define LANE_IMM0_1(x)  ((x) & 0b1)
#define LANE_IMM0_3(x)  ((x) & 0b11)
#define LANE_IMM0_7(x)  ((x) & 0b111)
#define LANE_IMM0_15(x) ((x) & 0b1111)
#endif

#define msa_get_lane_u8(__a, __b)        ((uint8_t)(__a)[LANE_IMM0_7(__b)])
#define msa_get_lane_s8(__a, __b)        ((int8_t)(__a)[LANE_IMM0_7(__b)])
#define msa_get_lane_u16(__a, __b)       ((uint16_t)(__a)[LANE_IMM0_3(__b)])
#define msa_get_lane_s16(__a, __b)       ((int16_t)(__a)[LANE_IMM0_3(__b)])
#define msa_get_lane_u32(__a, __b)       ((uint32_t)(__a)[LANE_IMM0_1(__b)])
#define msa_get_lane_s32(__a, __b)       ((int32_t)(__a)[LANE_IMM0_1(__b)])
#define msa_get_lane_f32(__a, __b)       ((float)(__a)[LANE_IMM0_3(__b)])
#define msa_get_lane_s64(__a, __b)       ((int64_t)(__a)[LANE_IMM0_1(__b)])
#define msa_get_lane_u64(__a, __b)       ((uint64_t)(__a)[LANE_IMM0_1(__b)])
#define msa_get_lane_f64(__a, __b)       ((double)(__a)[LANE_IMM0_1(__b)])
#define msa_getq_lane_u8(__a, imm0_15)   ((uint8_t)__builtin_msa_copy_u_b((v16i8)(__a), imm0_15))
#define msa_getq_lane_s8(__a, imm0_15)   ((int8_t)__builtin_msa_copy_s_b(__a, imm0_15))
#define msa_getq_lane_u16(__a, imm0_7)   ((uint16_t)__builtin_msa_copy_u_h((v8i16)(__a), imm0_7))
#define msa_getq_lane_s16(__a, imm0_7)   ((int16_t)__builtin_msa_copy_s_h(__a, imm0_7))
#define msa_getq_lane_u32(__a, imm0_3)   __builtin_msa_copy_u_w((v4i32)(__a), imm0_3)
#define msa_getq_lane_s32                __builtin_msa_copy_s_w
#define msa_getq_lane_f32(__a, __b)      ((float)(__a)[LANE_IMM0_3(__b)])
#define msa_getq_lane_f64(__a, __b)      ((double)(__a)[LANE_IMM0_1(__b)])
#if (__mips == 64)
#define msa_getq_lane_u64(__a, imm0_1)   __builtin_msa_copy_u_d((v2i64)(__a), imm0_1)
#define msa_getq_lane_s64                __builtin_msa_copy_s_d
#else
#define msa_getq_lane_u64(__a, imm0_1)   ((uint64_t)(__a)[LANE_IMM0_1(imm0_1)])
#define msa_getq_lane_s64(__a, imm0_1)   ((int64_t)(__a)[LANE_IMM0_1(imm0_1)])
#endif

/* combine */
#if (__mips == 64)
#define __COMBINE_64_64(__TYPE, a, b)    ((__TYPE)((v2u64){((v1u64)(a))[0], ((v1u64)(b))[0]}))
#else
#define __COMBINE_64_64(__TYPE, a, b)    ((__TYPE)((v4u32){((v2u32)(a))[0], ((v2u32)(a))[1],  \
                                                           ((v2u32)(b))[0], ((v2u32)(b))[1]}))
#endif

/* v16i8 msa_combine_s8 (v8i8 __a, v8i8 __b) */
#define msa_combine_s8(__a, __b)  __COMBINE_64_64(v16i8, __a, __b)

/* v8i16 msa_combine_s16(v4i16 __a, v4i16 __b) */
#define msa_combine_s16(__a, __b)  __COMBINE_64_64(v8i16, __a, __b)

/* v4i32 msa_combine_s32(v2i32 __a, v2i32 __b) */
#define msa_combine_s32(__a, __b)  __COMBINE_64_64(v4i32, __a, __b)

/* v2i64 msa_combine_s64(v1i64 __a, v1i64 __b) */
#define msa_combine_s64(__a, __b)  __COMBINE_64_64(v2i64, __a, __b)

/* v4f32 msa_combine_f32(v2f32 __a, v2f32 __b) */
#define msa_combine_f32(__a, __b)  __COMBINE_64_64(v4f32, __a, __b)

/* v16u8 msa_combine_u8(v8u8 __a, v8u8 __b) */
#define msa_combine_u8(__a, __b)  __COMBINE_64_64(v16u8, __a, __b)

/* v8u16 msa_combine_u16(v4u16 __a, v4u16 __b) */
#define msa_combine_u16(__a, __b)  __COMBINE_64_64(v8u16, __a, __b)

/* v4u32 msa_combine_u32(v2u32 __a, v2u32 __b) */
#define msa_combine_u32(__a, __b)  __COMBINE_64_64(v4u32, __a, __b)

/* v2u64 msa_combine_u64(v1u64 __a, v1u64 __b) */
#define msa_combine_u64(__a, __b)  __COMBINE_64_64(v2u64, __a, __b)

/* v2f64 msa_combine_f64(v1f64 __a, v1f64 __b) */
#define msa_combine_f64(__a, __b)  __COMBINE_64_64(v2f64, __a, __b)

/* get_low, get_high */
#if (__mips == 64)
#define __GET_LOW(__TYPE, a)   ((__TYPE)((v1u64)(__builtin_msa_copy_u_d((v2i64)(a), 0))))
#define __GET_HIGH(__TYPE, a)  ((__TYPE)((v1u64)(__builtin_msa_copy_u_d((v2i64)(a), 1))))
#else
#define __GET_LOW(__TYPE, a)   ((__TYPE)(((v2u64)(a))[0]))
#define __GET_HIGH(__TYPE, a)  ((__TYPE)(((v2u64)(a))[1]))
#endif

/* v8i8 msa_get_low_s8(v16i8 __a) */
#define msa_get_low_s8(__a)  __GET_LOW(v8i8, __a)

/* v4i16 msa_get_low_s16(v8i16 __a) */
#define msa_get_low_s16(__a)  __GET_LOW(v4i16, __a)

/* v2i32 msa_get_low_s32(v4i32 __a) */
#define msa_get_low_s32(__a)  __GET_LOW(v2i32, __a)

/* v1i64 msa_get_low_s64(v2i64 __a) */
#define msa_get_low_s64(__a)  __GET_LOW(v1i64, __a)

/* v8u8 msa_get_low_u8(v16u8 __a) */
#define msa_get_low_u8(__a)  __GET_LOW(v8u8, __a)

/* v4u16 msa_get_low_u16(v8u16 __a) */
#define msa_get_low_u16(__a)  __GET_LOW(v4u16, __a)

/* v2u32 msa_get_low_u32(v4u32 __a) */
#define msa_get_low_u32(__a)  __GET_LOW(v2u32, __a)

/* v1u64 msa_get_low_u64(v2u64 __a) */
#define msa_get_low_u64(__a)  __GET_LOW(v1u64, __a)

/* v2f32 msa_get_low_f32(v4f32 __a) */
#define msa_get_low_f32(__a)  __GET_LOW(v2f32, __a)

/* v1f64 msa_get_low_f64(v2f64 __a) */
#define msa_get_low_f64(__a)  __GET_LOW(v1f64, __a)

/* v8i8 msa_get_high_s8(v16i8 __a) */
#define msa_get_high_s8(__a)  __GET_HIGH(v8i8, __a)

/* v4i16 msa_get_high_s16(v8i16 __a) */
#define msa_get_high_s16(__a)  __GET_HIGH(v4i16, __a)

/* v2i32 msa_get_high_s32(v4i32 __a) */
#define msa_get_high_s32(__a)  __GET_HIGH(v2i32, __a)

/* v1i64 msa_get_high_s64(v2i64 __a) */
#define msa_get_high_s64(__a)  __GET_HIGH(v1i64, __a)

/* v8u8 msa_get_high_u8(v16u8 __a) */
#define msa_get_high_u8(__a)  __GET_HIGH(v8u8, __a)

/* v4u16 msa_get_high_u16(v8u16 __a) */
#define msa_get_high_u16(__a)  __GET_HIGH(v4u16, __a)

/* v2u32 msa_get_high_u32(v4u32 __a) */
#define msa_get_high_u32(__a)  __GET_HIGH(v2u32, __a)

/* v1u64 msa_get_high_u64(v2u64 __a) */
#define msa_get_high_u64(__a)  __GET_HIGH(v1u64, __a)

/* v2f32 msa_get_high_f32(v4f32 __a) */
#define msa_get_high_f32(__a)  __GET_HIGH(v2f32, __a)

/* v1f64 msa_get_high_f64(v2f64 __a) */
#define msa_get_high_f64(__a)  __GET_HIGH(v1f64, __a)

/* ri = ai * b[lane] */
/* v4f32 msa_mulq_lane_f32(v4f32 __a, v4f32 __b, const int __lane) */
#define msa_mulq_lane_f32(__a, __b, __lane)  ((__a) * msa_getq_lane_f32(__b, __lane))

/* ri = ai + bi * c[lane] */
/* v4f32 msa_mlaq_lane_f32(v4f32 __a, v4f32 __b, v4f32 __c, const int __lane) */
#define msa_mlaq_lane_f32(__a, __b, __c, __lane)  ((__a) + ((__b) * msa_getq_lane_f32(__c, __lane)))

/* uint16_t msa_sum_u16(v8u16 __a)*/
#define msa_sum_u16(__a)                         \
({                                               \
  v4u32 _b;                                      \
  v2u64 _c;                                      \
  _b = __builtin_msa_hadd_u_w(__a, __a);         \
  _c = __builtin_msa_hadd_u_d(_b, _b);           \
  (uint16_t)(_c[0] + _c[1]);                     \
})

/* int16_t msa_sum_s16(v8i16 __a) */
#define msa_sum_s16(__a)                        \
({                                              \
  v4i32 _b;                                     \
  v2i64 _c;                                     \
  _b = __builtin_msa_hadd_s_w(__a, __a);        \
  _c = __builtin_msa_hadd_s_d(_b, _b);          \
  (int32_t)(_c[0] + _c[1]);                     \
})


/* uint32_t msa_sum_u32(v4u32 __a)*/
#define msa_sum_u32(__a)                       \
({                                             \
  v2u64 _b;                                    \
  _b = __builtin_msa_hadd_u_d(__a, __a);       \
  (uint32_t)(_b[0] + _b[1]);                   \
})

/* int32_t  msa_sum_s32(v4i32 __a)*/
#define msa_sum_s32(__a)                       \
({                                             \
  v2i64 _b;                                    \
  _b = __builtin_msa_hadd_s_d(__a, __a);       \
  (int64_t)(_b[0] + _b[1]);                    \
})

/* uint8_t msa_sum_u8(v16u8 __a)*/
#define msa_sum_u8(__a)                        \
({                                             \
  v8u16 _b16;                                    \
  v4u32 _c32;                                    \
  _b16 = __builtin_msa_hadd_u_h(__a, __a);       \
  _c32 = __builtin_msa_hadd_u_w(_b16, _b16);         \
  (uint8_t)msa_sum_u32(_c32);                    \
})

/* int8_t msa_sum_s8(v16s8 __a)*/
#define msa_sum_s8(__a)                        \
({                                             \
  v8i16 _b16;                                    \
  v4i32 _c32;                                    \
  _b16 = __builtin_msa_hadd_s_h(__a, __a);       \
  _c32 = __builtin_msa_hadd_s_w(_b16, _b16);         \
  (int16_t)msa_sum_s32(_c32);                     \
})

/* float msa_sum_f32(v4f32 __a)*/
#define msa_sum_f32(__a)  ((__a)[0] + (__a)[1] + (__a)[2] + (__a)[3])

/* v8u16 msa_paddlq_u8(v16u8 __a) */
#define msa_paddlq_u8(__a)  (__builtin_msa_hadd_u_h(__a, __a))

/* v8i16 msa_paddlq_s8(v16i8 __a) */
#define msa_paddlq_s8(__a)  (__builtin_msa_hadd_s_h(__a, __a))

/* v4u32 msa_paddlq_u16 (v8u16 __a)*/
#define msa_paddlq_u16(__a)  (__builtin_msa_hadd_u_w(__a, __a))

/* v4i32 msa_paddlq_s16 (v8i16 __a)*/
#define msa_paddlq_s16(__a)  (__builtin_msa_hadd_s_w(__a, __a))

/* v2u64 msa_paddlq_u32(v4u32 __a) */
#define msa_paddlq_u32(__a)  (__builtin_msa_hadd_u_d(__a, __a))

/* v2i64 msa_paddlq_s32(v4i32 __a) */
#define msa_paddlq_s32(__a)  (__builtin_msa_hadd_s_d(__a, __a))

#define V8U8_2_V8U16(x)   {(uint16_t)x[0], (uint16_t)x[1], (uint16_t)x[2], (uint16_t)x[3], \
                           (uint16_t)x[4], (uint16_t)x[5], (uint16_t)x[6], (uint16_t)x[7]}
#define V8U8_2_V8I16(x)   {(int16_t)x[0], (int16_t)x[1], (int16_t)x[2], (int16_t)x[3], \
                           (int16_t)x[4], (int16_t)x[5], (int16_t)x[6], (int16_t)x[7]}
#define V8I8_2_V8I16(x)   {(int16_t)x[0], (int16_t)x[1], (int16_t)x[2], (int16_t)x[3], \
                           (int16_t)x[4], (int16_t)x[5], (int16_t)x[6], (int16_t)x[7]}
#define V4U16_2_V4U32(x)  {(uint32_t)x[0], (uint32_t)x[1], (uint32_t)x[2], (uint32_t)x[3]}
#define V4U16_2_V4I32(x)  {(int32_t)x[0], (int32_t)x[1], (int32_t)x[2], (int32_t)x[3]}
#define V4I16_2_V4I32(x)  {(int32_t)x[0], (int32_t)x[1], (int32_t)x[2], (int32_t)x[3]}
#define V2U32_2_V2U64(x)  {(uint64_t)x[0], (uint64_t)x[1]}
#define V2U32_2_V2I64(x)  {(int64_t)x[0], (int64_t)x[1]}

/* v8u16 msa_mull_u8(v8u8 __a, v8u8 __b) */
#define msa_mull_u8(__a, __b)  ((v8u16)__builtin_msa_mulv_h((v8i16)V8U8_2_V8I16(__a), (v8i16)V8U8_2_V8I16(__b)))

/* v8i16 msa_mull_s8(v8i8 __a, v8i8 __b)*/
#define msa_mull_s8(__a, __b)  (__builtin_msa_mulv_h((v8i16)V8I8_2_V8I16(__a), (v8i16)V8I8_2_V8I16(__b)))

/* v4u32 msa_mull_u16(v4u16 __a, v4u16 __b) */
#define msa_mull_u16(__a, __b)  ((v4u32)__builtin_msa_mulv_w((v4i32)V4U16_2_V4I32(__a), (v4i32)V4U16_2_V4I32(__b)))

/* v4i32 msa_mull_s16(v4i16 __a, v4i16 __b) */
#define msa_mull_s16(__a, __b)  (__builtin_msa_mulv_w((v4i32)V4I16_2_V4I32(__a), (v4i32)V4I16_2_V4I32(__b)))

/* v2u64 msa_mull_u32(v2u32 __a, v2u32 __b) */
#define msa_mull_u32(__a, __b)  ((v2u64)__builtin_msa_mulv_d((v2i64)V2U32_2_V2I64(__a), (v2i64)V2U32_2_V2I64(__b)))

/* bitwise and: __builtin_msa_and_v */
#define msa_andq_u8(__a, __b)  ((v16u8)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_s8(__a, __b)  ((v16i8)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_u16(__a, __b) ((v8u16)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_s16(__a, __b) ((v8i16)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_u32(__a, __b) ((v4u32)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_s32(__a, __b) ((v4i32)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_u64(__a, __b) ((v2u64)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))
#define msa_andq_s64(__a, __b) ((v2i64)__builtin_msa_and_v((v16u8)(__a), (v16u8)(__b)))

/* bitwise or: __builtin_msa_or_v */
#define msa_orrq_u8(__a, __b)  ((v16u8)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_s8(__a, __b)  ((v16i8)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_u16(__a, __b) ((v8u16)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_s16(__a, __b) ((v8i16)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_u32(__a, __b) ((v4u32)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_s32(__a, __b) ((v4i32)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_u64(__a, __b) ((v2u64)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))
#define msa_orrq_s64(__a, __b) ((v2i64)__builtin_msa_or_v((v16u8)(__a), (v16u8)(__b)))

/* bitwise xor: __builtin_msa_xor_v */
#define msa_eorq_u8(__a, __b)  ((v16u8)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_s8(__a, __b)  ((v16i8)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_u16(__a, __b) ((v8u16)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_s16(__a, __b) ((v8i16)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_u32(__a, __b) ((v4u32)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_s32(__a, __b) ((v4i32)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_u64(__a, __b) ((v2u64)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))
#define msa_eorq_s64(__a, __b) ((v2i64)__builtin_msa_xor_v((v16u8)(__a), (v16u8)(__b)))

/* bitwise not: v16u8 __builtin_msa_xori_b (v16u8, 0xff) */
#define msa_mvnq_u8(__a)  ((v16u8)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_s8(__a)  ((v16i8)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_u16(__a) ((v8u16)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_s16(__a) ((v8i16)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_u32(__a) ((v4u32)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_s32(__a) ((v4i32)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_u64(__a) ((v2u64)__builtin_msa_xori_b((v16u8)(__a), 0xFF))
#define msa_mvnq_s64(__a) ((v2i64)__builtin_msa_xori_b((v16u8)(__a), 0xFF))

/* compare equal: ceq -> ri = ai == bi ? 1...1:0...0 */
#define msa_ceqq_u8(__a, __b)  ((v16u8)__builtin_msa_ceq_b((v16i8)(__a), (v16i8)(__b)))
#define msa_ceqq_s8(__a, __b)  ((v16u8)__builtin_msa_ceq_b((v16i8)(__a), (v16i8)(__b)))
#define msa_ceqq_u16(__a, __b) ((v8u16)__builtin_msa_ceq_h((v8i16)(__a), (v8i16)(__b)))
#define msa_ceqq_s16(__a, __b) ((v8u16)__builtin_msa_ceq_h((v8i16)(__a), (v8i16)(__b)))
#define msa_ceqq_u32(__a, __b) ((v4u32)__builtin_msa_ceq_w((v4i32)(__a), (v4i32)(__b)))
#define msa_ceqq_s32(__a, __b) ((v4u32)__builtin_msa_ceq_w((v4i32)(__a), (v4i32)(__b)))
#define msa_ceqq_f32(__a, __b) ((v4u32)__builtin_msa_fceq_w((v4f32)(__a), (v4f32)(__b)))
#define msa_ceqq_u64(__a, __b) ((v2u64)__builtin_msa_ceq_d((v2i64)(__a), (v2i64)(__b)))
#define msa_ceqq_s64(__a, __b) ((v2u64)__builtin_msa_ceq_d((v2i64)(__a), (v2i64)(__b)))
#define msa_ceqq_f64(__a, __b) ((v2u64)__builtin_msa_fceq_d((v2f64)(__a), (v2f64)(__b)))

/* Compare less-than: clt -> ri = ai < bi ? 1...1:0...0 */
#define msa_cltq_u8(__a, __b)  ((v16u8)__builtin_msa_clt_u_b((v16u8)(__a), (v16u8)(__b)))
#define msa_cltq_s8(__a, __b)  ((v16u8)__builtin_msa_clt_s_b((v16i8)(__a), (v16i8)(__b)))
#define msa_cltq_u16(__a, __b) ((v8u16)__builtin_msa_clt_u_h((v8u16)(__a), (v8u16)(__b)))
#define msa_cltq_s16(__a, __b) ((v8u16)__builtin_msa_clt_s_h((v8i16)(__a), (v8i16)(__b)))
#define msa_cltq_u32(__a, __b) ((v4u32)__builtin_msa_clt_u_w((v4u32)(__a), (v4u32)(__b)))
#define msa_cltq_s32(__a, __b) ((v4u32)__builtin_msa_clt_s_w((v4i32)(__a), (v4i32)(__b)))
#define msa_cltq_f32(__a, __b) ((v4u32)__builtin_msa_fclt_w((v4f32)(__a), (v4f32)(__b)))
#define msa_cltq_u64(__a, __b) ((v2u64)__builtin_msa_clt_u_d((v2u64)(__a), (v2u64)(__b)))
#define msa_cltq_s64(__a, __b) ((v2u64)__builtin_msa_clt_s_d((v2i64)(__a), (v2i64)(__b)))
#define msa_cltq_f64(__a, __b) ((v2u64)__builtin_msa_fclt_d((v2f64)(__a), (v2f64)(__b)))

/* compare greater-than: cgt -> ri = ai > bi ? 1...1:0...0 */
#define msa_cgtq_u8(__a, __b)  ((v16u8)__builtin_msa_clt_u_b((v16u8)(__b), (v16u8)(__a)))
#define msa_cgtq_s8(__a, __b)  ((v16u8)__builtin_msa_clt_s_b((v16i8)(__b), (v16i8)(__a)))
#define msa_cgtq_u16(__a, __b) ((v8u16)__builtin_msa_clt_u_h((v8u16)(__b), (v8u16)(__a)))
#define msa_cgtq_s16(__a, __b) ((v8u16)__builtin_msa_clt_s_h((v8i16)(__b), (v8i16)(__a)))
#define msa_cgtq_u32(__a, __b) ((v4u32)__builtin_msa_clt_u_w((v4u32)(__b), (v4u32)(__a)))
#define msa_cgtq_s32(__a, __b) ((v4u32)__builtin_msa_clt_s_w((v4i32)(__b), (v4i32)(__a)))
#define msa_cgtq_f32(__a, __b) ((v4u32)__builtin_msa_fclt_w((v4f32)(__b), (v4f32)(__a)))
#define msa_cgtq_u64(__a, __b) ((v2u64)__builtin_msa_clt_u_d((v2u64)(__b), (v2u64)(__a)))
#define msa_cgtq_s64(__a, __b) ((v2u64)__builtin_msa_clt_s_d((v2i64)(__b), (v2i64)(__a)))
#define msa_cgtq_f64(__a, __b) ((v2u64)__builtin_msa_fclt_d((v2f64)(__b), (v2f64)(__a)))

/* compare less-equal: cle -> ri = ai <= bi ? 1...1:0...0 */
#define msa_cleq_u8(__a, __b)  ((v16u8)__builtin_msa_cle_u_b((v16u8)(__a), (v16u8)(__b)))
#define msa_cleq_s8(__a, __b)  ((v16u8)__builtin_msa_cle_s_b((v16i8)(__a), (v16i8)(__b)))
#define msa_cleq_u16(__a, __b) ((v8u16)__builtin_msa_cle_u_h((v8u16)(__a), (v8u16)(__b)))
#define msa_cleq_s16(__a, __b) ((v8u16)__builtin_msa_cle_s_h((v8i16)(__a), (v8i16)(__b)))
#define msa_cleq_u32(__a, __b) ((v4u32)__builtin_msa_cle_u_w((v4u32)(__a), (v4u32)(__b)))
#define msa_cleq_s32(__a, __b) ((v4u32)__builtin_msa_cle_s_w((v4i32)(__a), (v4i32)(__b)))
#define msa_cleq_f32(__a, __b) ((v4u32)__builtin_msa_fcle_w((v4f32)(__a), (v4f32)(__b)))
#define msa_cleq_u64(__a, __b) ((v2u64)__builtin_msa_cle_u_d((v2u64)(__a), (v2u64)(__b)))
#define msa_cleq_s64(__a, __b) ((v2u64)__builtin_msa_cle_s_d((v2i64)(__a), (v2i64)(__b)))
#define msa_cleq_f64(__a, __b) ((v2u64)__builtin_msa_fcle_d((v2f64)(__a), (v2f64)(__b)))

/* compare greater-equal: cge -> ri = ai >= bi ? 1...1:0...0 */
#define msa_cgeq_u8(__a, __b)  ((v16u8)__builtin_msa_cle_u_b((v16u8)(__b), (v16u8)(__a)))
#define msa_cgeq_s8(__a, __b)  ((v16u8)__builtin_msa_cle_s_b((v16i8)(__b), (v16i8)(__a)))
#define msa_cgeq_u16(__a, __b) ((v8u16)__builtin_msa_cle_u_h((v8u16)(__b), (v8u16)(__a)))
#define msa_cgeq_s16(__a, __b) ((v8u16)__builtin_msa_cle_s_h((v8i16)(__b), (v8i16)(__a)))
#define msa_cgeq_u32(__a, __b) ((v4u32)__builtin_msa_cle_u_w((v4u32)(__b), (v4u32)(__a)))
#define msa_cgeq_s32(__a, __b) ((v4u32)__builtin_msa_cle_s_w((v4i32)(__b), (v4i32)(__a)))
#define msa_cgeq_f32(__a, __b) ((v4u32)__builtin_msa_fcle_w((v4f32)(__b), (v4f32)(__a)))
#define msa_cgeq_u64(__a, __b) ((v2u64)__builtin_msa_cle_u_d((v2u64)(__b), (v2u64)(__a)))
#define msa_cgeq_s64(__a, __b) ((v2u64)__builtin_msa_cle_s_d((v2i64)(__b), (v2i64)(__a)))
#define msa_cgeq_f64(__a, __b) ((v2u64)__builtin_msa_fcle_d((v2f64)(__b), (v2f64)(__a)))

/* Shift Left Logical: shl -> ri = ai << bi; */
#define msa_shlq_u8(__a, __b)  ((v16u8)__builtin_msa_sll_b((v16i8)(__a), (v16i8)(__b)))
#define msa_shlq_s8(__a, __b)  ((v16i8)__builtin_msa_sll_b((v16i8)(__a), (v16i8)(__b)))
#define msa_shlq_u16(__a, __b) ((v8u16)__builtin_msa_sll_h((v8i16)(__a), (v8i16)(__b)))
#define msa_shlq_s16(__a, __b) ((v8i16)__builtin_msa_sll_h((v8i16)(__a), (v8i16)(__b)))
#define msa_shlq_u32(__a, __b) ((v4u32)__builtin_msa_sll_w((v4i32)(__a), (v4i32)(__b)))
#define msa_shlq_s32(__a, __b) ((v4i32)__builtin_msa_sll_w((v4i32)(__a), (v4i32)(__b)))
#define msa_shlq_u64(__a, __b) ((v2u64)__builtin_msa_sll_d((v2i64)(__a), (v2i64)(__b)))
#define msa_shlq_s64(__a, __b) ((v2i64)__builtin_msa_sll_d((v2i64)(__a), (v2i64)(__b)))

/* Immediate Shift Left Logical: shl -> ri = ai << imm; */
#define msa_shlq_n_u8(__a, __imm)  ((v16u8)__builtin_msa_slli_b((v16i8)(__a), __imm))
#define msa_shlq_n_s8(__a, __imm)  ((v16i8)__builtin_msa_slli_b((v16i8)(__a), __imm))
#define msa_shlq_n_u16(__a, __imm) ((v8u16)__builtin_msa_slli_h((v8i16)(__a), __imm))
#define msa_shlq_n_s16(__a, __imm) ((v8i16)__builtin_msa_slli_h((v8i16)(__a), __imm))
#define msa_shlq_n_u32(__a, __imm) ((v4u32)__builtin_msa_slli_w((v4i32)(__a), __imm))
#define msa_shlq_n_s32(__a, __imm) ((v4i32)__builtin_msa_slli_w((v4i32)(__a), __imm))
#define msa_shlq_n_u64(__a, __imm) ((v2u64)__builtin_msa_slli_d((v2i64)(__a), __imm))
#define msa_shlq_n_s64(__a, __imm) ((v2i64)__builtin_msa_slli_d((v2i64)(__a), __imm))

/* shift right: shrq -> ri = ai >> bi; */
#define msa_shrq_u8(__a, __b)  ((v16u8)__builtin_msa_srl_b((v16i8)(__a), (v16i8)(__b)))
#define msa_shrq_s8(__a, __b)  ((v16i8)__builtin_msa_sra_b((v16i8)(__a), (v16i8)(__b)))
#define msa_shrq_u16(__a, __b) ((v8u16)__builtin_msa_srl_h((v8i16)(__a), (v8i16)(__b)))
#define msa_shrq_s16(__a, __b) ((v8i16)__builtin_msa_sra_h((v8i16)(__a), (v8i16)(__b)))
#define msa_shrq_u32(__a, __b) ((v4u32)__builtin_msa_srl_w((v4i32)(__a), (v4i32)(__b)))
#define msa_shrq_s32(__a, __b) ((v4i32)__builtin_msa_sra_w((v4i32)(__a), (v4i32)(__b)))
#define msa_shrq_u64(__a, __b) ((v2u64)__builtin_msa_srl_d((v2i64)(__a), (v2i64)(__b)))
#define msa_shrq_s64(__a, __b) ((v2i64)__builtin_msa_sra_d((v2i64)(__a), (v2i64)(__b)))

/* Immediate Shift Right: shr -> ri = ai >> imm; */
#define msa_shrq_n_u8(__a, __imm)  ((v16u8)__builtin_msa_srli_b((v16i8)(__a), __imm))
#define msa_shrq_n_s8(__a, __imm)  ((v16i8)__builtin_msa_srai_b((v16i8)(__a), __imm))
#define msa_shrq_n_u16(__a, __imm) ((v8u16)__builtin_msa_srli_h((v8i16)(__a), __imm))
#define msa_shrq_n_s16(__a, __imm) ((v8i16)__builtin_msa_srai_h((v8i16)(__a), __imm))
#define msa_shrq_n_u32(__a, __imm) ((v4u32)__builtin_msa_srli_w((v4i32)(__a), __imm))
#define msa_shrq_n_s32(__a, __imm) ((v4i32)__builtin_msa_srai_w((v4i32)(__a), __imm))
#define msa_shrq_n_u64(__a, __imm) ((v2u64)__builtin_msa_srli_d((v2i64)(__a), __imm))
#define msa_shrq_n_s64(__a, __imm) ((v2i64)__builtin_msa_srai_d((v2i64)(__a), __imm))

/* Immediate Shift Right Rounded: shr -> ri = ai >> (rounded)imm; */
#define msa_rshrq_n_u8(__a, __imm)  ((v16u8)__builtin_msa_srlri_b((v16i8)(__a), __imm))
#define msa_rshrq_n_s8(__a, __imm)  ((v16i8)__builtin_msa_srari_b((v16i8)(__a), __imm))
#define msa_rshrq_n_u16(__a, __imm) ((v8u16)__builtin_msa_srlri_h((v8i16)(__a), __imm))
#define msa_rshrq_n_s16(__a, __imm) ((v8i16)__builtin_msa_srari_h((v8i16)(__a), __imm))
#define msa_rshrq_n_u32(__a, __imm) ((v4u32)__builtin_msa_srlri_w((v4i32)(__a), __imm))
#define msa_rshrq_n_s32(__a, __imm) ((v4i32)__builtin_msa_srari_w((v4i32)(__a), __imm))
#define msa_rshrq_n_u64(__a, __imm) ((v2u64)__builtin_msa_srlri_d((v2i64)(__a), __imm))
#define msa_rshrq_n_s64(__a, __imm) ((v2i64)__builtin_msa_srari_d((v2i64)(__a), __imm))

/* Vector saturating rounding shift left, qrshl -> ri = ai << bi; */
#define msa_qrshrq_s32(a, b)  ((v4i32)__msa_srar_w((v4i32)(a), (v4i32)(b)))

/* Rename the msa builtin func to unify the name style for intrin_msa.hpp */
#define msa_qaddq_u8          __builtin_msa_adds_u_b
#define msa_qaddq_s8          __builtin_msa_adds_s_b
#define msa_qaddq_u16         __builtin_msa_adds_u_h
#define msa_qaddq_s16         __builtin_msa_adds_s_h
#define msa_qaddq_u32         __builtin_msa_adds_u_w
#define msa_qaddq_s32         __builtin_msa_adds_s_w
#define msa_qaddq_u64         __builtin_msa_adds_u_d
#define msa_qaddq_s64         __builtin_msa_adds_s_d
#define msa_addq_u8(a, b)     ((v16u8)__builtin_msa_addv_b((v16i8)(a), (v16i8)(b)))
#define msa_addq_s8           __builtin_msa_addv_b
#define msa_addq_u16(a, b)    ((v8u16)__builtin_msa_addv_h((v8i16)(a), (v8i16)(b)))
#define msa_addq_s16          __builtin_msa_addv_h
#define msa_addq_u32(a, b)    ((v4u32)__builtin_msa_addv_w((v4i32)(a), (v4i32)(b)))
#define msa_addq_s32          __builtin_msa_addv_w
#define msa_addq_f32          __builtin_msa_fadd_w
#define msa_addq_u64(a, b)    ((v2u64)__builtin_msa_addv_d((v2i64)(a), (v2i64)(b)))
#define msa_addq_s64          __builtin_msa_addv_d
#define msa_addq_f64          __builtin_msa_fadd_d
#define msa_qsubq_u8          __builtin_msa_subs_u_b
#define msa_qsubq_s8          __builtin_msa_subs_s_b
#define msa_qsubq_u16         __builtin_msa_subs_u_h
#define msa_qsubq_s16         __builtin_msa_subs_s_h
#define msa_subq_u8(a, b)     ((v16u8)__builtin_msa_subv_b((v16i8)(a), (v16i8)(b)))
#define msa_subq_s8           __builtin_msa_subv_b
#define msa_subq_u16(a, b)    ((v8u16)__builtin_msa_subv_h((v8i16)(a), (v8i16)(b)))
#define msa_subq_s16          __builtin_msa_subv_h
#define msa_subq_u32(a, b)    ((v4u32)__builtin_msa_subv_w((v4i32)(a), (v4i32)(b)))
#define msa_subq_s32          __builtin_msa_subv_w
#define msa_subq_f32          __builtin_msa_fsub_w
#define msa_subq_u64(a, b)    ((v2u64)__builtin_msa_subv_d((v2i64)(a), (v2i64)(b)))
#define msa_subq_s64          __builtin_msa_subv_d
#define msa_subq_f64          __builtin_msa_fsub_d
#define msa_mulq_u8(a, b)     ((v16u8)__builtin_msa_mulv_b((v16i8)(a), (v16i8)(b)))
#define msa_mulq_s8(a, b)     ((v16i8)__builtin_msa_mulv_b((v16i8)(a), (v16i8)(b)))
#define msa_mulq_u16(a, b)    ((v8u16)__builtin_msa_mulv_h((v8i16)(a), (v8i16)(b)))
#define msa_mulq_s16(a, b)    ((v8i16)__builtin_msa_mulv_h((v8i16)(a), (v8i16)(b)))
#define msa_mulq_u32(a, b)    ((v4u32)__builtin_msa_mulv_w((v4i32)(a), (v4i32)(b)))
#define msa_mulq_s32(a, b)    ((v4i32)__builtin_msa_mulv_w((v4i32)(a), (v4i32)(b)))
#define msa_mulq_u64(a, b)    ((v2u64)__builtin_msa_mulv_d((v2i64)(a), (v2i64)(b)))
#define msa_mulq_s64(a, b)    ((v2i64)__builtin_msa_mulv_d((v2i64)(a), (v2i64)(b)))
#define msa_mulq_f32          __builtin_msa_fmul_w
#define msa_mulq_f64          __builtin_msa_fmul_d
#define msa_divq_f32          __builtin_msa_fdiv_w
#define msa_divq_f64          __builtin_msa_fdiv_d
#define msa_dotp_s_h          __builtin_msa_dotp_s_h
#define msa_dotp_s_w          __builtin_msa_dotp_s_w
#define msa_dotp_s_d          __builtin_msa_dotp_s_d
#define msa_dotp_u_h          __builtin_msa_dotp_u_h
#define msa_dotp_u_w          __builtin_msa_dotp_u_w
#define msa_dotp_u_d          __builtin_msa_dotp_u_d
#define msa_dpadd_s_h         __builtin_msa_dpadd_s_h
#define msa_dpadd_s_w         __builtin_msa_dpadd_s_w
#define msa_dpadd_s_d         __builtin_msa_dpadd_s_d
#define msa_dpadd_u_h         __builtin_msa_dpadd_u_h
#define msa_dpadd_u_w         __builtin_msa_dpadd_u_w
#define msa_dpadd_u_d         __builtin_msa_dpadd_u_d

#define ILVRL_B2(RTYPE, in0, in1, low, hi) do {       \
      low = (RTYPE)__builtin_msa_ilvr_b((v16i8)(in0), (v16i8)(in1));  \
      hi  = (RTYPE)__builtin_msa_ilvl_b((v16i8)(in0), (v16i8)(in1));  \
    } while (0)
#define ILVRL_B2_UB(...) ILVRL_B2(v16u8, __VA_ARGS__)
#define ILVRL_B2_SB(...) ILVRL_B2(v16i8, __VA_ARGS__)
#define ILVRL_B2_UH(...) ILVRL_B2(v8u16, __VA_ARGS__)
#define ILVRL_B2_SH(...) ILVRL_B2(v8i16, __VA_ARGS__)
#define ILVRL_B2_SW(...) ILVRL_B2(v4i32, __VA_ARGS__)

#define ILVRL_H2(RTYPE, in0, in1, low, hi) do {       \
      low = (RTYPE)__builtin_msa_ilvr_h((v8i16)(in0), (v8i16)(in1));  \
      hi  = (RTYPE)__builtin_msa_ilvl_h((v8i16)(in0), (v8i16)(in1));  \
    } while (0)
#define ILVRL_H2_UB(...) ILVRL_H2(v16u8, __VA_ARGS__)
#define ILVRL_H2_SB(...) ILVRL_H2(v16i8, __VA_ARGS__)
#define ILVRL_H2_UH(...) ILVRL_H2(v8u16, __VA_ARGS__)
#define ILVRL_H2_SH(...) ILVRL_H2(v8i16, __VA_ARGS__)
#define ILVRL_H2_SW(...) ILVRL_H2(v4i32, __VA_ARGS__)
#define ILVRL_H2_UW(...) ILVRL_H2(v4u32, __VA_ARGS__)

#define ILVRL_W2(RTYPE, in0, in1, low, hi) do {       \
      low = (RTYPE)__builtin_msa_ilvr_w((v4i32)(in0), (v4i32)(in1));  \
      hi  = (RTYPE)__builtin_msa_ilvl_w((v4i32)(in0), (v4i32)(in1));  \
    } while (0)
#define ILVRL_W2_UB(...) ILVRL_W2(v16u8, __VA_ARGS__)
#define ILVRL_W2_SH(...) ILVRL_W2(v8i16, __VA_ARGS__)
#define ILVRL_W2_SW(...) ILVRL_W2(v4i32, __VA_ARGS__)
#define ILVRL_W2_UW(...) ILVRL_W2(v4u32, __VA_ARGS__)

/* absq, qabsq (r = |a|;) */
#define msa_absq_s8(a)        __builtin_msa_add_a_b(a, __builtin_msa_fill_b(0))
#define msa_absq_s16(a)       __builtin_msa_add_a_h(a, __builtin_msa_fill_h(0))
#define msa_absq_s32(a)       __builtin_msa_add_a_w(a, __builtin_msa_fill_w(0))
#define msa_absq_s64(a)       __builtin_msa_add_a_d(a, __builtin_msa_fill_d(0))
#define msa_absq_f32(a)       ((v4f32)__builtin_msa_bclri_w((v4u32)(a), 31))
#define msa_absq_f64(a)       ((v2f64)__builtin_msa_bclri_d((v2u64)(a), 63))
#define msa_qabsq_s8(a)       __builtin_msa_adds_a_b(a, __builtin_msa_fill_b(0))
#define msa_qabsq_s16(a)      __builtin_msa_adds_a_h(a, __builtin_msa_fill_h(0))
#define msa_qabsq_s32(a)      __builtin_msa_adds_a_w(a, __builtin_msa_fill_w(0))
#define msa_qabsq_s64(a)      __builtin_msa_adds_a_d(a, __builtin_msa_fill_d(0))

/* abdq, qabdq (r = |a - b|;) */
#define msa_abdq_u8           __builtin_msa_asub_u_b
#define msa_abdq_s8           __builtin_msa_asub_s_b
#define msa_abdq_u16          __builtin_msa_asub_u_h
#define msa_abdq_s16          __builtin_msa_asub_s_h
#define msa_abdq_u32          __builtin_msa_asub_u_w
#define msa_abdq_s32          __builtin_msa_asub_s_w
#define msa_abdq_u64          __builtin_msa_asub_u_d
#define msa_abdq_s64          __builtin_msa_asub_s_d
#define msa_abdq_f32(a, b)    msa_absq_f32(__builtin_msa_fsub_w(a, b))
#define msa_abdq_f64(a, b)    msa_absq_f64(__builtin_msa_fsub_d(a, b))
#define msa_qabdq_s8(a, b)    msa_qabsq_s8(__builtin_msa_subs_s_b(a, b))
#define msa_qabdq_s16(a, b)   msa_qabsq_s16(__builtin_msa_subs_s_h(a, b))
#define msa_qabdq_s32(a, b)   msa_qabsq_s32(__builtin_msa_subs_s_w(a, b))
#define msa_qabdq_s64(a, b)   msa_qabsq_s64(__builtin_msa_subs_s_d(a, b))

/* sqrtq, rsqrtq */
#define msa_sqrtq_f32         __builtin_msa_fsqrt_w
#define msa_sqrtq_f64         __builtin_msa_fsqrt_d
#define msa_rsqrtq_f32        __builtin_msa_frsqrt_w
#define msa_rsqrtq_f64        __builtin_msa_frsqrt_d


/* mlaq: r = a + b * c; */
__extension__ extern __inline v4i32
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
msa_mlaq_s32(v4i32 __a, v4i32 __b, v4i32 __c)
{
  __asm__ volatile("maddv.w %w[__a], %w[__b], %w[__c]\n"
               // Outputs
               : [__a] "+f"(__a)
               // Inputs
               : [__b] "f"(__b), [__c] "f"(__c));
  return __a;
}

__extension__ extern __inline v2i64
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
msa_mlaq_s64(v2i64 __a, v2i64 __b, v2i64 __c)
{
  __asm__ volatile("maddv.d %w[__a], %w[__b], %w[__c]\n"
               // Outputs
               : [__a] "+f"(__a)
               // Inputs
               : [__b] "f"(__b), [__c] "f"(__c));
  return __a;
}

__extension__ extern __inline v4f32
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
msa_mlaq_f32(v4f32 __a, v4f32 __b, v4f32 __c)
{
  __asm__ volatile("fmadd.w %w[__a], %w[__b], %w[__c]\n"
               // Outputs
               : [__a] "+f"(__a)
               // Inputs
               : [__b] "f"(__b), [__c] "f"(__c));
  return __a;
}

__extension__ extern __inline v2f64
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
msa_mlaq_f64(v2f64 __a, v2f64 __b, v2f64 __c)
{
  __asm__ volatile("fmadd.d %w[__a], %w[__b], %w[__c]\n"
               // Outputs
               : [__a] "+f"(__a)
               // Inputs
               : [__b] "f"(__b), [__c] "f"(__c));
  return __a;
}

/* cntq */
#define msa_cntq_s8           __builtin_msa_pcnt_b
#define msa_cntq_s16          __builtin_msa_pcnt_h
#define msa_cntq_s32          __builtin_msa_pcnt_w
#define msa_cntq_s64          __builtin_msa_pcnt_d

/* bslq (a: mask; r = b(if a == 0); r = c(if a == 1);) */
#define msa_bslq_u8           __builtin_msa_bsel_v

/* ilvrq, ilvlq (For EL only, ilvrq: b0, a0, b1, a1; ilvlq: b2, a2, b3, a3;) */
#define msa_ilvrq_s8          __builtin_msa_ilvr_b
#define msa_ilvrq_s16         __builtin_msa_ilvr_h
#define msa_ilvrq_s32         __builtin_msa_ilvr_w
#define msa_ilvrq_s64         __builtin_msa_ilvr_d
#define msa_ilvlq_s8          __builtin_msa_ilvl_b
#define msa_ilvlq_s16         __builtin_msa_ilvl_h
#define msa_ilvlq_s32         __builtin_msa_ilvl_w
#define msa_ilvlq_s64         __builtin_msa_ilvl_d

/* ilvevq, ilvodq (ilvevq: b0, a0, b2, a2; ilvodq: b1, a1, b3, a3; ) */
#define msa_ilvevq_s8         __builtin_msa_ilvev_b
#define msa_ilvevq_s16        __builtin_msa_ilvev_h
#define msa_ilvevq_s32        __builtin_msa_ilvev_w
#define msa_ilvevq_s64        __builtin_msa_ilvev_d
#define msa_ilvodq_s8         __builtin_msa_ilvod_b
#define msa_ilvodq_s16        __builtin_msa_ilvod_h
#define msa_ilvodq_s32        __builtin_msa_ilvod_w
#define msa_ilvodq_s64        __builtin_msa_ilvod_d

/* extq (r = (a || b); a concatenation b and get elements from index c) */
#ifdef _MIPSEB
#define msa_extq_s8(a, b, c)  \
(__builtin_msa_vshf_b(__builtin_msa_subv_b((v16i8)((v2i64){0x1716151413121110, 0x1F1E1D1C1B1A1918}), __builtin_msa_fill_b(c)), a, b))
#define msa_extq_s16(a, b, c) \
(__builtin_msa_vshf_h(__builtin_msa_subv_h((v8i16)((v2i64){0x000B000A00090008, 0x000F000E000D000C}), __builtin_msa_fill_h(c)), a, b))
#define msa_extq_s32(a, b, c) \
(__builtin_msa_vshf_w(__builtin_msa_subv_w((v4i32)((v2i64){0x0000000500000004, 0x0000000700000006}), __builtin_msa_fill_w(c)), a, b))
#define msa_extq_s64(a, b, c) \
(__builtin_msa_vshf_d(__builtin_msa_subv_d((v2i64){0x0000000000000002, 0x0000000000000003}, __builtin_msa_fill_d(c)), a, b))
#else
#define msa_extq_s8(a, b, c)  \
(__builtin_msa_vshf_b(__builtin_msa_addv_b((v16i8)((v2i64){0x0706050403020100, 0x0F0E0D0C0B0A0908}), __builtin_msa_fill_b(c)), b, a))
#define msa_extq_s16(a, b, c) \
(__builtin_msa_vshf_h(__builtin_msa_addv_h((v8i16)((v2i64){0x0003000200010000, 0x0007000600050004}), __builtin_msa_fill_h(c)), b, a))
#define msa_extq_s32(a, b, c) \
(__builtin_msa_vshf_w(__builtin_msa_addv_w((v4i32)((v2i64){0x0000000100000000, 0x0000000300000002}), __builtin_msa_fill_w(c)), b, a))
#define msa_extq_s64(a, b, c) \
(__builtin_msa_vshf_d(__builtin_msa_addv_d((v2i64){0x0000000000000000, 0x0000000000000001}, __builtin_msa_fill_d(c)), b, a))
#endif /* _MIPSEB */

/* cvttruncq, cvttintq, cvtrintq */
#define msa_cvttruncq_u32_f32 __builtin_msa_ftrunc_u_w
#define msa_cvttruncq_s32_f32 __builtin_msa_ftrunc_s_w
#define msa_cvttruncq_u64_f64 __builtin_msa_ftrunc_u_d
#define msa_cvttruncq_s64_f64 __builtin_msa_ftrunc_s_d
#define msa_cvttintq_u32_f32  __builtin_msa_ftint_u_w
#define msa_cvttintq_s32_f32  __builtin_msa_ftint_s_w
#define msa_cvttintq_u64_f64  __builtin_msa_ftint_u_d
#define msa_cvttintq_s64_f64  __builtin_msa_ftint_s_d
#define msa_cvtrintq_f32      __builtin_msa_frint_w
#define msa_cvtrintq_f64      __builtin_msa_frint_d

/* cvtfintq, cvtfq */
#define msa_cvtfintq_f32_u32  __builtin_msa_ffint_u_w
#define msa_cvtfintq_f32_s32  __builtin_msa_ffint_s_w
#define msa_cvtfintq_f64_u64  __builtin_msa_ffint_u_d
#define msa_cvtfintq_f64_s64  __builtin_msa_ffint_s_d
#define msa_cvtfq_f32_f64     __builtin_msa_fexdo_w
#define msa_cvtflq_f64_f32    __builtin_msa_fexupr_d
#define msa_cvtfhq_f64_f32    __builtin_msa_fexupl_d

#define msa_addl_u8(a, b)     ((v8u16)__builtin_msa_addv_h((v8i16)V8U8_2_V8I16(a), (v8i16)V8U8_2_V8I16(b)))
#define msa_addl_s8(a, b)     (__builtin_msa_addv_h((v8i16)V8I8_2_V8I16(a), (v8i16)V8I8_2_V8I16(b)))
#define msa_addl_u16(a, b)    ((v4u32)__builtin_msa_addv_w((v4i32)V4U16_2_V4I32(a), (v4i32)V4U16_2_V4I32(b)))
#define msa_addl_s16(a, b)    (__builtin_msa_addv_w((v4i32)V4I16_2_V4I32(a), (v4i32)V4I16_2_V4I32(b)))
#define msa_subl_s16(a, b)    (__builtin_msa_subv_w((v4i32)V4I16_2_V4I32(a), (v4i32)V4I16_2_V4I32(b)))
#define msa_recpeq_f32        __builtin_msa_frcp_w
#define msa_recpsq_f32(a, b)  (__builtin_msa_fsub_w(msa_dupq_n_f32(2.0f), __builtin_msa_fmul_w(a, b)))

#define MSA_INTERLEAVED_IMPL_LOAD2_STORE2(_Tp, _Tpv, _Tpvs, suffix, df, nlanes) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld2q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + nlanes); \
  *a = (_Tpv)__builtin_msa_pckev_##df((_Tpvs)v1, (_Tpvs)v0); \
  *b = (_Tpv)__builtin_msa_pckod_##df((_Tpvs)v1, (_Tpvs)v0); \
} \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st2q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b) \
{ \
  msa_st1q_##suffix(ptr, (_Tpv)__builtin_msa_ilvr_##df((_Tpvs)b, (_Tpvs)a)); \
  msa_st1q_##suffix(ptr + nlanes, (_Tpv)__builtin_msa_ilvl_##df((_Tpvs)b, (_Tpvs)a)); \
}

MSA_INTERLEAVED_IMPL_LOAD2_STORE2(uint8_t, v16u8, v16i8, u8, b, 16)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(int8_t, v16i8, v16i8, s8, b, 16)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(uint16_t, v8u16, v8i16, u16, h, 8)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(int16_t, v8i16, v8i16, s16, h, 8)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(uint32_t, v4u32, v4i32, u32, w, 4)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(int32_t, v4i32, v4i32, s32, w, 4)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(float, v4f32, v4i32, f32, w, 4)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(uint64_t, v2u64, v2i64, u64, d, 2)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(int64_t, v2i64, v2i64, s64, d, 2)
MSA_INTERLEAVED_IMPL_LOAD2_STORE2(double, v2f64, v2i64, f64, d, 2)

#ifdef _MIPSEB
#define MSA_INTERLEAVED_IMPL_LOAD3_8(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + 16); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + 32); \
  _Tpvs v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0704011F1F1F1F1F, 0x1F1C191613100D0A}), (_Tpvs)v0, (_Tpvs)v1); \
  *a = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x1716150E0B080502, 0x1F1E1D1C1B1A1918}), v3, (_Tpvs)v2); \
  v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0603001F1F1F1F1F, 0x1E1B1815120F0C09}), (_Tpvs)v0, (_Tpvs)v1); \
  *b = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x1716150D0A070401, 0x1F1E1D1C1B1A1918}), v3, (_Tpvs)v2); \
  v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x05021F1F1F1F1F1F, 0x1D1A1714110E0B08}), (_Tpvs)v0, (_Tpvs)v1); \
  *c = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x17160F0C09060300, 0x1F1E1D1C1B1A1918}), v3, (_Tpvs)v2); \
}
#else
#define MSA_INTERLEAVED_IMPL_LOAD3_8(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + 16); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + 32); \
  _Tpvs v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x15120F0C09060300, 0x00000000001E1B18}), (_Tpvs)v1, (_Tpvs)v0); \
  *a = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x0706050403020100, 0x1D1A1714110A0908}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x1613100D0A070401, 0x00000000001F1C19}), (_Tpvs)v1, (_Tpvs)v0); \
  *b = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x0706050403020100, 0x1E1B1815120A0908}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x1714110E0B080502, 0x0000000000001D1A}), (_Tpvs)v1, (_Tpvs)v0); \
  *c = (_Tpv)__builtin_msa_vshf_b((_Tpvs)((v2i64){0x0706050403020100, 0x1F1C191613100908}), (_Tpvs)v2, v3); \
}
#endif

MSA_INTERLEAVED_IMPL_LOAD3_8(uint8_t, v16u8, v16i8, u8)
MSA_INTERLEAVED_IMPL_LOAD3_8(int8_t, v16i8, v16i8, s8)

#ifdef _MIPSEB
#define MSA_INTERLEAVED_IMPL_LOAD3_16(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + 8); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + 16); \
  _Tpvs v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x00030000000F000F, 0x000F000C00090006}), (_Tpvs)v1, (_Tpvs)v0); \
  *a = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x000B000A00050002, 0x000F000E000D000C}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0002000F000F000F, 0x000E000B00080005}), (_Tpvs)v1, (_Tpvs)v0); \
  *b = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x000B000700040001, 0x000F000E000D000C}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0001000F000F000F, 0x000D000A00070004}), (_Tpvs)v1, (_Tpvs)v0); \
  *c = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x000B000600030000, 0x000F000E000D000C}), (_Tpvs)v2, v3); \
}
#else
#define MSA_INTERLEAVED_IMPL_LOAD3_16(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + 8); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + 16); \
  _Tpvs v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0009000600030000, 0x00000000000F000C}), (_Tpvs)v1, (_Tpvs)v0); \
  *a = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x0003000200010000, 0x000D000A00050004}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000A000700040001, 0x000000000000000D}), (_Tpvs)v1, (_Tpvs)v0); \
  *b = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x0003000200010000, 0x000E000B00080004}), (_Tpvs)v2, v3); \
  v3 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000B000800050002, 0x000000000000000E}), (_Tpvs)v1, (_Tpvs)v0); \
  *c = (_Tpv)__builtin_msa_vshf_h((_Tpvs)((v2i64){0x0003000200010000, 0x000F000C00090004}), (_Tpvs)v2, v3); \
}
#endif

MSA_INTERLEAVED_IMPL_LOAD3_16(uint16_t, v8u16, v8i16, u16)
MSA_INTERLEAVED_IMPL_LOAD3_16(int16_t, v8i16, v8i16, s16)

#define MSA_INTERLEAVED_IMPL_LOAD3_32(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  _Tpv v00 = msa_ld1q_##suffix(ptr); \
  _Tpv v01 = msa_ld1q_##suffix(ptr + 4); \
  _Tpv v02 = msa_ld1q_##suffix(ptr + 8); \
  _Tpvs v10 = __builtin_msa_ilvr_w((_Tpvs)__builtin_msa_ilvl_d((v2i64)v01, (v2i64)v01), (_Tpvs)v00); \
  _Tpvs v11 = __builtin_msa_ilvr_w((_Tpvs)v02, (_Tpvs)__builtin_msa_ilvl_d((v2i64)v00, (v2i64)v00)); \
  _Tpvs v12 = __builtin_msa_ilvr_w((_Tpvs)__builtin_msa_ilvl_d((v2i64)v02, (v2i64)v02), (_Tpvs)v01); \
  *a = (_Tpv)__builtin_msa_ilvr_w((_Tpvs)__builtin_msa_ilvl_d((v2i64)v11, (v2i64)v11), v10); \
  *b = (_Tpv)__builtin_msa_ilvr_w(v12, (_Tpvs)__builtin_msa_ilvl_d((v2i64)v10, (v2i64)v10)); \
  *c = (_Tpv)__builtin_msa_ilvr_w((_Tpvs)__builtin_msa_ilvl_d((v2i64)v12, (v2i64)v12), v11); \
}

MSA_INTERLEAVED_IMPL_LOAD3_32(uint32_t, v4u32, v4i32, u32)
MSA_INTERLEAVED_IMPL_LOAD3_32(int32_t, v4i32, v4i32, s32)
MSA_INTERLEAVED_IMPL_LOAD3_32(float, v4f32, v4i32, f32)

#define MSA_INTERLEAVED_IMPL_LOAD3_64(_Tp, _Tpv, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld3q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c) \
{ \
  *((_Tp*)a) = *ptr;           *((_Tp*)b) = *(ptr + 1);     *((_Tp*)c) = *(ptr + 2);     \
  *((_Tp*)a + 1) = *(ptr + 3); *((_Tp*)b + 1) = *(ptr + 4); *((_Tp*)c + 1) = *(ptr + 5); \
}

MSA_INTERLEAVED_IMPL_LOAD3_64(uint64_t, v2u64, u64)
MSA_INTERLEAVED_IMPL_LOAD3_64(int64_t, v2i64, s64)
MSA_INTERLEAVED_IMPL_LOAD3_64(double, v2f64, f64)

#ifdef _MIPSEB
#define MSA_INTERLEAVED_IMPL_STORE3_8(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0F0E0D0C0B1F1F1F, 0x1F1E1D1C1B1A1F1F}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0D1C140C1B130B1A, 0x1F170F1E160E1D15}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0A09080706051F1F, 0x19181716151F1F1F}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x1D14071C13061B12, 0x170A1F16091E1508}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 16, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x04030201001F1F1F, 0x14131211101F1F1F}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x15021C14011B1300, 0x051F17041E16031D}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 32, (_Tpv)v1); \
}
#else
#define MSA_INTERLEAVED_IMPL_STORE3_8(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0000050403020100, 0x0000001413121110}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0A02110901100800, 0x05140C04130B0312}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0000000A09080706, 0x00001A1918171615}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x170A011609001508, 0x0D04190C03180B02}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 16, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x0000000F0E0D0C0B, 0x0000001F1E1D1C1B}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_b((_Tpvs)((v2i64){0x021C09011B08001A, 0x1F0C041E0B031D0A}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 32, (_Tpv)v1); \
}
#endif

MSA_INTERLEAVED_IMPL_STORE3_8(uint8_t, v16u8, v16i8, u8)
MSA_INTERLEAVED_IMPL_STORE3_8(int8_t, v16i8, v16i8, s8)

#ifdef _MIPSEB
#define MSA_INTERLEAVED_IMPL_STORE3_16(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000700060005000F, 0x000F000E000D000F}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000A0006000D0009, 0x000F000B0007000E}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x00040003000F000F, 0x000C000B000A000F}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000E000A0003000D, 0x0005000F000B0004}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 8, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000200010000000F, 0x00090008000F000F}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0001000E00090000, 0x000B0002000F000A}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 16, (_Tpv)v1); \
}
#else
#define MSA_INTERLEAVED_IMPL_STORE3_16(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0000000200010000, 0x0000000A00090008}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0001000800040000, 0x0006000200090005}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0000000500040003, 0x00000000000C000B}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x000B00040000000A, 0x0002000C00050001}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 8, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x0000000000070006, 0x0000000F000E000D}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_h((_Tpvs)((v2i64){0x00050000000D0004, 0x000F00060001000E}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 16, (_Tpv)v1); \
}
#endif

MSA_INTERLEAVED_IMPL_STORE3_16(uint16_t, v8u16, v8i16, u16)
MSA_INTERLEAVED_IMPL_STORE3_16(int16_t, v8i16, v8i16, s16)

#ifdef _MIPSEB
#define MSA_INTERLEAVED_IMPL_STORE3_32(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000300000007, 0x0000000700000006}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000300000006, 0x0000000700000005}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000200000001, 0x0000000500000007}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000700000004, 0x0000000500000002}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 4, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000000000007, 0x0000000400000007}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000500000000, 0x0000000100000007}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 8, (_Tpv)v1); \
}
#else
#define MSA_INTERLEAVED_IMPL_STORE3_32(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  _Tpvs v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000100000000, 0x0000000000000004}), (_Tpvs)b, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000200000000, 0x0000000100000004}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000000000002, 0x0000000600000005}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000500000002, 0x0000000300000000}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 4, (_Tpv)v1); \
  v0 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000000000003, 0x0000000000000007}), (_Tpvs)b, (_Tpvs)a); \
  v1 = __builtin_msa_vshf_w((_Tpvs)((v2i64){0x0000000000000006, 0x0000000700000002}), (_Tpvs)c, (_Tpvs)v0); \
  msa_st1q_##suffix(ptr + 8, (_Tpv)v1); \
}
#endif

MSA_INTERLEAVED_IMPL_STORE3_32(uint32_t, v4u32, v4i32, u32)
MSA_INTERLEAVED_IMPL_STORE3_32(int32_t, v4i32, v4i32, s32)
MSA_INTERLEAVED_IMPL_STORE3_32(float, v4f32, v4i32, f32)

#define MSA_INTERLEAVED_IMPL_STORE3_64(_Tp, _Tpv, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st3q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c) \
{ \
  *ptr = a[0];       *(ptr + 1) = b[0]; *(ptr + 2) = c[0]; \
  *(ptr + 3) = a[1]; *(ptr + 4) = b[1]; *(ptr + 5) = c[1]; \
}

MSA_INTERLEAVED_IMPL_STORE3_64(uint64_t, v2u64, u64)
MSA_INTERLEAVED_IMPL_STORE3_64(int64_t, v2i64, s64)
MSA_INTERLEAVED_IMPL_STORE3_64(double, v2f64, f64)

#define MSA_INTERLEAVED_IMPL_LOAD4_STORE4(_Tp, _Tpv, _Tpvs, suffix, df, nlanes) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld4q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c, _Tpv* d) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + nlanes); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + nlanes * 2); \
  _Tpv v3 = msa_ld1q_##suffix(ptr + nlanes * 3); \
  _Tpvs t0 = __builtin_msa_pckev_##df((_Tpvs)v1, (_Tpvs)v0); \
  _Tpvs t1 = __builtin_msa_pckev_##df((_Tpvs)v3, (_Tpvs)v2); \
  _Tpvs t2 = __builtin_msa_pckod_##df((_Tpvs)v1, (_Tpvs)v0); \
  _Tpvs t3 = __builtin_msa_pckod_##df((_Tpvs)v3, (_Tpvs)v2); \
  *a = (_Tpv)__builtin_msa_pckev_##df(t1, t0); \
  *b = (_Tpv)__builtin_msa_pckev_##df(t3, t2); \
  *c = (_Tpv)__builtin_msa_pckod_##df(t1, t0); \
  *d = (_Tpv)__builtin_msa_pckod_##df(t3, t2); \
} \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st4q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c, const _Tpv d) \
{ \
  _Tpvs v0 = __builtin_msa_ilvr_##df((_Tpvs)c, (_Tpvs)a); \
  _Tpvs v1 = __builtin_msa_ilvr_##df((_Tpvs)d, (_Tpvs)b); \
  _Tpvs v2 = __builtin_msa_ilvl_##df((_Tpvs)c, (_Tpvs)a); \
  _Tpvs v3 = __builtin_msa_ilvl_##df((_Tpvs)d, (_Tpvs)b); \
  msa_st1q_##suffix(ptr, (_Tpv)__builtin_msa_ilvr_##df(v1, v0)); \
  msa_st1q_##suffix(ptr + nlanes, (_Tpv)__builtin_msa_ilvl_##df(v1, v0)); \
  msa_st1q_##suffix(ptr + 2 * nlanes, (_Tpv)__builtin_msa_ilvr_##df(v3, v2)); \
  msa_st1q_##suffix(ptr + 3 * nlanes, (_Tpv)__builtin_msa_ilvl_##df(v3, v2)); \
}

MSA_INTERLEAVED_IMPL_LOAD4_STORE4(uint8_t, v16u8, v16i8, u8, b, 16)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(int8_t, v16i8, v16i8, s8, b, 16)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(uint16_t, v8u16, v8i16, u16, h, 8)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(int16_t, v8i16, v8i16, s16, h, 8)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(uint32_t, v4u32, v4i32, u32, w, 4)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(int32_t, v4i32, v4i32, s32, w, 4)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4(float, v4f32, v4i32, f32, w, 4)

#define MSA_INTERLEAVED_IMPL_LOAD4_STORE4_64(_Tp, _Tpv, _Tpvs, suffix) \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_ld4q_##suffix(const _Tp* ptr, _Tpv* a, _Tpv* b, _Tpv* c, _Tpv* d) \
{ \
  _Tpv v0 = msa_ld1q_##suffix(ptr); \
  _Tpv v1 = msa_ld1q_##suffix(ptr + 2); \
  _Tpv v2 = msa_ld1q_##suffix(ptr + 4); \
  _Tpv v3 = msa_ld1q_##suffix(ptr + 6); \
  *a = (_Tpv)__builtin_msa_ilvr_d((_Tpvs)v2, (_Tpvs)v0); \
  *b = (_Tpv)__builtin_msa_ilvl_d((_Tpvs)v2, (_Tpvs)v0); \
  *c = (_Tpv)__builtin_msa_ilvr_d((_Tpvs)v3, (_Tpvs)v1); \
  *d = (_Tpv)__builtin_msa_ilvl_d((_Tpvs)v3, (_Tpvs)v1); \
} \
__extension__ extern __inline void \
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__)) \
msa_st4q_##suffix(_Tp* ptr, const _Tpv a, const _Tpv b, const _Tpv c, const _Tpv d) \
{ \
  msa_st1q_##suffix(ptr, (_Tpv)__builtin_msa_ilvr_d((_Tpvs)b, (_Tpvs)a)); \
  msa_st1q_##suffix(ptr + 2, (_Tpv)__builtin_msa_ilvr_d((_Tpvs)d, (_Tpvs)c)); \
  msa_st1q_##suffix(ptr + 4, (_Tpv)__builtin_msa_ilvl_d((_Tpvs)b, (_Tpvs)a)); \
  msa_st1q_##suffix(ptr + 6, (_Tpv)__builtin_msa_ilvl_d((_Tpvs)d, (_Tpvs)c)); \
}

MSA_INTERLEAVED_IMPL_LOAD4_STORE4_64(uint64_t, v2u64, v2i64, u64)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4_64(int64_t, v2i64, v2i64, s64)
MSA_INTERLEAVED_IMPL_LOAD4_STORE4_64(double, v2f64, v2i64, f64)

__extension__ extern __inline v8i16
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
msa_qdmulhq_n_s16(v8i16 a, int16_t b)
{
  v8i16 a_lo, a_hi;
  ILVRL_H2_SH(a, msa_dupq_n_s16(0), a_lo, a_hi);
  return msa_packr_s32(msa_shlq_n_s32(msa_mulq_s32(msa_paddlq_s16(a_lo), msa_dupq_n_s32(b)), 1),
                       msa_shlq_n_s32(msa_mulq_s32(msa_paddlq_s16(a_hi), msa_dupq_n_s32(b)), 1), 16);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif /*__mips_msa*/
#endif /* OPENCV_CORE_MSA_MACROS_H */
