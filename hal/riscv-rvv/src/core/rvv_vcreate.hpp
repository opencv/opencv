// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.

#ifndef OPENCV_HAL_RVV_VCREATE_HPP_INCLUDED
#define OPENCV_HAL_RVV_VCREATE_HPP_INCLUDED

// Tuple creation intrinsics were added in Clang 18.
// These shims are shared by flip, transpose and rotate.
#if defined(__clang__) && __clang_major__ < 18
#define OPENCV_HAL_IMPL_RVV_VCREATE_x3(suffix, width, v0, v1, v2) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x3(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2);
#define __riscv_vcreate_v_u8m2x3(v0, v1, v2)  OPENCV_HAL_IMPL_RVV_VCREATE_x3(u8, 2, v0, v1, v2)
#define __riscv_vcreate_v_u16m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u16, 2, v0, v1, v2)
#define __riscv_vcreate_v_u32m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u32, 2, v0, v1, v2)
#define __riscv_vcreate_v_u64m2x3(v0, v1, v2) OPENCV_HAL_IMPL_RVV_VCREATE_x3(u64, 2, v0, v1, v2)

#define OPENCV_HAL_IMPL_RVV_VCREATE_x4(suffix, width, v0, v1, v2, v3) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x4(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3);

#define OPENCV_HAL_IMPL_RVV_VCREATE_x8(suffix, width, v0, v1, v2, v3, v4, v5, v6, v7) \
    __riscv_vset_v_##suffix##m##width##_##suffix##m##width##x8(v, 0, v0); \
    v = __riscv_vset(v, 1, v1); \
    v = __riscv_vset(v, 2, v2); \
    v = __riscv_vset(v, 3, v3); \
    v = __riscv_vset(v, 4, v4); \
    v = __riscv_vset(v, 5, v5); \
    v = __riscv_vset(v, 6, v6); \
    v = __riscv_vset(v, 7, v7);

#define __riscv_vcreate_v_u8m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u8, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u16, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_i32m1x4(v0, v1, v2, v3) OPENCV_HAL_IMPL_RVV_VCREATE_x4(i32, 1, v0, v1, v2, v3)
#define __riscv_vcreate_v_i64m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(i64, 1, v0, v1, v2, v3, v4, v5, v6, v7)

#define __riscv_vcreate_v_u32m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u32, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#define __riscv_vcreate_v_u64m1x8(v0, v1, v2, v3, v4, v5, v6, v7) OPENCV_HAL_IMPL_RVV_VCREATE_x8(u64, 1, v0, v1, v2, v3, v4, v5, v6, v7)
#endif

#endif // OPENCV_HAL_RVV_VCREATE_HPP_INCLUDED
