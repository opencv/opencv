// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <numeric>
#include "precomp.hpp"
#include "warp_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

#define CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 M0 = vx_setall_f32(M[0]), \
              M3 = vx_setall_f32(M[3]); \
    v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])), \
              M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));
#define CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1() \
    CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 M6 = vx_setall_f32(M[6]); \
    v_float32 M_w = vx_setall_f32(static_cast<float>(y * M[7] + M[8]));
#define CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 dst_y = vx_setall_f32(float(y));

#define CV_WARP_VECTOR_GET_ADDR_C1() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0), \
            addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
#define CV_WARP_VECTOR_GET_ADDR_C3() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
#define CV_WARP_VECTOR_GET_ADDR_C4() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
#define CV_WARP_VECTOR_GET_ADDR(CN) \
    CV_WARP_VECTOR_GET_ADDR_##CN() \
    vx_store(addr, addr_0); \
    vx_store(addr + vlanes_32, addr_1);

#define CV_WARP_VECTOR_LINEAR_COMPUTE_COORD() \
    v_int32 src_ix0 = v_floor(src_x0), src_iy0 = v_floor(src_y0); \
    v_int32 src_ix1 = v_floor(src_x1), src_iy1 = v_floor(src_y1); \
    src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0)); \
    src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0)); \
    src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1)); \
    src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1));
#define CV_WARP_VECTOR_NEAREST_COMPUTE_COORD() \
    v_int32 src_ix0 = v_round(src_x0), src_iy0 = v_round(src_y0); \
    v_int32 src_ix1 = v_round(src_x1), src_iy1 = v_round(src_y1); \

#define CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN) \
    CV_WARP_VECTOR_##INTER##_COMPUTE_COORD() \
    v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols), \
             mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols); \
    mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows)); \
    mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows)); \
    v_uint16 inner_mask = v_pack(mask_0, mask_1); \
    CV_WARP_VECTOR_GET_ADDR(CN)

#define CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

#define CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_w0 = v_fma(M6, dst_x0, M_w), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y), \
              src_w1 = v_fma(M6, dst_x1, M_w); \
    src_x0 = v_div(src_x0, src_w0); \
    src_y0 = v_div(src_y0, src_w0); \
    src_x1 = v_div(src_x1, src_w1); \
    src_y1 = v_div(src_y1, src_w1); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

#define CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(INTER, CN) \
    v_float32 src_x0, src_y0, \
              src_x1, src_y1; \
    if (map1 == map2) { \
        v_load_deinterleave(sx_data + 2*x, src_x0, src_y0); \
        v_load_deinterleave(sy_data + 2*(x+vlanes_32), src_x1, src_y1); \
    } else { \
        src_x0 = vx_load(sx_data+x); \
        src_y0 = vx_load(sy_data+x); \
        src_x1 = vx_load(sx_data+x+vlanes_32); \
        src_y1 = vx_load(sy_data+x+vlanes_32); \
    } \
    if (relative) { \
        src_x0 = v_add(src_x0, dst_x0); \
        src_y0 = v_add(src_y0, dst_y); \
        src_x1 = v_add(src_x1, dst_x1); \
        src_y1 = v_add(src_y1, dst_y); \
        dst_x0 = v_add(dst_x0, delta); \
        dst_x1 = v_add(dst_x1, delta); \
    } \
    CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN)

namespace cv{
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

ImgWarpFunc getBicubicWarpFunc_(int type);

void warpAffineNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpAffineNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double M[6], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double M[9], int border_type, const double border_value[4]);
void remapNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);

void warpAffineLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
// Approximate branch that uses FP16 intrinsics if possible
void warpAffineLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[6], int border_type, const double border_value[4]);

void warpPerspectiveLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
// Approximate branch that uses FP16 intrinsics if possible
void warpPerspectiveLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double M[9], int border_type, const double border_value[4]);

void remapLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
// Approximate branch that uses FP16 intrinsics if possible
void remapLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);
void remapLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {
static inline int borderInterpolate_fast( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    return p;
}
} // anonymous

void warpAffineNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);

                CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 16U);

                CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);

                CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                    uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);

                CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);

                CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpAffineNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                    float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                    const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);

                CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            // CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            v_float32 dst_x0 = vx_load(start_indices.data());
            v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32)));
            v_float32 M0 = vx_setall_f32(M[0]), M3 = vx_setall_f32(M[3]);
            v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])),
                    M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));
            v_float32 M6 = vx_setall_f32(M[6]);
            v_float32 M_w = vx_setall_f32(static_cast<float>(y * M[7] + M[8]));
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                         uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void warpPerspectiveNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                         float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                         const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;
                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapNearestInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 8U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 8U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 8U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 8U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 8U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 8U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 8U);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative)  {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 16U, 16U);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 16U, 16U);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                               uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative)  {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 16U, 16U);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 16U, 16U);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 16U);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C1, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C1, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C1, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(NEAREST, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(NEAREST, C3, 32F, 32F);
                CV_WARP_VECTOR_INTER_STORE(NEAREST, C3, 32F, 32F);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C3, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}
void remapNearestInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                               float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                               int border_type, const double border_value[4],
                               const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(NEAREST, C4);
                if (v_reduce_min(inner_mask) != 0) {
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, NEAREST, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, NEAREST, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, NEAREST, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(NEAREST, C4, 32F);
                    CV_WARP_VECTOR_INTER_LOAD(NEAREST, C4, 32F, 32F);
                    CV_WARP_VECTOR_INTER_STORE(NEAREST, C4, 32F, 32F);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }
                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(NEAREST, C4, 32F);
                CV_WARP_SCALAR_STORE(NEAREST, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };

    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);

                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}


void warpAffineLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}


void warpAffineLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                  uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
                    // CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpAffineLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);

                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }

    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpAffineLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[6], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;

                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpAffineLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                                 dst_data, dst_step, dst_rows, dst_cols,
                                 dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
    #endif
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                       uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                       const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 16U);

                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);

                CV_WARP_SCALAR_STORE(LINEAR, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                                        uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);

                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);

                CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};

        int vlanes_32 = VTraits<v_float32>::vlanes();

        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void warpPerspectiveLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                             const double dM[9], int border_type, const double border_value[4]) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step, dststep = dst_step;
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    warpPerspectiveLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                                      dst_data, dst_step, dst_rows, dst_cols,
                                      dM, border_type, border_value);
#endif
}

void remapLinearInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C1);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif
                if (v_reduce_min(inner_mask) != 0) {
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 8U);
    #endif
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8U16_NEON(C3);
    #else
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 8U, 16U);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        const auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_16{VTraits<v_uint16>::max_nlanes};
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_16 = VTraits<v_uint16>::vlanes();
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 8U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 8U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 8U);
    #endif
                } else {
                    uint8_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 8U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC1(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4];

        uint16_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC3(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*3];

        uint16_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 16U);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 16U);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 16U, 16U);
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_16UC4(const uint16_t *src_data, size_t src_step, int src_rows, int src_cols,
                              uint16_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(uint16_t), dststep = dst_step/sizeof(uint16_t),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        uint16_t bval[] = {
            saturate_cast<uint16_t>(border_value[0]),
            saturate_cast<uint16_t>(border_value[1]),
            saturate_cast<uint16_t>(border_value[2]),
            saturate_cast<uint16_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        uint16_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint16 bval_v0 = vx_load(&bvalbuf[0]);
        v_uint16 bval_v1 = vx_load(&bvalbuf[uf]);
        v_uint16 bval_v2 = vx_load(&bvalbuf[uf*2]);
        v_uint16 bval_v3 = vx_load(&bvalbuf[uf*3]);
#endif

        for (int y = r.start; y < r.end; y++) {
            uint16_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 16U);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 16U);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 16U);
    #endif
                } else {
                    uint16_t pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 16U);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 16U, 16U);
                    CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 16U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 16U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4];

        float bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C1, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*3];

        float bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN(LINEAR, C3, 32F);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 32F);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C3, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(float), dststep = dst_step/sizeof(float),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        float bval[] = {
            saturate_cast<float>(border_value[0]),
            saturate_cast<float>(border_value[1]),
            saturate_cast<float>(border_value[2]),
            saturate_cast<float>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];

        float bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_float32 bval_v0_l = vx_load(&bvalbuf[0]);
        v_float32 bval_v0_h = vx_load(&bvalbuf[vlanes_32]);
        v_float32 bval_v1_l = vx_load(&bvalbuf[uf]);
        v_float32 bval_v1_h = vx_load(&bvalbuf[uf+vlanes_32]);
        v_float32 bval_v2_l = vx_load(&bvalbuf[uf*2]);
        v_float32 bval_v2_h = vx_load(&bvalbuf[uf*2+vlanes_32]);
        v_float32 bval_v3_l = vx_load(&bvalbuf[uf*3]);
        v_float32 bval_v3_h = vx_load(&bvalbuf[uf*3+vlanes_32]);
#endif

        for (int y = r.start; y < r.end; y++) {
            float* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    float valpha[max_uf], vbeta[max_uf];
                    vx_store(valpha, src_x0);
                    vx_store(valpha+vlanes_32, src_x1);
                    vx_store(vbeta, src_y0);
                    vx_store(vbeta+vlanes_32, src_y1);
    #if CV_SIMD256
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD256, LINEAR, 32F);
    #elif CV_SIMD128
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMD128, LINEAR, 32F);
    #elif CV_SIMD_SCALABLE
                    CV_WARP_VECTOR_SHUFFLE_INTER_STORE_C4(SIMDX, LINEAR, 32F);
    #endif
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 32F);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 32F);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 32F);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearApproxInvoker_8UC1(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4];

        uint8_t bvalbuf[max_uf];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i] = bval[0];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        uint8x8_t grays = {0, 8, 16, 24, 1, 9, 17, 25};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                uint8x8_t p00g, p01g, p10g, p11g;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC1(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}
void remapLinearApproxInvoker_8UC3(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), three = vx_setall_s32(3);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*3];

        uint8_t bvalbuf[max_uf*3];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*3] = bval[0];
            bvalbuf[i*3+1] = bval[1];
            bvalbuf[i*3+2] = bval[2];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        uint8x8_t reds = {0, 8, 16, 24, 3, 11, 19, 27},
                  greens = {1, 9, 17, 25, 4, 12, 20, 28},
                  blues = {2, 10, 18, 26, 5, 13, 21, 29};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C3);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C3, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C3, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C3);
                CV_WARP_SCALAR_STORE(LINEAR, C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC3(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}
void remapLinearApproxInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                                   uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                   int border_type, const double border_value[4],
                                   const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
#if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 && CV_SIMD128_FP16
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step, dststep = dst_step,
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;
        uint8_t bval[] = {
            saturate_cast<uint8_t>(border_value[0]),
            saturate_cast<uint8_t>(border_value[1]),
            saturate_cast<uint8_t>(border_value[2]),
            saturate_cast<uint8_t>(border_value[3]),
        };
        int border_type_x = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srccols <= 1 ? BORDER_REPLICATE : border_type;
        int border_type_y = border_type != BORDER_CONSTANT &&
                            border_type != BORDER_TRANSPARENT &&
                            srcrows <= 1 ? BORDER_REPLICATE : border_type;

        constexpr int max_vlanes_32{VTraits<v_float32>::max_nlanes};
        constexpr int max_uf{max_vlanes_32*2};
        int vlanes_32 = VTraits<v_float32>::vlanes();
        // unrolling_factor = lane_size / 16 = vlanes_32 * 32 / 16 = vlanes_32 * 2
        int uf = vlanes_32 * 2;

        std::array<float, max_vlanes_32> start_indices;
        std::iota(start_indices.data(), start_indices.data() + max_vlanes_32, 0.f);

        v_uint32 inner_srows = vx_setall_u32((unsigned)std::max(srcrows - 2, 0)),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint8_t pixbuf[max_uf*4*4];

        uint8_t bvalbuf[max_uf*4];
        for (int i = 0; i < uf; i++) {
            bvalbuf[i*4] = bval[0];
            bvalbuf[i*4+1] = bval[1];
            bvalbuf[i*4+2] = bval[2];
            bvalbuf[i*4+3] = bval[3];
        }
        v_uint8 bval_v0 = vx_load_low(&bvalbuf[0]);
        v_uint8 bval_v1 = vx_load_low(&bvalbuf[uf]);
        v_uint8 bval_v2 = vx_load_low(&bvalbuf[uf*2]);
        v_uint8 bval_v3 = vx_load_low(&bvalbuf[uf*3]);
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, 8U);
                    CV_WARP_VECTOR_LINEAR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map1 == map2) {
                    sx = sx_data[2*x];
                    sy = sy_data[2*x+1];
                } else {
                    sx = sx_data[x];
                    sy = sy_data[x];
                }

                if (relative) {
                    sx += x;
                    sy += y;
                }

                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, 8U);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
#else
    remapLinearInvoker_8UC4(src_data, src_step, src_rows, src_cols,
                            dst_data, dst_step, dst_rows, dst_cols,
                            border_type, border_value,
                            map1_data, map1_step, map2_data, map2_step, is_relative);
#endif
}


namespace {

static CV_ALWAYS_INLINE void
bicubicWeights(float alpha, float A, float& w0, float& w1, float& w2, float& w3)
{
    float a2 = alpha*alpha;
    float b = 1.f - alpha;
    float b2 = b*b;

    w0 = A*alpha*b2;
    w3 = A*a2*b;
    w1 = a2*((A + 2.f)*alpha - (A + 3.f)) + 1.f;
    w2 = 1.f - w0 - w1 - w3;
}

static CV_ALWAYS_INLINE int
bicubicCoeffs(float xs0, float ys0,
              size_t srcstep, Size size, int bpp, float A,
              int bordertype, int& tl_x, int& tl_y, int& tl_ofs,
              float& wx0, float& wx1, float& wx2, float& wx3,
              float& wy0, float& wy1, float& wy2, float& wy3)
{
    constexpr int MIN_SIZE = 16;
    int width = size.width, height = size.height;
    int bigwidth = std::max(width, MIN_SIZE);
    int bigheight = std::max(height, MIN_SIZE);
    float minx = float(-bigwidth), maxx = float(bigwidth*2);
    float miny = float(-bigheight), maxy = float(bigheight*2);

    // clamp coordinates in floating-point to try to avoid unpredictable
    // behavior if the floating-point coordinates are huge
    float vx0 = std::clamp(xs0, minx, maxx);
    float vy0 = std::clamp(ys0, miny, maxy);

    int ix0 = cvFloor(vx0);
    int iy0 = cvFloor(vy0);
    float alpha = vx0 - ix0;
    float beta = vy0 - iy0;

    ix0--;
    iy0--;

    int all_outliers = int((unsigned)(ix0 + 4) >= (unsigned)(width + 4)) |
                       int((unsigned)(iy0 + 4) >= (unsigned)(height + 4));
    if (all_outliers && (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT))
        return -1;

    int all_inliers = int((unsigned)ix0 < (unsigned)std::max(width - 3, 0)) &
                      int((unsigned)iy0 < (unsigned)std::max(height - 3, 0));

    tl_x = ix0;
    tl_y = iy0;
    tl_ofs = iy0*(int)srcstep + ix0*bpp;

    bicubicWeights(alpha, A, wx0, wx1, wx2, wx3);
    bicubicWeights(beta, A, wy0, wy1, wy2, wy3);

    return all_inliers;
}

template<typename _Tp, typename _Fp>
static void bicubicFetchPixels(const _Tp* src, size_t srcstep, Size size, int cn,
                               const int32_t* tl_x, const int32_t* tl_y,
                               int32_t* goodx, int row, _Fp* pixbuf, int len,
                               int borderType, const _Tp* defVal)
{
    int width = size.width, height = size.height;
    srcstep /= sizeof(_Tp);
    if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT) {
        _Tp defR = defVal[0], defG = cn > 1 ? defVal[1] : _Tp();
        _Tp defB = cn > 2 ? defVal[2] : _Tp(), defA = cn > 3 ? defVal[3] : _Tp();
        for (int i = 0; i < len; i++) {
            int x0 = tl_x[i], y0 = tl_y[i] + row;
            int x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
            int my = int((unsigned)y0 < (unsigned)height);
            int mx0 = int((unsigned)x0 < (unsigned)width) & my;
            int mx1 = int((unsigned)x1 < (unsigned)width) & my;
            int mx2 = int((unsigned)x2 < (unsigned)width) & my;
            int mx3 = int((unsigned)x3 < (unsigned)width) & my;
            y0 = std::clamp(y0, 0, height-1);
            x0 = std::clamp(x0, 0, width-1)*cn;
            x1 = std::clamp(x1, 0, width-1)*cn;
            x2 = std::clamp(x2, 0, width-1)*cn;
            x3 = std::clamp(x3, 0, width-1)*cn;
            const _Tp* srcrow = src + srcstep*y0;
            pixbuf[i] = (_Fp)(srcrow[x0]*mx0 + defR*(1 - mx0));
            pixbuf[i + len] = (_Fp)(srcrow[x1]*mx1 + defR*(1 - mx1));
            pixbuf[i + len*2] = (_Fp)(srcrow[x2]*mx2 + defR*(1 - mx2));
            pixbuf[i + len*3] = (_Fp)(srcrow[x3]*mx3 + defR*(1 - mx3));
            if (cn > 1) {
                pixbuf[i + len*4] = (_Fp)(srcrow[x0 + 1]*mx0 + defG*(1 - mx0));
                pixbuf[i + len*5] = (_Fp)(srcrow[x1 + 1]*mx1 + defG*(1 - mx1));
                pixbuf[i + len*6] = (_Fp)(srcrow[x2 + 1]*mx2 + defG*(1 - mx2));
                pixbuf[i + len*7] = (_Fp)(srcrow[x3 + 1]*mx3 + defG*(1 - mx3));
                if (cn > 2) {
                    pixbuf[i + len*8] = (_Fp)(srcrow[x0 + 2]*mx0 + defB*(1 - mx0));
                    pixbuf[i + len*9] = (_Fp)(srcrow[x1 + 2]*mx1 + defB*(1 - mx1));
                    pixbuf[i + len*10] = (_Fp)(srcrow[x2 + 2]*mx2 + defB*(1 - mx2));
                    pixbuf[i + len*11] = (_Fp)(srcrow[x3 + 2]*mx3 + defB*(1 - mx3));
                    if (cn > 3) {
                        pixbuf[i + len*12] = (_Fp)(srcrow[x0 + 3]*mx0 + defA*(1 - mx0));
                        pixbuf[i + len*13] = (_Fp)(srcrow[x1 + 3]*mx1 + defA*(1 - mx1));
                        pixbuf[i + len*14] = (_Fp)(srcrow[x2 + 3]*mx2 + defA*(1 - mx2));
                        pixbuf[i + len*15] = (_Fp)(srcrow[x3 + 3]*mx3 + defA*(1 - mx3));
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < len; i++) {
            int y0 = borderInterpolate_fast(tl_y[i] + row, height, borderType);
            int x0, x1, x2, x3;
            if (row == 0) {
                int x0_ = tl_x[i];
                if ((unsigned)x0_ < (unsigned)std::max(width - 3, 0)) {
                    x0 = x0_*cn;
                    x1 = (x0_ + 1)*cn;
                    x2 = (x0_ + 2)*cn;
                    x3 = (x0_ + 3)*cn;
                } else {
                    x0 = borderInterpolate_fast(x0_, width, borderType)*cn;
                    x1 = borderInterpolate_fast(x0_ + 1, width, borderType)*cn;
                    x2 = borderInterpolate_fast(x0_ + 2, width, borderType)*cn;
                    x3 = borderInterpolate_fast(x0_ + 3, width, borderType)*cn;
                }
                goodx[i*4] = x0;
                goodx[i*4 + 1] = x1;
                goodx[i*4 + 2] = x2;
                goodx[i*4 + 3] = x3;
            } else {
                x0 = goodx[i*4];
                x1 = goodx[i*4 + 1];
                x2 = goodx[i*4 + 2];
                x3 = goodx[i*4 + 3];
            }
            const _Tp* srcrow = src + srcstep*y0;
            pixbuf[i] = (_Fp)srcrow[x0];
            pixbuf[i + len] = (_Fp)srcrow[x1];
            pixbuf[i + len*2] = (_Fp)srcrow[x2];
            pixbuf[i + len*3] = (_Fp)srcrow[x3];
            if (cn > 1) {
                pixbuf[i + len*4] = (_Fp)srcrow[x0 + 1];
                pixbuf[i + len*5] = (_Fp)srcrow[x1 + 1];
                pixbuf[i + len*6] = (_Fp)srcrow[x2 + 1];
                pixbuf[i + len*7] = (_Fp)srcrow[x3 + 1];
                if (cn > 2) {
                    pixbuf[i + len*8] = (_Fp)srcrow[x0 + 2];
                    pixbuf[i + len*9] = (_Fp)srcrow[x1 + 2];
                    pixbuf[i + len*10] = (_Fp)srcrow[x2 + 2];
                    pixbuf[i + len*11] = (_Fp)srcrow[x3 + 2];
                    if (cn > 3) {
                        pixbuf[i + len*12] = (_Fp)srcrow[x0 + 3];
                        pixbuf[i + len*13] = (_Fp)srcrow[x1 + 3];
                        pixbuf[i + len*14] = (_Fp)srcrow[x2 + 3];
                        pixbuf[i + len*15] = (_Fp)srcrow[x3 + 3];
                    }
                }
            }
        }
    }
}

#undef BICUBIC_UPDATE_ACC
#define BICUBIC_UPDATE_ACC(acc, v0, v1, v2, v3, wy) \
    acc += (v0*wx0 + v1*wx1 + v2*wx2 + v3*wx3)*wy

#undef BICUBIC_UPDATE_ACC_VEC
#define BICUBIC_UPDATE_ACC_VEC(acc, v0, v1, v2, v3, wy) \
    sumwx = v_fma(v1, wx1, v_mul(v0, wx0)); \
    sumwx = v_fma(v2, wx2, sumwx); \
    sumwx = v_fma(v3, wx3, sumwx); \
    acc = v_fma(sumwx, wy, acc)

#undef BICUBIC_PROCESS_INLIERS_C1_SCALAR
#define BICUBIC_PROCESS_INLIERS_C1_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = float(srcrow[0]); \
    V1 = float(srcrow[1]); \
    V2 = float(srcrow[2]); \
    V3 = float(srcrow[3]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C2_SCALAR
#define BICUBIC_PROCESS_INLIERS_C2_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = float(srcrow[0]); \
    V1 = float(srcrow[2]); \
    V2 = float(srcrow[4]); \
    V3 = float(srcrow[6]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[1]); \
    V1 = float(srcrow[3]); \
    V2 = float(srcrow[5]); \
    V3 = float(srcrow[7]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C3_SCALAR
#define BICUBIC_PROCESS_INLIERS_C3_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = float(srcrow[0]); \
    V1 = float(srcrow[3]); \
    V2 = float(srcrow[6]); \
    V3 = float(srcrow[9]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[1]); \
    V1 = float(srcrow[4]); \
    V2 = float(srcrow[7]); \
    V3 = float(srcrow[10]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[2]); \
    V1 = float(srcrow[5]); \
    V2 = float(srcrow[8]); \
    V3 = float(srcrow[11]); \
    BICUBIC_UPDATE_ACC(acc_b, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C4_SCALAR
#define BICUBIC_PROCESS_INLIERS_C4_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = float(srcrow[0]); \
    V1 = float(srcrow[4]); \
    V2 = float(srcrow[8]); \
    V3 = float(srcrow[12]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[1]); \
    V1 = float(srcrow[5]); \
    V2 = float(srcrow[9]); \
    V3 = float(srcrow[13]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[2]); \
    V1 = float(srcrow[6]); \
    V2 = float(srcrow[10]); \
    V3 = float(srcrow[14]); \
    BICUBIC_UPDATE_ACC(acc_b, V0, V1, V2, V3, wy##row); \
    V0 = float(srcrow[3]); \
    V1 = float(srcrow[7]); \
    V2 = float(srcrow[11]); \
    V3 = float(srcrow[15]); \
    BICUBIC_UPDATE_ACC(acc_a, V0, V1, V2, V3, wy##row)

template<typename chtype, int NCHANNELS>
static void bicubicRef(const float* srcx, const float* srcy, int len,
                       const void* src, size_t srcstep, Size size,
                       chtype* dst, const float* params, int borderType, chtype* borderVal)
{
    constexpr float defaultA = -0.75f;
    float A = params ? *params : defaultA;

    constexpr int BPP = int(NCHANNELS*sizeof(dst[0]));
    using buftype = std::conditional_t<std::is_same_v<chtype, float>, float, int>;
    using pixtype = std::conditional_t<NCHANNELS == 1, chtype, Vec<chtype, NCHANNELS>>;

    pixtype bval = borderVal ? *(pixtype*)borderVal : pixtype();

    for (int i = 0; i < len; i++, dst += NCHANNELS) {
        buftype pixbuf[NCHANNELS][4];
        int tl_x, tl_y, tl_ofs, goodx[4];
        float V0, V1, V2, V3;
        float wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3;
        float xs = srcx[i], ys = srcy[i];

        int code = bicubicCoeffs(xs, ys, srcstep, size, BPP, A,
                                 borderType, tl_x, tl_y, tl_ofs,
                                 wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3);
        if (code > 0) {
            const chtype* srcrow;
            if constexpr (NCHANNELS == 1) {
                float acc_r = 0.f;
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
            } else if constexpr (NCHANNELS == 2) {
                float acc_r = 0.f, acc_g = 0.f;
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
                dst[1] = saturate_cast<chtype>(acc_g);
            } else if constexpr (NCHANNELS == 3) {
                float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f;
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
                dst[1] = saturate_cast<chtype>(acc_g);
                dst[2] = saturate_cast<chtype>(acc_b);
            } else if constexpr (NCHANNELS == 4) {
                float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f, acc_a = 0.f;
                BICUBIC_PROCESS_INLIERS_C4_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C4_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C4_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C4_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
                dst[1] = saturate_cast<chtype>(acc_g);
                dst[2] = saturate_cast<chtype>(acc_b);
                dst[3] = saturate_cast<chtype>(acc_a);
            }
        } else if (code < 0) {
            if (borderType == BORDER_CONSTANT) {
                *((pixtype*)dst) = bval;
            }
            continue;
        } else {
            float acc[NCHANNELS] = {};
            float wys[] = {wy0, wy1, wy2, wy3};
            const chtype* defVal = borderType == BORDER_TRANSPARENT ? dst : borderVal;
            for (int row = 0; row < 4; row++) {
                bicubicFetchPixels((const chtype*)src, srcstep, size, NCHANNELS, &tl_x, &tl_y,
                                   goodx, row, &pixbuf[0][0], 1, borderType, defVal);
                float wy = wys[row];
                for (int c = 0; c < NCHANNELS; c++) {
                    V0 = float(pixbuf[c][0]);
                    V1 = float(pixbuf[c][1]);
                    V2 = float(pixbuf[c][2]);
                    V3 = float(pixbuf[c][3]);
                    BICUBIC_UPDATE_ACC(acc[c], V0, V1, V2, V3, wy);
                }
            }
            for (int c = 0; c < NCHANNELS; c++) {
                dst[c] = saturate_cast<chtype>(acc[c]);
            }
        }
    }
}

#if CV_SIMD

CV_ALWAYS_INLINE void
bicubicWeights(const v_float32& alpha, float A,
               v_float32& w0, v_float32& w1,
               v_float32& w2, v_float32& w3)
{
    const v_float32 vA   = vx_setall_f32(A);
    const v_float32 vAp2 = vx_setall_f32(A + 2.0f);
    const v_float32 vAp3 = vx_setall_f32(-(A + 3.0f));
    const v_float32 v1   = vx_setall_f32(1.0f);

    const v_float32 a2 = v_mul(alpha, alpha);            // α²
    const v_float32 b  = v_sub(v1, alpha);               // b = 1-α
    const v_float32 b2 = v_mul(b, b);                    // b²

    w0 = v_mul(vA, v_mul(alpha, b2));                    // A·α·b²
    w3 = v_mul(vA, v_mul(a2, b));                        // A·α²·b
    w1 = v_fma(a2, v_fma(vAp2, alpha, vAp3), v1);       // a²·((A+2)α-(A+3))+1
    w2 = v_sub(v_sub(v_sub(v1, w0), w1), w3);
}

static CV_ALWAYS_INLINE int
bicubicCoeffs(const float* srcx, const float* srcy,
              size_t srcstep, Size size, int bpp, float A,
              int bordertype, int32_t* tl_x, int32_t* tl_y, int32_t* tl_ofs,
              v_float32& wx0, v_float32& wx1, v_float32& wx2, v_float32& wx3,
              v_float32& wy0, v_float32& wy1, v_float32& wy2, v_float32& wy3)
{
    constexpr int MIN_SIZE = 16;
    int width = size.width, height = size.height;
    int bigwidth = std::max(width, MIN_SIZE);
    int bigheight = std::max(height, MIN_SIZE);
    v_float32 minx = vx_setall_f32(float(-bigwidth)), maxx = vx_setall_f32(float(bigwidth*2));
    v_float32 miny = vx_setall_f32(float(-bigheight)), maxy = vx_setall_f32(float(bigheight*2));

    // clamp coordinates in floating-point to try to avoid unpredictable
    // behavior if the floating-point coordinates are huge
    v_float32 xs0 = vx_load(srcx), ys0 = vx_load(srcy);
    v_float32 vx0 = v_min(v_max(xs0, minx), maxx);
    v_float32 vy0 = v_min(v_max(ys0, miny), maxy);

    v_int32 ix0 = v_floor(vx0);
    v_int32 iy0 = v_floor(vy0);
    v_float32 alpha = v_sub(vx0, v_cvt_f32(ix0));
    v_float32 beta = v_sub(vy0, v_cvt_f32(iy0));

    v_int32 one_i = vx_setall_s32(1), four_i = vx_setall_s32(4);
    ix0 = v_sub(ix0, one_i);
    iy0 = v_sub(iy0, one_i);

    v_uint32 width_outer = vx_setall_u32((uint32_t)(width + 4));
    v_uint32 height_outer = vx_setall_u32((uint32_t)(height + 4));

    v_uint32 outliers_mask = v_or(v_ge(v_reinterpret_as_u32(v_add(ix0, four_i)), width_outer),
                                  v_ge(v_reinterpret_as_u32(v_add(iy0, four_i)), height_outer));

    bool all_outliers = v_check_all(outliers_mask);
    if (all_outliers && (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT))
        return -1;

    v_uint32 width_inner = vx_setall_u32((uint32_t)std::max(width - 3, 0));
    v_uint32 height_inner = vx_setall_u32((uint32_t)std::max(height - 5, 0));

    v_uint32 inliers_mask = v_and(v_lt(v_reinterpret_as_u32(ix0), width_inner),
                                  v_lt(v_reinterpret_as_u32(iy0), height_inner));

    bool all_inliers = v_check_all(inliers_mask);
    v_int32 tl_ofs0 = v_add(v_mul(iy0, vx_setall_s32((int)srcstep)), v_mul(ix0, vx_setall_s32(bpp)));

    v_store(tl_x, ix0);
    v_store(tl_y, iy0);
    v_store(tl_ofs, tl_ofs0);

    bicubicWeights(alpha, A, wx0, wx1, wx2, wx3);
    bicubicWeights(beta, A, wy0, wy1, wy2, wy3);

    return int(all_inliers);
}

#if CV_SIMD_FP16
CV_ALWAYS_INLINE void
bicubicWeights(const v_float16& alpha, float A,
               v_float16& w0, v_float16& w1,
               v_float16& w2, v_float16& w3)
{
    const v_float16 vA   = vx_setall_f16(hfloat(A));
    const v_float16 vAp2 = vx_setall_f16(hfloat(A + 2.0f));
    const v_float16 vAp3 = vx_setall_f16(hfloat(-(A + 3.0f)));
    const v_float16 v1   = vx_setall_f16(hfloat(1.0f));

    const v_float16 a2 = v_mul(alpha, alpha);            // α²
    const v_float16 b  = v_sub(v1, alpha);               // b = 1-α
    const v_float16 b2 = v_mul(b, b);                    // b²

    w0 = v_mul(vA, v_mul(alpha, b2));                    // A·α·b²
    w3 = v_mul(vA, v_mul(a2, b));                        // A·α²·b
    w1 = v_fma(a2, v_fma(vAp2, alpha, vAp3), v1);       // a²·((A+2)α-(A+3))+1
    w2 = v_sub(v_sub(v_sub(v1, w0), w1), w3);
}

static CV_ALWAYS_INLINE int
bicubicCoeffs(const float* srcx, const float* srcy,
              size_t srcstep, Size size, int bpp, float A,
              int bordertype, int32_t* tl_x, int32_t* tl_y, int32_t* tl_ofs,
              v_float16& wx0, v_float16& wx1, v_float16& wx2, v_float16& wx3,
              v_float16& wy0, v_float16& wy1, v_float16& wy2, v_float16& wy3)
{
    constexpr int nlanes32 = VTraits<v_float32>::nlanes;
    constexpr int MIN_SIZE = 16;
    int width = size.width, height = size.height;
    int bigwidth = std::max(width, MIN_SIZE);
    int bigheight = std::max(height, MIN_SIZE);
    v_float32 minx = vx_setall_f32(float(-bigwidth)), maxx = vx_setall_f32(float(bigwidth*2));
    v_float32 miny = vx_setall_f32(float(-bigheight)), maxy = vx_setall_f32(float(bigheight*2));

    // clamp coordinates in floating-point to try to avoid unpredictable
    // behavior if the floating-point coordinates are huge
    v_float32 xs0 = vx_load(srcx), ys0 = vx_load(srcy);
    v_float32 xs1 = vx_load(srcx + nlanes32), ys1 = vx_load(srcy + nlanes32);
    v_float32 vx0 = v_min(v_max(xs0, minx), maxx);
    v_float32 vy0 = v_min(v_max(ys0, miny), maxy);
    v_float32 vx1 = v_min(v_max(xs1, minx), maxx);
    v_float32 vy1 = v_min(v_max(ys1, miny), maxy);

    v_int32 ix0 = v_floor(vx0);
    v_int32 iy0 = v_floor(vy0);
    v_int32 ix1 = v_floor(vx1);
    v_int32 iy1 = v_floor(vy1);
    v_float32 alpha0 = v_sub(vx0, v_cvt_f32(ix0));
    v_float32 beta0 = v_sub(vy0, v_cvt_f32(iy0));
    v_float32 alpha1 = v_sub(vx1, v_cvt_f32(ix1));
    v_float32 beta1 = v_sub(vy1, v_cvt_f32(iy1));

    hfloat abuf[nlanes32*2], bbuf[nlanes32*2];

    v_pack_store(abuf, alpha0);
    v_pack_store(abuf + nlanes32, alpha1);
    v_pack_store(bbuf, beta0);
    v_pack_store(bbuf + nlanes32, beta1);

    v_float16 alpha = vx_load(abuf);
    v_float16 beta = vx_load(bbuf);

    v_int32 one_i = vx_setall_s32(1), four_i = vx_setall_s32(4);
    ix0 = v_sub(ix0, one_i);
    iy0 = v_sub(iy0, one_i);
    ix1 = v_sub(ix1, one_i);
    iy1 = v_sub(iy1, one_i);

    v_uint32 width_outer = vx_setall_u32((uint32_t)(width + 4));
    v_uint32 height_outer = vx_setall_u32((uint32_t)(height + 4));

    v_uint32 outliers_mask0 = v_or(v_ge(v_reinterpret_as_u32(v_add(ix0, four_i)), width_outer),
                                  v_ge(v_reinterpret_as_u32(v_add(iy0, four_i)), height_outer));
    v_uint32 outliers_mask1 = v_or(v_ge(v_reinterpret_as_u32(v_add(ix1, four_i)), width_outer),
                                  v_ge(v_reinterpret_as_u32(v_add(iy1, four_i)), height_outer));

    bool all_outliers = v_check_all(v_and(outliers_mask0, outliers_mask1));
    if (all_outliers && (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT))
        return -1;

    v_uint32 width_inner = vx_setall_u32((uint32_t)std::max(width - 3, 0));
    v_uint32 height_inner = vx_setall_u32((uint32_t)std::max(height - 5, 0));

    v_uint32 inliers_mask0 = v_and(v_lt(v_reinterpret_as_u32(ix0), width_inner),
                                   v_lt(v_reinterpret_as_u32(iy0), height_inner));
    v_uint32 inliers_mask1 = v_and(v_lt(v_reinterpret_as_u32(ix1), width_inner),
                                   v_lt(v_reinterpret_as_u32(iy1), height_inner));

    bool all_inliers = v_check_all(v_and(inliers_mask0, inliers_mask1));
    v_int32 tl_ofs0 = v_add(v_mul(iy0, vx_setall_s32((int)srcstep)), v_mul(ix0, vx_setall_s32(bpp)));
    v_int32 tl_ofs1 = v_add(v_mul(iy1, vx_setall_s32((int)srcstep)), v_mul(ix1, vx_setall_s32(bpp)));

    v_store(tl_x, ix0);
    v_store(tl_y, iy0);
    v_store(tl_ofs, tl_ofs0);
    v_store(tl_x + nlanes32, ix1);
    v_store(tl_y + nlanes32, iy1);
    v_store(tl_ofs + nlanes32, tl_ofs1);

    bicubicWeights(alpha, A, wx0, wx1, wx2, wx3);
    bicubicWeights(beta, A, wy0, wy1, wy2, wy3);

    return int(all_inliers);
}

CV_ALWAYS_INLINE v_float16 v_mul(const v_int16& a, const v_float16& b)
{
    return v_mul(v_cvt_f16(a), b);
}

CV_ALWAYS_INLINE v_float16 v_fma(const v_int16& a, const v_float16& b, const v_float16& c)
{
    return v_fma(v_cvt_f16(a), b, c);
}

#endif

CV_ALWAYS_INLINE v_float32 v_mul(const v_int32& a, const v_float32& b)
{
    return v_mul(v_cvt_f32(a), b);
}

CV_ALWAYS_INLINE v_float32 v_fma(const v_int32& a, const v_float32& b, const v_float32& c)
{
    return v_fma(v_cvt_f32(a), b, c);
}

#undef FETCH_INLIERS_C1_DEFAULT
#define FETCH_INLIERS_C1_DEFAULT(row) \
    for (int j = 0; j < BATCH; j++) { \
        const chtype* srcj = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs[j]); \
        pixbuf[0][j] = buftype(srcj[0]); \
        pixbuf[0][j + BATCH] = buftype(srcj[1]); \
        pixbuf[0][j + BATCH*2] = buftype(srcj[2]); \
        pixbuf[0][j + BATCH*3] = buftype(srcj[3]); \
    } \
    R0 = vx_load(&pixbuf[0][0]); \
    R1 = vx_load(&pixbuf[0][BATCH]); \
    R2 = vx_load(&pixbuf[0][BATCH*2]); \
    R3 = vx_load(&pixbuf[0][BATCH*3])

#undef FETCH_INLIERS_C3_DEFAULT
#define FETCH_INLIERS_C3_DEFAULT(row) \
    for (int j = 0; j < BATCH; j++) { \
        const chtype* srcj = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs[j]); \
        pixbuf[0][j] = buftype(srcj[0]); \
        pixbuf[0][j + BATCH] = buftype(srcj[3]); \
        pixbuf[0][j + BATCH*2] = buftype(srcj[6]); \
        pixbuf[0][j + BATCH*3] = buftype(srcj[9]); \
        pixbuf[1][j] = buftype(srcj[1]); \
        pixbuf[1][j + BATCH] = buftype(srcj[4]); \
        pixbuf[1][j + BATCH*2] = buftype(srcj[7]); \
        pixbuf[1][j + BATCH*3] = buftype(srcj[10]); \
        pixbuf[2][j] = buftype(srcj[2]); \
        pixbuf[2][j + BATCH] = buftype(srcj[5]); \
        pixbuf[2][j + BATCH*2] = buftype(srcj[8]); \
        pixbuf[2][j + BATCH*3] = buftype(srcj[11]); \
    } \
    R0 = vx_load(&pixbuf[0][0]); \
    R1 = vx_load(&pixbuf[0][BATCH]); \
    R2 = vx_load(&pixbuf[0][BATCH*2]); \
    R3 = vx_load(&pixbuf[0][BATCH*3]); \
    G0 = vx_load(&pixbuf[1][0]); \
    G1 = vx_load(&pixbuf[1][BATCH]); \
    G2 = vx_load(&pixbuf[1][BATCH*2]); \
    G3 = vx_load(&pixbuf[1][BATCH*3]); \
    B0 = vx_load(&pixbuf[2][0]); \
    B1 = vx_load(&pixbuf[2][BATCH]); \
    B2 = vx_load(&pixbuf[2][BATCH*2]); \
    B3 = vx_load(&pixbuf[2][BATCH*3])

#undef FETCH_INLIERS_C4_DEFAULT
#define FETCH_INLIERS_C4_DEFAULT(row) \
    for (int j = 0; j < BATCH; j++) { \
        const chtype* srcj = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs[j]); \
        pixbuf[0][j] = buftype(srcj[0]); \
        pixbuf[0][j + BATCH] = buftype(srcj[4]); \
        pixbuf[0][j + BATCH*2] = buftype(srcj[8]); \
        pixbuf[0][j + BATCH*3] = buftype(srcj[12]); \
        pixbuf[1][j] = buftype(srcj[1]); \
        pixbuf[1][j + BATCH] = buftype(srcj[5]); \
        pixbuf[1][j + BATCH*2] = buftype(srcj[9]); \
        pixbuf[1][j + BATCH*3] = buftype(srcj[13]); \
        pixbuf[2][j] = buftype(srcj[2]); \
        pixbuf[2][j + BATCH] = buftype(srcj[6]); \
        pixbuf[2][j + BATCH*2] = buftype(srcj[10]); \
        pixbuf[2][j + BATCH*3] = buftype(srcj[14]); \
        pixbuf[3][j] = buftype(srcj[3]); \
        pixbuf[3][j + BATCH] = buftype(srcj[7]); \
        pixbuf[3][j + BATCH*2] = buftype(srcj[11]); \
        pixbuf[3][j + BATCH*3] = buftype(srcj[15]); \
    } \
    R0 = vx_load(&pixbuf[0][0]); \
    R1 = vx_load(&pixbuf[0][BATCH]); \
    R2 = vx_load(&pixbuf[0][BATCH*2]); \
    R3 = vx_load(&pixbuf[0][BATCH*3]); \
    G0 = vx_load(&pixbuf[1][0]); \
    G1 = vx_load(&pixbuf[1][BATCH]); \
    G2 = vx_load(&pixbuf[1][BATCH*2]); \
    G3 = vx_load(&pixbuf[1][BATCH*3]); \
    B0 = vx_load(&pixbuf[2][0]); \
    B1 = vx_load(&pixbuf[2][BATCH]); \
    B2 = vx_load(&pixbuf[2][BATCH*2]); \
    B3 = vx_load(&pixbuf[2][BATCH*3]); \
    A0 = vx_load(&pixbuf[3][0]); \
    A1 = vx_load(&pixbuf[3][BATCH]); \
    A2 = vx_load(&pixbuf[3][BATCH*2]); \
    A3 = vx_load(&pixbuf[3][BATCH*3])

#undef FETCH_INLIERS_8UC1
#define FETCH_INLIERS_8UC1(row) \
    FETCH_INLIERS_C1_DEFAULT(row)

#undef FETCH_INLIERS_8UC3
#define FETCH_INLIERS_8UC3(row) \
    FETCH_INLIERS_C3_DEFAULT(row)

#undef FETCH_INLIERS_8UC4
#define FETCH_INLIERS_8UC4(row) \
    FETCH_INLIERS_C4_DEFAULT(row)

#undef FETCH_INLIERS_16UC1
#define FETCH_INLIERS_16UC1(row) \
    FETCH_INLIERS_C1_DEFAULT(row)

#undef FETCH_INLIERS_16UC3
#define FETCH_INLIERS_16UC3(row) \
    FETCH_INLIERS_C3_DEFAULT(row)

#undef FETCH_INLIERS_16UC4
#define FETCH_INLIERS_16UC4(row) \
    FETCH_INLIERS_C4_DEFAULT(row)

#undef FETCH_INLIERS_32FC1
#define FETCH_INLIERS_32FC1(row) \
    FETCH_INLIERS_C1_DEFAULT(row)

#undef FETCH_INLIERS_32FC3
#define FETCH_INLIERS_32FC3(row) \
    FETCH_INLIERS_C3_DEFAULT(row)

#undef FETCH_INLIERS_32FC4
#define FETCH_INLIERS_32FC4(row) \
    FETCH_INLIERS_C4_DEFAULT(row)

// NEON-optimized macros for super-fast pixels retrieval and reordering
#if defined __ARM_NEON && defined __aarch64__

#if CV_SIMD_FP16

#undef FETCH_INLIERS_8UC1
#define FETCH_INLIERS_8UC1(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x8_t _r0 = vld1_u8(srcrow + tl_ofs[0]); \
    uint8x8_t _r1 = vld1_u8(srcrow + tl_ofs[1]); \
    uint8x8_t _r2 = vld1_u8(srcrow + tl_ofs[2]); \
    uint8x8_t _r3 = vld1_u8(srcrow + tl_ofs[3]); \
    uint8x8_t _r4 = vld1_u8(srcrow + tl_ofs[4]); \
    uint8x8_t _r5 = vld1_u8(srcrow + tl_ofs[5]); \
    uint8x8_t _r6 = vld1_u8(srcrow + tl_ofs[6]); \
    uint8x8_t _r7 = vld1_u8(srcrow + tl_ofs[7]); \
    \
    uint8x16_t _r0415 = vcombine_u8(vzip1_u8(_r0, _r4), vzip1_u8(_r1, _r5)); \
    uint8x16_t _r2637 = vcombine_u8(vzip1_u8(_r2, _r6), vzip1_u8(_r3, _r7)); \
    uint8x16_t _r0246 = vzip1q_u8(_r0415, _r2637); \
    uint8x16_t _r1357 = vzip2q_u8(_r0415, _r2637); \
    uint8x16_t _r_c01 = vzip1q_u8(_r0246, _r1357); \
    uint8x16_t _r_c23 = vzip2q_u8(_r0246, _r1357); \
    \
    R0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c01))); \
    R1.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c01));         \
    R2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c23))); \
    R3.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c23)); }

#undef FETCH_INLIERS_8UC3
#define FETCH_INLIERS_8UC3(row) { \
    const uint8x16_t mask_rgb = { 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255 }; \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x16_t _r0 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[0])), mask_rgb); \
    uint8x16_t _r1 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[1])), mask_rgb); \
    uint8x16_t _r2 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[2])), mask_rgb); \
    uint8x16_t _r3 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[3])), mask_rgb); \
    uint8x16_t _r4 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[4])), mask_rgb); \
    uint8x16_t _r5 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[5])), mask_rgb); \
    uint8x16_t _r6 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[6])), mask_rgb); \
    uint8x16_t _r7 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[7])), mask_rgb); \
    \
    uint8x16_t _z04a = vzip1q_u8(_r0, _r4);        \
    uint8x16_t _z15a = vzip1q_u8(_r1, _r5);        \
    uint8x16_t _z26a = vzip1q_u8(_r2, _r6);        \
    uint8x16_t _z37a = vzip1q_u8(_r3, _r7);        \
    uint8x16_t _z04b = vzip2q_u8(_r0, _r4);        \
    uint8x16_t _z15b = vzip2q_u8(_r1, _r5);        \
    uint8x16_t _z26b = vzip2q_u8(_r2, _r6);        \
    uint8x16_t _z37b = vzip2q_u8(_r3, _r7);        \
    \
    uint8x16_t _r0246 = vzip1q_u8(_z04a, _z26a);   \
    uint8x16_t _g0246 = vzip2q_u8(_z04a, _z26a);   \
    uint8x16_t _r1357 = vzip1q_u8(_z15a, _z37a);   \
    uint8x16_t _g1357 = vzip2q_u8(_z15a, _z37a);   \
    uint8x16_t _b0246 = vzip1q_u8(_z04b, _z26b);   \
    uint8x16_t _b1357 = vzip1q_u8(_z15b, _z37b);   \
    \
    uint8x16_t _r_c01 = vzip1q_u8(_r0246, _r1357); \
    uint8x16_t _r_c23 = vzip2q_u8(_r0246, _r1357); \
    uint8x16_t _g_c01 = vzip1q_u8(_g0246, _g1357); \
    uint8x16_t _g_c23 = vzip2q_u8(_g0246, _g1357); \
    uint8x16_t _b_c01 = vzip1q_u8(_b0246, _b1357); \
    uint8x16_t _b_c23 = vzip2q_u8(_b0246, _b1357); \
    \
    R0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c01)));  \
    R1.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c01));          \
    R2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c23)));  \
    R3.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c23));          \
    G0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_g_c01)));  \
    G1.val = vreinterpretq_s16_u16(vmovl_high_u8(_g_c01));          \
    G2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_g_c23)));  \
    G3.val = vreinterpretq_s16_u16(vmovl_high_u8(_g_c23));          \
    B0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_b_c01)));  \
    B1.val = vreinterpretq_s16_u16(vmovl_high_u8(_b_c01));          \
    B2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_b_c23)));  \
    B3.val = vreinterpretq_s16_u16(vmovl_high_u8(_b_c23)); }

#undef FETCH_INLIERS_8UC4
#define FETCH_INLIERS_8UC4(row) { \
    const uint8x16_t mask_rgb = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 }; \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x16_t _r0 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[0])), mask_rgb); \
    uint8x16_t _r1 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[1])), mask_rgb); \
    uint8x16_t _r2 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[2])), mask_rgb); \
    uint8x16_t _r3 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[3])), mask_rgb); \
    uint8x16_t _r4 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[4])), mask_rgb); \
    uint8x16_t _r5 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[5])), mask_rgb); \
    uint8x16_t _r6 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[6])), mask_rgb); \
    uint8x16_t _r7 = vqtbl1q_u8(vld1q_u8((const chtype*)(srcrow + tl_ofs[7])), mask_rgb); \
    \
    uint8x16_t _z04a = vzip1q_u8(_r0, _r4);        \
    uint8x16_t _z15a = vzip1q_u8(_r1, _r5);        \
    uint8x16_t _z26a = vzip1q_u8(_r2, _r6);        \
    uint8x16_t _z37a = vzip1q_u8(_r3, _r7);        \
    uint8x16_t _z04b = vzip2q_u8(_r0, _r4);        \
    uint8x16_t _z15b = vzip2q_u8(_r1, _r5);        \
    uint8x16_t _z26b = vzip2q_u8(_r2, _r6);        \
    uint8x16_t _z37b = vzip2q_u8(_r3, _r7);        \
    \
    uint8x16_t _r0246 = vzip1q_u8(_z04a, _z26a);   \
    uint8x16_t _g0246 = vzip2q_u8(_z04a, _z26a);   \
    uint8x16_t _r1357 = vzip1q_u8(_z15a, _z37a);   \
    uint8x16_t _g1357 = vzip2q_u8(_z15a, _z37a);   \
    uint8x16_t _b0246 = vzip1q_u8(_z04b, _z26b);   \
    uint8x16_t _a0246 = vzip2q_u8(_z04b, _z26b);   \
    uint8x16_t _b1357 = vzip1q_u8(_z15b, _z37b);   \
    uint8x16_t _a1357 = vzip2q_u8(_z15b, _z37b);   \
    \
    uint8x16_t _r_c01 = vzip1q_u8(_r0246, _r1357); \
    uint8x16_t _r_c23 = vzip2q_u8(_r0246, _r1357); \
    uint8x16_t _g_c01 = vzip1q_u8(_g0246, _g1357); \
    uint8x16_t _g_c23 = vzip2q_u8(_g0246, _g1357); \
    uint8x16_t _b_c01 = vzip1q_u8(_b0246, _b1357); \
    uint8x16_t _b_c23 = vzip2q_u8(_b0246, _b1357); \
    uint8x16_t _a_c01 = vzip1q_u8(_a0246, _a1357); \
    uint8x16_t _a_c23 = vzip2q_u8(_a0246, _a1357); \
    \
    R0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c01)));  \
    R1.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c01));          \
    R2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_r_c23)));  \
    R3.val = vreinterpretq_s16_u16(vmovl_high_u8(_r_c23));          \
    G0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_g_c01)));  \
    G1.val = vreinterpretq_s16_u16(vmovl_high_u8(_g_c01));          \
    G2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_g_c23)));  \
    G3.val = vreinterpretq_s16_u16(vmovl_high_u8(_g_c23));          \
    B0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_b_c01)));  \
    B1.val = vreinterpretq_s16_u16(vmovl_high_u8(_b_c01));          \
    B2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_b_c23)));  \
    B3.val = vreinterpretq_s16_u16(vmovl_high_u8(_b_c23));          \
    A0.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_a_c01)));  \
    A1.val = vreinterpretq_s16_u16(vmovl_high_u8(_a_c01));          \
    A2.val = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(_a_c23)));  \
    A3.val = vreinterpretq_s16_u16(vmovl_high_u8(_a_c23)); }

#else
#undef FETCH_INLIERS_8UC1
#define FETCH_INLIERS_8UC1(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x8_t _r0 = vld1_u8(srcrow + tl_ofs[0]); \
    uint8x8_t _r1 = vld1_u8(srcrow + tl_ofs[1]); \
    uint8x8_t _r2 = vld1_u8(srcrow + tl_ofs[2]); \
    uint8x8_t _r3 = vld1_u8(srcrow + tl_ofs[3]); \
    uint16x4_t _rw0 = vget_low_u16(vmovl_u8(_r0)); \
    uint16x4_t _rw1 = vget_low_u16(vmovl_u8(_r1)); \
    uint16x4_t _rw2 = vget_low_u16(vmovl_u8(_r2)); \
    uint16x4_t _rw3 = vget_low_u16(vmovl_u8(_r3)); \
    \
    uint16x4_t _r02a = vzip1_u16(_rw0, _rw2);      \
    uint16x4_t _r02b = vzip2_u16(_rw0, _rw2);      \
    uint16x4_t _r13a = vzip1_u16(_rw1, _rw3);      \
    uint16x4_t _r13b = vzip2_u16(_rw1, _rw3);      \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02a, _r13a)));   \
    R1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02a, _r13a)));   \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02b, _r13b)));   \
    R3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02b, _r13b))); }

#undef FETCH_INLIERS_8UC3
#define FETCH_INLIERS_8UC3(row) { \
    const uint8x16_t mask_rgb = { 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255 }; \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x16_t _r0 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[0]), mask_rgb); \
    uint8x16_t _r1 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[1]), mask_rgb); \
    uint8x16_t _r2 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[2]), mask_rgb); \
    uint8x16_t _r3 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[3]), mask_rgb); \
    \
    uint8x16_t _z02a = vzip1q_u8(_r0, _r2);        \
    uint8x16_t _z02b = vzip2q_u8(_r0, _r2);        \
    uint8x16_t _z13a = vzip1q_u8(_r1, _r3);        \
    uint8x16_t _z13b = vzip2q_u8(_r1, _r3);        \
    \
    uint8x16_t _r_c0123 = vzip1q_u8(_z02a, _z13a); \
    uint8x16_t _g_c0123 = vzip2q_u8(_z02a, _z13a); \
    uint8x16_t _b_c0123 = vzip1q_u8(_z02b, _z13b); \
    \
    uint16x8_t _rl = vmovl_u8(vget_low_u8(_r_c0123)); \
    uint16x8_t _rh = vmovl_high_u8(_r_c0123); \
    uint16x8_t _gl = vmovl_u8(vget_low_u8(_g_c0123)); \
    uint16x8_t _gh = vmovl_high_u8(_g_c0123); \
    uint16x8_t _bl = vmovl_u8(vget_low_u8(_b_c0123)); \
    uint16x8_t _bh = vmovl_high_u8(_b_c0123); \
    \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_rl))); \
    R1.val = vreinterpretq_s32_u32(vmovl_high_u16(_rl)); \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_rh))); \
    R3.val = vreinterpretq_s32_u32(vmovl_high_u16(_rh)); \
    G0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_gl))); \
    G1.val = vreinterpretq_s32_u32(vmovl_high_u16(_gl)); \
    G2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_gh))); \
    G3.val = vreinterpretq_s32_u32(vmovl_high_u16(_gh)); \
    B0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_bl))); \
    B1.val = vreinterpretq_s32_u32(vmovl_high_u16(_bl)); \
    B2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_bh))); \
    B3.val = vreinterpretq_s32_u32(vmovl_high_u16(_bh)); }

#undef FETCH_INLIERS_8UC4
#define FETCH_INLIERS_8UC4(row) { \
    const uint8x16_t mask_rgb = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 }; \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint8x16_t _r0 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[0]), mask_rgb); \
    uint8x16_t _r1 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[1]), mask_rgb); \
    uint8x16_t _r2 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[2]), mask_rgb); \
    uint8x16_t _r3 = vqtbl1q_u8(vld1q_u8(srcrow + tl_ofs[3]), mask_rgb); \
    \
    uint8x16_t _z02a = vzip1q_u8(_r0, _r2);        \
    uint8x16_t _z02b = vzip2q_u8(_r0, _r2);        \
    uint8x16_t _z13a = vzip1q_u8(_r1, _r3);        \
    uint8x16_t _z13b = vzip2q_u8(_r1, _r3);        \
    \
    uint8x16_t _r_c0123 = vzip1q_u8(_z02a, _z13a); \
    uint8x16_t _g_c0123 = vzip2q_u8(_z02a, _z13a); \
    uint8x16_t _b_c0123 = vzip1q_u8(_z02b, _z13b); \
    uint8x16_t _a_c0123 = vzip2q_u8(_z02b, _z13b); \
    \
    uint16x8_t _rl = vmovl_u8(vget_low_u8(_r_c0123)); \
    uint16x8_t _rh = vmovl_high_u8(_r_c0123); \
    uint16x8_t _gl = vmovl_u8(vget_low_u8(_g_c0123)); \
    uint16x8_t _gh = vmovl_high_u8(_g_c0123); \
    uint16x8_t _bl = vmovl_u8(vget_low_u8(_b_c0123)); \
    uint16x8_t _bh = vmovl_high_u8(_b_c0123); \
    uint16x8_t _al = vmovl_u8(vget_low_u8(_a_c0123)); \
    uint16x8_t _ah = vmovl_high_u8(_a_c0123); \
    \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_rl))); \
    R1.val = vreinterpretq_s32_u32(vmovl_high_u16(_rl)); \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_rh))); \
    R3.val = vreinterpretq_s32_u32(vmovl_high_u16(_rh)); \
    G0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_gl))); \
    G1.val = vreinterpretq_s32_u32(vmovl_high_u16(_gl)); \
    G2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_gh))); \
    G3.val = vreinterpretq_s32_u32(vmovl_high_u16(_gh)); \
    B0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_bl))); \
    B1.val = vreinterpretq_s32_u32(vmovl_high_u16(_bl)); \
    B2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_bh))); \
    B3.val = vreinterpretq_s32_u32(vmovl_high_u16(_bh)); \
    A0.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_al))); \
    A1.val = vreinterpretq_s32_u32(vmovl_high_u16(_al)); \
    A2.val = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(_ah))); \
    A3.val = vreinterpretq_s32_u32(vmovl_high_u16(_ah)); }
#endif

#undef FETCH_INLIERS_16UC1
#define FETCH_INLIERS_16UC1(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint16x4_t _r0 = vld1_u16((const chtype*)(srcrow + tl_ofs[0])); \
    uint16x4_t _r1 = vld1_u16((const chtype*)(srcrow + tl_ofs[1])); \
    uint16x4_t _r2 = vld1_u16((const chtype*)(srcrow + tl_ofs[2])); \
    uint16x4_t _r3 = vld1_u16((const chtype*)(srcrow + tl_ofs[3])); \
    \
    uint16x4_t _r02a = vzip1_u16(_r0, _r2);       \
    uint16x4_t _r02b = vzip2_u16(_r0, _r2);       \
    uint16x4_t _r13a = vzip1_u16(_r1, _r3);       \
    uint16x4_t _r13b = vzip2_u16(_r1, _r3);       \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02a, _r13a)));   \
    R1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02a, _r13a)));   \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02b, _r13b)));   \
    R3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02b, _r13b))); }

#undef FETCH_INLIERS_16UC3
#define FETCH_INLIERS_16UC3(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint16x4x3_t _r0 = vld3_u16((const chtype*)(srcrow + tl_ofs[0])); \
    uint16x4x3_t _r1 = vld3_u16((const chtype*)(srcrow + tl_ofs[1])); \
    uint16x4x3_t _r2 = vld3_u16((const chtype*)(srcrow + tl_ofs[2])); \
    uint16x4x3_t _r3 = vld3_u16((const chtype*)(srcrow + tl_ofs[3])); \
    uint16x4_t _r02a = vzip1_u16(_r0.val[0], _r2.val[0]);   \
    uint16x4_t _r02b = vzip2_u16(_r0.val[0], _r2.val[0]);   \
    uint16x4_t _r13a = vzip1_u16(_r1.val[0], _r3.val[0]);   \
    uint16x4_t _r13b = vzip2_u16(_r1.val[0], _r3.val[0]);   \
    uint16x4_t _g02a = vzip1_u16(_r0.val[1], _r2.val[1]);   \
    uint16x4_t _g02b = vzip2_u16(_r0.val[1], _r2.val[1]);   \
    uint16x4_t _g13a = vzip1_u16(_r1.val[1], _r3.val[1]);   \
    uint16x4_t _g13b = vzip2_u16(_r1.val[1], _r3.val[1]);   \
    uint16x4_t _b02a = vzip1_u16(_r0.val[2], _r2.val[2]);   \
    uint16x4_t _b02b = vzip2_u16(_r0.val[2], _r2.val[2]);   \
    uint16x4_t _b13a = vzip1_u16(_r1.val[2], _r3.val[2]);   \
    uint16x4_t _b13b = vzip2_u16(_r1.val[2], _r3.val[2]);   \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02a, _r13a))); \
    R1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02a, _r13a))); \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02b, _r13b))); \
    R3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02b, _r13b))); \
    G0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_g02a, _g13a))); \
    G1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_g02a, _g13a))); \
    G2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_g02b, _g13b))); \
    G3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_g02b, _g13b))); \
    B0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_b02a, _b13a))); \
    B1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_b02a, _b13a))); \
    B2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_b02b, _b13b))); \
    B3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_b02b, _b13b))); }

#undef FETCH_INLIERS_16UC4
#define FETCH_INLIERS_16UC4(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    uint16x4x4_t _r0 = vld4_u16((const chtype*)(srcrow + tl_ofs[0])); \
    uint16x4x4_t _r1 = vld4_u16((const chtype*)(srcrow + tl_ofs[1])); \
    uint16x4x4_t _r2 = vld4_u16((const chtype*)(srcrow + tl_ofs[2])); \
    uint16x4x4_t _r3 = vld4_u16((const chtype*)(srcrow + tl_ofs[3])); \
    uint16x4_t _r02a = vzip1_u16(_r0.val[0], _r2.val[0]);   \
    uint16x4_t _r02b = vzip2_u16(_r0.val[0], _r2.val[0]);   \
    uint16x4_t _r13a = vzip1_u16(_r1.val[0], _r3.val[0]);   \
    uint16x4_t _r13b = vzip2_u16(_r1.val[0], _r3.val[0]);   \
    uint16x4_t _g02a = vzip1_u16(_r0.val[1], _r2.val[1]);   \
    uint16x4_t _g02b = vzip2_u16(_r0.val[1], _r2.val[1]);   \
    uint16x4_t _g13a = vzip1_u16(_r1.val[1], _r3.val[1]);   \
    uint16x4_t _g13b = vzip2_u16(_r1.val[1], _r3.val[1]);   \
    uint16x4_t _b02a = vzip1_u16(_r0.val[2], _r2.val[2]);   \
    uint16x4_t _b02b = vzip2_u16(_r0.val[2], _r2.val[2]);   \
    uint16x4_t _b13a = vzip1_u16(_r1.val[2], _r3.val[2]);   \
    uint16x4_t _b13b = vzip2_u16(_r1.val[2], _r3.val[2]);   \
    uint16x4_t _a02a = vzip1_u16(_r0.val[3], _r2.val[3]);   \
    uint16x4_t _a02b = vzip2_u16(_r0.val[3], _r2.val[3]);   \
    uint16x4_t _a13a = vzip1_u16(_r1.val[3], _r3.val[3]);   \
    uint16x4_t _a13b = vzip2_u16(_r1.val[3], _r3.val[3]);   \
    R0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02a, _r13a))); \
    R1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02a, _r13a))); \
    R2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_r02b, _r13b))); \
    R3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_r02b, _r13b))); \
    G0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_g02a, _g13a))); \
    G1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_g02a, _g13a))); \
    G2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_g02b, _g13b))); \
    G3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_g02b, _g13b))); \
    B0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_b02a, _b13a))); \
    B1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_b02a, _b13a))); \
    B2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_b02b, _b13b))); \
    B3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_b02b, _b13b))); \
    A0.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_a02a, _a13a))); \
    A1.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_a02a, _a13a))); \
    A2.val = vreinterpretq_s32_u32(vmovl_u16(vzip1_u16(_a02b, _a13b))); \
    A3.val = vreinterpretq_s32_u32(vmovl_u16(vzip2_u16(_a02b, _a13b))); }

#undef FETCH_INLIERS_32FC1
#define FETCH_INLIERS_32FC1(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    float32x4_t _r0 = vld1q_f32((const chtype*)(srcrow + tl_ofs[0])); \
    float32x4_t _r1 = vld1q_f32((const chtype*)(srcrow + tl_ofs[1])); \
    float32x4_t _r2 = vld1q_f32((const chtype*)(srcrow + tl_ofs[2])); \
    float32x4_t _r3 = vld1q_f32((const chtype*)(srcrow + tl_ofs[3])); \
    \
    float32x4_t _r02a = vzip1q_f32(_r0, _r2);       \
    float32x4_t _r02b = vzip2q_f32(_r0, _r2);       \
    float32x4_t _r13a = vzip1q_f32(_r1, _r3);       \
    float32x4_t _r13b = vzip2q_f32(_r1, _r3);       \
    R0.val = vzip1q_f32(_r02a, _r13a);   \
    R1.val = vzip2q_f32(_r02a, _r13a);   \
    R2.val = vzip1q_f32(_r02b, _r13b);   \
    R3.val = vzip2q_f32(_r02b, _r13b); }

#undef FETCH_INLIERS_32FC3
#define FETCH_INLIERS_32FC3(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    float32x4x3_t _r0 = vld3q_f32((const chtype*)(srcrow + tl_ofs[0])); \
    float32x4x3_t _r1 = vld3q_f32((const chtype*)(srcrow + tl_ofs[1])); \
    float32x4x3_t _r2 = vld3q_f32((const chtype*)(srcrow + tl_ofs[2])); \
    float32x4x3_t _r3 = vld3q_f32((const chtype*)(srcrow + tl_ofs[3])); \
    float32x4_t _r02a = vzip1q_f32(_r0.val[0], _r2.val[0]);   \
    float32x4_t _r02b = vzip2q_f32(_r0.val[0], _r2.val[0]);   \
    float32x4_t _r13a = vzip1q_f32(_r1.val[0], _r3.val[0]);   \
    float32x4_t _r13b = vzip2q_f32(_r1.val[0], _r3.val[0]);   \
    float32x4_t _g02a = vzip1q_f32(_r0.val[1], _r2.val[1]);   \
    float32x4_t _g02b = vzip2q_f32(_r0.val[1], _r2.val[1]);   \
    float32x4_t _g13a = vzip1q_f32(_r1.val[1], _r3.val[1]);   \
    float32x4_t _g13b = vzip2q_f32(_r1.val[1], _r3.val[1]);   \
    float32x4_t _b02a = vzip1q_f32(_r0.val[2], _r2.val[2]);   \
    float32x4_t _b02b = vzip2q_f32(_r0.val[2], _r2.val[2]);   \
    float32x4_t _b13a = vzip1q_f32(_r1.val[2], _r3.val[2]);   \
    float32x4_t _b13b = vzip2q_f32(_r1.val[2], _r3.val[2]);   \
    R0.val = vzip1q_f32(_r02a, _r13a);             \
    R1.val = vzip2q_f32(_r02a, _r13a);             \
    R2.val = vzip1q_f32(_r02b, _r13b);             \
    R3.val = vzip2q_f32(_r02b, _r13b);             \
    G0.val = vzip1q_f32(_g02a, _g13a);             \
    G1.val = vzip2q_f32(_g02a, _g13a);             \
    G2.val = vzip1q_f32(_g02b, _g13b);             \
    G3.val = vzip2q_f32(_g02b, _g13b);             \
    B0.val = vzip1q_f32(_b02a, _b13a);             \
    B1.val = vzip2q_f32(_b02a, _b13a);             \
    B2.val = vzip1q_f32(_b02b, _b13b);             \
    B3.val = vzip2q_f32(_b02b, _b13b); }

#undef FETCH_INLIERS_32FC4
#define FETCH_INLIERS_32FC4(row) { \
    const uint8_t* srcrow = (const uint8_t*)src + row*srcstep; \
    float32x4x4_t _r0 = vld4q_f32((const chtype*)(srcrow + tl_ofs[0])); \
    float32x4x4_t _r1 = vld4q_f32((const chtype*)(srcrow + tl_ofs[1])); \
    float32x4x4_t _r2 = vld4q_f32((const chtype*)(srcrow + tl_ofs[2])); \
    float32x4x4_t _r3 = vld4q_f32((const chtype*)(srcrow + tl_ofs[3])); \
    float32x4_t _r02a = vzip1q_f32(_r0.val[0], _r2.val[0]);   \
    float32x4_t _r02b = vzip2q_f32(_r0.val[0], _r2.val[0]);   \
    float32x4_t _r13a = vzip1q_f32(_r1.val[0], _r3.val[0]);   \
    float32x4_t _r13b = vzip2q_f32(_r1.val[0], _r3.val[0]);   \
    float32x4_t _g02a = vzip1q_f32(_r0.val[1], _r2.val[1]);   \
    float32x4_t _g02b = vzip2q_f32(_r0.val[1], _r2.val[1]);   \
    float32x4_t _g13a = vzip1q_f32(_r1.val[1], _r3.val[1]);   \
    float32x4_t _g13b = vzip2q_f32(_r1.val[1], _r3.val[1]);   \
    float32x4_t _b02a = vzip1q_f32(_r0.val[2], _r2.val[2]);   \
    float32x4_t _b02b = vzip2q_f32(_r0.val[2], _r2.val[2]);   \
    float32x4_t _b13a = vzip1q_f32(_r1.val[2], _r3.val[2]);   \
    float32x4_t _b13b = vzip2q_f32(_r1.val[2], _r3.val[2]);   \
    float32x4_t _a02a = vzip1q_f32(_r0.val[3], _r2.val[3]);   \
    float32x4_t _a02b = vzip2q_f32(_r0.val[3], _r2.val[3]);   \
    float32x4_t _a13a = vzip1q_f32(_r1.val[3], _r3.val[3]);   \
    float32x4_t _a13b = vzip2q_f32(_r1.val[3], _r3.val[3]);   \
    R0.val = vzip1q_f32(_r02a, _r13a);  \
    R1.val = vzip2q_f32(_r02a, _r13a);  \
    R2.val = vzip1q_f32(_r02b, _r13b);  \
    R3.val = vzip2q_f32(_r02b, _r13b);  \
    G0.val = vzip1q_f32(_g02a, _g13a);  \
    G1.val = vzip2q_f32(_g02a, _g13a);  \
    G2.val = vzip1q_f32(_g02b, _g13b);  \
    G3.val = vzip2q_f32(_g02b, _g13b);  \
    B0.val = vzip1q_f32(_b02a, _b13a);  \
    B1.val = vzip2q_f32(_b02a, _b13a);  \
    B2.val = vzip1q_f32(_b02b, _b13b);  \
    B3.val = vzip2q_f32(_b02b, _b13b);  \
    A0.val = vzip1q_f32(_a02a, _a13a);  \
    A1.val = vzip2q_f32(_a02a, _a13a);  \
    A2.val = vzip1q_f32(_a02b, _a13b);  \
    A3.val = vzip2q_f32(_a02b, _a13b); }

#endif

#undef BICUBIC_C1_PROCESS_ROW
#define BICUBIC_C1_PROCESS_ROW(row, fetch_inliers) \
    fetch_inliers(row); \
    BICUBIC_UPDATE_ACC_VEC(acc_r, R0, R1, R2, R3, wy##row)

#undef BICUBIC_C3_PROCESS_ROW
#define BICUBIC_C3_PROCESS_ROW(row, fetch_inliers) \
    fetch_inliers(row); \
    BICUBIC_UPDATE_ACC_VEC(acc_r, R0, R1, R2, R3, wy##row); \
    BICUBIC_UPDATE_ACC_VEC(acc_g, G0, G1, G2, G3, wy##row); \
    BICUBIC_UPDATE_ACC_VEC(acc_b, B0, B1, B2, B3, wy##row)

#undef BICUBIC_C4_PROCESS_ROW
#define BICUBIC_C4_PROCESS_ROW(row, fetch_inliers) \
    fetch_inliers(row); \
    BICUBIC_UPDATE_ACC_VEC(acc_r, R0, R1, R2, R3, wy##row); \
    BICUBIC_UPDATE_ACC_VEC(acc_g, G0, G1, G2, G3, wy##row); \
    BICUBIC_UPDATE_ACC_VEC(acc_b, B0, B1, B2, B3, wy##row); \
    BICUBIC_UPDATE_ACC_VEC(acc_a, A0, A1, A2, A3, wy##row)

template<typename chtype, int NCHANNELS, typename vecfptype, typename vecbuftype>
static void
bicubicVec(const float* srcx, const float* srcy, int len,
           const void* src, size_t srcstep, Size size,
           chtype* dst, const float* params,
           int borderType, chtype* borderVal)
{
    constexpr float defaultA = -0.75f;
    float A = params ? *params : defaultA;

    using pixtype = std::conditional_t<NCHANNELS == 1, chtype, Vec<chtype, NCHANNELS>>;
    using buftype = typename VTraits<vecbuftype>::lane_type;

    constexpr int BPP = int(NCHANNELS*sizeof(dst[0]));
    constexpr int BATCH = VTraits<vecfptype>::nlanes;
    float xbuf[BATCH] = {}, ybuf[BATCH] = {};
    pixtype bval = borderVal ? *(pixtype*)borderVal : pixtype();

    int savestorelen = borderType == BORDER_TRANSPARENT ? 0 : len;

    vecfptype acc_r = v_setzero_<vecfptype>(), acc_g = v_setzero_<vecfptype>(),
              acc_b = v_setzero_<vecfptype>(), acc_a = v_setzero_<vecfptype>();

    for (int i = 0; i < len; i += BATCH, dst += BATCH*NCHANNELS) {
        if (i + BATCH > len && i > 0 && borderType != BORDER_TRANSPARENT) {
            int i1 = len - BATCH;
            dst -= (i - i1)*NCHANNELS;
            i = i1;
        }
        int dlen = std::min(BATCH, len - i);
        int32_t tl_x[BATCH], tl_y[BATCH], tl_ofs[BATCH], goodx[BATCH*4];
        buftype pixbuf[NCHANNELS][BATCH*4];
        vecfptype wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3, sumwx;
        const float *xptr = srcx + i, *yptr = srcy + i;
        if (dlen < BATCH) {
            memcpy(xbuf, srcx + i, dlen*sizeof(srcx[0]));
            memcpy(ybuf, srcy + i, dlen*sizeof(srcy[0]));
            xptr = xbuf;
            yptr = ybuf;
        }
        int code = bicubicCoeffs(xptr, yptr, srcstep, size, BPP, A,
                                 borderType, tl_x, tl_y, tl_ofs,
                                 wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3);

        if (code > 0) {
            if constexpr (NCHANNELS == 1) {
                vecbuftype R0, R1, R2, R3;
                acc_r = v_setzero_<vecfptype>();
                if constexpr (std::is_same_v<chtype, uint8_t>) {
                    BICUBIC_C1_PROCESS_ROW(0, FETCH_INLIERS_8UC1);
                    BICUBIC_C1_PROCESS_ROW(1, FETCH_INLIERS_8UC1);
                    BICUBIC_C1_PROCESS_ROW(2, FETCH_INLIERS_8UC1);
                    BICUBIC_C1_PROCESS_ROW(3, FETCH_INLIERS_8UC1);
                } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                    BICUBIC_C1_PROCESS_ROW(0, FETCH_INLIERS_16UC1);
                    BICUBIC_C1_PROCESS_ROW(1, FETCH_INLIERS_16UC1);
                    BICUBIC_C1_PROCESS_ROW(2, FETCH_INLIERS_16UC1);
                    BICUBIC_C1_PROCESS_ROW(3, FETCH_INLIERS_16UC1);
                } else if constexpr (std::is_same_v<chtype, float>) {
                    BICUBIC_C1_PROCESS_ROW(0, FETCH_INLIERS_32FC1);
                    BICUBIC_C1_PROCESS_ROW(1, FETCH_INLIERS_32FC1);
                    BICUBIC_C1_PROCESS_ROW(2, FETCH_INLIERS_32FC1);
                    BICUBIC_C1_PROCESS_ROW(3, FETCH_INLIERS_32FC1);
                }
            } else if constexpr (NCHANNELS == 3) {
                vecbuftype R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3;
                acc_r = acc_g = acc_b = v_setzero_<vecfptype>();
                if constexpr (std::is_same_v<chtype, uint8_t>) {
                    BICUBIC_C3_PROCESS_ROW(0, FETCH_INLIERS_8UC3);
                    BICUBIC_C3_PROCESS_ROW(1, FETCH_INLIERS_8UC3);
                    BICUBIC_C3_PROCESS_ROW(2, FETCH_INLIERS_8UC3);
                    BICUBIC_C3_PROCESS_ROW(3, FETCH_INLIERS_8UC3);
                } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                    BICUBIC_C3_PROCESS_ROW(0, FETCH_INLIERS_16UC3);
                    BICUBIC_C3_PROCESS_ROW(1, FETCH_INLIERS_16UC3);
                    BICUBIC_C3_PROCESS_ROW(2, FETCH_INLIERS_16UC3);
                    BICUBIC_C3_PROCESS_ROW(3, FETCH_INLIERS_16UC3);
                } else if constexpr (std::is_same_v<chtype, float>) {
                    BICUBIC_C3_PROCESS_ROW(0, FETCH_INLIERS_32FC3);
                    BICUBIC_C3_PROCESS_ROW(1, FETCH_INLIERS_32FC3);
                    BICUBIC_C3_PROCESS_ROW(2, FETCH_INLIERS_32FC3);
                    BICUBIC_C3_PROCESS_ROW(3, FETCH_INLIERS_32FC3);
                }
            } else if constexpr (NCHANNELS == 4) {
                vecbuftype R0, R1, R2, R3, G0, G1, G2, G3, B0, B1, B2, B3, A0, A1, A2, A3;
                acc_r = acc_g = acc_b = acc_a = v_setzero_<vecfptype>();
                if constexpr (std::is_same_v<chtype, uint8_t>) {
                    BICUBIC_C4_PROCESS_ROW(0, FETCH_INLIERS_8UC4);
                    BICUBIC_C4_PROCESS_ROW(1, FETCH_INLIERS_8UC4);
                    BICUBIC_C4_PROCESS_ROW(2, FETCH_INLIERS_8UC4);
                    BICUBIC_C4_PROCESS_ROW(3, FETCH_INLIERS_8UC4);
                } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                    BICUBIC_C4_PROCESS_ROW(0, FETCH_INLIERS_16UC4);
                    BICUBIC_C4_PROCESS_ROW(1, FETCH_INLIERS_16UC4);
                    BICUBIC_C4_PROCESS_ROW(2, FETCH_INLIERS_16UC4);
                    BICUBIC_C4_PROCESS_ROW(3, FETCH_INLIERS_16UC4);
                } else if constexpr (std::is_same_v<chtype, float>) {
                    BICUBIC_C4_PROCESS_ROW(0, FETCH_INLIERS_32FC4);
                    BICUBIC_C4_PROCESS_ROW(1, FETCH_INLIERS_32FC4);
                    BICUBIC_C4_PROCESS_ROW(2, FETCH_INLIERS_32FC4);
                    BICUBIC_C4_PROCESS_ROW(3, FETCH_INLIERS_32FC4);
                }
            }
        } else if (code < 0) {
            if (borderType == BORDER_CONSTANT) {
                for (int j = 0; j < dlen; j++) {
                    ((pixtype*)dst)[j] = bval;
                }
            }
            continue;
        } else {
            const chtype* defVal = borderType == BORDER_TRANSPARENT ? dst : borderVal;
            vecfptype wys[4] = {wy0, wy1, wy2, wy3};
            vecbuftype V0, V1, V2, V3;
            acc_r = acc_g = acc_b = acc_a = v_setzero_<vecfptype>();

            for (int row = 0; row < 4; row++) {
                bicubicFetchPixels((const chtype*)src, srcstep, size, NCHANNELS, tl_x, tl_y,
                                    goodx, row, &pixbuf[0][0], BATCH, borderType, defVal);
                vecfptype wy = wys[row];
                V0 = vx_load(&pixbuf[0][0]);
                V1 = vx_load(&pixbuf[0][BATCH]);
                V2 = vx_load(&pixbuf[0][BATCH*2]);
                V3 = vx_load(&pixbuf[0][BATCH*3]);
                BICUBIC_UPDATE_ACC_VEC(acc_r, V0, V1, V2, V3, wy);
                if constexpr (NCHANNELS > 1) {
                    V0 = vx_load(&pixbuf[1][0]);
                    V1 = vx_load(&pixbuf[1][BATCH]);
                    V2 = vx_load(&pixbuf[1][BATCH*2]);
                    V3 = vx_load(&pixbuf[1][BATCH*3]);
                    BICUBIC_UPDATE_ACC_VEC(acc_g, V0, V1, V2, V3, wy);
                    V0 = vx_load(&pixbuf[2][0]);
                    V1 = vx_load(&pixbuf[2][BATCH]);
                    V2 = vx_load(&pixbuf[2][BATCH*2]);
                    V3 = vx_load(&pixbuf[2][BATCH*3]);
                    BICUBIC_UPDATE_ACC_VEC(acc_b, V0, V1, V2, V3, wy);
                    if constexpr (NCHANNELS > 3) {
                        V0 = vx_load(&pixbuf[3][0]);
                        V1 = vx_load(&pixbuf[3][BATCH]);
                        V2 = vx_load(&pixbuf[3][BATCH*2]);
                        V3 = vx_load(&pixbuf[3][BATCH*3]);
                        BICUBIC_UPDATE_ACC_VEC(acc_a, V0, V1, V2, V3, wy);
                    }
                }
            }
        }

        // store the result
        chtype outbuf[BATCH*NCHANNELS*4];
        if constexpr (NCHANNELS == 1) {
            if constexpr (std::is_same_v<chtype, uint8_t>) {
            #if CV_SIMD_FP16
                v_int16 wacc_r = v_round(acc_r);
            #else
                v_int32 iacc_r = v_round(acc_r);
                v_int16 wacc_r = v_pack(iacc_r, iacc_r);
            #endif
                v_uint8 bacc_r = v_pack_u(wacc_r, wacc_r);
                if (i + BATCH*4 <= savestorelen) {
                    v_store(dst, bacc_r);
                    continue;
                }
                v_store(outbuf, bacc_r);
            } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                v_int32 iacc_r = v_round(acc_r);
                v_uint16 wacc_r = v_pack_u(iacc_r, iacc_r);
                if (i + BATCH*2 <= savestorelen) {
                    v_store(dst, wacc_r);
                    continue;
                }
                v_store(outbuf, wacc_r);
            } else if constexpr (std::is_same_v<chtype, float>) {
                if (i + BATCH <= savestorelen) {
                    v_store(dst, acc_r);
                    continue;
                }
                v_store(outbuf, acc_r);
            }
        } else if constexpr (NCHANNELS == 3) {
            if constexpr (std::is_same_v<chtype, uint8_t>) {
            #if CV_SIMD_FP16
                v_int16 wacc_r = v_round(acc_r);
                v_int16 wacc_g = v_round(acc_g);
                v_int16 wacc_b = v_round(acc_b);
            #else
                v_int32 iacc_r = v_round(acc_r);
                v_int32 iacc_g = v_round(acc_g);
                v_int32 iacc_b = v_round(acc_b);
                v_int16 wacc_r = v_pack(iacc_r, iacc_r);
                v_int16 wacc_g = v_pack(iacc_g, iacc_g);
                v_int16 wacc_b = v_pack(iacc_b, iacc_b);
            #endif
                v_uint8 bacc_r = v_pack_u(wacc_r, wacc_r);
                v_uint8 bacc_g = v_pack_u(wacc_g, wacc_g);
                v_uint8 bacc_b = v_pack_u(wacc_b, wacc_b);
                if (i + BATCH*4 <= savestorelen) {
                    v_store_interleave(dst, bacc_r, bacc_g, bacc_b);
                    continue;
                }
                v_store_interleave(outbuf, bacc_r, bacc_g, bacc_b);
            } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                v_int32 iacc_r = v_round(acc_r), iacc_g = v_round(acc_g), iacc_b = v_round(acc_b);
                v_uint16 wacc_r = v_pack_u(iacc_r, iacc_r);
                v_uint16 wacc_g = v_pack_u(iacc_g, iacc_g);
                v_uint16 wacc_b = v_pack_u(iacc_b, iacc_b);
                if (i + BATCH*2 <= savestorelen) {
                    v_store_interleave(dst, wacc_r, wacc_g, wacc_b);
                    continue;
                }
                v_store_interleave(outbuf, wacc_r, wacc_g, wacc_b);
            } else if constexpr (std::is_same_v<chtype, float>) {
                if (i + BATCH <= savestorelen) {
                    v_store_interleave(dst, acc_r, acc_g, acc_b);
                    continue;
                }
                v_store_interleave(outbuf, acc_r, acc_g, acc_b);
            }
        } else if constexpr (NCHANNELS == 4) {
            if constexpr (std::is_same_v<chtype, uint8_t>) {
            #if CV_SIMD_FP16
                v_int16 wacc_r = v_round(acc_r);
                v_int16 wacc_g = v_round(acc_g);
                v_int16 wacc_b = v_round(acc_b);
                v_int16 wacc_a = v_round(acc_a);
            #else
                v_int32 iacc_r = v_round(acc_r);
                v_int32 iacc_g = v_round(acc_g);
                v_int32 iacc_b = v_round(acc_b);
                v_int32 iacc_a = v_round(acc_a);
                v_int16 wacc_r = v_pack(iacc_r, iacc_r);
                v_int16 wacc_g = v_pack(iacc_g, iacc_g);
                v_int16 wacc_b = v_pack(iacc_b, iacc_b);
                v_int16 wacc_a = v_pack(iacc_a, iacc_a);
            #endif
                v_uint8 bacc_r = v_pack_u(wacc_r, wacc_r);
                v_uint8 bacc_g = v_pack_u(wacc_g, wacc_g);
                v_uint8 bacc_b = v_pack_u(wacc_b, wacc_b);
                v_uint8 bacc_a = v_pack_u(wacc_a, wacc_a);
                if (i + BATCH*4 <= savestorelen) {
                    v_store_interleave(dst, bacc_r, bacc_g, bacc_b, bacc_a);
                    continue;
                }
                v_store_interleave(outbuf, bacc_r, bacc_g, bacc_b, bacc_a);
            } else if constexpr (std::is_same_v<chtype, uint16_t>) {
                v_int32 iacc_r = v_round(acc_r), iacc_g = v_round(acc_g);
                v_int32 iacc_b = v_round(acc_b), iacc_a = v_round(acc_a);
                v_uint16 wacc_r = v_pack_u(iacc_r, iacc_r);
                v_uint16 wacc_g = v_pack_u(iacc_g, iacc_g);
                v_uint16 wacc_b = v_pack_u(iacc_b, iacc_b);
                v_uint16 wacc_a = v_pack_u(iacc_a, iacc_a);
                if (i + BATCH*2 <= savestorelen) {
                    v_store_interleave(dst, wacc_r, wacc_g, wacc_b, wacc_a);
                    continue;
                }
                v_store_interleave(outbuf, wacc_r, wacc_g, wacc_b, wacc_a);
            } else if constexpr (std::is_same_v<chtype, float>) {
                if (i + BATCH <= savestorelen) {
                    v_store_interleave(dst, acc_r, acc_g, acc_b, acc_a);
                    continue;
                }
                v_store_interleave(outbuf, acc_r, acc_g, acc_b, acc_a);
            }
        }

        memcpy(dst, outbuf, dlen*NCHANNELS*sizeof(dst[0]));
    }
}
#endif

static void
bicubic8uC1(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
            uint8_t* dst, const float* params,
            int borderType, uint8_t* borderVal)
{
#if CV_SIMD_FP16
    bicubicVec<uint8_t, 1, v_float16, v_int16>(srcx, srcy, len, src, srcstep, size,
                                               dst, params, borderType, borderVal);
#elif CV_SIMD
    bicubicVec<uint8_t, 1, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                               dst, params, borderType, borderVal);
#else
    bicubicRef<uint8_t, 1>(srcx, srcy, len, src, srcstep, size,
                           dst, params, borderType, borderVal);
#endif
}

static void
bicubic8uC2(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
            uint8_t* dst, const float* params,
            int borderType, uint8_t* borderVal)
{
    bicubicRef<uint8_t, 2>(srcx, srcy, len, src, srcstep, size,
                           dst, params, borderType, borderVal);
}

static void
bicubic8uC3(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
            uint8_t* dst, const float* params,
            int borderType, uint8_t* borderVal)
{
#if CV_SIMD_FP16
    bicubicVec<uint8_t, 3, v_float16, v_int16>(srcx, srcy, len, src, srcstep, size,
                                               dst, params, borderType, borderVal);
#elif CV_SIMD
    bicubicVec<uint8_t, 3, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                               dst, params, borderType, borderVal);
#else
    bicubicRef<uint8_t, 3>(srcx, srcy, len, src, srcstep, size,
                           dst, params, borderType, borderVal);
#endif
}

static void
bicubic8uC4(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
            uint8_t* dst, const float* params,
            int borderType, uint8_t* borderVal)
{
#if CV_SIMD_FP16
    bicubicVec<uint8_t, 4, v_float16, v_int16>(srcx, srcy, len, src, srcstep, size,
                                                    dst, params, borderType, borderVal);
#elif CV_SIMD
    bicubicVec<uint8_t, 4, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                                    dst, params, borderType, borderVal);
#else
    bicubicRef<uint8_t, 4>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic16uC1(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
            uint16_t* dst, const float* params,
            int borderType, uint16_t* borderVal)
{
#if CV_SIMD
    bicubicVec<uint16_t, 1, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                                    dst, params, borderType, borderVal);
#else
    bicubicRef<uint16_t, 1>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic16uC2(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             uint16_t* dst, const float* params,
             int borderType, uint16_t* borderVal)
{
    bicubicRef<uint16_t, 2>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic16uC3(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
             uint16_t* dst, const float* params,
            int borderType, uint16_t* borderVal)
{
#if CV_SIMD
    bicubicVec<uint16_t, 3, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                                    dst, params, borderType, borderVal);
#else
    bicubicRef<uint16_t, 3>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic16uC4(const float* srcx, const float* srcy, int len,
            const void* src, size_t srcstep, Size size,
             uint16_t* dst, const float* params,
            int borderType, uint16_t* borderVal)
{
#if CV_SIMD
    bicubicVec<uint16_t, 4, v_float32, v_int32>(srcx, srcy, len, src, srcstep, size,
                                                    dst, params, borderType, borderVal);
#else
    bicubicRef<uint16_t, 4>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic16sC1(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             int16_t* dst, const float* params,
             int borderType, int16_t* borderVal)
{
    bicubicRef<int16_t, 1>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic16sC2(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             int16_t* dst, const float* params,
             int borderType, int16_t* borderVal)
{
    bicubicRef<int16_t, 2>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic16sC3(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             int16_t* dst, const float* params,
             int borderType, int16_t* borderVal)
{
    bicubicRef<int16_t, 3>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic16sC4(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             int16_t* dst, const float* params,
             int borderType, int16_t* borderVal)
{
    bicubicRef<int16_t, 4>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic32fC1(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             float* dst, const float* params,
             int borderType, float* borderVal)
{
#if CV_SIMD
    bicubicVec<float, 1, v_float32, v_float32>(srcx, srcy, len, src, srcstep, size,
                                                      dst, params, borderType, borderVal);
#else
    bicubicRef<float, 1>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic32fC2(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             float* dst, const float* params,
             int borderType, float* borderVal)
{
    bicubicRef<float, 2>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
}

static void
bicubic32fC3(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             float* dst, const float* params,
             int borderType, float* borderVal)
{
#if CV_SIMD
    bicubicVec<float, 3, v_float32, v_float32>(srcx, srcy, len, src, srcstep, size,
                                                      dst, params, borderType, borderVal);
#else
    bicubicRef<float, 3>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic32fC4(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             float* dst, const float* params,
             int borderType, float* borderVal)
{
#if CV_SIMD
    bicubicVec<float, 4, v_float32, v_float32>(srcx, srcy, len, src, srcstep, size,
                                                      dst, params, borderType, borderVal);
#else
    bicubicRef<float, 4>(srcx, srcy, len, src, srcstep, size,
                                dst, params, borderType, borderVal);
#endif
}

static void
bicubic64fC1(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             double* dst, const float* params,
             int borderType, double* borderVal)
{
    bicubicRef<double, 1>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic64fC2(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             double* dst, const float* params,
             int borderType, double* borderVal)
{
    bicubicRef<double, 2>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic64fC3(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             double* dst, const float* params,
             int borderType, double* borderVal)
{
    bicubicRef<double, 3>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

static void
bicubic64fC4(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             double* dst, const float* params,
             int borderType, double* borderVal)
{
    bicubicRef<double, 4>(srcx, srcy, len, src, srcstep, size,
                            dst, params, borderType, borderVal);
}

}

ImgWarpFunc getBicubicWarpFunc_(int type)
{
    if (type == CV_8UC1) {
        return (ImgWarpFunc)bicubic8uC1;
    }
    if (type == CV_8UC2) {
        return (ImgWarpFunc)bicubic8uC2;
    }
    if (type == CV_8UC3) {
        return (ImgWarpFunc)bicubic8uC3;
    }
    if (type == CV_8UC4) {
        return (ImgWarpFunc)bicubic8uC4;
    }
    if (type == CV_16UC1) {
        return (ImgWarpFunc)bicubic16uC1;
    }
    if (type == CV_16UC2) {
        return (ImgWarpFunc)bicubic16uC2;
    }
    if (type == CV_16UC3) {
        return (ImgWarpFunc)bicubic16uC3;
    }
    if (type == CV_16UC4) {
        return (ImgWarpFunc)bicubic16uC4;
    }
    if (type == CV_16SC1) {
        return (ImgWarpFunc)bicubic16sC1;
    }
    if (type == CV_16SC2) {
        return (ImgWarpFunc)bicubic16sC2;
    }
    if (type == CV_16SC3) {
        return (ImgWarpFunc)bicubic16sC3;
    }
    if (type == CV_16SC4) {
        return (ImgWarpFunc)bicubic16sC4;
    }
    if (type == CV_32FC1) {
        return (ImgWarpFunc)bicubic32fC1;
    }
    if (type == CV_32FC2) {
        return (ImgWarpFunc)bicubic32fC2;
    }
    if (type == CV_32FC3) {
        return (ImgWarpFunc)bicubic32fC3;
    }
    if (type == CV_32FC4) {
        return (ImgWarpFunc)bicubic32fC4;
    }
    if (type == CV_64FC1) {
        return (ImgWarpFunc)bicubic64fC1;
    }
    if (type == CV_64FC2) {
        return (ImgWarpFunc)bicubic64fC2;
    }
    if (type == CV_64FC3) {
        return (ImgWarpFunc)bicubic64fC3;
    }
    if (type == CV_64FC4) {
        return (ImgWarpFunc)bicubic64fC4;
    }
    return (ImgWarpFunc)nullptr;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
