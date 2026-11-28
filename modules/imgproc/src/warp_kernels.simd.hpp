// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <numeric>
#include "precomp.hpp"
#if CV_AVX512_SKX
#define CV__SIMD_FORCE_WIDTH 256
#endif
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
    v_int32 addr_0 = v_muladd(v_srcstep, src_iy0, src_ix0), \
            addr_1 = v_muladd(v_srcstep, src_iy1, src_ix1);
#define CV_WARP_VECTOR_GET_ADDR_C3() \
    v_int32 addr_0 = v_muladd(v_srcstep, src_iy0, v_mul(src_ix0, three)), \
            addr_1 = v_muladd(v_srcstep, src_iy1, v_mul(src_ix1, three));
#define CV_WARP_VECTOR_GET_ADDR_C4() \
    v_int32 addr_0 = v_muladd(v_srcstep, src_iy0, v_mul(src_ix0, four)), \
            addr_1 = v_muladd(v_srcstep, src_iy1, v_mul(src_ix1, four));
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

// Half-precision warps sanitize coordinates that cannot be floored safely (NaN, infinity, beyond int range)
#define CV_WARP_VECTOR_SANITIZE_COORDS() \
    if constexpr (sanitize_coords) { \
        const v_float32 coord_limit = vx_setall_f32(2147483648.f); \
        v_float32 coord_ok = v_and(v_and(v_lt(v_abs(src_x0), coord_limit), v_lt(v_abs(src_y0), coord_limit)), \
                                  v_and(v_lt(v_abs(src_x1), coord_limit), v_lt(v_abs(src_y1), coord_limit))); \
        if (!v_check_all(coord_ok)) { \
            float coord_x[max_uf], coord_y[max_uf]; \
            vx_store(coord_x, src_x0); \
            vx_store(coord_x + vlanes_32, src_x1); \
            vx_store(coord_y, src_y0); \
            vx_store(coord_y + vlanes_32, src_y1); \
            for (int k = 0; k < uf; k++) { \
                coord_x[k] = warpSanitizeCoordinate(coord_x[k], srccols, border_type); \
                coord_y[k] = warpSanitizeCoordinate(coord_y[k], srcrows, border_type); \
            } \
            src_x0 = vx_load(coord_x); \
            src_x1 = vx_load(coord_x + vlanes_32); \
            src_y0 = vx_load(coord_y); \
            src_y1 = vx_load(coord_y + vlanes_32); \
        } \
    }
#define CV_WARP_SCALAR_SANITIZE_COORDS() \
    if constexpr (sanitize_coords) { \
        sx = warpSanitizeCoordinate(sx, srccols, border_type); \
        sy = warpSanitizeCoordinate(sy, srcrows, border_type); \
    }

#define CV_WARP_VECTOR_COMPUTE_MAPPED_COORD(INTER, CN) \
    CV_WARP_VECTOR_SANITIZE_COORDS() \
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

// Only the half-precision invokers override this
static constexpr bool sanitize_coords = false;

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
// Maps a coordinate that cannot be floored safely to one with the same border behavior.
// Coordinates beyond the image are outside for constant borders; otherwise the value is
// reduced modulo the period of the border mode, which keeps floats beyond 2^24 exact.
static inline float warpSanitizeCoordinate(float v, int size, int border_type)
{
    if (std::fabs(v) < 2147483648.f)
        return v;
    if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT)
        return -2.f;
    if (std::isnan(v))
        return 0.f;
    if (border_type == BORDER_REPLICATE)
        return v > 0 ? float(size) : 0.f;
    if (std::isinf(v))
        return 0.f;
    double period = 2.*size*std::max(size - 1, 1);
    if (period > 16777216.)
        period = 2.*size;
    return float(std::fmod(double(v), period));
}
#endif

ImgWarpFunc getBicubicWarpFunc_(int type);
ImgWarpFunc getBilinearHalfWarpFunc_();
RemapBilinearHalfRGBFunc getRemapBilinearHalfRGB_(bool relative);
RemapLanczosHalfRGBFunc getRemapLanczosHalfRGB_();

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
void warpAffineLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                  hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double M[6], int border_type, const double border_value[4]);
void warpAffineLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                  hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
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
void warpPerspectiveLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                        hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC3(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double M[9], int border_type, const double border_value[4]);
void warpPerspectiveLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                        hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
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
void remapLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                              hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
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
void remapLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                              hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
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
        const int delta = borderType == BORDER_REFLECT_101;
        if( len == 1 )
            return 0;

        const int64_t len64 = len;
        const int64_t p64 = p;
        // Fast path: single reflection without division for small deviations.
        if( -len64 + delta <= p64 && p64 < 2 * len64 - delta )
            return (int)(p64 < 0 ? -p64 - 1 + delta : 2 * len64 - 1 - delta - p64);
        // Bounded fallback: O(1) modulo for large |p|
        const int64_t period = 2 * (len64 - delta);
        int64_t r64 = p64 % period;
        if( r64 < 0 )
            r64 += period;
        if( r64 >= len64 )
            r64 = period - r64 - 1 + delta;
        p = (int)r64;
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 || p >= len )
        {
            int64_t p64 = p;
            p64 %= (int64_t)len;
            if( p64 < 0 )
                p64 += len;
            p = (int)p64;
        }
    }
    else
        CV_Error( cv::Error::StsBadArg, "Unknown/unsupported border type" );
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

template<typename T>
static void warpAffineLinearInvokerFT_C1(const T *src_data, size_t src_step, int src_rows, int src_cols,
                                  T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, FT);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    warpAffineLinearInvokerFT_C1<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
}

void warpAffineLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                  hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    warpAffineLinearInvokerFT_C1<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
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

template<typename T>
static void warpAffineLinearInvokerFT_C4(const T *src_data, size_t src_step, int src_rows, int src_cols,
                                  T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[6];
        for (int i = 0; i < 6; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPAFFINE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    if constexpr (std::is_same_v<T, hfloat>) {
                        float pixbuf[max_uf*4*4];
                        CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C4);
                        CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                        CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                        CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
                    } else {
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
                    }
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, FT);
                    // CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpAffineLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                  float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    warpAffineLinearInvokerFT_C4<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
}

void warpAffineLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                  hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                  const double dM[6], int border_type, const double border_value[4]) {
    warpAffineLinearInvokerFT_C4<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
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

template<typename T>
static void warpPerspectiveLinearInvokerFT_C1(const T *src_data, size_t src_step, int src_rows, int src_cols,
                                        T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, FT);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    warpPerspectiveLinearInvokerFT_C1<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
}

void warpPerspectiveLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                        hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    warpPerspectiveLinearInvokerFT_C1<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
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

template<typename T>
static void warpPerspectiveLinearInvokerFT_C4(const T *src_data, size_t src_step, int src_rows, int src_cols,
                                        T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T);
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        float M[9];
        for (int i = 0; i < 9; i++) {
            M[i] = static_cast<float>(dM[i]);
        }
        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_WARPPERSPECTIVE_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    if constexpr (std::is_same_v<T, hfloat>) {
                        float pixbuf[max_uf*4*4];
                        CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C4);
                        CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                        CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                        CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
                    } else {
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
                    }
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, FT);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void warpPerspectiveLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                                        float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    warpPerspectiveLinearInvokerFT_C4<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
}

void warpPerspectiveLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                                        hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                                        const double dM[9], int border_type, const double border_value[4]) {
    warpPerspectiveLinearInvokerFT_C4<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, dM, border_type, border_value);
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

template<typename T>
static void remapLinearInvokerFT_C1(const T *src_data, size_t src_step, int src_rows, int src_cols,
                              T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C1);
                if (v_reduce_min(inner_mask) != 0) {
                    CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C1);
                } else {
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C1, FT);
                }
                CV_WARP_VECTOR_INTER_LOAD(LINEAR, C1, 32F, 32F);
                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);
                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C1);
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

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C1, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C1);
                CV_WARP_SCALAR_STORE(LINEAR, C1, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC1(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    remapLinearInvokerFT_C1<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
}

void remapLinearInvoker_16FC1(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                              hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    remapLinearInvokerFT_C1<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
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

template<typename T>
static void remapLinearInvokerFT_C4(const T *src_data, size_t src_step, int src_rows, int src_cols,
                              T *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    auto worker = [&](const Range &r) {
        CV_INSTRUMENT_REGION();
        constexpr bool sanitize_coords = std::is_same_v<T, hfloat>;

        const auto *src = src_data;
        auto *dst = dst_data;
        auto *map1 = map1_data, *map2 = map2_data;
        size_t srcstep = src_step/sizeof(T), dststep = dst_step/sizeof(T),
               map1step = map1_step/sizeof(float), map2step=map2_step/sizeof(float);
        if (map2 == nullptr) {
            map2 = map1;
            map2step = map1step;
        }
        int srccols = src_cols, srcrows = src_rows;
        int dstcols = dst_cols;
        bool relative = is_relative;

        T bval[] = {
            saturate_cast<T>(border_value[0]),
            saturate_cast<T>(border_value[1]),
            saturate_cast<T>(border_value[2]),
            saturate_cast<T>(border_value[3]),
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
            T* dstptr = dst + y*dststep;
            const float *sx_data = map1 + y*map1step;
            const float *sy_data = map2 + y*map2step;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD1();
            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick
                CV_REMAP_VECTOR_COMPUTE_MAPPED_COORD2(LINEAR, C4);
                if (v_reduce_min(inner_mask) != 0) {
                    if constexpr (std::is_same_v<T, hfloat>) {
                        float pixbuf[max_uf*4*4];
                        CV_WARP_VECTOR_SHUFFLE_ALLWITHIN_FT(LINEAR, C4);
                        CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                        CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                        CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
                    } else {
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
                    }
                } else {
                    float pixbuf[max_uf*4*4];
                    CV_WARP_VECTOR_SHUFFLE_NOTALLWITHIN(LINEAR, C4, FT);
                    CV_WARP_VECTOR_INTER_LOAD(LINEAR, C4, 32F, 32F);
                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32FT(C4);
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

                CV_WARP_SCALAR_SANITIZE_COORDS();
                CV_WARP_SCALAR_SHUFFLE(LINEAR, C4, FT);
                CV_WARP_SCALAR_LINEAR_INTER_CALC_F32(C4);
                CV_WARP_SCALAR_STORE(LINEAR, C4, FT);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_32FC4(const float *src_data, size_t src_step, int src_rows, int src_cols,
                              float *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    remapLinearInvokerFT_C4<float>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
}

void remapLinearInvoker_16FC4(const hfloat *src_data, size_t src_step, int src_rows, int src_cols,
                              hfloat *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                              int border_type, const double border_value[4],
                              const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    remapLinearInvokerFT_C4<hfloat>(src_data, src_step, src_rows, src_cols, dst_data, dst_step, dst_rows, dst_cols, border_type, border_value, map1_data, map1_step, map2_data, map2_step, is_relative);
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

constexpr int BICUBIC_LAST_TAP = 3;
constexpr int BICUBIC_MIN_COORD_LIMIT = 16;

static CV_ALWAYS_INLINE int
bicubicCoeffs(float xs0, float ys0, Size size, float A,
              int bordertype, int& tl_x, int& tl_y,
              float& wx0, float& wx1, float& wx2, float& wx3,
              float& wy0, float& wy1, float& wy2, float& wy3)
{
    int width = size.width, height = size.height;
    int bigwidth = std::max(width, BICUBIC_MIN_COORD_LIMIT);
    int bigheight = std::max(height, BICUBIC_MIN_COORD_LIMIT);
    float minx = float(-bigwidth), maxx = float(bigwidth*2);
    float miny = float(-bigheight), maxy = float(bigheight*2);

    // Clamp before converting to integers, with NaN mapped to the lower bound.
    float vx0 = std::clamp(std::isnan(xs0) ? minx : xs0, minx, maxx);
    float vy0 = std::clamp(std::isnan(ys0) ? miny : ys0, miny, maxy);

    int ix0 = cvFloor(vx0);
    int iy0 = cvFloor(vy0);
    float alpha = vx0 - ix0;
    float beta = vy0 - iy0;

    ix0--;
    iy0--;

    int all_outliers = int((unsigned)(ix0 + BICUBIC_LAST_TAP) >= (unsigned)(width + BICUBIC_LAST_TAP)) |
                       int((unsigned)(iy0 + BICUBIC_LAST_TAP) >= (unsigned)(height + BICUBIC_LAST_TAP));
    if (all_outliers && (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT))
        return -1;

    int all_inliers = int((unsigned)ix0 < (unsigned)std::max(width - BICUBIC_LAST_TAP, 0)) &
                      int((unsigned)iy0 < (unsigned)std::max(height - BICUBIC_LAST_TAP, 0));

    tl_x = ix0;
    tl_y = iy0;

    bicubicWeights(alpha, A, wx0, wx1, wx2, wx3);
    bicubicWeights(beta, A, wy0, wy1, wy2, wy3);

    return all_inliers;
}

template<typename _Tp, typename _Fp>
CV_DISABLE_UBSAN
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
    V0 = acctype(srcrow[0]); \
    V1 = acctype(srcrow[1]); \
    V2 = acctype(srcrow[2]); \
    V3 = acctype(srcrow[3]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C2_SCALAR
#define BICUBIC_PROCESS_INLIERS_C2_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = acctype(srcrow[0]); \
    V1 = acctype(srcrow[2]); \
    V2 = acctype(srcrow[4]); \
    V3 = acctype(srcrow[6]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[1]); \
    V1 = acctype(srcrow[3]); \
    V2 = acctype(srcrow[5]); \
    V3 = acctype(srcrow[7]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C3_SCALAR
#define BICUBIC_PROCESS_INLIERS_C3_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = acctype(srcrow[0]); \
    V1 = acctype(srcrow[3]); \
    V2 = acctype(srcrow[6]); \
    V3 = acctype(srcrow[9]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[1]); \
    V1 = acctype(srcrow[4]); \
    V2 = acctype(srcrow[7]); \
    V3 = acctype(srcrow[10]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[2]); \
    V1 = acctype(srcrow[5]); \
    V2 = acctype(srcrow[8]); \
    V3 = acctype(srcrow[11]); \
    BICUBIC_UPDATE_ACC(acc_b, V0, V1, V2, V3, wy##row)

#undef BICUBIC_PROCESS_INLIERS_C4_SCALAR
#define BICUBIC_PROCESS_INLIERS_C4_SCALAR(row) \
    srcrow = (const chtype*)((const uint8_t*)src + row*srcstep + tl_ofs); \
    V0 = acctype(srcrow[0]); \
    V1 = acctype(srcrow[4]); \
    V2 = acctype(srcrow[8]); \
    V3 = acctype(srcrow[12]); \
    BICUBIC_UPDATE_ACC(acc_r, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[1]); \
    V1 = acctype(srcrow[5]); \
    V2 = acctype(srcrow[9]); \
    V3 = acctype(srcrow[13]); \
    BICUBIC_UPDATE_ACC(acc_g, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[2]); \
    V1 = acctype(srcrow[6]); \
    V2 = acctype(srcrow[10]); \
    V3 = acctype(srcrow[14]); \
    BICUBIC_UPDATE_ACC(acc_b, V0, V1, V2, V3, wy##row); \
    V0 = acctype(srcrow[3]); \
    V1 = acctype(srcrow[7]); \
    V2 = acctype(srcrow[11]); \
    V3 = acctype(srcrow[15]); \
    BICUBIC_UPDATE_ACC(acc_a, V0, V1, V2, V3, wy##row)

template<typename chtype, int NCHANNELS>
static void bicubicRef(const float* srcx, const float* srcy, int len,
                       const void* src, size_t srcstep, Size size,
                       chtype* dst, const float* params, int borderType, chtype* borderVal)
{
    constexpr float defaultA = -0.75f;
    float A = params ? *params : defaultA;

    constexpr int BPP = int(NCHANNELS*sizeof(dst[0]));
    using buftype = std::conditional_t<std::is_floating_point_v<chtype> || std::is_same_v<chtype, hfloat>,
                                       std::conditional_t<std::is_same_v<chtype, hfloat>, float, chtype>, int>;
    using acctype = std::conditional_t<std::is_same_v<chtype, double>, double, float>;
    using pixtype = std::conditional_t<NCHANNELS == 1, chtype, Vec<chtype, NCHANNELS>>;

    pixtype bval = borderVal ? *(pixtype*)borderVal : pixtype();

    for (int i = 0; i < len; i++, dst += NCHANNELS) {
        buftype pixbuf[NCHANNELS][4];
        int tl_x, tl_y, goodx[4];
        acctype V0, V1, V2, V3;
        float wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3;
        float xs = srcx[i], ys = srcy[i];

        int code = bicubicCoeffs(xs, ys, size, A, borderType, tl_x, tl_y,
                                 wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3);
        if (code > 0) {
            const size_t tl_ofs = size_t(tl_y)*srcstep + size_t(tl_x)*BPP;
            const chtype* srcrow;
            if constexpr (NCHANNELS == 1) {
                acctype acc_r = 0;
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C1_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
            } else if constexpr (NCHANNELS == 2) {
                acctype acc_r = 0, acc_g = 0;
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C2_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
                dst[1] = saturate_cast<chtype>(acc_g);
            } else if constexpr (NCHANNELS == 3) {
                acctype acc_r = 0, acc_g = 0, acc_b = 0;
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(0);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(1);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(2);
                BICUBIC_PROCESS_INLIERS_C3_SCALAR(3);
                dst[0] = saturate_cast<chtype>(acc_r);
                dst[1] = saturate_cast<chtype>(acc_g);
                dst[2] = saturate_cast<chtype>(acc_b);
            } else if constexpr (NCHANNELS == 4) {
                acctype acc_r = 0, acc_g = 0, acc_b = 0, acc_a = 0;
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
            acctype acc[NCHANNELS] = {};
            float wys[] = {wy0, wy1, wy2, wy3};
            const chtype* defVal = borderType == BORDER_TRANSPARENT ? dst : borderVal;
            for (int row = 0; row < 4; row++) {
                bicubicFetchPixels((const chtype*)src, srcstep, size, NCHANNELS, &tl_x, &tl_y,
                                   goodx, row, &pixbuf[0][0], 1, borderType, defVal);
                float wy = wys[row];
                for (int c = 0; c < NCHANNELS; c++) {
                    V0 = acctype(pixbuf[c][0]);
                    V1 = acctype(pixbuf[c][1]);
                    V2 = acctype(pixbuf[c][2]);
                    V3 = acctype(pixbuf[c][3]);
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

static CV_ALWAYS_INLINE v_float32
bicubicClampCoords(const float* coords, int length)
{
    const int limit = std::max(length, BICUBIC_MIN_COORD_LIMIT);
    const v_float32 lower = vx_setall_f32(float(-limit));
    const v_float32 upper = vx_setall_f32(float(limit*2));
    const v_float32 values = vx_load(coords);
    return v_min(v_max(v_select(v_not_nan(values), values, lower), lower), upper);
}

constexpr int BICUBIC_MIXED_OUTLIERS = -2;

static CV_ALWAYS_INLINE v_uint32
bicubicOutliers(const v_int32& x, const v_int32& y, Size size)
{
    const v_int32 last = vx_setall_s32(BICUBIC_LAST_TAP);
    const v_uint32 width = vx_setall_u32(size.width + BICUBIC_LAST_TAP);
    const v_uint32 height = vx_setall_u32(size.height + BICUBIC_LAST_TAP);
    return v_or(v_ge(v_reinterpret_as_u32(v_add(x, last)), width),
                v_ge(v_reinterpret_as_u32(v_add(y, last)), height));
}

static CV_ALWAYS_INLINE int
bicubicCoeffs(const float* srcx, const float* srcy,
              size_t srcstep, Size size, int bpp, float A,
              int bordertype, int32_t* tl_x, int32_t* tl_y, int32_t* tl_ofs,
              v_float32& wx0, v_float32& wx1, v_float32& wx2, v_float32& wx3,
              v_float32& wy0, v_float32& wy1, v_float32& wy2, v_float32& wy3)
{
    const int width = size.width, height = size.height;
    const v_float32 vx0 = bicubicClampCoords(srcx, width);
    const v_float32 vy0 = bicubicClampCoords(srcy, height);

    v_int32 ix0 = v_floor(vx0);
    v_int32 iy0 = v_floor(vy0);
    v_float32 alpha = v_sub(vx0, v_cvt_f32(ix0));
    v_float32 beta = v_sub(vy0, v_cvt_f32(iy0));

    const v_int32 one_i = vx_setall_s32(1);
    ix0 = v_sub(ix0, one_i);
    iy0 = v_sub(iy0, one_i);

    if (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT) {
        const v_uint32 outliers = bicubicOutliers(ix0, iy0, size);
        if (v_check_any(outliers))
            return v_check_all(outliers) ? -1 : BICUBIC_MIXED_OUTLIERS;
    }

    v_uint32 width_inner = vx_setall_u32((uint32_t)std::max(width - BICUBIC_LAST_TAP, 0));
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
    const int width = size.width, height = size.height;
    const v_float32 vx0 = bicubicClampCoords(srcx, width);
    const v_float32 vy0 = bicubicClampCoords(srcy, height);
    const v_float32 vx1 = bicubicClampCoords(srcx + nlanes32, width);
    const v_float32 vy1 = bicubicClampCoords(srcy + nlanes32, height);

    v_int32 ix0 = v_floor(vx0);
    v_int32 iy0 = v_floor(vy0);
    v_int32 ix1 = v_floor(vx1);
    v_int32 iy1 = v_floor(vy1);
    v_float32 alpha0 = v_sub(vx0, v_cvt_f32(ix0));
    v_float32 beta0 = v_sub(vy0, v_cvt_f32(iy0));
    v_float32 alpha1 = v_sub(vx1, v_cvt_f32(ix1));
    v_float32 beta1 = v_sub(vy1, v_cvt_f32(iy1));

    const v_int32 one_i = vx_setall_s32(1);
    ix0 = v_sub(ix0, one_i);
    iy0 = v_sub(iy0, one_i);
    ix1 = v_sub(ix1, one_i);
    iy1 = v_sub(iy1, one_i);

    if (bordertype == BORDER_CONSTANT || bordertype == BORDER_TRANSPARENT) {
        const v_uint32 outliers0 = bicubicOutliers(ix0, iy0, size);
        const v_uint32 outliers1 = bicubicOutliers(ix1, iy1, size);
        if (v_check_any(v_or(outliers0, outliers1)))
            return v_check_all(v_and(outliers0, outliers1)) ? -1 : BICUBIC_MIXED_OUTLIERS;
    }

    v_uint32 width_inner = vx_setall_u32((uint32_t)std::max(width - BICUBIC_LAST_TAP, 0));
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

    hfloat abuf[nlanes32*2], bbuf[nlanes32*2];

    v_pack_store(abuf, alpha0);
    v_pack_store(abuf + nlanes32, alpha1);
    v_pack_store(bbuf, beta0);
    v_pack_store(bbuf + nlanes32, beta1);

    v_float16 alpha = vx_load(abuf);
    v_float16 beta = vx_load(bbuf);

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
    // SIMD bicubicCoeffs computes 32-bit signed byte offsets; fall back to
    // bicubicRef (which uses ptrdiff_t) for images larger than INT_MAX bytes.
    if (srcstep > (size_t)INT_MAX ||
        (srcstep > 0 && (size_t)size.height > (size_t)INT_MAX / srcstep)) {
        bicubicRef<chtype, NCHANNELS>(srcx, srcy, len, src, srcstep, size,
                                      dst, params, borderType, borderVal);
        return;
    }

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
        if (code == BICUBIC_MIXED_OUTLIERS) {
            // Fully outside lanes need per-pixel handling for exact border values.
            bicubicRef<chtype, NCHANNELS>(xptr, yptr, dlen, src, srcstep, size,
                                          dst, params, borderType, borderVal);
            continue;
        }

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
                } else if constexpr (std::is_same_v<chtype, hfloat>) {
                    BICUBIC_C1_PROCESS_ROW(0, FETCH_INLIERS_C1_DEFAULT);
                    BICUBIC_C1_PROCESS_ROW(1, FETCH_INLIERS_C1_DEFAULT);
                    BICUBIC_C1_PROCESS_ROW(2, FETCH_INLIERS_C1_DEFAULT);
                    BICUBIC_C1_PROCESS_ROW(3, FETCH_INLIERS_C1_DEFAULT);
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
        if constexpr (std::is_same_v<chtype, hfloat>) {
            static_assert(NCHANNELS == 1, "half cubic uses bicubicVec for one channel only");
            if (i + BATCH <= savestorelen) {
                v_pack_store(dst, acc_r);
            } else {
                float fbuf[BATCH];
                v_store(fbuf, acc_r);
                for (int k = 0; k < dlen; k++)
                    dst[k] = hfloat(fbuf[k]);
            }
            continue;
        }
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

#if CV_SIMD
static void
bicubic16fC1(const float* srcx, const float* srcy, int len,
             const void* src, size_t srcstep, Size size,
             hfloat* dst, const float* params,
             int borderType, hfloat* borderVal)
{
    bicubicVec<hfloat, 1, v_float32, v_float32>(srcx, srcy, len, src, srcstep, size,
                                                dst, params, borderType, borderVal);
}
#endif

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

#if (CV_SIMD || CV_SIMD_SCALABLE)
// LSX and LASX use a 16-byte load even when expanding fewer bytes.
constexpr size_t BICUBIC_EXPAND_LOAD_BYTES = (CV_LSX || CV_LASX) ? 16 : VTraits<v_uint8>::max_nlanes;
constexpr bool BICUBIC_EXPAND_LOAD_IS_EXACT = CV_SSE2 || CV_NEON || CV_RVV;

template<typename T, size_t N>
static inline const T* bicubicExpandInput(const T* src, [[maybe_unused]] const uchar* end,
                                          [[maybe_unused]] int lanes, T (&scratch)[N])
{
    if constexpr (!BICUBIC_EXPAND_LOAD_IS_EXACT)
    {
        const size_t bytes = lanes*sizeof(T);
        if (bytes < sizeof(scratch) && size_t(end - reinterpret_cast<const uchar*>(src)) < sizeof(scratch))
        {
            std::memcpy(scratch, src, bytes);
            std::fill(scratch + lanes, scratch + N, T{});
            return scratch;
        }
    }
    return src;
}

template<typename T>
static inline v_float32 bicubicLoadChannels(const T* src, [[maybe_unused]] const uchar* end,
                                          [[maybe_unused]] int lanes)
{
    if constexpr (std::is_same_v<T, float>)
        return vx_load(src);
    else
    {
        alignas(CV_SIMD_WIDTH) T scratch[BICUBIC_EXPAND_LOAD_BYTES/sizeof(T)];
        src = bicubicExpandInput(src, end, lanes, scratch);
        if constexpr (std::is_same_v<T, uchar>)
            return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(src)));
        else if constexpr (std::is_same_v<T, ushort>)
            return v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(src)));
        else
            return v_cvt_f32(vx_load_expand(src));
    }
}

template<typename T>
static inline void bicubicStoreChannels(T* dst, const v_float32& value, [[maybe_unused]] int lanes)
{
    if constexpr (std::is_same_v<T, float>)
        vx_store(dst, value);
    else
    {
        const v_int32 rounded = v_round(value);
        if constexpr (std::is_same_v<T, uchar>)
        {
            const v_uint16 words = v_pack_u(rounded, rounded);
            alignas(CV_SIMD_WIDTH) uchar packed[VTraits<v_uint8>::max_nlanes];
            // Store packed lanes to scratch to avoid writing past the channel tail.
            v_store_low(packed, v_pack(words, words));
            std::memcpy(dst, packed, lanes);
        }
        else if constexpr (std::is_same_v<T, ushort>)
            v_store_low(dst, v_pack_u(rounded, rounded));
        else
            v_store_low(dst, v_pack(rounded, rounded));
    }
}
#endif

#if CV_SIMD128
// Keep the four-lane cleanup path for wide vectors when fewer than a full vector remains.
template<typename T>
static inline v_float32x4 bicubicLoadChannels4(const T* src, [[maybe_unused]] const uchar* end)
{
    if constexpr (std::is_same_v<T, float>)
        return v_load(src);
    else if constexpr (std::is_same_v<T, hfloat>)
    {
        // Half expansion can read eight samples for a four-lane result.
        alignas(16) hfloat values[8] = {};
        std::memcpy(values, src, 4*sizeof(hfloat));
        return v_load_expand(values);
    }
    else
    {
        alignas(16) T scratch[16/sizeof(T)];
        src = bicubicExpandInput(src, end, 4, scratch);
        if constexpr (std::is_same_v<T, uchar>)
            return v_cvt_f32(v_reinterpret_as_s32(v_load_expand_q(src)));
        else if constexpr (std::is_same_v<T, ushort>)
            return v_cvt_f32(v_reinterpret_as_s32(v_load_expand(src)));
        else
            return v_cvt_f32(v_load_expand(src));
    }
}

template<typename T>
static inline void bicubicStoreChannels4(T* dst, const v_float32x4& value)
{
    if constexpr (std::is_same_v<T, float>)
        v_store(dst, value);
    else if constexpr (std::is_same_v<T, hfloat>)
        v_pack_store(dst, value);
    else
    {
        const v_int32x4 rounded = v_round(value);
        if constexpr (std::is_same_v<T, uchar>)
        {
            const v_uint16x8 words = v_pack_u(rounded, rounded);
            const unsigned packed = v_extract_n<0>(v_reinterpret_as_u32(v_pack(words, words)));
            std::memcpy(dst, &packed, sizeof(packed));
        }
        else if constexpr (std::is_same_v<T, ushort>)
            v_store_low(dst, v_pack_u(rounded, rounded));
        else
            v_store_low(dst, v_pack(rounded, rounded));
    }
}
#endif

template<typename T>
static inline int bicubicNextChannel(int channel, int channels, int width)
{
    const int next = channel + width;
    // Overlap the final vector so CV_8U rounding is independent of channel position.
    if constexpr (std::is_same_v<T, uchar>)
    {
        if (next < channels)
            return std::min(next, channels - width);
    }
    return next;
}

#if (CV_SIMD || CV_SIMD_SCALABLE)
template<typename T, typename V>
static inline V bicubicChannelMulAdd(const V& value, const V& weight, const V& sum)
{
    if constexpr (std::is_same_v<T, uchar>)
        return v_fma(value, weight, sum);
    else
        return v_add(sum, v_mul(value, weight));
}
#endif

struct BilinearHalfAxis
{
    int first;
    int second;
    float fraction;
};

static inline BilinearHalfAxis bilinearHalfAxis(float value, int length, int borderType)
{
    double position = value;
    if (!std::isfinite(position))
        position = borderType == BORDER_REPLICATE && !std::isnan(position) ?
                   (position > 0 ? double(length) : -1.) : 0.;
    const double base = std::floor(position);
    const float fraction = float(position - base);
    if (base >= 0 && base + 1 < length)
        return {int(base), int(base + 1), fraction};
    if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT)
        return {base >= 0 && base < length ? int(base) : -1,
                base + 1 >= 0 && base + 1 < length ? int(base + 1) : -1, fraction};
    if (borderType == BORDER_REPLICATE)
    {
        const auto clampIndex = [length](double index) {
            return index < 0 ? 0 : index >= length ? length - 1 : int(index);
        };
        return {clampIndex(base), clampIndex(base + 1), fraction};
    }
    if (length == 1)
        return {0, 0, 0.f};
    if (borderType != BORDER_WRAP && borderType != BORDER_REFLECT && borderType != BORDER_REFLECT_101)
        CV_Error(Error::StsBadArg, "Unsupported border mode");
    if (base >= std::numeric_limits<int>::min()/2 && base <= std::numeric_limits<int>::max()/2 &&
        length <= std::numeric_limits<int>::max()/2)
    {
        const int index = int(base);
        return {borderInterpolate_fast(index, length, borderType),
                borderInterpolate_fast(index + 1, length, borderType), fraction};
    }
    const int delta = borderType == BORDER_REFLECT_101;
    const double period = borderType == BORDER_WRAP ? double(length) : 2.*(length - delta);
    double phase = std::fmod(base, period);
    if (phase < 0)
        phase += period;
    const auto borderIndex = [=](double index) {
        if (index >= period)
            index -= period;
        return int(index < length ? index : period - index - 1 + delta);
    };
    return {borderIndex(phase), borderIndex(phase + 1), fraction};
}

static void bilinearWarpHalf(const float* x, const float* y, int len,
                              const void* source, size_t step, Size size, void* destination,
                              const float*, int borderType, const void* borderValue, int channels)
{
    const hfloat* fill = static_cast<const hfloat*>(borderValue);
    hfloat* dst = static_cast<hfloat*>(destination);
    for (int i = 0; i < len; ++i, dst += channels)
    {
        const float sx = x[i], sy = y[i];
        if (sx >= 0.f && sx < size.width - 1 && sy >= 0.f && sy < size.height - 1)
        {
            const int col = static_cast<int>(sx), row = static_cast<int>(sy);
            const float alpha = sx - col, beta = sy - row;
            const float w0 = (1.f - alpha)*(1.f - beta), w1 = alpha*(1.f - beta);
            const float w2 = (1.f - alpha)*beta, w3 = alpha*beta;
            const hfloat* top = reinterpret_cast<const hfloat*>(
                static_cast<const uchar*>(source) + size_t(row)*step) + 3*size_t(col);
            const hfloat* bottom = reinterpret_cast<const hfloat*>(reinterpret_cast<const uchar*>(top) + step);
            bilinearHalfRGB(top, bottom, w0, w1, w2, w3, dst);
            continue;
        }
        if (borderType == BORDER_CONSTANT || borderType == BORDER_TRANSPARENT)
        {
            const bool outside = !std::isfinite(sx) || !std::isfinite(sy) ||
                                 (borderType == BORDER_TRANSPARENT ? sx < -1.f || sy < -1.f : sx <= -1.f || sy <= -1.f) ||
                                 double(sx) >= size.width || double(sy) >= size.height;
            if (outside)
            {
                if (borderType == BORDER_CONSTANT)
                    std::copy(fill, fill + channels, dst);
                continue;
            }
        }
        const BilinearHalfAxis horizontal = bilinearHalfAxis(sx, size.width, borderType);
        const BilinearHalfAxis vertical = bilinearHalfAxis(sy, size.height, borderType);
        const float alpha = horizontal.fraction, beta = vertical.fraction;
        const float weights[] = {(1.f - alpha)*(1.f - beta), alpha*(1.f - beta),
                                 (1.f - alpha)*beta, alpha*beta};
        const int columns[] = {horizontal.first, horizontal.second};
        const int rows[] = {vertical.first, vertical.second};
        const hfloat* pixels[4];
        for (int row = 0; row < 2; ++row)
        {
            const hfloat* rowData = rows[row] < 0 ? nullptr : reinterpret_cast<const hfloat*>(
                static_cast<const uchar*>(source) + size_t(rows[row])*step);
            for (int column = 0; column < 2; ++column)
                pixels[row*2 + column] = rowData && columns[column] >= 0 ?
                                         rowData + size_t(columns[column])*channels : nullptr;
        }
        for (int channel = 0; channel < channels; ++channel)
        {
            // Taps outside the image read the existing destination pixel under BORDER_TRANSPARENT, like float32.
            const float border = borderType == BORDER_CONSTANT ? float(fill[channel]) :
                                 borderType == BORDER_TRANSPARENT ? float(dst[channel]) : 0.f;
            const float v0 = pixels[0] ? float(pixels[0][channel]) : border;
            const float v1 = pixels[1] ? float(pixels[1][channel]) : border;
            const float v2 = pixels[2] ? float(pixels[2][channel]) : border;
            const float v3 = pixels[3] ? float(pixels[3][channel]) : border;
            dst[channel] = hfloat(v0*weights[0] + v1*weights[1] + v2*weights[2] + v3*weights[3]);
        }
    }
}


#if CV_AVX2 && CV_FP16
static inline __m256 loadEightWarpHalf(const hfloat* src)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
}
#endif

#if CV_FP16 && (CV_AVX2 || (CV_NEON && CV_SIMD128))
static inline void storeThreeWarpHalf(const v_float32x4& rgb, hfloat* dst)
{
    hfloat packed[4];
    v_pack_store(packed, rgb);
    std::memcpy(dst, packed, 3*sizeof(hfloat));
}

#endif

#if CV_AVX2 && CV_FP16 && CV_SIMD_WIDTH == 32
static inline void storeWarpWeights8(__m256 w0, __m256 w1, __m256 w2, __m256 w3,
                                    float (&weights)[8][4])
{
    __m128 lower0 = _mm256_castps256_ps128(w0), lower1 = _mm256_castps256_ps128(w1);
    __m128 lower2 = _mm256_castps256_ps128(w2), lower3 = _mm256_castps256_ps128(w3);
    __m128 upper0 = _mm256_extractf128_ps(w0, 1), upper1 = _mm256_extractf128_ps(w1, 1);
    __m128 upper2 = _mm256_extractf128_ps(w2, 1), upper3 = _mm256_extractf128_ps(w3, 1);
    _MM_TRANSPOSE4_PS(lower0, lower1, lower2, lower3);
    _MM_TRANSPOSE4_PS(upper0, upper1, upper2, upper3);
    _mm_storeu_ps(weights[0], lower0);
    _mm_storeu_ps(weights[1], lower1);
    _mm_storeu_ps(weights[2], lower2);
    _mm_storeu_ps(weights[3], lower3);
    _mm_storeu_ps(weights[4], upper0);
    _mm_storeu_ps(weights[5], upper1);
    _mm_storeu_ps(weights[6], upper2);
    _mm_storeu_ps(weights[7], upper3);
}

static inline bool fullWarpFootprints8(__m256 x, __m256 y, __m256 minimum, __m256 maxX, __m256 maxY)
{
    const __m256 validX = _mm256_and_ps(_mm256_cmp_ps(x, minimum, _CMP_GE_OQ),
                                        _mm256_cmp_ps(x, maxX, _CMP_LT_OQ));
    const __m256 validY = _mm256_and_ps(_mm256_cmp_ps(y, minimum, _CMP_GE_OQ),
                                        _mm256_cmp_ps(y, maxY, _CMP_LT_OQ));
    return _mm256_movemask_ps(_mm256_and_ps(validX, validY)) == 0xff;
}

static inline void bilinearWarpHalfRGBWide(const hfloat* top, size_t step, const float* weights, hfloat* dst)
{
    const __m256 row0 = loadEightWarpHalf(top), row1 = loadEightWarpHalf(top + step);
    const __m128 tl = _mm256_castps256_ps128(row0), bl = _mm256_castps256_ps128(row1);
    const __m128 tr = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(_mm256_extractf128_ps(row0, 1)),
                                                     _mm_castps_si128(tl), 12));
    const __m128 br = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(_mm256_extractf128_ps(row1, 1)),
                                                     _mm_castps_si128(bl), 12));
    __m128 sum = _mm_add_ps(_mm_mul_ps(tl, _mm_set1_ps(weights[0])), _mm_mul_ps(tr, _mm_set1_ps(weights[1])));
    sum = _mm_add_ps(sum, _mm_mul_ps(bl, _mm_set1_ps(weights[2])));
    sum = _mm_add_ps(sum, _mm_mul_ps(br, _mm_set1_ps(weights[3])));
    storeThreeWarpHalf(v_float32x4(sum), dst);
}

static void bilinearWarpHalfRGBBlocks(const float* x, const float* y, int len,
                                      const void* source, size_t step, Size size, void* destination,
                                      const float* params, int borderType, const void* borderValue, int channels)
{
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.f);
    // Each load covers eight halves. A third source pixel must be available at the right.
    const __m256 maxX = _mm256_set1_ps(float(size.width - 2));
    const __m256 maxY = _mm256_set1_ps(float(size.height - 1));
    const size_t halfStep = step/sizeof(hfloat);
    const hfloat* src = static_cast<const hfloat*>(source);
    hfloat* dst = static_cast<hfloat*>(destination);
    const int block = 8;
    int i = 0;
    for (; i + block <= len; i += block, dst += 3*block)
    {
        const __m256 sx = _mm256_loadu_ps(x + i), sy = _mm256_loadu_ps(y + i);
        if (!fullWarpFootprints8(sx, sy, zero, maxX, maxY))
        {
            bilinearWarpHalf(x + i, y + i, block, source, step, size, dst, params,
                                   borderType, borderValue, channels);
            continue;
        }

        const __m256i ix = _mm256_cvttps_epi32(sx), iy = _mm256_cvttps_epi32(sy);
        int ixValues[block], iyValues[block];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ixValues), ix);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(iyValues), iy);
        const __m256 ax = _mm256_sub_ps(sx, _mm256_cvtepi32_ps(ix));
        const __m256 ay = _mm256_sub_ps(sy, _mm256_cvtepi32_ps(iy));
        const __m256 nx = _mm256_sub_ps(one, ax), ny = _mm256_sub_ps(one, ay);
        float weights[block][4];
        storeWarpWeights8(_mm256_mul_ps(nx, ny), _mm256_mul_ps(ax, ny),
                          _mm256_mul_ps(nx, ay), _mm256_mul_ps(ax, ay), weights);
        for (int lane = 0; lane < block; ++lane)
        {
            const hfloat* top = src + size_t(iyValues[lane])*halfStep + 3*size_t(ixValues[lane]);
            bilinearWarpHalfRGBWide(top, halfStep, weights[lane], dst + 3*lane);
        }
    }

    if (i < len)
        bilinearWarpHalf(x + i, y + i, len - i, source, step, size, dst, params,
                               borderType, borderValue, channels);
}
#endif

ImgWarpFunc getBilinearHalfWarpFunc_()
{
#if CV_AVX2 && CV_FP16 && CV_SIMD_WIDTH == 32
    return bilinearWarpHalfRGBBlocks;
#else
    return bilinearWarpHalf;
#endif
}

#if CV_AVX2 && CV_FP16
template<bool relative>
static void remapBilinearHalfRGBFixed(const hfloat* src, size_t step, hfloat* dst,
                                      const short* xy, const ushort* fractions,
                                      const float* weights, int length, const Point& offset)
{
    for (int i = 0; i < length; ++i, dst += 3)
    {
        const int x = xy[2*i] + (relative ? offset.x + i : 0);
        const int y = xy[2*i + 1] + (relative ? offset.y : 0);
        const hfloat* top = src + size_t(y)*step + 3*size_t(x);
        const float* w = weights + 4*fractions[i];
        bilinearHalfRGB(top, top + step, w[0], w[1], w[2], w[3], dst);
    }
}
#endif

RemapBilinearHalfRGBFunc getRemapBilinearHalfRGB_(bool relative)
{
#if CV_AVX2 && CV_FP16
    return relative ? remapBilinearHalfRGBFixed<true> : remapBilinearHalfRGBFixed<false>;
#else
    CV_UNUSED(relative);
    return nullptr;
#endif
}

#if CV_AVX2 && CV_FP16
static inline void sumPackedLanczosRGB(const float* values, hfloat* dst)
{
    float red = values[0], green = values[1], blue = values[2];
    for (int tap = 1; tap < 8; ++tap)
    {
        red += values[3*tap];
        green += values[3*tap + 1];
        blue += values[3*tap + 2];
    }
    storeThreeWarpHalf(v_float32x4(red, green, blue, 0.f), dst);
}

static inline void bicubicWarpHalfRGB(const hfloat* src, size_t step,
                                      const float* wx, const float* wy, hfloat* dst)
{
    // Four RGB taps occupy 12 halves. The loads read exactly eight and four per row.
    __m256 low = _mm256_setzero_ps();
    __m128 high = _mm_setzero_ps();
    for (int row = 0; row < 4; ++row)
    {
        const hfloat* samples = src + size_t(row)*step;
        low = _mm256_add_ps(low, _mm256_mul_ps(loadEightWarpHalf(samples), _mm256_set1_ps(wy[row])));
        high = _mm_add_ps(high, _mm_mul_ps(loadBilinearFourHalf(samples + 8).val, _mm_set1_ps(wy[row])));
    }

    const __m128 w = _mm_loadu_ps(wx);
    const __m256 lowWeights = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(w),
                                                       _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2));
    const __m128 highWeights = _mm_shuffle_ps(w, w, _MM_SHUFFLE(3, 3, 3, 2));
    low = _mm256_mul_ps(low, lowWeights);
    high = _mm_mul_ps(high, highWeights);

    // Shift the packed groups into RGB lanes before adding the four column taps.
    const __m128i first = _mm_castps_si128(_mm256_castps256_ps128(low));
    const __m128i middle = _mm_castps_si128(_mm256_extractf128_ps(low, 1));
    const __m128i last = _mm_castps_si128(high);
    const __m128 tap1 = _mm_castsi128_ps(_mm_alignr_epi8(middle, first, 12));
    const __m128 tap2 = _mm_castsi128_ps(_mm_alignr_epi8(last, middle, 8));
    const __m128 tap3 = _mm_castsi128_ps(_mm_srli_si128(last, 4));
    const __m128 rgb = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_castsi128_ps(first), tap1), tap2), tap3);
    storeThreeWarpHalf(v_float32x4(rgb), dst);
}

static void remapLanczosHalfRGB(const hfloat* src, size_t step, const float* weights, hfloat* dst)
{
    // Three eight-half loads cover one row of eight RGB taps.
    const __m256i taps0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i taps1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i taps2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int row = 0; row < 8; ++row)
    {
        const hfloat* samples = src + size_t(row)*step;
        const __m256 w = _mm256_loadu_ps(weights + 8*row);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(loadEightWarpHalf(samples), _mm256_permutevar8x32_ps(w, taps0)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(loadEightWarpHalf(samples + 8), _mm256_permutevar8x32_ps(w, taps1)));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(loadEightWarpHalf(samples + 16), _mm256_permutevar8x32_ps(w, taps2)));
    }

    float values[24];
    _mm256_storeu_ps(values, sum0);
    _mm256_storeu_ps(values + 8, sum1);
    _mm256_storeu_ps(values + 16, sum2);
    sumPackedLanczosRGB(values, dst);
}
#elif CV_NEON && CV_FP16 && CV_SIMD128
static inline void expandFourPackedRGBWeights(const float* w, v_float32x4* packed)
{
    packed[0] = v_float32x4(w[0], w[0], w[0], w[1]);
    packed[1] = v_float32x4(w[1], w[1], w[2], w[2]);
    packed[2] = v_float32x4(w[2], w[3], w[3], w[3]);
}

static inline v_float32x4 sumFourPackedWarpRGB(const v_float32x4* taps)
{
    // Align the start of each RGB tap to the first lane before summing.
    const v_float32x4 tap1 = v_rotate_right<3>(taps[0], taps[1]);
    const v_float32x4 tap2 = v_rotate_right<2>(taps[1], taps[2]);
    const v_float32x4 tap3 = v_rotate_right<1>(taps[2]);
    return v_add(v_add(v_add(taps[0], tap1), tap2), tap3);
}

static inline void bicubicWarpHalfRGB(const hfloat* src, size_t step,
                                      const float* wx, const float* wy, hfloat* dst)
{
    const v_float32x4 zero = v_setall_f32(0.f);
    v_float32x4 columns[3] = {zero, zero, zero};
    for (int row = 0; row < 4; ++row)
    {
        const hfloat* samples = src + size_t(row)*step;
        const v_float32x4 weight = v_setall_f32(wy[row]);
        for (int group = 0; group < 3; ++group)
            columns[group] = v_fma(loadBilinearFourHalf(samples + 4*group), weight, columns[group]);
    }

    v_float32x4 weights[3];
    expandFourPackedRGBWeights(wx, weights);
    for (int group = 0; group < 3; ++group)
        columns[group] = v_mul(columns[group], weights[group]);
    storeThreeWarpHalf(sumFourPackedWarpRGB(columns), dst);
}

static void remapLanczosHalfRGB(const hfloat* src, size_t step, const float* weights, hfloat* dst)
{
    const v_float32x4 zero = v_setall_f32(0.f);
    v_float32x4 sums[6] = {zero, zero, zero, zero, zero, zero};
    for (int row = 0; row < 8; ++row)
    {
        const hfloat* samples = src + size_t(row)*step;
        const float* w = weights + 8*row;
        v_float32x4 packed[6];
        expandFourPackedRGBWeights(w, packed);
        expandFourPackedRGBWeights(w + 4, packed + 3);
        for (int group = 0; group < 6; ++group)
            sums[group] = v_fma(loadBilinearFourHalf(samples + 4*group), packed[group], sums[group]);
    }
    storeThreeWarpHalf(v_add(sumFourPackedWarpRGB(sums), sumFourPackedWarpRGB(sums + 3)), dst);
}
#endif

RemapLanczosHalfRGBFunc getRemapLanczosHalfRGB_()
{
#if CV_FP16 && (CV_AVX2 || (CV_NEON && CV_SIMD128))
    return remapLanczosHalfRGB;
#else
    return nullptr;
#endif
}

template<typename T>
static void bicubicWarpChannels(const float* x, const float* y, int len,
                                const void* source, size_t step, Size size, void* destination,
                                const float* params, int borderType, const void* borderValue, int channels)
{
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const float coefficient = params ? *params : -0.75f;
    const T* fill = static_cast<const T*>(borderValue);
    T* dst = static_cast<T*>(destination);
    const int sampleBorder = borderType == BORDER_TRANSPARENT ? BORDER_CONSTANT : borderType;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int lanes = VTraits<v_float32>::vlanes();
    [[maybe_unused]] const uchar* srcEnd = static_cast<const uchar*>(source) +
        size_t(size.height - 1)*step + size_t(size.width)*channels*sizeof(T);
#endif
    for (int i = 0; i < len; ++i, dst += channels)
    {
        int left, top;
        float wx[4], wy[4];
        const int code = bicubicCoeffs(x[i], y[i], size, coefficient, borderType, left, top,
                                      wx[0], wx[1], wx[2], wx[3], wy[0], wy[1], wy[2], wy[3]);
        if (code < 0)
        {
            if (borderType == BORDER_CONSTANT)
                std::copy(fill, fill + channels, dst);
            continue;
        }
        if (code > 0)
        {
#if CV_FP16 && (CV_AVX2 || (CV_NEON && CV_SIMD128))
            if constexpr (std::is_same_v<T, hfloat>)
            {
                if (channels == 3)
                {
                    const hfloat* row = reinterpret_cast<const hfloat*>(
                        static_cast<const uchar*>(source) + size_t(top)*step) + 3*size_t(left);
                    bicubicWarpHalfRGB(row, step/sizeof(hfloat), wx, wy, dst);
                    continue;
                }
            }
#endif
            const T* rows[4];
            for (int row = 0; row < 4; ++row)
                rows[row] = reinterpret_cast<const T*>(static_cast<const uchar*>(source) + size_t(top + row)*step)
                            + left*channels;
            int channel = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            if constexpr (!std::is_same_v<T, double> && !std::is_same_v<T, hfloat>)
            {
                const v_float32 w0 = vx_setall_f32(wx[0]), w1 = vx_setall_f32(wx[1]);
                const v_float32 w2 = vx_setall_f32(wx[2]), w3 = vx_setall_f32(wx[3]);
                for (; channel <= channels - lanes; channel = bicubicNextChannel<T>(channel, channels, lanes))
                {
                    v_float32 sum = vx_setzero_f32();
                    for (int row = 0; row < 4; ++row)
                    {
                        const T* ptr = rows[row] + channel;
                        v_float32 horizontal = v_mul(bicubicLoadChannels(ptr, srcEnd, lanes), w0);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels(ptr + channels, srcEnd, lanes), w1, horizontal);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels(ptr + channels*2, srcEnd, lanes), w2, horizontal);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels(ptr + channels*3, srcEnd, lanes), w3, horizontal);
                        sum = bicubicChannelMulAdd<T>(horizontal, vx_setall_f32(wy[row]), sum);
                    }
                    bicubicStoreChannels(dst + channel, sum, lanes);
                }
            }
#endif
#if CV_SIMD128
            if constexpr (!std::is_same_v<T, double>)
            {
                const v_float32x4 w0 = v_setall_f32(wx[0]), w1 = v_setall_f32(wx[1]);
                const v_float32x4 w2 = v_setall_f32(wx[2]), w3 = v_setall_f32(wx[3]);
                for (; channel <= channels - 4; channel = bicubicNextChannel<T>(channel, channels, 4))
                {
                    v_float32x4 sum = v_setzero_f32();
                    for (int row = 0; row < 4; ++row)
                    {
                        const T* ptr = rows[row] + channel;
                        v_float32x4 horizontal = v_mul(bicubicLoadChannels4(ptr, srcEnd), w0);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels4(ptr + channels, srcEnd), w1, horizontal);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels4(ptr + channels*2, srcEnd), w2, horizontal);
                        horizontal = bicubicChannelMulAdd<T>(bicubicLoadChannels4(ptr + channels*3, srcEnd), w3, horizontal);
                        sum = bicubicChannelMulAdd<T>(horizontal, v_setall_f32(wy[row]), sum);
                    }
                    bicubicStoreChannels4(dst + channel, sum);
                }
            }
#endif
            for (; channel < channels; ++channel)
            {
                Acc sum = 0;
                for (int row = 0; row < 4; ++row)
                {
                    const T* ptr = rows[row] + channel;
                    sum += (Acc(ptr[0])*wx[0] + Acc(ptr[channels])*wx[1] +
                            Acc(ptr[channels*2])*wx[2] + Acc(ptr[channels*3])*wx[3])*wy[row];
                }
                dst[channel] = saturate_cast<T>(sum);
            }
            continue;
        }
        int columns[4], rows[4];
        for (int tap = 0; tap < 4; ++tap)
        {
            columns[tap] = borderInterpolate(left + tap, size.width, sampleBorder);
            rows[tap] = borderInterpolate(top + tap, size.height, sampleBorder);
        }
        const T* pixels[4][4];
        for (int row = 0; row < 4; ++row)
        {
            const int sourceRow = rows[row];
            const T* rowData = sourceRow < 0 ? nullptr : reinterpret_cast<const T*>(
                static_cast<const uchar*>(source) + size_t(sourceRow)*step);
            for (int column = 0; column < 4; ++column)
                pixels[row][column] = rowData && columns[column] >= 0 ? rowData + columns[column]*channels : nullptr;
        }
        for (int channel = 0; channel < channels; ++channel)
        {
            const Acc border = Acc(borderType == BORDER_TRANSPARENT ? dst[channel] : fill[channel]);
            Acc sum = 0;
            for (int row = 0; row < 4; ++row)
            {
                const Acc v0 = pixels[row][0] ? Acc(pixels[row][0][channel]) : border;
                const Acc v1 = pixels[row][1] ? Acc(pixels[row][1][channel]) : border;
                const Acc v2 = pixels[row][2] ? Acc(pixels[row][2][channel]) : border;
                const Acc v3 = pixels[row][3] ? Acc(pixels[row][3][channel]) : border;
                sum += (v0*wx[0] + v1*wx[1] + v2*wx[2] + v3*wx[3])*wy[row];
            }
            dst[channel] = saturate_cast<T>(sum);
        }
    }
}

#if CV_AVX2 && CV_FP16 && CV_SIMD_WIDTH == 32
static inline void cubicRGBBlockWeights(__m256 fraction, float coefficient, float (&weights)[8][4])
{
    v_float32 w0, w1, w2, w3;
    bicubicWeights(v_float32(fraction), coefficient, w0, w1, w2, w3);
    storeWarpWeights8(w0.val, w1.val, w2.val, w3.val, weights);
}

static void bicubicWarpHalfRGBBlocks(const float* x, const float* y, int len,
                                     const void* source, size_t step, Size size, void* destination,
                                     const float* params, int borderType, const void* borderValue, int channels)
{
    const float coefficient = params ? *params : -0.75f;
    const __m256 minimum = _mm256_set1_ps(1.f);
    const __m256 maxX = _mm256_set1_ps(float(size.width - 2));
    const __m256 maxY = _mm256_set1_ps(float(size.height - 2));
    const size_t halfStep = step/sizeof(hfloat);
    const hfloat* src = static_cast<const hfloat*>(source);
    hfloat* dst = static_cast<hfloat*>(destination);
    const int block = 8;
    int i = 0;
    for (; i + block <= len; i += block, dst += 3*block)
    {
        const __m256 sx = _mm256_loadu_ps(x + i), sy = _mm256_loadu_ps(y + i);
        if (!fullWarpFootprints8(sx, sy, minimum, maxX, maxY))
        {
            bicubicWarpChannels<hfloat>(x + i, y + i, block, source, step, size, dst, params,
                                       borderType, borderValue, channels);
            continue;
        }

        // Complete footprints have positive coordinates, so truncation gives the floor.
        const __m256i ix = _mm256_cvttps_epi32(sx), iy = _mm256_cvttps_epi32(sy);
        int ixValues[block], iyValues[block];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ixValues), ix);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(iyValues), iy);
        float wx[block][4], wy[block][4];
        cubicRGBBlockWeights(_mm256_sub_ps(sx, _mm256_cvtepi32_ps(ix)), coefficient, wx);
        cubicRGBBlockWeights(_mm256_sub_ps(sy, _mm256_cvtepi32_ps(iy)), coefficient, wy);

        for (int lane = 0; lane < block; ++lane)
        {
            const hfloat* top = src + size_t(iyValues[lane] - 1)*halfStep + 3*size_t(ixValues[lane] - 1);
            bicubicWarpHalfRGB(top, halfStep, wx[lane], wy[lane], dst + 3*lane);
        }
    }

    if (i < len)
        bicubicWarpChannels<hfloat>(x + i, y + i, len - i, source, step, size, dst, params,
                                   borderType, borderValue, channels);
}
#elif CV_NEON && CV_FP16 && CV_SIMD128 && CV_SIMD_WIDTH == 16
static inline void cubicRGBBlockWeights(const v_float32x4& fraction, float coefficient, float (&weights)[4][4])
{
    v_float32x4 w0, w1, w2, w3;
    v_float32x4 p0, p1, p2, p3;
    bicubicWeights(fraction, coefficient, w0, w1, w2, w3);
    v_transpose4x4(w0, w1, w2, w3, p0, p1, p2, p3);
    v_store(weights[0], p0);
    v_store(weights[1], p1);
    v_store(weights[2], p2);
    v_store(weights[3], p3);
}

static void bicubicWarpHalfRGBBlocks(const float* x, const float* y, int len,
                                     const void* source, size_t step, Size size, void* destination,
                                     const float* params, int borderType, const void* borderValue, int channels)
{
    const float coefficient = params ? *params : -0.75f;
    const v_float32x4 minimum = v_setall_f32(1.f);
    const v_float32x4 maxX = v_setall_f32(float(size.width - 2));
    const v_float32x4 maxY = v_setall_f32(float(size.height - 2));
    const size_t halfStep = step/sizeof(hfloat);
    const hfloat* src = static_cast<const hfloat*>(source);
    hfloat* dst = static_cast<hfloat*>(destination);
    const int block = 4;
    int i = 0;
    for (; i + block <= len; i += block, dst += 3*block)
    {
        const v_float32x4 sx = v_load(x + i), sy = v_load(y + i);
        const v_float32x4 inX = v_and(v_ge(sx, minimum), v_lt(sx, maxX));
        const v_float32x4 inY = v_and(v_ge(sy, minimum), v_lt(sy, maxY));
        if (!v_check_all(v_and(inX, inY)))
        {
            bicubicWarpChannels<hfloat>(x + i, y + i, block, source, step, size, dst, params,
                                       borderType, borderValue, channels);
            continue;
        }

        const v_int32x4 ix = v_trunc(sx), iy = v_trunc(sy);
        int ixValues[block], iyValues[block];
        v_store(ixValues, ix);
        v_store(iyValues, iy);
        float wx[block][4], wy[block][4];
        cubicRGBBlockWeights(v_sub(sx, v_cvt_f32(ix)), coefficient, wx);
        cubicRGBBlockWeights(v_sub(sy, v_cvt_f32(iy)), coefficient, wy);

        for (int lane = 0; lane < block; ++lane)
        {
            const hfloat* top = src + size_t(iyValues[lane] - 1)*halfStep + 3*size_t(ixValues[lane] - 1);
            bicubicWarpHalfRGB(top, halfStep, wx[lane], wy[lane], dst + 3*lane);
        }
    }

    if (i < len)
        bicubicWarpChannels<hfloat>(x + i, y + i, len - i, source, step, size, dst, params,
                                   borderType, borderValue, channels);
}
#endif

template<typename T, void (*fn)(const float*, const float*, int, const void*, size_t, Size, T*, const float*, int, T*)>
static void bicubicWarpWrap(const float* x, const float* y, int len, const void* src, size_t srcstep, Size srcsize, void* dst, const float* coeffs, int flags, const void* fillval, int)
{
    fn(x, y, len, src, srcstep, srcsize, (T*)dst, coeffs, flags, (T*)fillval);
}

ImgWarpFunc getBicubicWarpFunc_(int type)
{
#if CV_FP16 && ((CV_AVX2 && CV_SIMD_WIDTH == 32) || (CV_NEON && CV_SIMD128 && CV_SIMD_WIDTH == 16))
    if (type == CV_16FC3)
        return bicubicWarpHalfRGBBlocks;
#endif
#if CV_SIMD
    if (type == CV_16FC1)
        return bicubicWarpWrap<hfloat, bicubic16fC1>;
#endif
    if (CV_MAT_DEPTH(type) == CV_16F)
        return bicubicWarpChannels<hfloat>;
    if (CV_MAT_CN(type) > 4)
    {
        switch (CV_MAT_DEPTH(type))
        {
        case CV_8U: return bicubicWarpChannels<uchar>;
        case CV_16U: return bicubicWarpChannels<ushort>;
        case CV_16S: return bicubicWarpChannels<short>;
        case CV_32F: return bicubicWarpChannels<float>;
        case CV_64F: return bicubicWarpChannels<double>;
        default: return nullptr;
        }
    }
    if (type == CV_8UC1) {
        return bicubicWarpWrap<uint8_t, bicubic8uC1>;
    }
    if (type == CV_8UC2) {
        return bicubicWarpWrap<uint8_t, bicubic8uC2>;
    }
    if (type == CV_8UC3) {
        return bicubicWarpWrap<uint8_t, bicubic8uC3>;
    }
    if (type == CV_8UC4) {
        return bicubicWarpWrap<uint8_t, bicubic8uC4>;
    }
    if (type == CV_16UC1) {
        return bicubicWarpWrap<uint16_t, bicubic16uC1>;
    }
    if (type == CV_16UC2) {
        return bicubicWarpWrap<uint16_t, bicubic16uC2>;
    }
    if (type == CV_16UC3) {
        return bicubicWarpWrap<uint16_t, bicubic16uC3>;
    }
    if (type == CV_16UC4) {
        return bicubicWarpWrap<uint16_t, bicubic16uC4>;
    }
    if (type == CV_16SC1) {
        return bicubicWarpWrap<int16_t, bicubic16sC1>;
    }
    if (type == CV_16SC2) {
        return bicubicWarpWrap<int16_t, bicubic16sC2>;
    }
    if (type == CV_16SC3) {
        return bicubicWarpWrap<int16_t, bicubic16sC3>;
    }
    if (type == CV_16SC4) {
        return bicubicWarpWrap<int16_t, bicubic16sC4>;
    }
    if (type == CV_32FC1) {
        return bicubicWarpWrap<float, bicubic32fC1>;
    }
    if (type == CV_32FC2) {
        return bicubicWarpWrap<float, bicubic32fC2>;
    }
    if (type == CV_32FC3) {
        return bicubicWarpWrap<float, bicubic32fC3>;
    }
    if (type == CV_32FC4) {
        return bicubicWarpWrap<float, bicubic32fC4>;
    }
    if (type == CV_64FC1) {
        return bicubicWarpWrap<double, bicubic64fC1>;
    }
    if (type == CV_64FC2) {
        return bicubicWarpWrap<double, bicubic64fC2>;
    }
    if (type == CV_64FC3) {
        return bicubicWarpWrap<double, bicubic64fC3>;
    }
    if (type == CV_64FC4) {
        return bicubicWarpWrap<double, bicubic64fC4>;
    }
    return (ImgWarpFunc)nullptr;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
