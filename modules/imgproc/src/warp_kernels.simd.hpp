// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <numeric>
#include "precomp.hpp"
#include "warp_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

#define CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 M0 = vx_setall_f32(M[0]), \
              M3 = vx_setall_f32(M[3]); \
    v_float32 M_x = vx_setall_f32(static_cast<float>(y * M[1] + M[2])), \
              M_y = vx_setall_f32(static_cast<float>(y * M[4] + M[5]));
#define CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1() \
    CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 M6 = vx_setall_f32(M[6]); \
    v_float32 M_w = vx_setall_f32(static_cast<float>(y * M[7] + M[8]));
#define CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1() \
    v_float32 dst_x0 = vx_load(start_indices.data()); \
    v_float32 dst_x1 = v_add(dst_x0, vx_setall_f32(float(vlanes_32))); \
    v_float32 dst_y = vx_setall_f32(float(y));

#define CV_WARP_LINEAR_VECTOR_GET_ADDR_C1() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, src_ix0), \
            addr_1 = v_fma(v_srcstep, src_iy1, src_ix1);
#define CV_WARP_LINEAR_VECTOR_GET_ADDR_C3() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, three)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, three));
#define CV_WARP_LINEAR_VECTOR_GET_ADDR_C4() \
    v_int32 addr_0 = v_fma(v_srcstep, src_iy0, v_mul(src_ix0, four)), \
            addr_1 = v_fma(v_srcstep, src_iy1, v_mul(src_ix1, four));
#define CV_WARP_LINEAR_VECTOR_GET_ADDR(CN) \
    CV_WARP_LINEAR_VECTOR_GET_ADDR_##CN() \
    vx_store(addr, addr_0); \
    vx_store(addr + vlanes_32, addr_1);

#define CV_WARP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD(CN) \
    v_int32 src_ix0 = v_floor(src_x0), \
            src_iy0 = v_floor(src_y0), \
            src_ix1 = v_floor(src_x1), \
            src_iy1 = v_floor(src_y1); \
    v_uint32 mask_0 = v_lt(v_reinterpret_as_u32(src_ix0), inner_scols), \
             mask_1 = v_lt(v_reinterpret_as_u32(src_ix1), inner_scols); \
    mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(src_iy0), inner_srows)); \
    mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(src_iy1), inner_srows)); \
    v_uint16 inner_mask = v_pack(mask_0, mask_1); \
    src_x0 = v_sub(src_x0, v_cvt_f32(src_ix0)); \
    src_y0 = v_sub(src_y0, v_cvt_f32(src_iy0)); \
    src_x1 = v_sub(src_x1, v_cvt_f32(src_ix1)); \
    src_y1 = v_sub(src_y1, v_cvt_f32(src_iy1)); \
    CV_WARP_LINEAR_VECTOR_GET_ADDR(CN);

#define CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
    CV_WARP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD(CN)

#define CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(CN) \
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
    CV_WARP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD(CN)

#define CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(CN) \
    v_float32 src_x0, src_y0, \
              src_x1, src_y1; \
    if (map2 == nullptr) { \
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
    CV_WARP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD(CN)

namespace cv{
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C1);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C1);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C3);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C3);
    #endif
                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};
    #endif
#endif

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C4)
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C4);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C4);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*4];

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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*4];

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
            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                uint8x8_t p00g, p01g, p10g, p11g;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx = x*M[0] + y*M[1] + M[2];
                float sy = x*M[3] + y*M[4] + M[5];

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C1);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C1);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C3);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C3);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
        uint8x8_t reds = {0, 8, 16, 24, 4, 12, 20, 28},
                  greens = {1, 9, 17, 25, 5, 13, 21, 29},
                  blues = {2, 10, 18, 26, 6, 14, 22, 30},
                  alphas = {3, 11, 19, 27, 7, 15, 23, 31};
    #endif
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

        for (int y = r.start; y < r.end; y++) {
            uint8_t* dstptr = dst + y*dststep;
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C4);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C4);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*4];

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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*4];

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
            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                uint8x8_t p00g, p01g, p10g, p11g;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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

            CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_WARPPERSPECTIVE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float w = x*M[6] + y*M[7] + M[8];
                float sx = (x*M[0] + y*M[1] + M[2]) / w;
                float sy = (x*M[3] + y*M[4] + M[5]) / w;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00g, p01g, p10g, p11g;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C1);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C1);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;
    #endif

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
    #else
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 8U);
    #endif
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
    #endif
                }

    #if defined(CV_NEON_AARCH64) && CV_NEON_AARCH64 // In case neon fp16 intrinsics are not available; still requires A64
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16_NEON(C3);
    #else
                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8S16(C3);
    #endif

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_S16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
            }
        }
    };
    parallel_for_(Range(0, dst_rows), worker);
}

void remapLinearInvoker_8UC4(const uint8_t *src_data, size_t src_step, int src_rows, int src_cols,
                             uint8_t *dst_data, size_t dst_step, int dst_rows, int dst_cols,
                             int border_type, const double border_value[4],
                             const float *map1_data, size_t map1_step, const float *map2_data, size_t map2_step, bool is_relative) {
    // printf("In remapLinearInvoker_8UC4\n");
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    float valpha[max_uf], vbeta[max_uf];
                    v_store_low(valpha, src_x0);
                    v_store_high(valpha+vlanes_32, src_x1);
                    v_store_low(vbeta, src_y0);
                    v_store_high(vbeta+vlanes_32, src_y1);
        #if CV_SIMD128
                    for (int i = 0; i < uf; i+=vlanes_32) {
                        #define VECTOR_LOAD_AND_INTER(ofs) \
                            const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
                            auto i##ofs##_pix01 = v_reinterpret_as_s16(vx_load_expand(srcptr##ofs)); \
                            auto i##ofs##_pix23 = v_reinterpret_as_s16(vx_load_expand(srcptr##ofs+srcstep)); \
                            v_float32 i##ofs##_pix0 = v_cvt_f32(v_expand_low( i##ofs##_pix01)); \
                            v_float32 i##ofs##_pix1 = v_cvt_f32(v_expand_high(i##ofs##_pix01)); \
                            v_float32 i##ofs##_pix2 = v_cvt_f32(v_expand_low( i##ofs##_pix23)); \
                            v_float32 i##ofs##_pix3 = v_cvt_f32(v_expand_high(i##ofs##_pix23)); \
                            v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
                                      i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]);  \
                            i##ofs##_pix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix1, i##ofs##_pix0), i##ofs##_pix0); \
                            i##ofs##_pix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_pix3, i##ofs##_pix2), i##ofs##_pix2); \
                            i##ofs##_pix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_pix2, i##ofs##_pix0), i##ofs##_pix0);
                            VECTOR_LOAD_AND_INTER(0);
                            VECTOR_LOAD_AND_INTER(1);
                            VECTOR_LOAD_AND_INTER(2);
                            VECTOR_LOAD_AND_INTER(3);
                        #undef VECTOR_LOAD_AND_INTER

                        // pack and store
                        auto i01_pix = v_pack_u(v_round(i0_pix0), v_round(i1_pix0)),
                             i23_pix = v_pack_u(v_round(i2_pix0), v_round(i3_pix0));
                        v_pack_store(dstptr + 4*(x+i), i01_pix);
                        v_pack_store(dstptr + 4*(x+i+2), i23_pix);
                    }
        #elif CV_SIMD256
                    for (int i = 0; i < uf; i+=vlanes_32) {
                        #define SIMD256_LOAD_SHUFFLE_INTER(ofs0, ofs1) \
                            const uint8_t *srcptr##ofs0 = src + addr[i+ofs0]; \
                            const uint8_t *srcptr##ofs1 = src + addr[i+ofs1]; \
                            v_int32 i##ofs0##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0)), \
                                    i##ofs0##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs0+srcstep)); \
                            v_int32 i##ofs1##_pix01 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1)), \
                                    i##ofs1##_pix23 = v_reinterpret_as_s32(v256_load_expand_q(srcptr##ofs1+srcstep)); \
                            v_float32 i##ofs0##_fpix01 = v_cvt_f32(i##ofs0##_pix01), i##ofs0##_fpix23 = v_cvt_f32(i##ofs0##_pix23); \
                            v_float32 i##ofs1##_fpix01 = v_cvt_f32(i##ofs1##_pix01), i##ofs1##_fpix23 = v_cvt_f32(i##ofs1##_pix23); \
                            v_float32 i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11, \
                                      i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33; \
                            v256_zip(i##ofs0##_fpix01, i##ofs1##_fpix01, i##ofs0##ofs1##_fpix00, i##ofs0##ofs1##_fpix11); \
                            v256_zip(i##ofs0##_fpix23, i##ofs1##_fpix23, i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix33); \
                            v_float32 i##ofs0##_alpha = vx_setall_f32(valpha[i+ofs0]), \
                                      i##ofs1##_alpha = vx_setall_f32(valpha[i+ofs1]), \
                                      i##ofs0##_beta  = vx_setall_f32(vbeta[i+ofs0]), \
                                      i##ofs1##_beta  = vx_setall_f32(vbeta[i+ofs1]); \
                            v_float32 i##ofs0##ofs1##_alpha = v_combine_low(i##ofs0##_alpha, i##ofs1##_alpha), \
                                      i##ofs0##ofs1##_beta  = v_combine_low(i##ofs0##_beta,  i##ofs1##_beta); \
                            i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix11, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00); \
                            i##ofs0##ofs1##_fpix22 = v_fma(i##ofs0##ofs1##_alpha, v_sub(i##ofs0##ofs1##_fpix33, i##ofs0##ofs1##_fpix22), i##ofs0##ofs1##_fpix22); \
                            i##ofs0##ofs1##_fpix00 = v_fma(i##ofs0##ofs1##_beta,  v_sub(i##ofs0##ofs1##_fpix22, i##ofs0##ofs1##_fpix00), i##ofs0##ofs1##_fpix00);
                            SIMD256_LOAD_SHUFFLE_INTER(0, 1);
                            SIMD256_LOAD_SHUFFLE_INTER(2, 3);
                        #undef SIMD256_LOAD_SHUFFLE_INTER

                        // Store
                        auto i01_pix = v_round(i01_fpix00), i23_pix = v_round(i23_fpix00);
                        v_pack_store(dstptr + 4*x, v_pack_u(i01_pix, i23_pix));
                    }
        #else // CV_SIMD_SCALABLE
                    for (int i = 0; i < uf; i+=vlanes_32) {
                        #define VECTOR_LOAD_INTER(ofs) \
                            const uint8_t *srcptr##ofs = src + addr[i+ofs]; \
                            v_uint32 i##ofs##_pix0 = v_load_expand_q<4>(srcptr##ofs), \
                                     i##ofs##_pix1 = v_load_expand_q<4>(srcptr##ofs+4), \
                                     i##ofs##_pix2 = v_load_expand_q<4>(srcptr##ofs+srcstep), \
                                     i##ofs##_pix3 = v_load_expand_q<4>(srcptr##ofs+srcstep+4); \
                            v_float32 i##ofs##_fpix0 = v_cvt_f32(v_reinterpret_as_s32(i##ofs##_pix0)), \
                                      i##ofs##_fpix1 = v_cvt_f32(v_reinterpret_as_s32(i##ofs##_pix1)), \
                                      i##ofs##_fpix2 = v_cvt_f32(v_reinterpret_as_s32(i##ofs##_pix2)), \
                                      i##ofs##_fpix3 = v_cvt_f32(v_reinterpret_as_s32(i##ofs##_pix3)); \
                            v_float32 i##ofs##_alpha = vx_setall_f32(valpha[i+ofs]), \
                                      i##ofs##_beta  = vx_setall_f32(vbeta[i+ofs]); \
                            i##ofs##_fpix0 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix1, i##ofs##_fpix0), i##ofs##_fpix0); \
                            i##ofs##_fpix2 = v_fma(i##ofs##_alpha, v_sub(i##ofs##_fpix3, i##ofs##_fpix2), i##ofs##_fpix2); \
                            i##ofs##_fpix0 = v_fma(i##ofs##_beta,  v_sub(i##ofs##_fpix2, i##ofs##_fpix0), i##ofs##_fpix0);
                            VECTOR_LOAD_INTER(0);
                            VECTOR_LOAD_INTER(1);
                            VECTOR_LOAD_INTER(2);
                            VECTOR_LOAD_INTER(3);
                        #undef VECTOR_LOAD_INTER

                        // Pack and store
                        auto i01_pix = v_pack(v_round(i0_fpix0), v_round(i1_fpix0)),
                             i23_pix = v_pack(v_round(i2_fpix0), v_round(i3_fpix0));
                        v_pack_u_store<8>(dstptr + 4*(x+i), i01_pix);
                        v_pack_u_store<8>(dstptr + 4*(x+i+2), i23_pix);
                    }
        #endif
                } else {
                    int pixbuf[max_uf*4*4];
                    // CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);
                    if (border_type == BORDER_CONSTANT || border_type == BORDER_TRANSPARENT) {
                        mask_0 = v_lt(v_reinterpret_as_u32(v_add(src_ix0, one)), outer_scols);
                        mask_1 = v_lt(v_reinterpret_as_u32(v_add(src_ix1, one)), outer_scols);
                        mask_0 = v_and(mask_0, v_lt(v_reinterpret_as_u32(v_add(src_iy0, one)), outer_srows));
                        mask_1 = v_and(mask_1, v_lt(v_reinterpret_as_u32(v_add(src_iy1, one)), outer_srows));
                        v_uint16 outer_mask = v_pack(mask_0, mask_1);
                        if (v_reduce_max(outer_mask) == 0) {
                            if (border_type == BORDER_CONSTANT) {
                                v_store_low(dstptr + x*4,        bval_v0);
                                v_store_low(dstptr + x*4 + uf,   bval_v1);
                                v_store_low(dstptr + x*4 + uf*2, bval_v2);
                                v_store_low(dstptr + x*4 + uf*3, bval_v3);
                            }
                            continue;
                        }
                    }
                    vx_store(src_ix, src_ix0);
                    vx_store(src_iy, src_iy0);
                    vx_store(src_ix + vlanes_32, src_ix1);
                    vx_store(src_iy + vlanes_32, src_iy1);
                    for (int i = 0; i < uf; i++) {
                        int ix = src_ix[i], iy = src_iy[i];
                        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C4(0, 0, 0);
                        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C4(0, 1, uf);
                        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C4(1, 0, uf*2);
                        CV_WARP_LINEAR_VECTOR_FETCH_PIXEL_C4(1, 1, uf*3);
                    }
                    v_int32  f00r = vx_load(pixbuf + uf * 0),
                             f01r = vx_load(pixbuf + uf * (0+1)),
                             f10r = vx_load(pixbuf + uf * (0+2)),
                             f11r = vx_load(pixbuf + uf * (0+3));
                    v_int32  f00g = vx_load(pixbuf + uf * 4),
                             f01g = vx_load(pixbuf + uf * (4+1)),
                             f10g = vx_load(pixbuf + uf * (4+2)),
                             f11g = vx_load(pixbuf + uf * (4+3));
                    v_int32  f00b = vx_load(pixbuf + uf * 8),
                             f01b = vx_load(pixbuf + uf * (8+1)),
                             f10b = vx_load(pixbuf + uf * (8+2)),
                             f11b = vx_load(pixbuf + uf * (8+3));
                    v_int32  f00a = vx_load(pixbuf + uf * 12),
                             f01a = vx_load(pixbuf + uf * (12+1)),
                             f10a = vx_load(pixbuf + uf * (12+2)),
                             f11a = vx_load(pixbuf + uf * (12+3));

                    v_float32 f00rl = v_cvt_f32(f00r), f00rh = v_cvt_f32(f00r),
                              f01rl = v_cvt_f32(f01r), f01rh = v_cvt_f32(f01r),
                              f10rl = v_cvt_f32(f10r), f10rh = v_cvt_f32(f10r),
                              f11rl = v_cvt_f32(f11r), f11rh = v_cvt_f32(f11r);
                    v_float32 f00gl = v_cvt_f32(f00g), f00gh = v_cvt_f32(f00g),
                              f01gl = v_cvt_f32(f01g), f01gh = v_cvt_f32(f01g),
                              f10gl = v_cvt_f32(f10g), f10gh = v_cvt_f32(f10g),
                              f11gl = v_cvt_f32(f11g), f11gh = v_cvt_f32(f11g);
                    v_float32 f00bl = v_cvt_f32(f00b), f00bh = v_cvt_f32(f00b),
                              f01bl = v_cvt_f32(f01b), f01bh = v_cvt_f32(f01b),
                              f10bl = v_cvt_f32(f10b), f10bh = v_cvt_f32(f10b),
                              f11bl = v_cvt_f32(f11b), f11bh = v_cvt_f32(f11b);
                    v_float32 f00al = v_cvt_f32(f00a), f00ah = v_cvt_f32(f00a),
                              f01al = v_cvt_f32(f01a), f01ah = v_cvt_f32(f01a),
                              f10al = v_cvt_f32(f10a), f10ah = v_cvt_f32(f10a),
                              f11al = v_cvt_f32(f11a), f11ah = v_cvt_f32(f11a);

                    CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);
                    CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U8(C4);
                }
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2 == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        uint16_t pixbuf[max_uf*4*4];

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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 16U);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 16U);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CONVERT_U16F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32U16(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 16U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 16U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C1, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C1);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C3, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C3);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
                 inner_scols = vx_setall_u32((unsigned)srccols - 1),
                 outer_srows = vx_setall_u32((unsigned)srcrows + 1),
                 outer_scols = vx_setall_u32((unsigned)srccols + 1);
        v_float32 delta = vx_setall_f32(static_cast<float>(uf));
        v_int32 one = vx_setall_s32(1), four = vx_setall_s32(4);
        v_int32 v_srcstep = vx_setall_s32(int(srcstep));
        int32_t addr[max_uf],
                src_ix[max_uf],
                src_iy[max_uf];
        float pixbuf[max_uf*4*4];

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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN(C4, 32F);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 32F);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F32F32(C4);
            }
#endif // (CV_SIMD || CV_SIMD_SCALABLE)

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 32F);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 32F);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C1);

                uint8x8_t p00g, p01g, p10g, p11g;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C1);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C1, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C1);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C1);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C1);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C1);

                CV_WARP_LINEAR_SCALAR_STORE(C1, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C3);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C3);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C3, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C3);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C3);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C3);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C3);

                CV_WARP_LINEAR_SCALAR_STORE(C3, 8U);
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

        v_uint32 inner_srows = vx_setall_u32((unsigned)srcrows - 2),
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
            if (map2_data == nullptr) {
                sy_data = sx_data;
            }
            int x = 0;

            CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD1();

            for (; x <= dstcols - uf; x += uf) {
                // [TODO] apply halide trick

                CV_REMAP_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(C4);

                uint8x8_t p00r, p01r, p10r, p11r,
                          p00g, p01g, p10g, p11g,
                          p00b, p01b, p10b, p11b,
                          p00a, p01a, p10a, p11a;

                if (v_reduce_min(inner_mask) != 0) { // all loaded pixels are completely inside the image
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_ALLWITHIN_NEON_U8(C4);
                } else {
                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN(C4, 8U);

                    CV_WARP_LINEAR_VECTOR_SHUFFLE_NOTALLWITHIN_NEON_U8(C4);
                }

                CV_WARP_LINEAR_VECTOR_INTER_LOAD_U8F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_CALC_F16(C4);

                CV_WARP_LINEAR_VECTOR_INTER_STORE_F16U8(C4);
            }

            for (; x < dstcols; x++) {
                float sx, sy;
                if (map2_data == nullptr) {
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

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4, 8U);

                CV_WARP_LINEAR_SCALAR_INTER_CALC_F32(C4);

                CV_WARP_LINEAR_SCALAR_STORE(C4, 8U);
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

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
