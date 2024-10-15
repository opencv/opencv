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

#define CV_WARPAFFINE_LINEAR_VECTOR_COMPUTE_MAPPED_COORD2(CN) \
    v_float32 src_x0 = v_fma(M0, dst_x0, M_x), \
              src_y0 = v_fma(M3, dst_x0, M_y), \
              src_x1 = v_fma(M0, dst_x1, M_x), \
              src_y1 = v_fma(M3, dst_x1, M_y); \
    dst_x0 = v_add(dst_x0, delta); \
    dst_x1 = v_add(dst_x1, delta); \
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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint8_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t* srcptr = src + srcstep*iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint8_t* srcptr = src + srcstep*iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint16_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint16_t *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint16_t *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00g, p01g, p10g, p11g;
                const float *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p01r, p01g, p01b;
                float p10r, p10g, p10b, p11r, p11g, p11b;
                const float *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                float p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const float *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint8_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint8_t *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint8_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint8_t *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint16_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint16_t *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint16_t *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00g, p01g, p10g, p11g;
                const float *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p01r, p01g, p01b;
                float p10r, p10g, p10b, p11r, p11g, p11b;
                const float *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                float p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                float p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const float *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00g, p01g, p10g, p11g;
                const uint8_t *srcptr = src + srcstep * iy + ix;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C1);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p01r, p01g, p01b;
                int p10r, p10g, p10b, p11r, p11g, p11b;
                const uint8_t *srcptr = src + srcstep * iy + ix*3;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C3);

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
                int ix = cvFloor(sx), iy = cvFloor(sy);
                sx -= ix; sy -= iy;
                int p00r, p00g, p00b, p00a, p01r, p01g, p01b, p01a;
                int p10r, p10g, p10b, p10a, p11r, p11g, p11b, p11a;
                const uint8_t *srcptr = src + srcstep * iy + ix*4;

                CV_WARP_LINEAR_SCALAR_SHUFFLE(C4);

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

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY


CV_CPU_OPTIMIZATION_NAMESPACE_END
} // cv
