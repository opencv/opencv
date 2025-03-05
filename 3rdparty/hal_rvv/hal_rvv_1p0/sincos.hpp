// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv { namespace detail {

static constexpr size_t sincos_90 = 16;
static constexpr size_t sincos_180 = sincos_90 * 2;
static constexpr size_t sincos_mask = sincos_180 * 2 - 1;

static constexpr float sincos_rad_scale = (float)sincos_180 / CV_PI;
static constexpr float sincos_deg_scale = (float)sincos_180 / 180.f;

static constexpr float sincos_table[] =
{
     0.00000000000000000000,     0.09801714032956060400,
     0.19509032201612825000,     0.29028467725446233000,
     0.38268343236508978000,     0.47139673682599764000,
     0.55557023301960218000,     0.63439328416364549000,
     0.70710678118654746000,     0.77301045336273699000,
     0.83146961230254524000,     0.88192126434835494000,
     0.92387953251128674000,     0.95694033573220894000,
     0.98078528040323043000,     0.99518472667219682000,
     1.00000000000000000000,     0.99518472667219693000,
     0.98078528040323043000,     0.95694033573220894000,
     0.92387953251128674000,     0.88192126434835505000,
     0.83146961230254546000,     0.77301045336273710000,
     0.70710678118654757000,     0.63439328416364549000,
     0.55557023301960218000,     0.47139673682599786000,
     0.38268343236508989000,     0.29028467725446239000,
     0.19509032201612861000,     0.09801714032956082600,
     0.00000000000000012246,    -0.09801714032956059000,
    -0.19509032201612836000,    -0.29028467725446211000,
    -0.38268343236508967000,    -0.47139673682599764000,
    -0.55557023301960196000,    -0.63439328416364527000,
    -0.70710678118654746000,    -0.77301045336273666000,
    -0.83146961230254524000,    -0.88192126434835494000,
    -0.92387953251128652000,    -0.95694033573220882000,
    -0.98078528040323032000,    -0.99518472667219693000,
    -1.00000000000000000000,    -0.99518472667219693000,
    -0.98078528040323043000,    -0.95694033573220894000,
    -0.92387953251128663000,    -0.88192126434835505000,
    -0.83146961230254546000,    -0.77301045336273688000,
    -0.70710678118654768000,    -0.63439328416364593000,
    -0.55557023301960218000,    -0.47139673682599792000,
    -0.38268343236509039000,    -0.29028467725446250000,
    -0.19509032201612872000,    -0.09801714032956050600,
};

static constexpr float sincos_sin_p1 = 0.098174770424681;
static constexpr float sincos_sin_p3 = -0.000157706078493;
static constexpr float sincos_cos_p0 = 1.000000000000000;
static constexpr float sincos_cos_p2 = -0.004819142773969;

struct SinCosVlen128
{
    static constexpr size_t table_size = 128 * 4 / 32;

    static inline void lookup(vint32m4_t idx,
                              vfloat32m4_t table,
                              size_t vl,
                              vfloat32m4_t& sinval,
                              vfloat32m4_t& cosval)
    {
        auto sin_idx = __riscv_vand(__riscv_vreinterpret_u32m4(idx), sincos_mask, vl);
        auto cos_idx = __riscv_vand(__riscv_vadd(sin_idx, sincos_90, vl), sincos_mask, vl);

        auto sin180 = __riscv_vmsgtu(sin_idx, sincos_180, vl);
        auto cos180 = __riscv_vmsgtu(cos_idx, sincos_180, vl);

        sin_idx = __riscv_vsub_mu(sin180, sin_idx, sin_idx, sincos_180, vl);
        cos_idx = __riscv_vsub_mu(cos180, cos_idx, cos_idx, sincos_180, vl);

        auto sin90 = __riscv_vmsgtu(sin_idx, sincos_90, vl);
        auto cos90 = __riscv_vmsgtu(cos_idx, sincos_90, vl);

        sin_idx = __riscv_vrsub_mu(sin90, sin_idx, sin_idx, sincos_180, vl);
        cos_idx = __riscv_vrsub_mu(cos90, cos_idx, cos_idx, sincos_180, vl);

        sinval = __riscv_vrgather(table, sin_idx, vl);
        cosval = __riscv_vrgather(table, cos_idx, vl);

        sinval = __riscv_vfmerge(sinval, 1.f, __riscv_vmseq(sin_idx, sincos_90, vl), vl);
        cosval = __riscv_vfmerge(cosval, 1.f, __riscv_vmseq(cos_idx, sincos_90, vl), vl);

        sinval = __riscv_vfneg_mu(sin180, sinval, sinval, vl);
        cosval = __riscv_vfneg_mu(cos180, cosval, cosval, vl);
    }
};

struct SinCosVlen256
{
    static constexpr size_t table_size = 256 * 4 / 32;
 
    static inline void lookup(vint32m4_t idx,
                              vfloat32m4_t table,
                              size_t vl,
                              vfloat32m4_t& sinval,
                              vfloat32m4_t& cosval)
    {
        auto sin_idx = __riscv_vand(__riscv_vreinterpret_u32m4(idx), sincos_mask, vl);
        auto cos_idx = __riscv_vand(__riscv_vadd(sin_idx, sincos_90, vl), sincos_mask, vl);

        auto sin180 = __riscv_vmsgeu(sin_idx, sincos_180, vl);
        auto cos180 = __riscv_vmsgeu(cos_idx, sincos_180, vl);

        sin_idx = __riscv_vsub_mu(sin180, sin_idx, sin_idx, sincos_180, vl);
        cos_idx = __riscv_vsub_mu(cos180, cos_idx, cos_idx, sincos_180, vl);

        sinval = __riscv_vrgather(table, sin_idx, vl);
        cosval = __riscv_vrgather(table, cos_idx, vl);

        sinval = __riscv_vfneg_mu(sin180, sinval, sinval, vl);
        cosval = __riscv_vfneg_mu(cos180, cosval, cosval, vl);
    }
};

struct SinCosVlen512
{
    static constexpr size_t table_size = 512 * 4 / 32;

    static inline void lookup(vint32m4_t idx,
                              vfloat32m4_t table,
                              size_t vl,
                              vfloat32m4_t& sinval,
                              vfloat32m4_t& cosval)
    {
        auto sin_idx = __riscv_vand(__riscv_vreinterpret_u32m4(idx), sincos_mask, vl);
        auto cos_idx = __riscv_vand(__riscv_vadd(sin_idx, sincos_90, vl), sincos_mask, vl);

        sinval = __riscv_vrgather(table, sin_idx, vl);
        cosval = __riscv_vrgather(table, cos_idx, vl);
    }
};

template <typename SinCosVlen>
static inline vfloat32m4_t SinCosLoadTab()
{
    return __riscv_vle32_v_f32m4(sincos_table, SinCosVlen::table_size);
}

template <typename SinCosVlen>
static inline void SinCos32f(vfloat32m4_t angle,
                              float scale,
                              vfloat32m4_t table,
                              size_t vl,
                              vfloat32m4_t& sinval,
                              vfloat32m4_t& cosval)
{
    angle = __riscv_vfmul(angle, scale, vl);
    auto round_angle = __riscv_vfcvt_x_f_v_i32m4(angle, vl);
    auto delta_angle =
        __riscv_vfsub(angle, __riscv_vfcvt_f_x_v_f32m4(round_angle, vl), vl);

    vfloat32m4_t sin_a, cos_a;
    SinCosVlen::lookup(round_angle, table, vl, sin_a, cos_a);

    auto delta_angle2 = __riscv_vfmul(delta_angle, delta_angle, vl);

    auto sin_b = __riscv_vfmul(delta_angle2, sincos_sin_p3, vl);
    sin_b = __riscv_vfmul(__riscv_vfadd(sin_b, sincos_sin_p1, vl), delta_angle, vl);
    auto cos_b = __riscv_vfmul(delta_angle2, sincos_cos_p2, vl);
    cos_b = __riscv_vfadd(cos_b, sincos_cos_p0, vl);

    sinval = __riscv_vfadd(__riscv_vfmul(sin_a, cos_b, vl), __riscv_vfmul(cos_a, sin_b, vl), vl);
    cosval = __riscv_vfsub(__riscv_vfmul(cos_a, cos_b, vl), __riscv_vfmul(sin_a, sin_b, vl), vl);
}

}}}  // namespace cv::cv_hal_rvv::detail
