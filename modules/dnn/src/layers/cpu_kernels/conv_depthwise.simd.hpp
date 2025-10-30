// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void fastDepthwiseConv(const float* weights,
                        int kernel_h, int kernel_w,
                        int stride_h, int stride_w,
                        int dilation_h, int dilation_w,
                        int pad_t, int pad_l,
                        const float* bias, const float* relu,
                        const float* inptr,
                        int height, int width,
                        float* outptr,
                        int out_d, int outH, int outW);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_AVX

#if !CV_FMA3 // AVX workaround
#undef _mm256_fmadd_ps
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))
#endif

static inline void _mm256_load_deinterleave(const float* ptr, __m256& a, __m256& b)
{
    __m256 t0 = _mm256_loadu_ps(ptr);
    __m256 t1 = _mm256_loadu_ps(ptr + 8);

    __m256 lo = _mm256_permute2f128_ps(t0, t1, 0+2*16);
    __m256 hi = _mm256_permute2f128_ps(t0, t1, 1+3*16);
    a = _mm256_shuffle_ps(lo, hi, 0x88);
    b = _mm256_shuffle_ps(lo, hi, 0xdd);
}

void fastDepthwiseConv( const float* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const float* biasptr, const float* relu,
                     const float* inptr_,
                     int height, int width,
                     float* outptr_,
                     int out_d, int outH, int outW )
{
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }
        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            const int VECSZ = 8;
            __m256 vw00 = _mm256_set1_ps(w00), vw01 = _mm256_set1_ps(w01), vw02 = _mm256_set1_ps(w02),
                      vw10 = _mm256_set1_ps(w10), vw11 = _mm256_set1_ps(w11), vw12 = _mm256_set1_ps(w12),
                      vw20 = _mm256_set1_ps(w20), vw21 = _mm256_set1_ps(w21), vw22 = _mm256_set1_ps(w22);
            __m256 z = _mm256_setzero_ps(), vbias = _mm256_set1_ps(bias), vrc = _mm256_set1_ps(relu_coeff);

            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00 = _mm256_loadu_ps(imgptr0 + in_j),
                           v01 = _mm256_loadu_ps(imgptr0 + in_j + dilation_w),
                           v02 = _mm256_loadu_ps(imgptr0 + in_j + dilation_w*2),
                           v10 = _mm256_loadu_ps(imgptr1 + in_j),
                           v11 = _mm256_loadu_ps(imgptr1 + in_j + dilation_w),
                           v12 = _mm256_loadu_ps(imgptr1 + in_j + dilation_w*2),
                           v20 = _mm256_loadu_ps(imgptr2 + in_j),
                           v21 = _mm256_loadu_ps(imgptr2 + in_j + dilation_w),
                           v22 = _mm256_loadu_ps(imgptr2 + in_j + dilation_w*2);

                    __m256 vout0 = _mm256_fmadd_ps(v00, vw00, vbias);
                    __m256 vout1 = _mm256_mul_ps(v01, vw01);
                    __m256 vout2 = _mm256_mul_ps(v02, vw02);

                    vout0 = _mm256_fmadd_ps(v10, vw10, vout0);
                    vout1 = _mm256_fmadd_ps(v11, vw11, vout1);
                    vout2 = _mm256_fmadd_ps(v12, vw12, vout2);

                    vout0 = _mm256_fmadd_ps(v20, vw20, vout0);
                    vout1 = _mm256_fmadd_ps(v21, vw21, vout1);
                    vout2 = _mm256_fmadd_ps(v22, vw22, vout2);

                    vout0 = _mm256_add_ps(_mm256_add_ps(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256 m = _mm256_cmp_ps(vout0, z, _CMP_GT_OQ);
                        vout0 = _mm256_blendv_ps(_mm256_mul_ps(vout0, vrc), vout0, m);
                    }
                    _mm256_storeu_ps(outptr + out_j, vout0);
                }
            else
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    _mm256_load_deinterleave(imgptr0 + in_j, v00, v01);
                    _mm256_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    _mm256_load_deinterleave(imgptr1 + in_j, v10, v11);
                    _mm256_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    _mm256_load_deinterleave(imgptr2 + in_j, v20, v21);
                    _mm256_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    __m256 vout0 = _mm256_fmadd_ps(v00, vw00, vbias);
                    __m256 vout1 = _mm256_mul_ps(v01, vw01);
                    __m256 vout2 = _mm256_mul_ps(v02, vw02);

                    vout0 = _mm256_fmadd_ps(v10, vw10, vout0);
                    vout1 = _mm256_fmadd_ps(v11, vw11, vout1);
                    vout2 = _mm256_fmadd_ps(v12, vw12, vout2);

                    vout0 = _mm256_fmadd_ps(v20, vw20, vout0);
                    vout1 = _mm256_fmadd_ps(v21, vw21, vout1);
                    vout2 = _mm256_fmadd_ps(v22, vw22, vout2);

                    vout0 = _mm256_add_ps(_mm256_add_ps(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256 m = _mm256_cmp_ps(vout0, z, _CMP_GT_OQ);
                        vout0 = _mm256_blendv_ps(_mm256_mul_ps(vout0, vrc), vout0, m);
                    }
                    _mm256_storeu_ps(outptr + out_j, vout0);
                }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }
    }
    _mm256_zeroupper();
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_RVV


void fastDepthwiseConv( const float* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const float* biasptr, const float* relu,
                     const float* inptr_,
                     int height, int width,
                     float* outptr_,
                     int out_d, int outH, int outW )
{
    int vl;
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = std::min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }
        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            int avl = outW1 - out_j;
            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += vl, avl -= vl)
                {
                    vl = __riscv_vsetvl_e32m8(avl);
                    int in_j = out_j * stride_w - pad_l;
                    vfloat32m8_t vout0 = __riscv_vfmacc_vf_f32m8(__riscv_vfmv_v_f_f32m8(bias, vl), w00, __riscv_vle32_v_f32m8(imgptr0 + in_j, vl), vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w01, __riscv_vle32_v_f32m8(imgptr0 + in_j + dilation_w, vl), vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w02, __riscv_vle32_v_f32m8(imgptr0 + in_j + dilation_w*2, vl), vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w10, __riscv_vle32_v_f32m8(imgptr1 + in_j, vl),vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w11, __riscv_vle32_v_f32m8(imgptr1 + in_j + dilation_w, vl),vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w12, __riscv_vle32_v_f32m8(imgptr1 + in_j + dilation_w*2, vl),vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w20, __riscv_vle32_v_f32m8(imgptr2 + in_j, vl), vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w21, __riscv_vle32_v_f32m8(imgptr2 + in_j + dilation_w, vl), vl);
                    vout0 = __riscv_vfmacc_vf_f32m8(vout0, w22, __riscv_vle32_v_f32m8(imgptr2 + in_j + dilation_w*2, vl), vl);
                    if (relu)
                    {
                        vbool4_t m = __riscv_vmfgt_vf_f32m8_b4(vout0, 0, vl);
                        vout0 = __riscv_vmerge_vvm_f32m8(__riscv_vfmul_vf_f32m8(vout0, relu_coeff, vl), vout0, m, vl);
                    }
                    __riscv_vse32_v_f32m8(outptr + out_j, vout0, vl);
                }
            else //stride_w == 2 && dilation_w == 1
                for( ; out_j < outW1; out_j += vl, avl -= vl)
                {
                    vl = __riscv_vsetvl_e32m2(avl);
                    int in_j = out_j * stride_w - pad_l;
                    vfloat32m2_t vout0 = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(bias, vl), w00, __riscv_vlse32_v_f32m2(imgptr0+in_j  , 8, vl), vl);
                    vfloat32m2_t vout1 = __riscv_vfmul_vf_f32m2(__riscv_vlse32_v_f32m2(imgptr0+in_j+1, 8, vl), w01, vl);
                    vfloat32m2_t vout2 = __riscv_vfmul_vf_f32m2(__riscv_vlse32_v_f32m2(imgptr0+in_j+2, 8, vl), w02, vl);

                    vout0 = __riscv_vfmacc_vf_f32m2(vout0, w10, __riscv_vlse32_v_f32m2(imgptr1+in_j  , 8, vl), vl);
                    vout1 = __riscv_vfmacc_vf_f32m2(vout1, w11, __riscv_vlse32_v_f32m2(imgptr1+in_j+1, 8, vl), vl);
                    vout2 = __riscv_vfmacc_vf_f32m2(vout2, w12, __riscv_vlse32_v_f32m2(imgptr1+in_j+2, 8, vl), vl);

                    vout0 = __riscv_vfmacc_vf_f32m2(vout0, w20, __riscv_vlse32_v_f32m2(imgptr2+in_j  , 8, vl), vl);
                    vout1 = __riscv_vfmacc_vf_f32m2(vout1, w21, __riscv_vlse32_v_f32m2(imgptr2+in_j+1, 8, vl), vl);
                    vout2 = __riscv_vfmacc_vf_f32m2(vout2, w22, __riscv_vlse32_v_f32m2(imgptr2+in_j+2, 8, vl), vl);

                    vout0 = __riscv_vfadd_vv_f32m2(__riscv_vfadd_vv_f32m2(vout0, vout1, vl), vout2, vl);
                    if (relu)
                    {
                        vbool16_t m = __riscv_vmfgt_vf_f32m2_b16(vout0, 0, vl);
                        vout0 = __riscv_vmerge_vvm_f32m2(__riscv_vfmul_vf_f32m2(vout0, relu_coeff, vl), vout0, m, vl);
                    }
                    __riscv_vse32_v_f32m2(outptr + out_j, vout0, vl);
                }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }
    }
}

#endif // CV_RVV

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && CV_LASX

static inline void _v256_load_deinterleave(const float* ptr, __m256& a, __m256& b)
{
    __m256 t0 = (__m256)__lasx_xvld(ptr, 0);
    __m256 t1 = (__m256)__lasx_xvld(ptr, 8*4);

    __m256 lo = (__m256)__lasx_xvpermi_q(t0, t1, 2+0*16);
    __m256 hi = (__m256)__lasx_xvpermi_q(t0, t1, 3+1*16);

    a = (__m256)__lasx_xvpermi_w(hi, lo, 0x88);
    b = (__m256)__lasx_xvpermi_w(hi, lo, 0xdd);
}

void fastDepthwiseConv( const float* wptr,
                     int kernel_h, int kernel_w,
                     int stride_h, int stride_w,
                     int dilation_h, int dilation_w,
                     int pad_t, int pad_l,
                     const float* biasptr, const float* relu,
                     const float* inptr_,
                     int height, int width,
                     float* outptr_,
                     int out_d, int outH, int outW )
{
    const float w00_ = wptr[0], w01_ = wptr[1], w02_ = wptr[2],
                w10 = wptr[3], w11 = wptr[4], w12 = wptr[5],
                w20_ = wptr[6], w21_ = wptr[7], w22_ = wptr[8];
    int outW1 = min(outW, (width - dilation_w*(kernel_w - 1) + pad_l)/stride_w);
    float relu_coeff = relu ? relu[out_d] : 1.f, bias = biasptr[out_d];

    for (int out_i = 0; out_i < outH; out_i++)
    {
        int in_i = out_i * stride_h - pad_t, out_j = 0;
        const float* imgptr0 = inptr_ + in_i*width;
        const float* imgptr1 = imgptr0 + dilation_h*width;
        const float* imgptr2 = imgptr0 + (dilation_h*2)*width;
        float out, w00 = w00_, w01 = w01_, w02 = w02_;
        float w20 = w20_, w21 = w21_, w22 = w22_;
        if (in_i < 0)
        {
            w00 = w01 = w02 = 0.f;
            imgptr0 = imgptr1;
        }
        else if (in_i + dilation_h*(kernel_h-1) >= height)
        {
            w20 = w21 = w22 = 0.f;
            imgptr2 = imgptr1;
        }
        float* outptr = outptr_ + out_i*outW;
        if (pad_l > 0)
        {
            out = imgptr0[0]*w01 + imgptr0[dilation_w]*w02 +
                  imgptr1[0]*w11 + imgptr1[dilation_w]*w12 +
                  imgptr2[0]*w21 + imgptr2[dilation_w]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[0] = out;
            out_j = 1;
        }

        if (stride_w == 1 || (stride_w == 2 && dilation_w == 1))
        {
            const int VECSZ = 8;
            __m256 vw00 = _v256_setall_ps(w00), vw01 = _v256_setall_ps(w01), vw02 = _v256_setall_ps(w02),
                   vw10 = _v256_setall_ps(w10), vw11 = _v256_setall_ps(w11), vw12 = _v256_setall_ps(w12),
                   vw20 = _v256_setall_ps(w20), vw21 = _v256_setall_ps(w21), vw22 = _v256_setall_ps(w22);
            __m256 z = (__m256)__lasx_xvxor_v((__m256i)vw00, (__m256i)vw00),
            vbias = _v256_setall_ps(bias), vrc = _v256_setall_ps(relu_coeff);

            if( stride_w == 1 )
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00 = (__m256)__lasx_xvld(imgptr0 + in_j, 0),
                           v01 = (__m256)__lasx_xvld(imgptr0 + in_j + dilation_w, 0),
                           v02 = (__m256)__lasx_xvld(imgptr0 + in_j + dilation_w*2, 0),
                           v10 = (__m256)__lasx_xvld(imgptr1 + in_j, 0),
                           v11 = (__m256)__lasx_xvld(imgptr1 + in_j + dilation_w, 0),
                           v12 = (__m256)__lasx_xvld(imgptr1 + in_j + dilation_w*2, 0),
                           v20 = (__m256)__lasx_xvld(imgptr2 + in_j, 0),
                           v21 = (__m256)__lasx_xvld(imgptr2 + in_j + dilation_w, 0),
                           v22 = (__m256)__lasx_xvld(imgptr2 + in_j + dilation_w*2, 0);

                    __m256 vout0 = __lasx_xvfmadd_s(v00, vw00, vbias);
                    __m256 vout1 = __lasx_xvfmul_s(v01, vw01);
                    __m256 vout2 = __lasx_xvfmul_s(v02, vw02);

                    vout0 = __lasx_xvfmadd_s(v10, vw10, vout0);
                    vout1 = __lasx_xvfmadd_s(v11, vw11, vout1);
                    vout2 = __lasx_xvfmadd_s(v12, vw12, vout2);

                    vout0 = __lasx_xvfmadd_s(v20, vw20, vout0);
                    vout1 = __lasx_xvfmadd_s(v21, vw21, vout1);
                    vout2 = __lasx_xvfmadd_s(v22, vw22, vout2);

                    vout0 = __lasx_xvfadd_s(__lasx_xvfadd_s(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256i m = __lasx_xvfcmp_clt_s(z, vout0);
                        vout0 = (__m256)__lasx_xvbitsel_v((__m256i)__lasx_xvfmul_s(vout0, vrc), (__m256i)vout0, m);
                    }
                    __lasx_xvst(vout0, outptr + out_j, 0);
                }
            else
                for( ; out_j < outW1; out_j += VECSZ )
                {
                    if (out_j + VECSZ > outW1 && out_j > pad_l)
                        out_j = outW1 - VECSZ;
                    int in_j = out_j * stride_w - pad_l;
                    __m256 v00, v01, v02, v10, v11, v12, v20, v21, v22, unused;
                    _v256_load_deinterleave(imgptr0 + in_j, v00, v01);
                    _v256_load_deinterleave(imgptr0 + in_j + 2, v02, unused);
                    _v256_load_deinterleave(imgptr1 + in_j, v10, v11);
                    _v256_load_deinterleave(imgptr1 + in_j + 2, v12, unused);
                    _v256_load_deinterleave(imgptr2 + in_j, v20, v21);
                    _v256_load_deinterleave(imgptr2 + in_j + 2, v22, unused);

                    __m256 vout0 = __lasx_xvfmadd_s(v00, vw00, vbias);
                    __m256 vout1 = __lasx_xvfmul_s(v01, vw01);
                    __m256 vout2 = __lasx_xvfmul_s(v02, vw02);

                    vout0 = __lasx_xvfmadd_s(v10, vw10, vout0);
                    vout1 = __lasx_xvfmadd_s(v11, vw11, vout1);
                    vout2 = __lasx_xvfmadd_s(v12, vw12, vout2);

                    vout0 = __lasx_xvfmadd_s(v20, vw20, vout0);
                    vout1 = __lasx_xvfmadd_s(v21, vw21, vout1);
                    vout2 = __lasx_xvfmadd_s(v22, vw22, vout2);

                    vout0 = __lasx_xvfadd_s(__lasx_xvfadd_s(vout0, vout1), vout2);
                    if (relu)
                    {
                        __m256i m = __lasx_xvfcmp_clt_s(z, vout0);
                        vout0 = (__m256)__lasx_xvbitsel_v((__m256i)__lasx_xvfmul_s(vout0, vrc), (__m256i)vout0, m);
                    }
                    __lasx_xvst(vout0, outptr + out_j, 0);
                }
        }

        for (; out_j < outW1; out_j++)
        {
            int in_j = out_j * stride_w - pad_l;
            out = imgptr0[in_j]*w00 + imgptr0[in_j + dilation_w]*w01 + imgptr0[in_j + dilation_w*2]*w02 +
                  imgptr1[in_j]*w10 + imgptr1[in_j + dilation_w]*w11 + imgptr1[in_j + dilation_w*2]*w12 +
                  imgptr2[in_j]*w20 + imgptr2[in_j + dilation_w]*w21 + imgptr2[in_j + dilation_w*2]*w22 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }

        for (; out_j < outW; out_j++ )
        {
            int in_j0 = out_j * stride_w - pad_l, in_j1 = in_j0 + dilation_w, in_j2 = in_j0 + dilation_w*2;
            float s0 = 1.f, s1 = 1.f, s2 = 1.f;
            if (in_j0 >= width)
            {
                in_j0 = 0;
                s0 = 0.f;
            }
            if (in_j1 >= width)
            {
                in_j1 = 0;
                s1 = 0.f;
            }
            if (in_j2 >= width)
            {
                in_j2 = 0;
                s2 = 0.f;
            }
            out = imgptr0[in_j0]*w00*s0 + imgptr0[in_j1]*w01*s1 + imgptr0[in_j2]*w02*s2 +
                  imgptr1[in_j0]*w10*s0 + imgptr1[in_j1]*w11*s1 + imgptr1[in_j2]*w12*s2 +
                  imgptr2[in_j0]*w20*s0 + imgptr2[in_j1]*w21*s1 + imgptr2[in_j2]*w22*s2 + bias;
            if (relu)
                out = out > 0.f ? out : out*relu_coeff;
            outptr[out_j] = out;
        }
    }
}

#endif // CV_LASX

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace
