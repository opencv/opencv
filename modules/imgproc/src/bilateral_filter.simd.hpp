/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Copyright (C) 2025, Advanced Micro Devices, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/core/hal/intrin.hpp"

/****************************************************************************************\
                                   Bilateral Filtering
\****************************************************************************************/

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
void bilateralFilterInvoker_8u(
        Mat& dst, const Mat& temp, int radius, int maxk,
        int* space_ofs, float *space_weight, float *color_weight);
void bilateralFilterInvoker_32f(
        int cn, int radius, int maxk, int *space_ofs,
        const Mat& temp, Mat& dst, float scale_index, float *space_weight, float *expLUT);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {
class BilateralFilter_8u_Invoker :
    public ParallelLoopBody
{
public:
    BilateralFilter_8u_Invoker(Mat& _dest, const Mat& _temp, int _radius, int _maxk,
        int* _space_ofs, float *_space_weight, float *_color_weight) :
        temp(&_temp), dest(&_dest), radius(_radius),
        maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), color_weight(_color_weight)
    {
    }

#if (CV_SIMD || CV_SIMD_SCALABLE)
    static void expand(const v_uint8& v_input, v_uint32& v_out0, v_uint32& v_out1, v_uint32& v_out2, v_uint32& v_out3)
    {
        v_uint16 d0, d1;
        v_expand(v_input, d0, d1);
        v_expand(d0, v_out0, v_out1);
        v_expand(d1, v_out2, v_out3);
    }

    static void computeBilateral(const v_uint8& v_val_u8,const v_uint8& v_abs_diff, v_float32& kweight, const float* color_weight,
                                        v_float32& v_wsum0, v_float32& v_sum0, v_float32& v_wsum1, v_float32& v_sum1, v_float32& v_wsum2, v_float32& v_sum2, v_float32& v_wsum3, v_float32& v_sum3)
    {
        v_uint32 d0, d1, d2, d3;
        v_float32 w0, w1, w2, w3;
        v_uint32 val0, val1, val2, val3;
        expand(v_abs_diff, d0, d1, d2, d3);
        expand(v_val_u8, val0, val1, val2, val3);
        w0 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(d0)));
        w1 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(d1)));
        w2 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(d2)));
        w3 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(d3)));
        v_wsum0 = v_add(v_wsum0, w0);
        v_sum0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum0);
        v_wsum1 = v_add(v_wsum1, w1);
        v_sum1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum1);
        v_wsum2 = v_add(v_wsum2, w2);
        v_sum2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum2);
        v_wsum3 = v_add(v_wsum3, w3);
        v_sum3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum3);
    }

    static void computeBilateral(const v_uint8& v_val_u8,const v_uint8& v_abs_diff, const float* color_weight,
                                        v_float32& v_wsum0, v_float32& v_sum0, v_float32& v_wsum1, v_float32& v_sum1, v_float32& v_wsum2, v_float32& v_sum2, v_float32& v_wsum3, v_float32& v_sum3)
    {
        v_uint32 d0, d1, d2, d3;
        v_float32 w0, w1, w2, w3;
        v_uint32 val0, val1, val2, val3;
        expand(v_abs_diff, d0, d1, d2, d3);
        expand(v_val_u8, val0, val1, val2, val3);
        w0 = v_lut(color_weight, v_reinterpret_as_s32(d0));
        w1 = v_lut(color_weight, v_reinterpret_as_s32(d1));
        w2 = v_lut(color_weight, v_reinterpret_as_s32(d2));
        w3 = v_lut(color_weight, v_reinterpret_as_s32(d3));
        v_wsum0 = v_add(v_wsum0, w0);
        v_sum0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum0);
        v_wsum1 = v_add(v_wsum1, w1);
        v_sum1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum1);
        v_wsum2 = v_add(v_wsum2, w2);
        v_sum2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum2);
        v_wsum3 = v_add(v_wsum3, w3);
        v_sum3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum3);
    }
#endif

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i, j, cn = dest->channels(), k;
        Size size = dest->size();
#if (CV_SIMD || CV_SIMD_SCALABLE)
        int nlanes = VTraits<v_float32>::vlanes();
        int nlanes_2 = 2*nlanes;
        int nlanes_4 = 4*nlanes;
#endif
        for( i = range.start; i < range.end; i++ )
        {
            const uchar* sptr = temp->ptr(i+radius) + radius*cn;
            uchar* dptr = dest->ptr(i);

            if( cn == 1 )
            {
                k = 0; j=0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
                for ( ;j <= size.width - nlanes_4; j += nlanes_4)
                {
                    const uchar* sptr_j = sptr + j;
                    v_float32 v_wsum0 = vx_setzero_f32();
                    v_float32 v_wsum1 = vx_setzero_f32();
                    v_float32 v_wsum2 = vx_setzero_f32();
                    v_float32 v_wsum3 = vx_setzero_f32();
                    v_float32 v_sum0 = vx_setzero_f32();
                    v_float32 v_sum1 = vx_setzero_f32();
                    v_float32 v_sum2 = vx_setzero_f32();
                    v_float32 v_sum3 = vx_setzero_f32();
                    v_uint8 v_sptr8 = vx_load(sptr_j);

                    k=0;
                    if(maxk==5)
                    {
                        const uchar* ksptrline1 = sptr_j + space_ofs[0];
                        const uchar* ksptrline2 = sptr_j + space_ofs[1];
                        const uchar* ksptrline3 = sptr_j + space_ofs[4];

                        v_float32 kweight = vx_setall_f32(space_weight[0]);//same weight for all, expect centre one

                        v_uint8 v_val_u8_line1 = vx_load(ksptrline1);
                        v_uint8 v_val_u8_line2_0 = vx_load(ksptrline2);
                        v_uint8 v_val_u8_line2_1 = vx_load(ksptrline2 + 1);
                        v_uint8 v_val_u8_line2_2 = vx_load(ksptrline2 + 2);
                        v_uint8 v_val_u8_line3 = vx_load(ksptrline3);

                        //compute abs diff
                        v_uint8 v_abs_diff_line1 = v_absdiff(v_val_u8_line1, v_sptr8);
                        v_uint8 v_abs_diff_line2_0 = v_absdiff(v_val_u8_line2_0, v_sptr8);
                        v_uint8 v_abs_diff_line2_1 = v_absdiff(v_val_u8_line2_1, v_sptr8);
                        v_uint8 v_abs_diff_line2_2 = v_absdiff(v_val_u8_line2_2, v_sptr8);
                        v_uint8 v_abs_diff_line3 = v_absdiff(v_val_u8_line3, v_sptr8);

                        computeBilateral( v_val_u8_line1, v_abs_diff_line1, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line2_0, v_abs_diff_line2_0, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line2_1, v_abs_diff_line2_1, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line2_2, v_abs_diff_line2_2, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line3, v_abs_diff_line3, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);

                        k = maxk;
                    }
                    else if(maxk==13)
                    {
                        const uchar* ksptrline1 = sptr_j + space_ofs[0];
                        const uchar* ksptrline5 = sptr_j + space_ofs[12];//last element
                        const uchar* ksptrline2 = sptr_j + space_ofs[1];
                        const uchar* ksptrline3 = sptr_j + space_ofs[4];
                        const uchar* ksptrline4 = sptr_j + space_ofs[9];

                        v_float32 kweight = vx_setall_f32(space_weight[0]);

                        //compute line 1 and 5
                        v_uint8 v_val_u8_line1 = vx_load(ksptrline1);
                        v_uint8 v_val_u8_line5 = vx_load(ksptrline5);
                        v_uint8 v_abs_diff_line1 = v_absdiff(v_val_u8_line1, v_sptr8);
                        v_uint8 v_abs_diff_line5 = v_absdiff(v_val_u8_line5, v_sptr8);
                        computeBilateral( v_val_u8_line1, v_abs_diff_line1, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line5, v_abs_diff_line5, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);

                        //compute line 2 and 4
                        v_uint8 v_val_u8_line2_0 = vx_load(ksptrline2);
                        v_uint8 v_val_u8_line2_1 = vx_load(ksptrline2 + 1);
                        v_uint8 v_val_u8_line2_2 = vx_load(ksptrline2 + 2);

                        v_uint8 v_val_u8_line4_0 = vx_load(ksptrline4);
                        v_uint8 v_val_u8_line4_1 = vx_load(ksptrline4 + 1);
                        v_uint8 v_val_u8_line4_2 = vx_load(ksptrline4 + 2);

                        v_uint8 v_abs_diff_line2_0 = v_absdiff(v_val_u8_line2_0, v_sptr8);
                        v_uint8 v_abs_diff_line2_1 = v_absdiff(v_val_u8_line2_1, v_sptr8);
                        v_uint8 v_abs_diff_line2_2 = v_absdiff(v_val_u8_line2_2, v_sptr8);
                        v_uint8 v_abs_diff_line4_0 = v_absdiff(v_val_u8_line4_0, v_sptr8);
                        v_uint8 v_abs_diff_line4_1 = v_absdiff(v_val_u8_line4_1, v_sptr8);
                        v_uint8 v_abs_diff_line4_2 = v_absdiff(v_val_u8_line4_2, v_sptr8);
                        v_float32 kweight_1 = vx_setall_f32(space_weight[1]);
                        v_float32 kweight_2 = vx_setall_f32(space_weight[2]);
                        computeBilateral( v_val_u8_line2_0, v_abs_diff_line2_0, kweight_1, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line2_1, v_abs_diff_line2_1, kweight_2, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line2_2, v_abs_diff_line2_2, kweight_1, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);

                        computeBilateral( v_val_u8_line4_0, v_abs_diff_line4_0, kweight_1, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line4_1, v_abs_diff_line4_1, kweight_2, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line4_2, v_abs_diff_line4_2, kweight_1, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);

                        //compute line 3
                        v_uint8 v_val_u8_line3_0 = vx_load(ksptrline3);
                        v_uint8 v_val_u8_line3_1 = vx_load(ksptrline3 + 1);
                        v_uint8 v_val_u8_line3_2 = vx_load(ksptrline3 + 2);
                        v_uint8 v_val_u8_line3_3 = vx_load(ksptrline3 + 3);
                        v_uint8 v_val_u8_line3_4 = vx_load(ksptrline3 + 4);

                        v_uint8 v_abs_diff_line3_0 = v_absdiff(v_val_u8_line3_0, v_sptr8);
                        v_uint8 v_abs_diff_line3_1 = v_absdiff(v_val_u8_line3_1, v_sptr8);
                        v_uint8 v_abs_diff_line3_2 = v_absdiff(v_val_u8_line3_2, v_sptr8);
                        v_uint8 v_abs_diff_line3_3 = v_absdiff(v_val_u8_line3_3, v_sptr8);
                        v_uint8 v_abs_diff_line3_4 = v_absdiff(v_val_u8_line3_4, v_sptr8);

                        computeBilateral( v_val_u8_line3_0, v_abs_diff_line3_0, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line3_1, v_abs_diff_line3_1, kweight_2, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line3_2, v_abs_diff_line3_2, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line3_3, v_abs_diff_line3_3, kweight_2, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);
                        computeBilateral( v_val_u8_line3_4, v_abs_diff_line3_4, kweight, color_weight,
                                          v_wsum0, v_sum0, v_wsum1, v_sum1, v_wsum2, v_sum2, v_wsum3, v_sum3);

                        k = maxk;
                    }

                    for(; k < maxk; k++)
                    {
                        const uchar* ksptr = sptr_j + space_ofs[k];
                        v_float32 kweight = vx_setall_f32(space_weight[k]);

                        v_uint8 v_val_u8 = vx_load(ksptr);
                        v_uint8 v_abs_diff = v_absdiff(v_val_u8, v_sptr8);

                        v_uint16 d0, d1;
                        v_uint32 diff0, diff1, diff2, diff3;
                        v_expand(v_abs_diff, d0, d1);
                        v_expand(d0, diff0, diff1);
                        v_expand(d1, diff2, diff3);

                        v_uint16 v0, v1;
                        v_uint32 val0, val1, val2, val3;
                        v_expand(v_val_u8, v0, v1);
                        v_expand(v0, val0, val1);
                        v_expand(v1, val2, val3);

                        v_float32 w0 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(diff0)));
                        v_float32 w1 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(diff1)));
                        v_float32 w2 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(diff2)));
                        v_float32 w3 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(diff3)));

                        v_wsum0 = v_add(v_wsum0, w0);
                        v_sum0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum0);
                        v_wsum1 = v_add(v_wsum1, w1);
                        v_sum1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum1);
                        v_wsum2 = v_add(v_wsum2, w2);
                        v_sum2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum2);
                        v_wsum3 = v_add(v_wsum3, w3);
                        v_sum3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum3);
                    }
                    v_pack_u_store(dptr + j, v_pack(v_round(v_div(v_sum0, v_wsum0)),
                                v_round(v_div(v_sum1, v_wsum1))));

                    v_pack_u_store(dptr + j + nlanes_2, v_pack(v_round(v_div(v_sum2, v_wsum2)),
                                v_round(v_div(v_sum3, v_wsum3))));
                }
#endif
                for (; j < size.width; j++)
                {
                    uchar val0 = sptr[j];
                    float wsumT = 0;
                    float sumT = 0;
                    for(k=0; k < maxk; k++)
                    {
                        const uchar* ksptr = sptr + space_ofs[k];
                        uchar val = ksptr[j];
                        float w = space_weight[k] * color_weight[std::abs(val - val0)];
                        wsumT += w;
                        sumT += val * w;
                    }

                    // overflow is not possible here => there is no need to use cv::saturate_cast
                    CV_DbgAssert(fabs(wsumT) > 0);
                    dptr[j] = (uchar)cvRound(sumT/wsumT);
                }

            }
            else
            {
                CV_Assert( cn == 3 );
                j = 0;
                const uchar* sptr_j = sptr;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                int n_8_lanes = VTraits<v_uint8>::vlanes();
                for (; j <= size.width - n_8_lanes; j += n_8_lanes, sptr_j += 3*n_8_lanes, dptr += 3*n_8_lanes)
                {
                    const uchar* rsptr = sptr_j;
                    v_float32 v_wsum_0  = vx_setzero_f32();
                    v_float32 v_wsum_1  = vx_setzero_f32();
                    v_float32 v_wsum_2  = vx_setzero_f32();
                    v_float32 v_wsum_3  = vx_setzero_f32();

                    v_float32 v_sum_b_0 = vx_setzero_f32();
                    v_float32 v_sum_b_1 = vx_setzero_f32();
                    v_float32 v_sum_b_2 = vx_setzero_f32();
                    v_float32 v_sum_b_3 = vx_setzero_f32();

                    v_float32 v_sum_g_0 = vx_setzero_f32();
                    v_float32 v_sum_g_1 = vx_setzero_f32();
                    v_float32 v_sum_g_2 = vx_setzero_f32();
                    v_float32 v_sum_g_3 = vx_setzero_f32();

                    v_float32 v_sum_r_0 = vx_setzero_f32();
                    v_float32 v_sum_r_1 = vx_setzero_f32();
                    v_float32 v_sum_r_2 = vx_setzero_f32();
                    v_float32 v_sum_r_3 = vx_setzero_f32();

                    v_float32 v_one = vx_setall_f32(1.f);

                    for(k=0; k < maxk; k++)
                    {
                        const uchar* ksptr = sptr_j + space_ofs[k];
                        v_float32 kweight = vx_setall_f32(space_weight[k]);

                        v_uint8 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(ksptr, kb, kg, kr);
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_uint16 b_l, b_h, g_l, g_h, r_l, r_h;
                        v_expand(v_absdiff(kb, rb), b_l, b_h);
                        v_expand(v_absdiff(kg, rg), g_l, g_h);
                        v_expand(v_absdiff(kr, rr), r_l, r_h);

                        v_uint32 val0, val1, val2, val3;
                        v_expand(v_add(v_add(b_l, g_l), r_l), val0, val1);
                        v_expand(v_add(v_add(b_h, g_h), r_h), val2, val3);

                        v_expand(kb, b_l, b_h);
                        v_expand(kg, g_l, g_h);
                        v_expand(kr, r_l, r_h);

                        v_float32 w0 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(val0)));
                        v_float32 w1 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(val1)));
                        v_float32 w2 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(val2)));
                        v_float32 w3 = v_mul(kweight, v_lut(color_weight, v_reinterpret_as_s32(val3)));
                        v_wsum_0 = v_add(w0, v_wsum_0);
                        v_wsum_1 = v_add(w1, v_wsum_1);
                        v_wsum_2 = v_add(w2, v_wsum_2);
                        v_wsum_3 = v_add(w3, v_wsum_3);

                        v_expand(b_l, val0, val1);
                        v_expand(b_h, val2, val3);
                        v_sum_b_0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum_b_0);
                        v_sum_b_1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum_b_1);
                        v_sum_b_2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum_b_2);
                        v_sum_b_3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum_b_3);

                        v_expand(g_l, val0, val1);
                        v_expand(g_h, val2, val3);
                        v_sum_g_0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum_g_0);
                        v_sum_g_1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum_g_1);
                        v_sum_g_2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum_g_2);
                        v_sum_g_3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum_g_3);

                        v_expand(r_l, val0, val1);
                        v_expand(r_h, val2, val3);
                        v_sum_r_0 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, v_sum_r_0);
                        v_sum_r_1 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, v_sum_r_1);
                        v_sum_r_2 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, v_sum_r_2);
                        v_sum_r_3 = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, v_sum_r_3);
                    }
                    v_float32 w0 = v_div(v_one, v_wsum_0);
                    v_float32 w1 = v_div(v_one, v_wsum_1);
                    v_float32 w2 = v_div(v_one, v_wsum_2);
                    v_float32 w3 = v_div(v_one, v_wsum_3);

                    v_store_interleave(dptr, v_pack_u(v_pack(v_round(v_mul(w0, v_sum_b_0)),
                                                             v_round(v_mul(w1, v_sum_b_1))),
                                                      v_pack(v_round(v_mul(w2, v_sum_b_2)),
                                                             v_round(v_mul(w3, v_sum_b_3)))),
                                             v_pack_u(v_pack(v_round(v_mul(w0, v_sum_g_0)),
                                                             v_round(v_mul(w1, v_sum_g_1))),
                                                      v_pack(v_round(v_mul(w2, v_sum_g_2)),
                                                             v_round(v_mul(w3, v_sum_g_3)))),
                                             v_pack_u(v_pack(v_round(v_mul(w0, v_sum_r_0)),
                                                             v_round(v_mul(w1, v_sum_r_1))),
                                                      v_pack(v_round(v_mul(w2, v_sum_r_2)),
                                                             v_round(v_mul(w3, v_sum_r_3)))));
                }
#endif
                for(; j < size.width; j++, sptr_j += 3)
                {
                    const uchar* rsptr = sptr_j;
                    float wsum = 0.f;
                    float sum_b = 0.f, sum_g = 0.f, sum_r = 0.f;
                    for(k=0; k < maxk; k++)
                    {
                        const uchar* ksptr = sptr_j + space_ofs[k];

                        int b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        float w = space_weight[k]*color_weight[std::abs(b - rsptr[0]) + std::abs(g - rsptr[1]) + std::abs(r - rsptr[2])];
                        wsum += w;
                        sum_b += b*w; sum_g += g*w; sum_r += r*w;
                    }

                    CV_DbgAssert(fabs(wsum) > 0);
                    wsum = 1.f/wsum;
                    *(dptr++) = (uchar)cvRound(sum_b*wsum);
                    *(dptr++) = (uchar)cvRound(sum_g*wsum);
                    *(dptr++) = (uchar)cvRound(sum_r*wsum);
                }
            }
        }
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vx_cleanup();
#endif
    }

private:
    const Mat *temp;
    Mat *dest;
    int radius, maxk, *space_ofs;
    float *space_weight, *color_weight;
};

}  // namespace anon

void bilateralFilterInvoker_8u(
        Mat& dst, const Mat& temp, int radius, int maxk,
        int* space_ofs, float *space_weight, float *color_weight)
{
    CV_INSTRUMENT_REGION();

    BilateralFilter_8u_Invoker body(dst, temp, radius, maxk, space_ofs, space_weight, color_weight);
    parallel_for_(Range(0, dst.rows), body, dst.total()/(double)(1<<16));
}


namespace {

class BilateralFilter_32f_Invoker :
    public ParallelLoopBody
{
public:

    BilateralFilter_32f_Invoker(int _cn, int _radius, int _maxk, int *_space_ofs,
        const Mat& _temp, Mat& _dest, float _scale_index, float *_space_weight, float *_expLUT) :
        cn(_cn), radius(_radius), maxk(_maxk), space_ofs(_space_ofs),
        temp(&_temp), dest(&_dest), scale_index(_scale_index), space_weight(_space_weight), expLUT(_expLUT)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i, j, k;
        Size size = dest->size();
#if (CV_SIMD || CV_SIMD_SCALABLE)
        int nlanes = VTraits<v_float32>::vlanes();
        int nlanes_2 = 2 * nlanes;
        int nlanes_3 = 3 * nlanes;
        int nlanes_4 = 4 * nlanes;
#endif
        for( i = range.start; i < range.end; i++ )
        {
            const float* sptr = temp->ptr<float>(i+radius) + radius*cn;
            float* dptr = dest->ptr<float>(i);

            if( cn == 1 )
            {
                j = 0;
                const float* sptr_j = sptr;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 v_one = vx_setall_f32(1.f);
                v_float32 sindex = vx_setall_f32(scale_index);

                for(; j <= size.width - nlanes_4; j += nlanes_4, sptr_j += nlanes_4, dptr += nlanes_4)
                {
                    v_float32 v_wsum0 = vx_setzero_f32();
                    v_float32 v_wsum1 = vx_setzero_f32();
                    v_float32 v_wsum2 = vx_setzero_f32();
                    v_float32 v_wsum3 = vx_setzero_f32();
                    v_float32 v_sum0 = vx_setzero_f32();
                    v_float32 v_sum1 = vx_setzero_f32();
                    v_float32 v_sum2 = vx_setzero_f32();
                    v_float32 v_sum3 = vx_setzero_f32();

                    v_float32 rval0 = vx_load(sptr_j);
                    v_float32 rval1 = vx_load(sptr_j + nlanes);
                    v_float32 rval2 = vx_load(sptr_j + nlanes_2);
                    v_float32 rval3 = vx_load(sptr_j + nlanes_3);
                    for(k = 0; k < maxk; k++)
                    {
                        const float* ksptr = sptr_j + space_ofs[k];
                        v_float32 kweight = vx_setall_f32(space_weight[k]);

                        //0th
                        v_float32 val0 = vx_load(ksptr);
                        v_float32 knan0 = v_not_nan(val0);
                        v_float32 alpha0 = v_and(v_and(v_mul(v_absdiff(val0, rval0), sindex), v_not_nan(rval0)), knan0);
                        v_int32 idx0 = v_trunc(alpha0);
                        alpha0 = v_sub(alpha0, v_cvt_f32(idx0));
                        v_float32 w0 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx0), alpha0, v_mul(v_lut(this->expLUT, idx0), v_sub(v_one, alpha0)))), knan0);
                        v_wsum0 = v_add(v_wsum0, w0);
                        v_sum0 = v_muladd(v_and(val0, knan0), w0, v_sum0);

                        //1st
                        v_float32 val1 = vx_load(ksptr + nlanes);
                        v_float32 knan1 = v_not_nan(val1);
                        v_float32 alpha1 = v_and(v_and(v_mul(v_absdiff(val1, rval1), sindex), v_not_nan(rval1)), knan1);
                        v_int32 idx1 = v_trunc(alpha1);
                        alpha1 = v_sub(alpha1, v_cvt_f32(idx1));
                        v_float32 w1 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx1), alpha1, v_mul(v_lut(this->expLUT, idx1), v_sub(v_one, alpha1)))), knan1);
                        v_wsum1 = v_add(v_wsum1, w1);
                        v_sum1 = v_muladd(v_and(val1, knan1), w1, v_sum1);

                        //2nd
                        v_float32 val2 = vx_load(ksptr + nlanes_2);
                        v_float32 knan2 = v_not_nan(val2);
                        v_float32 alpha2 = v_and(v_and(v_mul(v_absdiff(val2, rval2), sindex), v_not_nan(rval2)), knan2);
                        v_int32 idx2 = v_trunc(alpha2);
                        alpha2 = v_sub(alpha2, v_cvt_f32(idx2));
                        v_float32 w2 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx2), alpha2, v_mul(v_lut(this->expLUT, idx2), v_sub(v_one, alpha2)))), knan2);
                        v_wsum2 = v_add(v_wsum2, w2);
                        v_sum2 = v_muladd(v_and(val2, knan2), w2, v_sum2);

                        //3rd
                        v_float32 val3 = vx_load(ksptr + nlanes_3);
                        v_float32 knan3 = v_not_nan(val3);
                        v_float32 alpha3 = v_and(v_and(v_mul(v_absdiff(val3, rval3), sindex), v_not_nan(rval3)), knan3);
                        v_int32 idx3 = v_trunc(alpha3);
                        alpha3 = v_sub(alpha3, v_cvt_f32(idx3));
                        v_float32 w3 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx3), alpha3, v_mul(v_lut(this->expLUT, idx3), v_sub(v_one, alpha3)))), knan3);
                        v_wsum3 = v_add(v_wsum3, w3);
                        v_sum3 = v_muladd(v_and(val3, knan3), w3, v_sum3);
                    }
                    v_store(dptr , v_div(v_add(v_sum0, v_and(rval0, v_not_nan(rval0))), v_add(v_wsum0, v_and(v_one, v_not_nan(rval0)))));
                    v_store(dptr + nlanes, v_div(v_add(v_sum1, v_and(rval1, v_not_nan(rval1))), v_add(v_wsum1, v_and(v_one, v_not_nan(rval1)))));
                    v_store(dptr + nlanes_2, v_div(v_add(v_sum2, v_and(rval2, v_not_nan(rval2))), v_add(v_wsum2, v_and(v_one, v_not_nan(rval2)))));
                    v_store(dptr + nlanes_3, v_div(v_add(v_sum3, v_and(rval3, v_not_nan(rval3))), v_add(v_wsum3, v_and(v_one, v_not_nan(rval3)))));
                }
                for (; j <= size.width - nlanes_2; j += nlanes_2, sptr_j += nlanes_2, dptr += nlanes_2)
                {
                    v_float32 v_wsum0 = vx_setzero_f32();
                    v_float32 v_wsum1 = vx_setzero_f32();
                    v_float32 v_sum0 = vx_setzero_f32();
                    v_float32 v_sum1 = vx_setzero_f32();
                    v_float32 rval0 = vx_load(sptr_j);
                    v_float32 rval1 = vx_load(sptr_j + nlanes);
                    v_float32 rval0_not_nan = v_not_nan(rval0);
                    v_float32 rval1_not_nan = v_not_nan(rval1);

                    for (k = 0; k < maxk; k++)
                    {
                        v_float32 kweight = vx_setall_f32(space_weight[k]);
                        const float* ksptr = sptr_j + space_ofs[k];

                        //0th
                        v_float32 val0 = vx_load(ksptr);
                        v_float32 knan0 = v_not_nan(val0);
                        v_float32 alpha0 = v_and(v_and(v_mul(v_absdiff(val0, rval0), sindex), rval0_not_nan), knan0);
                        v_int32 idx0 = v_trunc(alpha0);
                        alpha0 = v_sub(alpha0, v_cvt_f32(idx0));
                        v_float32 w0 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx0), alpha0, v_mul(v_lut(this->expLUT, idx0), v_sub(v_one, alpha0)))), knan0);
                        v_wsum0 = v_add(v_wsum0, w0);
                        v_sum0 = v_muladd(v_and(val0, knan0), w0, v_sum0);

                        //1st
                        v_float32 val1 = vx_load(ksptr + nlanes);
                        v_float32 knan1 = v_not_nan(val1);
                        v_float32 alpha1 = v_and(v_and(v_mul(v_absdiff(val1, rval1), sindex), rval1_not_nan), knan1);
                        v_int32 idx1 = v_trunc(alpha1);
                        alpha1 = v_sub(alpha1, v_cvt_f32(idx1));
                        v_float32 w1 = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx1), alpha1, v_mul(v_lut(this->expLUT, idx1), v_sub(v_one, alpha1)))), knan1);
                        v_wsum1 = v_add(v_wsum1, w1);
                        v_sum1 = v_muladd(v_and(val1, knan1), w1, v_sum1);
                    }
                    v_store(dptr, v_div(v_add(v_sum0, v_and(rval0, rval0_not_nan)), v_add(v_wsum0, v_and(v_one, rval0_not_nan))));
                    v_store(dptr + nlanes, v_div(v_add(v_sum1, v_and(rval1, rval1_not_nan)), v_add(v_wsum1, v_and(v_one, rval1_not_nan))));
                }
#endif
                for (; j < size.width; j++, sptr_j++, dptr++)
                {
                    float rval = *sptr_j;
                    float wsum = 0.f;
                    float sum = 0.f;
                    for (k = 0; k < maxk; k++)
                    {
                        const float* ksptr = sptr_j + space_ofs[k];
                        float val = *ksptr;
                        float alpha = std::abs(val - rval) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha * (expLUT[idx + 1] - expLUT[idx])));
                            wsum += w;
                            sum += val * w;
                        }
                    }
                    CV_DbgAssert(fabs(wsum) >= 0);
                    *dptr = cvIsNaN(rval) ? sum / wsum : (sum + rval) / (wsum + 1.f);
                }
            }
            else
            {
                CV_Assert(cn == 3);
                j = 0;
                const float* sptr_j = sptr;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                v_float32 v_one = vx_setall_f32(1.f);
                v_float32 sindex = vx_setall_f32(scale_index);

                for (; j <= size.width - nlanes; j += nlanes, sptr_j += nlanes_3, dptr += nlanes_3)
                {
                    const float* rsptr = sptr_j;
                    v_float32 v_wsum = vx_setzero_f32();
                    v_float32 v_sum_b = vx_setzero_f32();
                    v_float32 v_sum_g = vx_setzero_f32();
                    v_float32 v_sum_r = vx_setzero_f32();
                    for (k = 0; k < maxk; k++)
                    {
                        const float* ksptr = sptr_j + space_ofs[k];
                        v_float32 kweight = vx_setall_f32(space_weight[k]);

                        v_float32 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(ksptr, kb, kg, kr);
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_float32 knan = v_and(v_and(v_not_nan(kb), v_not_nan(kg)), v_not_nan(kr));
                        v_float32 alpha = v_and(v_and(v_and(v_and(v_mul(v_add(v_add(v_absdiff(kb, rb), v_absdiff(kg, rg)), v_absdiff(kr, rr)), sindex), v_not_nan(rb)), v_not_nan(rg)), v_not_nan(rr)), knan);
                        v_int32 idx = v_trunc(alpha);
                        alpha = v_sub(alpha, v_cvt_f32(idx));

                        v_float32 w = v_and(v_mul(kweight, v_muladd(v_lut(this->expLUT + 1, idx), alpha, v_mul(v_lut(this->expLUT, idx), v_sub(v_one, alpha)))), knan);
                        v_wsum = v_add(v_wsum, w);
                        v_sum_b = v_muladd(v_and(kb, knan), w, v_sum_b);
                        v_sum_g = v_muladd(v_and(kg, knan), w, v_sum_g);
                        v_sum_r = v_muladd(v_and(kr, knan), w, v_sum_r);
                    }

                    v_float32 b, g, r;
                    v_load_deinterleave(sptr_j, b, g, r);
                    v_float32 mask = v_and(v_and(v_not_nan(b), v_not_nan(g)), v_not_nan(r));
                    v_float32 w = v_div(v_one, v_add(v_wsum, v_and(v_one, mask)));
                    v_store_interleave(dptr, v_mul(v_add(v_sum_b, v_and(b, mask)), w), v_mul(v_add(v_sum_g, v_and(g, mask)), w), v_mul(v_add(v_sum_r, v_and(r, mask)), w));
                }
#endif
                for (; j < size.width; j++, sptr_j += 3)
                {
                    const float* rsptr = sptr_j;
                    float wsum = 0.f, sum_b = 0.f, sum_g = 0.f, sum_r = 0.f;
                    for (k = 0; k < maxk; k++)
                    {
                        const float* ksptr = sptr_j + space_ofs[k];
                        float b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        bool v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        float rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];
                        bool r_NAN = cvIsNaN(rb) || cvIsNaN(rg) || cvIsNaN(rr);
                        float alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k] * (r_NAN ? 1.f : (expLUT[idx] + alpha * (expLUT[idx + 1] - expLUT[idx])));
                            wsum += w;
                            sum_b += b * w;
                            sum_g += g * w;
                            sum_r += r * w;
                        }
                    }

                    CV_DbgAssert(fabs(wsum) >= 0);
                    float b = *(sptr_j);
                    float g = *(sptr_j+1);
                    float r = *(sptr_j+2);
                    if (cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r))
                    {
                        wsum = 1.f / wsum;
                        *(dptr++) = sum_b * wsum;
                        *(dptr++) = sum_g * wsum;
                        *(dptr++) = sum_r * wsum;
                    }
                    else
                    {
                        wsum = 1.f / (wsum + 1.f);
                        *(dptr++) = (sum_b + b) * wsum;
                        *(dptr++) = (sum_g + g) * wsum;
                        *(dptr++) = (sum_r + r) * wsum;
                    }
                }
            }
        }
#if (CV_SIMD || CV_SIMD_SCALABLE)
        vx_cleanup();
#endif
    }

private:
    int cn, radius, maxk, *space_ofs;
    const Mat* temp;
    Mat *dest;
    float scale_index, *space_weight, *expLUT;
};

} // namespace anon

void bilateralFilterInvoker_32f(
        int cn, int radius, int maxk, int *space_ofs,
        const Mat& temp, Mat& dst, float scale_index, float *space_weight, float *expLUT)
{
    CV_INSTRUMENT_REGION();

    BilateralFilter_32f_Invoker body(cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT);
    parallel_for_(Range(0, dst.rows), body, dst.total()/(double)(1<<16));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
