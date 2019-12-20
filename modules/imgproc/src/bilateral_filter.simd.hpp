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

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

        int i, j, cn = dest->channels(), k;
        Size size = dest->size();

        for( i = range.start; i < range.end; i++ )
        {
            const uchar* sptr = temp->ptr(i+radius) + radius*cn;
            uchar* dptr = dest->ptr(i);

            if( cn == 1 )
            {
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH) + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum = alignPtr(buf.data(), CV_SIMD_WIDTH);
                float *wsum = sum + alignSize(size.width, CV_SIMD_WIDTH);
                k = 0;
                for(; k <= maxk-4; k+=4)
                {
                    const uchar* ksptr0 = sptr + space_ofs[k];
                    const uchar* ksptr1 = sptr + space_ofs[k+1];
                    const uchar* ksptr2 = sptr + space_ofs[k+2];
                    const uchar* ksptr3 = sptr + space_ofs[k+3];
                    j = 0;
#if CV_SIMD
                    v_float32 kweight0 = vx_setall_f32(space_weight[k]);
                    v_float32 kweight1 = vx_setall_f32(space_weight[k+1]);
                    v_float32 kweight2 = vx_setall_f32(space_weight[k+2]);
                    v_float32 kweight3 = vx_setall_f32(space_weight[k+3]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    {
                        v_uint32 rval = vx_load_expand_q(sptr + j);

                        v_uint32 val = vx_load_expand_q(ksptr0 + j);
                        v_float32 w = kweight0 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, rval)));
                        v_float32 v_wsum = vx_load_aligned(wsum + j) + w;
                        v_float32 v_sum = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val)), w, vx_load_aligned(sum + j));

                        val = vx_load_expand_q(ksptr1 + j);
                        w = kweight1 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, rval)));
                        v_wsum += w;
                        v_sum = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val)), w, v_sum);

                        val = vx_load_expand_q(ksptr2 + j);
                        w = kweight2 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, rval)));
                        v_wsum += w;
                        v_sum = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val)), w, v_sum);

                        val = vx_load_expand_q(ksptr3 + j);
                        w = kweight3 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, rval)));
                        v_wsum += w;
                        v_sum = v_muladd(v_cvt_f32(v_reinterpret_as_s32(val)), w, v_sum);

                        v_store_aligned(wsum + j, v_wsum);
                        v_store_aligned(sum + j, v_sum);
                    }
#endif
#if CV_SIMD128
                    v_float32x4 kweight4 = v_load(space_weight + k);
#endif
                    for (; j < size.width; j++)
                    {
#if CV_SIMD128
                        v_uint32x4 rval = v_setall_u32(sptr[j]);
                        v_uint32x4 val(ksptr0[j], ksptr1[j], ksptr2[j], ksptr3[j]);
                        v_float32x4 w = kweight4 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, rval)));
                        wsum[j] += v_reduce_sum(w);
                        sum[j] += v_reduce_sum(v_cvt_f32(v_reinterpret_as_s32(val)) * w);
#else
                        int rval = sptr[j];

                        int val = ksptr0[j];
                        float w = space_weight[k] * color_weight[std::abs(val - rval)];
                        wsum[j] += w;
                        sum[j] += val * w;

                        val = ksptr1[j];
                        w = space_weight[k+1] * color_weight[std::abs(val - rval)];
                        wsum[j] += w;
                        sum[j] += val * w;

                        val = ksptr2[j];
                        w = space_weight[k+2] * color_weight[std::abs(val - rval)];
                        wsum[j] += w;
                        sum[j] += val * w;

                        val = ksptr3[j];
                        w = space_weight[k+3] * color_weight[std::abs(val - rval)];
                        wsum[j] += w;
                        sum[j] += val * w;
#endif
                    }
                }
                for(; k < maxk; k++)
                {
                    const uchar* ksptr = sptr + space_ofs[k];
                    j = 0;
#if CV_SIMD
                    v_float32 kweight = vx_setall_f32(space_weight[k]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    {
                        v_uint32 val = vx_load_expand_q(ksptr + j);
                        v_float32 w = kweight * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(val, vx_load_expand_q(sptr + j))));
                        v_store_aligned(wsum + j, vx_load_aligned(wsum + j) + w);
                        v_store_aligned(sum + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val)), w, vx_load_aligned(sum + j)));
                    }
#endif
                    for (; j < size.width; j++)
                    {
                        int val = ksptr[j];
                        float w = space_weight[k] * color_weight[std::abs(val - sptr[j])];
                        wsum[j] += w;
                        sum[j] += val * w;
                    }
                }
                j = 0;
#if CV_SIMD
                for (; j <= size.width - 2*v_float32::nlanes; j += 2*v_float32::nlanes)
                    v_pack_u_store(dptr + j, v_pack(v_round(vx_load_aligned(sum + j                    ) / vx_load_aligned(wsum + j                    )),
                                                    v_round(vx_load_aligned(sum + j + v_float32::nlanes) / vx_load_aligned(wsum + j + v_float32::nlanes))));
#endif
                for (; j < size.width; j++)
                {
                    // overflow is not possible here => there is no need to use cv::saturate_cast
                    CV_DbgAssert(fabs(wsum[j]) > 0);
                    dptr[j] = (uchar)cvRound(sum[j]/wsum[j]);
                }
            }
            else
            {
                assert( cn == 3 );
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH)*3 + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum_b = alignPtr(buf.data(), CV_SIMD_WIDTH);
                float *sum_g = sum_b + alignSize(size.width, CV_SIMD_WIDTH);
                float *sum_r = sum_g + alignSize(size.width, CV_SIMD_WIDTH);
                float *wsum = sum_r + alignSize(size.width, CV_SIMD_WIDTH);
                k = 0;
                for(; k <= maxk-4; k+=4)
                {
                    const uchar* ksptr0 = sptr + space_ofs[k];
                    const uchar* ksptr1 = sptr + space_ofs[k+1];
                    const uchar* ksptr2 = sptr + space_ofs[k+2];
                    const uchar* ksptr3 = sptr + space_ofs[k+3];
                    const uchar* rsptr = sptr;
                    j = 0;
#if CV_SIMD
                    v_float32 kweight0 = vx_setall_f32(space_weight[k]);
                    v_float32 kweight1 = vx_setall_f32(space_weight[k+1]);
                    v_float32 kweight2 = vx_setall_f32(space_weight[k+2]);
                    v_float32 kweight3 = vx_setall_f32(space_weight[k+3]);
                    for (; j <= size.width - v_uint8::nlanes; j += v_uint8::nlanes, rsptr += 3*v_uint8::nlanes,
                                                              ksptr0 += 3*v_uint8::nlanes, ksptr1 += 3*v_uint8::nlanes, ksptr2 += 3*v_uint8::nlanes, ksptr3 += 3*v_uint8::nlanes)
                    {
                        v_uint8 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_load_deinterleave(ksptr0, kb, kg, kr);
                        v_uint16 val0, val1, val2, val3, val4;
                        v_expand(v_absdiff(kb, rb), val0, val1);
                        v_expand(v_absdiff(kg, rg), val2, val3);
                        val0 += val2; val1 += val3;
                        v_expand(v_absdiff(kr, rr), val2, val3);
                        val0 += val2; val1 += val3;

                        v_uint32 vall, valh;
                        v_expand(val0, vall, valh);
                        v_float32 w0 = kweight0 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        v_float32 w1 = kweight0 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j, w0 + vx_load_aligned(wsum + j));
                        v_store_aligned(wsum + j + v_float32::nlanes, w1 + vx_load_aligned(wsum + j + v_float32::nlanes));
                        v_expand(kb, val0, val2);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_b + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_b + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + v_float32::nlanes)));
                        v_expand(kg, val0, val3);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_g + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_g + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + v_float32::nlanes)));
                        v_expand(kr, val0, val4);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_r + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j)));
                        v_store_aligned(sum_r + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + v_float32::nlanes)));

                        v_expand(val1, vall, valh);
                        w0 = kweight0 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight0 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j + 2 * v_float32::nlanes, w0 + vx_load_aligned(wsum + j + 2 * v_float32::nlanes));
                        v_store_aligned(wsum + j + 3 * v_float32::nlanes, w1 + vx_load_aligned(wsum + j + 3 * v_float32::nlanes));
                        v_expand(val2, vall, valh);
                        v_store_aligned(sum_b + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + 3 * v_float32::nlanes)));
                        v_expand(val3, vall, valh);
                        v_store_aligned(sum_g + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + 3 * v_float32::nlanes)));
                        v_expand(val4, vall, valh);
                        v_store_aligned(sum_r + j + 2*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j + 2*v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 3*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + 3*v_float32::nlanes)));

                        v_load_deinterleave(ksptr1, kb, kg, kr);
                        v_expand(v_absdiff(kb, rb), val0, val1);
                        v_expand(v_absdiff(kg, rg), val2, val3);
                        val0 += val2; val1 += val3;
                        v_expand(v_absdiff(kr, rr), val2, val3);
                        val0 += val2; val1 += val3;

                        v_expand(val0, vall, valh);
                        w0 = kweight1 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight1 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j, w0 + vx_load_aligned(wsum + j));
                        v_store_aligned(wsum + j + v_float32::nlanes, w1 + vx_load_aligned(wsum + j + v_float32::nlanes));
                        v_expand(kb, val0, val2);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_b + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_b + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + v_float32::nlanes)));
                        v_expand(kg, val0, val3);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_g + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_g + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + v_float32::nlanes)));
                        v_expand(kr, val0, val4);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_r + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j)));
                        v_store_aligned(sum_r + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + v_float32::nlanes)));

                        v_expand(val1, vall, valh);
                        w0 = kweight1 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight1 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j + 2 * v_float32::nlanes, w0 + vx_load_aligned(wsum + j + 2 * v_float32::nlanes));
                        v_store_aligned(wsum + j + 3 * v_float32::nlanes, w1 + vx_load_aligned(wsum + j + 3 * v_float32::nlanes));
                        v_expand(val2, vall, valh);
                        v_store_aligned(sum_b + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + 3 * v_float32::nlanes)));
                        v_expand(val3, vall, valh);
                        v_store_aligned(sum_g + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + 3 * v_float32::nlanes)));
                        v_expand(val4, vall, valh);
                        v_store_aligned(sum_r + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + 3 * v_float32::nlanes)));

                        v_load_deinterleave(ksptr2, kb, kg, kr);
                        v_expand(v_absdiff(kb, rb), val0, val1);
                        v_expand(v_absdiff(kg, rg), val2, val3);
                        val0 += val2; val1 += val3;
                        v_expand(v_absdiff(kr, rr), val2, val3);
                        val0 += val2; val1 += val3;

                        v_expand(val0, vall, valh);
                        w0 = kweight2 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight2 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j, w0 + vx_load_aligned(wsum + j));
                        v_store_aligned(wsum + j + v_float32::nlanes, w1 + vx_load_aligned(wsum + j + v_float32::nlanes));
                        v_expand(kb, val0, val2);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_b + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_b + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + v_float32::nlanes)));
                        v_expand(kg, val0, val3);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_g + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_g + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + v_float32::nlanes)));
                        v_expand(kr, val0, val4);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_r + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j)));
                        v_store_aligned(sum_r + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + v_float32::nlanes)));

                        v_expand(val1, vall, valh);
                        w0 = kweight2 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight2 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j + 2 * v_float32::nlanes, w0 + vx_load_aligned(wsum + j + 2 * v_float32::nlanes));
                        v_store_aligned(wsum + j + 3 * v_float32::nlanes, w1 + vx_load_aligned(wsum + j + 3 * v_float32::nlanes));
                        v_expand(val2, vall, valh);
                        v_store_aligned(sum_b + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + 3 * v_float32::nlanes)));
                        v_expand(val3, vall, valh);
                        v_store_aligned(sum_g + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + 3 * v_float32::nlanes)));
                        v_expand(val4, vall, valh);
                        v_store_aligned(sum_r + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + 3 * v_float32::nlanes)));

                        v_load_deinterleave(ksptr3, kb, kg, kr);
                        v_expand(v_absdiff(kb, rb), val0, val1);
                        v_expand(v_absdiff(kg, rg), val2, val3);
                        val0 += val2; val1 += val3;
                        v_expand(v_absdiff(kr, rr), val2, val3);
                        val0 += val2; val1 += val3;

                        v_expand(val0, vall, valh);
                        w0 = kweight3 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight3 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j, w0 + vx_load_aligned(wsum + j));
                        v_store_aligned(wsum + j + v_float32::nlanes, w1 + vx_load_aligned(wsum + j + v_float32::nlanes));
                        v_expand(kb, val0, val2);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_b + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_b + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + v_float32::nlanes)));
                        v_expand(kg, val0, val3);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_g + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_g + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + v_float32::nlanes)));
                        v_expand(kr, val0, val4);
                        v_expand(val0, vall, valh);
                        v_store_aligned(sum_r + j, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j)));
                        v_store_aligned(sum_r + j + v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + v_float32::nlanes)));

                        v_expand(val1, vall, valh);
                        w0 = kweight3 * v_lut(color_weight, v_reinterpret_as_s32(vall));
                        w1 = kweight3 * v_lut(color_weight, v_reinterpret_as_s32(valh));
                        v_store_aligned(wsum + j + 2 * v_float32::nlanes, w0 + vx_load_aligned(wsum + j + 2 * v_float32::nlanes));
                        v_store_aligned(wsum + j + 3 * v_float32::nlanes, w1 + vx_load_aligned(wsum + j + 3 * v_float32::nlanes));
                        v_expand(val2, vall, valh);
                        v_store_aligned(sum_b + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_b + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_b + j + 3 * v_float32::nlanes)));
                        v_expand(val3, vall, valh);
                        v_store_aligned(sum_g + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_g + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_g + j + 3 * v_float32::nlanes)));
                        v_expand(val4, vall, valh);
                        v_store_aligned(sum_r + j + 2 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(vall)), w0, vx_load_aligned(sum_r + j + 2 * v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 3 * v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(valh)), w1, vx_load_aligned(sum_r + j + 3 * v_float32::nlanes)));
                    }
#endif
#if CV_SIMD128
                    v_float32x4 kweight4 = v_load(space_weight + k);
#endif
                    for(; j < size.width; j++, rsptr += 3, ksptr0 += 3, ksptr1 += 3, ksptr2 += 3, ksptr3 += 3)
                    {
#if CV_SIMD128
                            v_uint32x4 rb = v_setall_u32(rsptr[0]);
                            v_uint32x4 rg = v_setall_u32(rsptr[1]);
                            v_uint32x4 rr = v_setall_u32(rsptr[2]);
                            v_uint32x4 b(ksptr0[0], ksptr1[0], ksptr2[0], ksptr3[0]);
                            v_uint32x4 g(ksptr0[1], ksptr1[1], ksptr2[1], ksptr3[1]);
                            v_uint32x4 r(ksptr0[2], ksptr1[2], ksptr2[2], ksptr3[2]);
                            v_float32x4 w = kweight4 * v_lut(color_weight, v_reinterpret_as_s32(v_absdiff(b, rb) + v_absdiff(g, rg) + v_absdiff(r, rr)));
                            wsum[j] += v_reduce_sum(w);
                            sum_b[j] += v_reduce_sum(v_cvt_f32(v_reinterpret_as_s32(b)) * w);
                            sum_g[j] += v_reduce_sum(v_cvt_f32(v_reinterpret_as_s32(g)) * w);
                            sum_r[j] += v_reduce_sum(v_cvt_f32(v_reinterpret_as_s32(r)) * w);
#else
                        int rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];

                        int b = ksptr0[0], g = ksptr0[1], r = ksptr0[2];
                        float w = space_weight[k]*color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                        wsum[j] += w;
                        sum_b[j] += b*w; sum_g[j] += g*w; sum_r[j] += r*w;

                        b = ksptr1[0]; g = ksptr1[1]; r = ksptr1[2];
                        w = space_weight[k+1] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                        wsum[j] += w;
                        sum_b[j] += b*w; sum_g[j] += g*w; sum_r[j] += r*w;

                        b = ksptr2[0]; g = ksptr2[1]; r = ksptr2[2];
                        w = space_weight[k+2] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                        wsum[j] += w;
                        sum_b[j] += b*w; sum_g[j] += g*w; sum_r[j] += r*w;

                        b = ksptr3[0]; g = ksptr3[1]; r = ksptr3[2];
                        w = space_weight[k+3] * color_weight[std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)];
                        wsum[j] += w;
                        sum_b[j] += b*w; sum_g[j] += g*w; sum_r[j] += r*w;
#endif
                    }
                }
                for(; k < maxk; k++)
                {
                    const uchar* ksptr = sptr + space_ofs[k];
                    const uchar* rsptr = sptr;
                    j = 0;
#if CV_SIMD
                    v_float32 kweight = vx_setall_f32(space_weight[k]);
                    for (; j <= size.width - v_uint8::nlanes; j += v_uint8::nlanes, ksptr += 3*v_uint8::nlanes, rsptr += 3*v_uint8::nlanes)
                    {
                        v_uint8 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(ksptr, kb, kg, kr);
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_uint16 b_l, b_h, g_l, g_h, r_l, r_h;
                        v_expand(v_absdiff(kb, rb), b_l, b_h);
                        v_expand(v_absdiff(kg, rg), g_l, g_h);
                        v_expand(v_absdiff(kr, rr), r_l, r_h);

                        v_uint32 val0, val1, val2, val3;
                        v_expand(b_l + g_l + r_l, val0, val1);
                        v_expand(b_h + g_h + r_h, val2, val3);

                        v_expand(kb, b_l, b_h);
                        v_expand(kg, g_l, g_h);
                        v_expand(kr, r_l, r_h);

                        v_float32 w0 = kweight * v_lut(color_weight, v_reinterpret_as_s32(val0));
                        v_float32 w1 = kweight * v_lut(color_weight, v_reinterpret_as_s32(val1));
                        v_float32 w2 = kweight * v_lut(color_weight, v_reinterpret_as_s32(val2));
                        v_float32 w3 = kweight * v_lut(color_weight, v_reinterpret_as_s32(val3));
                        v_store_aligned(wsum + j                      , w0 + vx_load_aligned(wsum + j));
                        v_store_aligned(wsum + j +   v_float32::nlanes, w1 + vx_load_aligned(wsum + j + v_float32::nlanes));
                        v_store_aligned(wsum + j + 2*v_float32::nlanes, w2 + vx_load_aligned(wsum + j + 2*v_float32::nlanes));
                        v_store_aligned(wsum + j + 3*v_float32::nlanes, w3 + vx_load_aligned(wsum + j + 3*v_float32::nlanes));
                        v_expand(b_l, val0, val1);
                        v_expand(b_h, val2, val3);
                        v_store_aligned(sum_b + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_b + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, vx_load_aligned(sum_b + j + v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 2*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, vx_load_aligned(sum_b + j + 2*v_float32::nlanes)));
                        v_store_aligned(sum_b + j + 3*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, vx_load_aligned(sum_b + j + 3*v_float32::nlanes)));
                        v_expand(g_l, val0, val1);
                        v_expand(g_h, val2, val3);
                        v_store_aligned(sum_g + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_g + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, vx_load_aligned(sum_g + j + v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 2*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, vx_load_aligned(sum_g + j + 2*v_float32::nlanes)));
                        v_store_aligned(sum_g + j + 3*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, vx_load_aligned(sum_g + j + 3*v_float32::nlanes)));
                        v_expand(r_l, val0, val1);
                        v_expand(r_h, val2, val3);
                        v_store_aligned(sum_r + j                      , v_muladd(v_cvt_f32(v_reinterpret_as_s32(val0)), w0, vx_load_aligned(sum_r + j)));
                        v_store_aligned(sum_r + j +   v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val1)), w1, vx_load_aligned(sum_r + j + v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 2*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val2)), w2, vx_load_aligned(sum_r + j + 2*v_float32::nlanes)));
                        v_store_aligned(sum_r + j + 3*v_float32::nlanes, v_muladd(v_cvt_f32(v_reinterpret_as_s32(val3)), w3, vx_load_aligned(sum_r + j + 3*v_float32::nlanes)));
                    }
#endif
                    for(; j < size.width; j++, ksptr += 3, rsptr += 3)
                    {
                        int b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        float w = space_weight[k]*color_weight[std::abs(b - rsptr[0]) + std::abs(g - rsptr[1]) + std::abs(r - rsptr[2])];
                        wsum[j] += w;
                        sum_b[j] += b*w; sum_g[j] += g*w; sum_r[j] += r*w;
                    }
                }
                j = 0;
#if CV_SIMD
                v_float32 v_one = vx_setall_f32(1.f);
                for(; j <= size.width - v_uint8::nlanes; j += v_uint8::nlanes, dptr += 3*v_uint8::nlanes)
                {
                    v_float32 w0 = v_one / vx_load_aligned(wsum + j);
                    v_float32 w1 = v_one / vx_load_aligned(wsum + j + v_float32::nlanes);
                    v_float32 w2 = v_one / vx_load_aligned(wsum + j + 2*v_float32::nlanes);
                    v_float32 w3 = v_one / vx_load_aligned(wsum + j + 3*v_float32::nlanes);

                    v_store_interleave(dptr, v_pack_u(v_pack(v_round(w0 * vx_load_aligned(sum_b + j)),
                                                             v_round(w1 * vx_load_aligned(sum_b + j + v_float32::nlanes))),
                                                      v_pack(v_round(w2 * vx_load_aligned(sum_b + j + 2*v_float32::nlanes)),
                                                             v_round(w3 * vx_load_aligned(sum_b + j + 3*v_float32::nlanes)))),
                                             v_pack_u(v_pack(v_round(w0 * vx_load_aligned(sum_g + j)),
                                                             v_round(w1 * vx_load_aligned(sum_g + j + v_float32::nlanes))),
                                                      v_pack(v_round(w2 * vx_load_aligned(sum_g + j + 2*v_float32::nlanes)),
                                                             v_round(w3 * vx_load_aligned(sum_g + j + 3*v_float32::nlanes)))),
                                             v_pack_u(v_pack(v_round(w0 * vx_load_aligned(sum_r + j)),
                                                             v_round(w1 * vx_load_aligned(sum_r + j + v_float32::nlanes))),
                                                      v_pack(v_round(w2 * vx_load_aligned(sum_r + j + 2*v_float32::nlanes)),
                                                             v_round(w3 * vx_load_aligned(sum_r + j + 3*v_float32::nlanes)))));
                }
#endif
                for(; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) > 0);
                    wsum[j] = 1.f/wsum[j];
                    *(dptr++) = (uchar)cvRound(sum_b[j]*wsum[j]);
                    *(dptr++) = (uchar)cvRound(sum_g[j]*wsum[j]);
                    *(dptr++) = (uchar)cvRound(sum_r[j]*wsum[j]);
                }
            }
        }
#if CV_SIMD
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

        for( i = range.start; i < range.end; i++ )
        {
            const float* sptr = temp->ptr<float>(i+radius) + radius*cn;
            float* dptr = dest->ptr<float>(i);

            if( cn == 1 )
            {
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH) + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum = alignPtr(buf.data(), CV_SIMD_WIDTH);
                float *wsum = sum + alignSize(size.width, CV_SIMD_WIDTH);
#if CV_SIMD
                v_float32 v_one = vx_setall_f32(1.f);
                v_float32 sindex = vx_setall_f32(scale_index);
#endif
                k = 0;
                for(; k <= maxk - 4; k+=4)
                {
                    const float* ksptr0 = sptr + space_ofs[k];
                    const float* ksptr1 = sptr + space_ofs[k + 1];
                    const float* ksptr2 = sptr + space_ofs[k + 2];
                    const float* ksptr3 = sptr + space_ofs[k + 3];
                    j = 0;
#if CV_SIMD
                    v_float32 kweight0 = vx_setall_f32(space_weight[k]);
                    v_float32 kweight1 = vx_setall_f32(space_weight[k+1]);
                    v_float32 kweight2 = vx_setall_f32(space_weight[k+2]);
                    v_float32 kweight3 = vx_setall_f32(space_weight[k+3]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    {
                        v_float32 rval = vx_load(sptr + j);

                        v_float32 val = vx_load(ksptr0 + j);
                        v_float32 knan = v_not_nan(val);
                        v_float32 alpha = (v_absdiff(val, rval) * sindex) & v_not_nan(rval) & knan;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        v_float32 w = (kweight0 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one-alpha))) & knan;
                        v_float32 v_wsum = vx_load_aligned(wsum + j) + w;
                        v_float32 v_sum = v_muladd(val & knan, w, vx_load_aligned(sum + j));

                        val = vx_load(ksptr1 + j);
                        knan = v_not_nan(val);
                        alpha = (v_absdiff(val, rval) * sindex) & v_not_nan(rval) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight1 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum = v_muladd(val & knan, w, v_sum);

                        val = vx_load(ksptr2 + j);
                        knan = v_not_nan(val);
                        alpha = (v_absdiff(val, rval) * sindex) & v_not_nan(rval) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight2 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum = v_muladd(val & knan, w, v_sum);

                        val = vx_load(ksptr3 + j);
                        knan = v_not_nan(val);
                        alpha = (v_absdiff(val, rval) * sindex) & v_not_nan(rval) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight3 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum = v_muladd(val & knan, w, v_sum);

                        v_store_aligned(wsum + j, v_wsum);
                        v_store_aligned(sum + j, v_sum);
                    }
#endif
#if CV_SIMD128
                    v_float32x4 v_one4 = v_setall_f32(1.f);
                    v_float32x4 sindex4 = v_setall_f32(scale_index);
                    v_float32x4 kweight4 = v_load(space_weight + k);
#endif
                    for (; j < size.width; j++)
                    {
#if CV_SIMD128
                        v_float32x4 rval = v_setall_f32(sptr[j]);
                        v_float32x4 val(ksptr0[j], ksptr1[j], ksptr2[j], ksptr3[j]);
                        v_float32x4 knan = v_not_nan(val);
                        v_float32x4 alpha = (v_absdiff(val, rval) * sindex4) & v_not_nan(rval) & knan;
                        v_int32x4 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        v_float32x4 w = (kweight4 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one4 - alpha))) & knan;
                        wsum[j] += v_reduce_sum(w);
                        sum[j] += v_reduce_sum((val & knan) * w);
#else
                        float rval = sptr[j];

                        float val = ksptr0[j];
                        float alpha = std::abs(val - rval) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }

                        val = ksptr1[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+1] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }

                        val = ksptr2[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+2] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }

                        val = ksptr3[j];
                        alpha = std::abs(val - rval) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k+3] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }
#endif
                    }
                }
                for(; k < maxk; k++)
                {
                    const float* ksptr = sptr + space_ofs[k];
                    j = 0;
#if CV_SIMD
                    v_float32 kweight = vx_setall_f32(space_weight[k]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    {
                        v_float32 val = vx_load(ksptr + j);
                        v_float32 rval = vx_load(sptr + j);
                        v_float32 knan = v_not_nan(val);
                        v_float32 alpha = (v_absdiff(val, rval) * sindex) & v_not_nan(rval) & knan;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);

                        v_float32 w = (kweight * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one-alpha))) & knan;
                        v_store_aligned(wsum + j, vx_load_aligned(wsum + j) + w);
                        v_store_aligned(sum + j, v_muladd(val & knan, w, vx_load_aligned(sum + j)));
                    }
#endif
                    for (; j < size.width; j++)
                    {
                        float val = ksptr[j];
                        float rval = sptr[j];
                        float alpha = std::abs(val - rval) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!cvIsNaN(val))
                        {
                            float w = space_weight[k] * (cvIsNaN(rval) ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum[j] += val * w;
                        }
                    }
                }
                j = 0;
#if CV_SIMD
                for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                {
                    v_float32 v_val = vx_load(sptr + j);
                    v_store(dptr + j, (vx_load_aligned(sum + j) + (v_val & v_not_nan(v_val))) / (vx_load_aligned(wsum + j) + (v_one & v_not_nan(v_val))));
                }
#endif
                for (; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) >= 0);
                    dptr[j] = cvIsNaN(sptr[j]) ? sum[j] / wsum[j] : (sum[j] + sptr[j]) / (wsum[j] + 1.f);
                }
            }
            else
            {
                CV_Assert( cn == 3 );
                AutoBuffer<float> buf(alignSize(size.width, CV_SIMD_WIDTH)*3 + size.width + CV_SIMD_WIDTH - 1);
                memset(buf.data(), 0, buf.size() * sizeof(float));
                float *sum_b = alignPtr(buf.data(), CV_SIMD_WIDTH);
                float *sum_g = sum_b + alignSize(size.width, CV_SIMD_WIDTH);
                float *sum_r = sum_g + alignSize(size.width, CV_SIMD_WIDTH);
                float *wsum = sum_r + alignSize(size.width, CV_SIMD_WIDTH);
#if CV_SIMD
                v_float32 v_one = vx_setall_f32(1.f);
                v_float32 sindex = vx_setall_f32(scale_index);
#endif
                k = 0;
                for (; k <= maxk-4; k+=4)
                {
                    const float* ksptr0 = sptr + space_ofs[k];
                    const float* ksptr1 = sptr + space_ofs[k+1];
                    const float* ksptr2 = sptr + space_ofs[k+2];
                    const float* ksptr3 = sptr + space_ofs[k+3];
                    const float* rsptr = sptr;
                    j = 0;
#if CV_SIMD
                    v_float32 kweight0 = vx_setall_f32(space_weight[k]);
                    v_float32 kweight1 = vx_setall_f32(space_weight[k+1]);
                    v_float32 kweight2 = vx_setall_f32(space_weight[k+2]);
                    v_float32 kweight3 = vx_setall_f32(space_weight[k+3]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes, rsptr += 3 * v_float32::nlanes,
                                                                ksptr0 += 3 * v_float32::nlanes, ksptr1 += 3 * v_float32::nlanes, ksptr2 += 3 * v_float32::nlanes, ksptr3 += 3 * v_float32::nlanes)
                    {
                        v_float32 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_load_deinterleave(ksptr0, kb, kg, kr);
                        v_float32 knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        v_float32 alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        v_float32 w = (kweight0 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_float32 v_wsum = vx_load_aligned(wsum + j) + w;
                        v_float32 v_sum_b = v_muladd(kb & knan, w, vx_load_aligned(sum_b + j));
                        v_float32 v_sum_g = v_muladd(kg & knan, w, vx_load_aligned(sum_g + j));
                        v_float32 v_sum_r = v_muladd(kr & knan, w, vx_load_aligned(sum_r + j));

                        v_load_deinterleave(ksptr1, kb, kg, kr);
                        knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight1 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum_b = v_muladd(kb & knan, w, v_sum_b);
                        v_sum_g = v_muladd(kg & knan, w, v_sum_g);
                        v_sum_r = v_muladd(kr & knan, w, v_sum_r);

                        v_load_deinterleave(ksptr2, kb, kg, kr);
                        knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight2 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum_b = v_muladd(kb & knan, w, v_sum_b);
                        v_sum_g = v_muladd(kg & knan, w, v_sum_g);
                        v_sum_r = v_muladd(kr & knan, w, v_sum_r);

                        v_load_deinterleave(ksptr3, kb, kg, kr);
                        knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        w = (kweight3 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_wsum += w;
                        v_sum_b = v_muladd(kb & knan, w, v_sum_b);
                        v_sum_g = v_muladd(kg & knan, w, v_sum_g);
                        v_sum_r = v_muladd(kr & knan, w, v_sum_r);

                        v_store_aligned(wsum + j, v_wsum);
                        v_store_aligned(sum_b + j, v_sum_b);
                        v_store_aligned(sum_g + j, v_sum_g);
                        v_store_aligned(sum_r + j, v_sum_r);
                    }
#endif
#if CV_SIMD128
                    v_float32x4 v_one4 = v_setall_f32(1.f);
                    v_float32x4 sindex4 = v_setall_f32(scale_index);
                    v_float32x4 kweight4 = v_load(space_weight + k);
#endif
                    for (; j < size.width; j++, rsptr += 3, ksptr0 += 3, ksptr1 += 3, ksptr2 += 3, ksptr3 += 3)
                    {
#if CV_SIMD128
                        v_float32x4 rb = v_setall_f32(rsptr[0]);
                        v_float32x4 rg = v_setall_f32(rsptr[1]);
                        v_float32x4 rr = v_setall_f32(rsptr[2]);
                        v_float32x4 kb(ksptr0[0], ksptr1[0], ksptr2[0], ksptr3[0]);
                        v_float32x4 kg(ksptr0[1], ksptr1[1], ksptr2[1], ksptr3[1]);
                        v_float32x4 kr(ksptr0[2], ksptr1[2], ksptr2[2], ksptr3[2]);
                        v_float32x4 knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        v_float32x4 alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex4) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        v_int32x4 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);
                        v_float32x4 w = (kweight4 * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one4 - alpha))) & knan;
                        wsum[j] += v_reduce_sum(w);
                        sum_b[j] += v_reduce_sum((kb & knan) * w);
                        sum_g[j] += v_reduce_sum((kg & knan) * w);
                        sum_r[j] += v_reduce_sum((kr & knan) * w);
#else
                        float rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];
                        bool r_NAN = cvIsNaN(rb) || cvIsNaN(rg) || cvIsNaN(rr);

                        float b = ksptr0[0], g = ksptr0[1], r = ksptr0[2];
                        bool v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        float alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr1[0]; g = ksptr1[1]; r = ksptr1[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+1] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr2[0]; g = ksptr2[1]; r = ksptr2[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+2] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }

                        b = ksptr3[0]; g = ksptr3[1]; r = ksptr3[2];
                        v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k+3] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }
#endif
                    }
                }
                for (; k < maxk; k++)
                {
                    const float* ksptr = sptr + space_ofs[k];
                    const float* rsptr = sptr;
                    j = 0;
#if CV_SIMD
                    v_float32 kweight = vx_setall_f32(space_weight[k]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes, ksptr += 3*v_float32::nlanes, rsptr += 3*v_float32::nlanes)
                    {
                        v_float32 kb, kg, kr, rb, rg, rr;
                        v_load_deinterleave(ksptr, kb, kg, kr);
                        v_load_deinterleave(rsptr, rb, rg, rr);

                        v_float32 knan = v_not_nan(kb) & v_not_nan(kg) & v_not_nan(kr);
                        v_float32 alpha = ((v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex) & v_not_nan(rb) & v_not_nan(rg) & v_not_nan(rr) & knan;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);

                        v_float32 w = (kweight * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha))) & knan;
                        v_store_aligned(wsum + j, vx_load_aligned(wsum + j) + w);
                        v_store_aligned(sum_b + j, v_muladd(kb & knan, w, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_g + j, v_muladd(kg & knan, w, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_r + j, v_muladd(kr & knan, w, vx_load_aligned(sum_r + j)));
                    }
#endif
                    for (; j < size.width; j++, ksptr += 3, rsptr += 3)
                    {
                        float b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        bool v_NAN = cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r);
                        float rb = rsptr[0], rg = rsptr[1], rr = rsptr[2];
                        bool r_NAN = cvIsNaN(rb) || cvIsNaN(rg) || cvIsNaN(rr);
                        float alpha = (std::abs(b - rb) + std::abs(g - rg) + std::abs(r - rr)) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        if (!v_NAN)
                        {
                            float w = space_weight[k] * (r_NAN ? 1.f : (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx])));
                            wsum[j] += w;
                            sum_b[j] += b*w;
                            sum_g[j] += g*w;
                            sum_r[j] += r*w;
                        }
                    }
                }
                j = 0;
#if CV_SIMD
                for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes, sptr += 3*v_float32::nlanes, dptr += 3*v_float32::nlanes)
                {
                    v_float32 b, g, r;
                    v_load_deinterleave(sptr, b, g, r);
                    v_float32 mask = v_not_nan(b) & v_not_nan(g) & v_not_nan(r);
                    v_float32 w = v_one / (vx_load_aligned(wsum + j) + (v_one & mask));
                    v_store_interleave(dptr, (vx_load_aligned(sum_b + j) + (b & mask)) * w, (vx_load_aligned(sum_g + j) + (g & mask)) * w, (vx_load_aligned(sum_r + j) + (r & mask)) * w);
                }
#endif
                for (; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) >= 0);
                    float b = *(sptr++);
                    float g = *(sptr++);
                    float r = *(sptr++);
                    if (cvIsNaN(b) || cvIsNaN(g) || cvIsNaN(r))
                    {
                        wsum[j] = 1.f / wsum[j];
                        *(dptr++) = sum_b[j] * wsum[j];
                        *(dptr++) = sum_g[j] * wsum[j];
                        *(dptr++) = sum_r[j] * wsum[j];
                    }
                    else
                    {
                        wsum[j] = 1.f / (wsum[j] + 1.f);
                        *(dptr++) = (sum_b[j] + b) * wsum[j];
                        *(dptr++) = (sum_g[j] + g) * wsum[j];
                        *(dptr++) = (sum_r[j] + r) * wsum[j];
                    }
                }
            }
        }
#if CV_SIMD
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
