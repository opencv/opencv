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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

/****************************************************************************************\
                                   Bilateral Filtering
\****************************************************************************************/

namespace cv
{

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
                for( k = 0; k < maxk; k++ )
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
                for(k = 0; k < maxk; k++ )
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

#ifdef HAVE_OPENCL

static bool ocl_bilateralFilter_8u(InputArray _src, OutputArray _dst, int d,
                                   double sigma_color, double sigma_space,
                                   int borderType)
{
#ifdef __ANDROID__
    if (ocl::Device::getDefault().isNVidia())
        return false;
#endif

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int i, j, maxk, radius;

    if (depth != CV_8U || cn > 4)
        return false;

    if (sigma_color <= 0)
        sigma_color = 1;
    if (sigma_space <= 0)
        sigma_space = 1;

    double gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    double gauss_space_coeff = -0.5 / (sigma_space * sigma_space);

    if ( d <= 0 )
        radius = cvRound(sigma_space * 1.5);
    else
        radius = d / 2;
    radius = MAX(radius, 1);
    d = radius * 2 + 1;

    UMat src = _src.getUMat(), dst = _dst.getUMat(), temp;
    if (src.u == dst.u)
        return false;

    copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);
    std::vector<float> _space_weight(d * d);
    std::vector<int> _space_ofs(d * d);
    float * const space_weight = &_space_weight[0];
    int * const space_ofs = &_space_ofs[0];

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i * i + (double)j * j);
            if ( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r * r * gauss_space_coeff);
            space_ofs[maxk++] = (int)(i * temp.step + j * cn);
        }

    char cvt[3][40];
    String cnstr = cn > 1 ? format("%d", cn) : "";
    String kernelName("bilateral");
    size_t sizeDiv = 1;
    if ((ocl::Device::getDefault().isIntel()) &&
        (ocl::Device::getDefault().type() == ocl::Device::TYPE_GPU))
    {
            //Intel GPU
            if (dst.cols % 4 == 0 && cn == 1) // For single channel x4 sized images.
            {
                kernelName = "bilateral_float4";
                sizeDiv = 4;
            }
     }
     ocl::Kernel k(kernelName.c_str(), ocl::imgproc::bilateral_oclsrc,
            format("-D radius=%d -D maxk=%d -D cn=%d -D int_t=%s -D uint_t=uint%s -D convert_int_t=%s"
            " -D uchar_t=%s -D float_t=%s -D convert_float_t=%s -D convert_uchar_t=%s -D gauss_color_coeff=(float)%f",
            radius, maxk, cn, ocl::typeToStr(CV_32SC(cn)), cnstr.c_str(),
            ocl::convertTypeStr(CV_8U, CV_32S, cn, cvt[0]),
            ocl::typeToStr(type), ocl::typeToStr(CV_32FC(cn)),
            ocl::convertTypeStr(CV_32S, CV_32F, cn, cvt[1]),
            ocl::convertTypeStr(CV_32F, CV_8U, cn, cvt[2]), gauss_color_coeff));
    if (k.empty())
        return false;

    Mat mspace_weight(1, d * d, CV_32FC1, space_weight);
    Mat mspace_ofs(1, d * d, CV_32SC1, space_ofs);
    UMat ucolor_weight, uspace_weight, uspace_ofs;

    mspace_weight.copyTo(uspace_weight);
    mspace_ofs.copyTo(uspace_ofs);

    k.args(ocl::KernelArg::ReadOnlyNoSize(temp), ocl::KernelArg::WriteOnly(dst),
           ocl::KernelArg::PtrReadOnly(uspace_weight),
           ocl::KernelArg::PtrReadOnly(uspace_ofs));

    size_t globalsize[2] = { (size_t)dst.cols / sizeDiv, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}

#endif
static void
bilateralFilter_8u( const Mat& src, Mat& dst, int d,
    double sigma_color, double sigma_space,
    int borderType )
{
    int cn = src.channels();
    int i, j, maxk, radius;
    Size size = src.size();

    CV_Assert( (src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;

    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

    std::vector<float> _color_weight(cn*256);
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* color_weight = &_color_weight[0];
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // initialize color-related bilateral filter coefficients

    for( i = 0; i < 256*cn; i++ )
        color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
    {
        j = -radius;

        for( ; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*temp.step + j*cn);
        }
    }

    BilateralFilter_8u_Invoker body(dst, temp, radius, maxk, space_ofs, space_weight, color_weight);
    parallel_for_(Range(0, size.height), body, dst.total()/(double)(1<<16));
}


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
                for( k = 0; k < maxk; k++ )
                {
                    const float* ksptr = sptr + space_ofs[k];
                    j = 0;
#if CV_SIMD
                    v_float32 kweight = vx_setall_f32(space_weight[k]);
                    for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    {
                        v_float32 val = vx_load(ksptr + j);

                        v_float32 alpha = v_absdiff(val, vx_load(sptr + j)) * sindex;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);

                        v_float32 w = kweight * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one-alpha));
                        v_store_aligned(wsum + j, vx_load_aligned(wsum + j) + w);
                        v_store_aligned(sum + j, v_muladd(val, w, vx_load_aligned(sum + j)));
                    }
#endif
                    for (; j < size.width; j++)
                    {
                        float val = ksptr[j];
                        float alpha = std::abs(val - sptr[j]) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k] * (expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                        wsum[j] += w;
                        sum[j] += val * w;
                    }
                }
                j = 0;
#if CV_SIMD
                for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes)
                    v_store(dptr + j, vx_load_aligned(sum + j) / vx_load_aligned(wsum + j));
#endif
                for (; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) > 0);
                    dptr[j] = sum[j] / wsum[j];
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
                for (k = 0; k < maxk; k++)
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

                        v_float32 alpha = (v_absdiff(kb, rb) + v_absdiff(kg, rg) + v_absdiff(kr, rr)) * sindex;
                        v_int32 idx = v_trunc(alpha);
                        alpha -= v_cvt_f32(idx);

                        v_float32 w = kweight * v_muladd(v_lut(expLUT + 1, idx), alpha, v_lut(expLUT, idx) * (v_one - alpha));
                        v_store_aligned(wsum + j, vx_load_aligned(wsum + j) + w);
                        v_store_aligned(sum_b + j, v_muladd(kb, w, vx_load_aligned(sum_b + j)));
                        v_store_aligned(sum_g + j, v_muladd(kg, w, vx_load_aligned(sum_g + j)));
                        v_store_aligned(sum_r + j, v_muladd(kr, w, vx_load_aligned(sum_r + j)));
                    }
#endif
                    for (; j < size.width; j++, ksptr += 3, rsptr += 3)
                    {
                        float b = ksptr[0], g = ksptr[1], r = ksptr[2];
                        float alpha = (std::abs(b - rsptr[0]) + std::abs(g - rsptr[1]) + std::abs(r - rsptr[2])) * scale_index;
                        int idx = cvFloor(alpha);
                        alpha -= idx;
                        float w = space_weight[k] * (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx]));
                        wsum[j] += w;
                        sum_b[j] += b*w;
                        sum_g[j] += g*w;
                        sum_r[j] += r*w;
                    }
                }
                j = 0;
#if CV_SIMD
                for (; j <= size.width - v_float32::nlanes; j += v_float32::nlanes, dptr += 3*v_float32::nlanes)
                {
                    v_float32 w = v_one / vx_load_aligned(wsum + j);
                    v_store_interleave(dptr, vx_load_aligned(sum_b + j) * w, vx_load_aligned(sum_g + j) * w, vx_load_aligned(sum_r + j) * w);
                }
#endif
                for (; j < size.width; j++)
                {
                    CV_DbgAssert(fabs(wsum[j]) > 0);
                    wsum[j] = 1.f / wsum[j];
                    *(dptr++) = sum_b[j] * wsum[j];
                    *(dptr++) = sum_g[j] * wsum[j];
                    *(dptr++) = sum_r[j] * wsum[j];
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


static void
bilateralFilter_32f( const Mat& src, Mat& dst, int d,
                     double sigma_color, double sigma_space,
                     int borderType )
{
    int cn = src.channels();
    int i, j, maxk, radius;
    double minValSrc=-1, maxValSrc=1;
    const int kExpNumBinsPerChannel = 1 << 12;
    int kExpNumBins = 0;
    float lastExpVal = 1.f;
    float len, scale_index;
    Size size = src.size();

    CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.data != dst.data );

    if( sigma_color <= 0 )
        sigma_color = 1;
    if( sigma_space <= 0 )
        sigma_space = 1;

    double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
    double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

    if( d <= 0 )
        radius = cvRound(sigma_space*1.5);
    else
        radius = d/2;
    radius = MAX(radius, 1);
    d = radius*2 + 1;
    // compute the min/max range for the input image (even if multichannel)

    minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
    if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
    {
        src.copyTo(dst);
        return;
    }

    // temporary copy of the image with borders for easy processing
    Mat temp;
    copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );
    minValSrc -= 5. * sigma_color;
    patchNaNs( temp, minValSrc ); // this replacement of NaNs makes the assumption that depth values are nonnegative
                                  // TODO: make replacement parameter avalible in the outside function interface
    // allocate lookup tables
    std::vector<float> _space_weight(d*d);
    std::vector<int> _space_ofs(d*d);
    float* space_weight = &_space_weight[0];
    int* space_ofs = &_space_ofs[0];

    // assign a length which is slightly more than needed
    len = (float)(maxValSrc - minValSrc) * cn;
    kExpNumBins = kExpNumBinsPerChannel * cn;
    std::vector<float> _expLUT(kExpNumBins+2);
    float* expLUT = &_expLUT[0];

    scale_index = kExpNumBins/len;

    // initialize the exp LUT
    for( i = 0; i < kExpNumBins+2; i++ )
    {
        if( lastExpVal > 0.f )
        {
            double val =  i / scale_index;
            expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
            lastExpVal = expLUT[i];
        }
        else
            expLUT[i] = 0.f;
    }

    // initialize space-related bilateral filter coefficients
    for( i = -radius, maxk = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            double r = std::sqrt((double)i*i + (double)j*j);
            if( r > radius )
                continue;
            space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
            space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
        }

    // parallel_for usage

    BilateralFilter_32f_Invoker body(cn, radius, maxk, space_ofs, temp, dst, scale_index, space_weight, expLUT);
    parallel_for_(Range(0, size.height), body, dst.total()/(double)(1<<16));
}

#ifdef HAVE_IPP
#define IPP_BILATERAL_PARALLEL 1

#ifdef HAVE_IPP_IW
class ipp_bilateralFilterParallel: public ParallelLoopBody
{
public:
    ipp_bilateralFilterParallel(::ipp::IwiImage &_src, ::ipp::IwiImage &_dst, int _radius, Ipp32f _valSquareSigma, Ipp32f _posSquareSigma, ::ipp::IwiBorderType _borderType, bool *_ok):
        src(_src), dst(_dst)
    {
        pOk = _ok;

        radius          = _radius;
        valSquareSigma  = _valSquareSigma;
        posSquareSigma  = _posSquareSigma;
        borderType      = _borderType;

        *pOk = true;
    }
    ~ipp_bilateralFilterParallel() {}

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        if(*pOk == false)
            return;

        try
        {
            ::ipp::IwiTile tile = ::ipp::IwiRoi(0, range.start, dst.m_size.width, range.end - range.start);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, src, dst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), borderType, tile);
        }
        catch(const ::ipp::IwException &)
        {
            *pOk = false;
            return;
        }
    }
private:
    ::ipp::IwiImage &src;
    ::ipp::IwiImage &dst;

    int                  radius;
    Ipp32f               valSquareSigma;
    Ipp32f               posSquareSigma;
    ::ipp::IwiBorderType borderType;

    bool  *pOk;
    const ipp_bilateralFilterParallel& operator= (const ipp_bilateralFilterParallel&);
};
#endif

static bool ipp_bilateralFilter(Mat &src, Mat &dst, int d, double sigmaColor, double sigmaSpace, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP();

    int         radius         = IPP_MAX(((d <= 0)?cvRound(sigmaSpace*1.5):d/2), 1);
    Ipp32f      valSquareSigma = (Ipp32f)((sigmaColor <= 0)?1:sigmaColor*sigmaColor);
    Ipp32f      posSquareSigma = (Ipp32f)((sigmaSpace <= 0)?1:sigmaSpace*sigmaSpace);

    // Acquire data and begin processing
    try
    {
        ::ipp::IwiImage      iwSrc = ippiGetImage(src);
        ::ipp::IwiImage      iwDst = ippiGetImage(dst);
        ::ipp::IwiBorderSize borderSize(radius);
        ::ipp::IwiBorderType ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        const int threads = ippiSuggestThreadsNum(iwDst, 2);
        if(IPP_BILATERAL_PARALLEL && threads > 1) {
            bool  ok      = true;
            Range range(0, (int)iwDst.m_size.height);
            ipp_bilateralFilterParallel invoker(iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ippBorder, &ok);
            if(!ok)
                return false;

            parallel_for_(range, invoker, threads*4);

            if(!ok)
                return false;
        } else {
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterBilateral, iwSrc, iwDst, radius, valSquareSigma, posSquareSigma, ::ipp::IwDefault(), ippBorder);
        }
    }
    catch (const ::ipp::IwException &)
    {
        return false;
    }
    return true;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(d); CV_UNUSED(sigmaColor); CV_UNUSED(sigmaSpace); CV_UNUSED(borderType);
    return false;
#endif
}
#endif

}

void cv::bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType )
{
    CV_INSTRUMENT_REGION();

    _dst.create( _src.size(), _src.type() );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_bilateralFilter_8u(_src, _dst, d, sigmaColor, sigmaSpace, borderType))

    Mat src = _src.getMat(), dst = _dst.getMat();

    CV_IPP_RUN_FAST(ipp_bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType));

    if( src.depth() == CV_8U )
        bilateralFilter_8u( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else if( src.depth() == CV_32F )
        bilateralFilter_32f( src, dst, d, sigmaColor, sigmaSpace, borderType );
    else
        CV_Error( CV_StsUnsupportedFormat,
        "Bilateral filtering is only implemented for 8u and 32f images" );
}

/* End of file. */
