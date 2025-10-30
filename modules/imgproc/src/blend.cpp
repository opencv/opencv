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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Nathan, liujun@multicorewareinc.com
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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_float32 blend(const v_float32& v_src1, const v_float32& v_src2, const v_float32& v_w1, const v_float32& v_w2)
{
    const v_float32 v_eps = vx_setall_f32(1e-5f);
    v_float32 v_denom = v_add(v_add(v_w1, v_w2), v_eps);
    return v_div(v_add(v_mul(v_src1, v_w1), v_mul(v_src2, v_w2)), v_denom);
}
static inline v_float32 blend(const v_float32& v_src1, const v_float32& v_src2, const float* w_ptr1, const float* w_ptr2, int offset)
{
    v_float32 v_w1 = vx_load(w_ptr1 + offset);
    v_float32 v_w2 = vx_load(w_ptr2 + offset);
    return blend(v_src1, v_src2, v_w1, v_w2);
}
static inline v_uint32 saturate_f32_u32(const v_float32& vec)
{
    const v_int32 z = vx_setzero_s32();
    const v_int32 x = vx_setall_s32(255);
    return v_reinterpret_as_u32(v_min(v_max(v_round(vec), z), x));
}
static inline v_uint8 pack_f32tou8(v_float32& val0, v_float32& val1, v_float32& val2, v_float32& val3)
{
    v_uint32 a = saturate_f32_u32(val0);
    v_uint32 b = saturate_f32_u32(val1);
    v_uint32 c = saturate_f32_u32(val2);
    v_uint32 d = saturate_f32_u32(val3);
    v_uint16 e = v_pack(a, b);
    v_uint16 f = v_pack(c, d);
    return v_pack(e, f);
}
static inline void store_pack_f32tou8(uchar* ptr, v_float32& val0, v_float32& val1, v_float32& val2, v_float32& val3)
{
    v_store((ptr), pack_f32tou8(val0, val1, val2, val3));
}
static inline void expand_u8tof32(const v_uint8& src, v_float32& dst0, v_float32& dst1, v_float32& dst2, v_float32& dst3)
{
    v_uint16 a0, a1;
    v_expand(src, a0, a1);
    v_uint32 b0, b1,b2,b3;
    v_expand(a0, b0, b1);
    v_expand(a1, b2, b3);
    dst0 = v_cvt_f32(v_reinterpret_as_s32(b0));
    dst1 = v_cvt_f32(v_reinterpret_as_s32(b1));
    dst2 = v_cvt_f32(v_reinterpret_as_s32(b2));
    dst3 = v_cvt_f32(v_reinterpret_as_s32(b3));
}
static inline void load_expand_u8tof32(const uchar* ptr, v_float32& dst0, v_float32& dst1, v_float32& dst2, v_float32& dst3)
{
    v_uint8 a = vx_load((ptr));
    expand_u8tof32(a, dst0, dst1, dst2, dst3);
}
int blendLinearSimd(const uchar* src1, const uchar* src2, const float* weights1, const float* weights2, uchar* dst, int x, int width, int cn);
int blendLinearSimd(const float* src1, const float* src2, const float* weights1, const float* weights2, float* dst, int x, int width, int cn);
int blendLinearSimd(const uchar* src1, const uchar* src2, const float* weights1, const float* weights2, uchar* dst, int x, int width, int cn)
{
    switch(cn)
    {
    case 1:
        for(int weight_offset = 0 ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes(), weight_offset += VTraits<v_uint8>::vlanes())
        {
            v_float32 v_src10, v_src11, v_src12, v_src13;
            v_float32 v_src20, v_src21, v_src22, v_src23;
            load_expand_u8tof32(src1 + x, v_src10, v_src11, v_src12, v_src13);
            load_expand_u8tof32(src2 + x, v_src20, v_src21, v_src22, v_src23);

            v_float32 v_dst0 = blend(v_src10, v_src20, weights1, weights2, weight_offset);
            v_float32 v_dst1 = blend(v_src11, v_src21, weights1, weights2, weight_offset + VTraits<v_float32>::vlanes());
            v_float32 v_dst2 = blend(v_src12, v_src22, weights1, weights2, weight_offset + 2*VTraits<v_float32>::vlanes());
            v_float32 v_dst3 = blend(v_src13, v_src23, weights1, weights2, weight_offset + 3*VTraits<v_float32>::vlanes());

            store_pack_f32tou8(dst + x, v_dst0, v_dst1, v_dst2, v_dst3);
        }
        break;
    case 2:
        for(int weight_offset = 0 ; x <= width - 2*VTraits<v_uint8>::vlanes(); x += 2*VTraits<v_uint8>::vlanes(), weight_offset += VTraits<v_uint8>::vlanes())
        {
            v_uint8 v_src10, v_src11, v_src20, v_src21;
            v_load_deinterleave(src1 + x, v_src10, v_src11);
            v_load_deinterleave(src2 + x, v_src20, v_src21);
            v_float32 v_src100, v_src101, v_src102, v_src103, v_src110, v_src111, v_src112, v_src113;
            v_float32 v_src200, v_src201, v_src202, v_src203, v_src210, v_src211, v_src212, v_src213;
            expand_u8tof32(v_src10, v_src100, v_src101, v_src102, v_src103);
            expand_u8tof32(v_src11, v_src110, v_src111, v_src112, v_src113);
            expand_u8tof32(v_src20, v_src200, v_src201, v_src202, v_src203);
            expand_u8tof32(v_src21, v_src210, v_src211, v_src212, v_src213);

            v_float32 v_dst0 = blend(v_src100, v_src200, weights1, weights2, weight_offset);
            v_float32 v_dst1 = blend(v_src110, v_src210, weights1, weights2, weight_offset);
            v_float32 v_dst2 = blend(v_src101, v_src201, weights1, weights2, weight_offset + VTraits<v_float32>::vlanes());
            v_float32 v_dst3 = blend(v_src111, v_src211, weights1, weights2, weight_offset + VTraits<v_float32>::vlanes());
            v_float32 v_dst4 = blend(v_src102, v_src202, weights1, weights2, weight_offset + 2*VTraits<v_float32>::vlanes());
            v_float32 v_dst5 = blend(v_src112, v_src212, weights1, weights2, weight_offset + 2*VTraits<v_float32>::vlanes());
            v_float32 v_dst6 = blend(v_src103, v_src203, weights1, weights2, weight_offset + 3*VTraits<v_float32>::vlanes());
            v_float32 v_dst7 = blend(v_src113, v_src213, weights1, weights2, weight_offset + 3*VTraits<v_float32>::vlanes());

            v_uint8 v_dsta = pack_f32tou8(v_dst0, v_dst2, v_dst4, v_dst6);
            v_uint8 v_dstb = pack_f32tou8(v_dst1, v_dst3, v_dst5, v_dst7);
            v_store_interleave(dst + x, v_dsta, v_dstb);
        }
        break;
    case 3:
        for(int weight_offset = 0 ; x <= width - 3*VTraits<v_uint8>::vlanes(); x += 3*VTraits<v_uint8>::vlanes(), weight_offset += VTraits<v_uint8>::vlanes())
        {
            v_uint8 v_src10, v_src11, v_src12, v_src20, v_src21, v_src22;
            v_load_deinterleave(src1 + x, v_src10, v_src11, v_src12);
            v_load_deinterleave(src2 + x, v_src20, v_src21, v_src22);

            v_float32 v_src100, v_src101, v_src102, v_src103, v_src110, v_src111, v_src112, v_src113, v_src120, v_src121, v_src122, v_src123;
            v_float32 v_src200, v_src201, v_src202, v_src203, v_src210, v_src211, v_src212, v_src213, v_src220, v_src221, v_src222, v_src223;
            expand_u8tof32(v_src10, v_src100, v_src101, v_src102, v_src103);
            expand_u8tof32(v_src11, v_src110, v_src111, v_src112, v_src113);
            expand_u8tof32(v_src12, v_src120, v_src121, v_src122, v_src123);
            expand_u8tof32(v_src20, v_src200, v_src201, v_src202, v_src203);
            expand_u8tof32(v_src21, v_src210, v_src211, v_src212, v_src213);
            expand_u8tof32(v_src22, v_src220, v_src221, v_src222, v_src223);

            v_float32 v_w10 = vx_load(weights1 + weight_offset);
            v_float32 v_w11 = vx_load(weights1 + weight_offset + VTraits<v_float32>::vlanes());
            v_float32 v_w12 = vx_load(weights1 + weight_offset + 2*VTraits<v_float32>::vlanes());
            v_float32 v_w13 = vx_load(weights1 + weight_offset + 3*VTraits<v_float32>::vlanes());
            v_float32 v_w20 = vx_load(weights2 + weight_offset);
            v_float32 v_w21 = vx_load(weights2 + weight_offset + VTraits<v_float32>::vlanes());
            v_float32 v_w22 = vx_load(weights2 + weight_offset + 2*VTraits<v_float32>::vlanes());
            v_float32 v_w23 = vx_load(weights2 + weight_offset + 3*VTraits<v_float32>::vlanes());
            v_src100 = blend(v_src100, v_src200, v_w10, v_w20);
            v_src110 = blend(v_src110, v_src210, v_w10, v_w20);
            v_src120 = blend(v_src120, v_src220, v_w10, v_w20);
            v_src101 = blend(v_src101, v_src201, v_w11, v_w21);
            v_src111 = blend(v_src111, v_src211, v_w11, v_w21);
            v_src121 = blend(v_src121, v_src221, v_w11, v_w21);
            v_src102 = blend(v_src102, v_src202, v_w12, v_w22);
            v_src112 = blend(v_src112, v_src212, v_w12, v_w22);
            v_src122 = blend(v_src122, v_src222, v_w12, v_w22);
            v_src103 = blend(v_src103, v_src203, v_w13, v_w23);
            v_src113 = blend(v_src113, v_src213, v_w13, v_w23);
            v_src123 = blend(v_src123, v_src223, v_w13, v_w23);


            v_uint8 v_dst0 = pack_f32tou8(v_src100, v_src101, v_src102, v_src103);
            v_uint8 v_dst1 = pack_f32tou8(v_src110, v_src111, v_src112, v_src113);
            v_uint8 v_dst2 = pack_f32tou8(v_src120, v_src121, v_src122, v_src123);
            v_store_interleave(dst + x, v_dst0, v_dst1, v_dst2);
        }
        break;
    case 4:
        for(int weight_offset = 0 ; x <= width - VTraits<v_uint8>::vlanes(); x += VTraits<v_uint8>::vlanes(), weight_offset += VTraits<v_float32>::vlanes())
        {
            v_float32 v_src10, v_src11, v_src12, v_src13;
            v_float32 v_src20, v_src21, v_src22, v_src23;
            load_expand_u8tof32(src1 + x, v_src10, v_src11, v_src12, v_src13);
            load_expand_u8tof32(src2 + x, v_src20, v_src21, v_src22, v_src23);

            v_float32 v_w10, v_w11, v_w12, v_w13, v_w20, v_w21, v_w22, v_w23, v_w0, v_w1;
            v_w10 = vx_load(weights1 + weight_offset);
            v_zip(v_w10, v_w10, v_w0, v_w1);
            v_zip(v_w0, v_w0, v_w10, v_w11);
            v_zip(v_w1, v_w1, v_w12, v_w13);
            v_w20 = vx_load(weights2 + weight_offset);
            v_zip(v_w20, v_w20, v_w0, v_w1);
            v_zip(v_w0, v_w0, v_w20, v_w21);
            v_zip(v_w1, v_w1, v_w22, v_w23);

            v_float32 v_dst0, v_dst1, v_dst2, v_dst3;
            v_dst0 = blend(v_src10, v_src20, v_w10, v_w20);
            v_dst1 = blend(v_src11, v_src21, v_w11, v_w21);
            v_dst2 = blend(v_src12, v_src22, v_w12, v_w22);
            v_dst3 = blend(v_src13, v_src23, v_w13, v_w23);

            store_pack_f32tou8(dst + x, v_dst0, v_dst1, v_dst2, v_dst3);
        }
        break;
    default:
        break;
    }
    return x;
}

int blendLinearSimd(const float* src1, const float* src2, const float* weights1, const float* weights2, float* dst, int x, int width, int cn)
{
    switch(cn)
    {
    case 1:
        for(int weight_offset = 0 ; x <= width - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes(), weight_offset += VTraits<v_float32>::vlanes())
        {
            v_float32 v_src1 = vx_load(src1 + x);
            v_float32 v_src2 = vx_load(src2 + x);
            v_float32 v_w1 = vx_load(weights1 + weight_offset);
            v_float32 v_w2 = vx_load(weights2 + weight_offset);

            v_float32 v_dst = blend(v_src1, v_src2, v_w1, v_w2);

            v_store(dst + x, v_dst);
        }
        break;
    case 2:
        for(int weight_offset = 0 ; x <= width - 2*VTraits<v_float32>::vlanes(); x += 2*VTraits<v_float32>::vlanes(), weight_offset += VTraits<v_float32>::vlanes())
        {
            v_float32 v_src10, v_src11, v_src20, v_src21;
            v_load_deinterleave(src1 + x, v_src10, v_src11);
            v_load_deinterleave(src2 + x, v_src20, v_src21);
            v_float32 v_w1 = vx_load(weights1 + weight_offset);
            v_float32 v_w2 = vx_load(weights2 + weight_offset);

            v_float32 v_dst0 = blend(v_src10, v_src20, v_w1, v_w2);
            v_float32 v_dst1 = blend(v_src11, v_src21, v_w1, v_w2);

            v_store_interleave(dst + x, v_dst0, v_dst1);
        }
        break;
    case 3:
        for(int weight_offset = 0 ; x <= width - 3*VTraits<v_float32>::vlanes(); x += 3*VTraits<v_float32>::vlanes(), weight_offset += VTraits<v_float32>::vlanes())
        {
            v_float32 v_src10, v_src11, v_src12, v_src20, v_src21, v_src22;
            v_load_deinterleave(src1 + x, v_src10, v_src11, v_src12);
            v_load_deinterleave(src2 + x, v_src20, v_src21, v_src22);
            v_float32 v_w1 = vx_load(weights1 + weight_offset);
            v_float32 v_w2 = vx_load(weights2 + weight_offset);

            v_float32 v_dst0 = blend(v_src10, v_src20, v_w1, v_w2);
            v_float32 v_dst1 = blend(v_src11, v_src21, v_w1, v_w2);
            v_float32 v_dst2 = blend(v_src12, v_src22, v_w1, v_w2);

            v_store_interleave(dst + x, v_dst0, v_dst1, v_dst2);
        }
        break;
    case 4:
        for(int weight_offset = 0 ; x <= width - 4*VTraits<v_float32>::vlanes(); x += 4*VTraits<v_float32>::vlanes(), weight_offset += VTraits<v_float32>::vlanes())
        {
            v_float32 v_src10, v_src11, v_src12, v_src13, v_src20, v_src21, v_src22, v_src23;
            v_load_deinterleave(src1 + x, v_src10, v_src11, v_src12, v_src13);
            v_load_deinterleave(src2 + x, v_src20, v_src21, v_src22, v_src23);
            v_float32 v_w1 = vx_load(weights1 + weight_offset);
            v_float32 v_w2 = vx_load(weights2 + weight_offset);

            v_float32 v_dst0 = blend(v_src10, v_src20, v_w1, v_w2);
            v_float32 v_dst1 = blend(v_src11, v_src21, v_w1, v_w2);
            v_float32 v_dst2 = blend(v_src12, v_src22, v_w1, v_w2);
            v_float32 v_dst3 = blend(v_src13, v_src23, v_w1, v_w2);

            v_store_interleave(dst + x, v_dst0, v_dst1, v_dst2, v_dst3);
        }
        break;
    default:
        break;
    }
    return x;
}
#endif

template <typename T>
class BlendLinearInvoker :
        public ParallelLoopBody
{
public:
    BlendLinearInvoker(const Mat & _src1, const Mat & _src2, const Mat & _weights1,
                       const Mat & _weights2, Mat & _dst) :
        src1(&_src1), src2(&_src2), weights1(&_weights1), weights2(&_weights2), dst(&_dst)
    {
    }

    virtual void operator() (const Range & range) const CV_OVERRIDE
    {
        int cn = src1->channels(), width = src1->cols * cn;

        for (int y = range.start; y < range.end; ++y)
        {
            const float * const weights1_row = weights1->ptr<float>(y);
            const float * const weights2_row = weights2->ptr<float>(y);
            const T * const src1_row = src1->ptr<T>(y);
            const T * const src2_row = src2->ptr<T>(y);
            T * const dst_row = dst->ptr<T>(y);

            int x = 0;
            #if (CV_SIMD || CV_SIMD_SCALABLE)
            x = blendLinearSimd(src1_row, src2_row, weights1_row, weights2_row, dst_row, x, width, cn);
            #endif

            for ( ; x < width; ++x)
            {
                int x1 = x / cn;
                float w1 = weights1_row[x1], w2 = weights2_row[x1];
                float den = (w1 + w2 + 1e-5f);
                float num = (src1_row[x] * w1 + src2_row[x] * w2);

                dst_row[x] = saturate_cast<T>(num / den);
            }
        }
    }

private:
    const BlendLinearInvoker & operator= (const BlendLinearInvoker &);
    BlendLinearInvoker(const BlendLinearInvoker &);

    const Mat * src1, * src2, * weights1, * weights2;
    Mat * dst;
};

#ifdef HAVE_OPENCL

static bool ocl_blendLinear( InputArray _src1, InputArray _src2, InputArray _weights1, InputArray _weights2, OutputArray _dst )
{
    int type = _src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    char cvt[50];
    ocl::Kernel k("blendLinear", ocl::imgproc::blend_linear_oclsrc,
                  format("-D T=%s -D cn=%d -D convertToT=%s", ocl::typeToStr(depth),
                         cn, ocl::convertTypeStr(CV_32F, depth, 1, cvt, sizeof(cvt))));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat(), weights1 = _weights1.getUMat(),
            weights2 = _weights2.getUMat(), dst = _dst.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(src1), ocl::KernelArg::ReadOnlyNoSize(src2),
           ocl::KernelArg::ReadOnlyNoSize(weights1), ocl::KernelArg::ReadOnlyNoSize(weights2),
           ocl::KernelArg::WriteOnly(dst));

    size_t globalsize[2] = { (size_t)dst.cols, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::blendLinear( InputArray _src1, InputArray _src2, InputArray _weights1, InputArray _weights2, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src1.type(), depth = CV_MAT_DEPTH(type);
    Size size = _src1.size();

    CV_Assert(depth == CV_8U || depth == CV_32F);
    CV_Assert(size == _src2.size() && size == _weights1.size() && size == _weights2.size());
    CV_Assert(type == _src2.type() && _weights1.type() == CV_32FC1 && _weights2.type() == CV_32FC1);

    _dst.create(size, type);

    CV_OCL_RUN(_dst.isUMat(),
               ocl_blendLinear(_src1, _src2, _weights1, _weights2, _dst))

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), weights1 = _weights1.getMat(),
            weights2 = _weights2.getMat(), dst = _dst.getMat();

    if (depth == CV_8U)
    {
        BlendLinearInvoker<uchar> invoker(src1, src2, weights1, weights2, dst);
        parallel_for_(Range(0, src1.rows), invoker, dst.total()/(double)(1<<16));
    }
    else if (depth == CV_32F)
    {
        BlendLinearInvoker<float> invoker(src1, src2, weights1, weights2, dst);
        parallel_for_(Range(0, src1.rows), invoker, dst.total()/(double)(1<<16));
    }
}
