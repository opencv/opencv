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

namespace cv {

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

    virtual void operator() (const Range & range) const
    {
        int cn = src1->channels(), width = src1->cols * cn;

        for (int y = range.start; y < range.end; ++y)
        {
            const float * const weights1_row = weights1->ptr<float>(y);
            const float * const weights2_row = weights2->ptr<float>(y);
            const T * const src1_row = src1->ptr<T>(y);
            const T * const src2_row = src2->ptr<T>(y);
            T * const dst_row = dst->ptr<T>(y);

            for (int x = 0; x < width; ++x)
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

    char cvt[30];
    ocl::Kernel k("blendLinear", ocl::imgproc::blend_linear_oclsrc,
                  format("-D T=%s -D cn=%d -D convertToT=%s", ocl::typeToStr(depth),
                         cn, ocl::convertTypeStr(CV_32F, depth, 1, cvt)));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat(), weights1 = _weights1.getUMat(),
            weights2 = _weights2.getUMat(), dst = _dst.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(src1), ocl::KernelArg::ReadOnlyNoSize(src2),
           ocl::KernelArg::ReadOnlyNoSize(weights1), ocl::KernelArg::ReadOnlyNoSize(weights2),
           ocl::KernelArg::WriteOnly(dst));

    size_t globalsize[2] = { dst.cols, dst.rows };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

void cv::blendLinear( InputArray _src1, InputArray _src2, InputArray _weights1, InputArray _weights2, OutputArray _dst )
{
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
