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
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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

#include "../../precomp.hpp"
#include <string>
#include <vector>
#include "../include/common.hpp"
#include "../include/ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"

namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNPool<Dtype>::OCL4DNNPool(OCL4DNNPoolConfig config)
{
    int dims = config.in_shape.size();
    int spatial_dims = config.in_shape.size()-2;

    channels_ = config.channels;
    pool_method_ = config.pool_method;
    avePoolPaddedArea = config.avePoolPaddedArea;
    computeMaxIdx = config.computeMaxIdx;
    use_half = config.use_half;
    kernel_shape_.push_back(config.kernel.height);
    kernel_shape_.push_back(config.kernel.width);
    stride_.push_back(config.stride.height);
    stride_.push_back(config.stride.width);

    for (int i = 0; i < spatial_dims; ++i)
    {
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    pad_t_ = config.pad_t;
    pad_l_ = config.pad_l;
    pad_r_ = config.pad_r;
    pad_b_ = config.pad_b;
    height_ = spatial_dims == 1? 1 : im_in_shape_[0];
    width_ = im_in_shape_.back();
    pooled_height_ = spatial_dims == 1? 1 : im_out_shape_[0];
    pooled_width_ = im_out_shape_.back();

    count_ = 1;
    for (int i = 0; i < config.out_shape.size(); ++i)
    {
        count_ *= config.out_shape[i];
    }
}

template<typename Dtype>
OCL4DNNPool<Dtype>::~OCL4DNNPool()
{
    // nothing
}

template<typename Dtype>
bool OCL4DNNPool<Dtype>::Forward(const UMat& bottom,
                                 UMat& top,
                                 UMat& top_mask)
{
    bool ret = true;
    size_t global[] = { (size_t)count_ };
    size_t local[] = { 128 };

    // support 2D case
    switch (pool_method_)
    {
    case LIBDNN_POOLING_METHOD_MAX:
        {
            String kname = computeMaxIdx ? "max_pool_forward_mask" : "max_pool_forward";
            kname += (use_half) ? "_half" : "_float";
            ocl::Kernel oclk_max_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_MAX_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_L=%d -D PAD_T=%d -D PAD_R=%d -D PAD_B=%d%s",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_l_, pad_t_, pad_r_, pad_b_,
                       computeMaxIdx ? " -D HAVE_MASK=1" : ""
                ));
            if (oclk_max_pool_forward.empty())
                return false;

            oclk_max_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );
            if (computeMaxIdx)
                oclk_max_pool_forward.set(8, ocl::KernelArg::PtrWriteOnly(top_mask));  // TODO remove magic number. Extend cv::ocl::Kernel API

            ret = oclk_max_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_AVE:
        {
            CV_Assert(top_mask.empty());

            String kname = format("ave_pool_forward_%s", (use_half) ? "half" : "float");
            ocl::Kernel oclk_ave_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_AVE_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_L=%d -D PAD_T=%d -D PAD_R=%d -D PAD_B=%d%s",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_l_, pad_t_, pad_r_, pad_b_,
                       avePoolPaddedArea ? " -D AVE_POOL_PADDING_AREA" : ""
                ));

            if (oclk_ave_pool_forward.empty())
                return false;

            oclk_ave_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );

            ret = oclk_ave_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_STO:
        {
            CV_Assert(top_mask.empty());

            String kname = format("sto_pool_forward_test_%s", (use_half) ? "half" : "float");
            ocl::Kernel oclk_sto_pool_forward(
                kname.c_str(),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format(" -D Dtype=%s -D KERNEL_STO_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d",
                       (use_half) ? "half" : "float",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_
                ));


            if (oclk_sto_pool_forward.empty())
                return false;

            oclk_sto_pool_forward.args(
                count_,
                ocl::KernelArg::PtrReadOnly(bottom),
                channels_,
                height_,
                width_,
                pooled_height_,
                pooled_width_,
                ocl::KernelArg::PtrWriteOnly(top)
            );

            ret = oclk_sto_pool_forward.run(1, global, local, false);
        }
        break;
    default:
        {
            ret = false;
            LOG(FATAL)<< "Unknown pooling method.";
        }
    }
    return ret;
}

template class OCL4DNNPool<float>;

}}} // namespace cv::dnn::ocl4dnn
