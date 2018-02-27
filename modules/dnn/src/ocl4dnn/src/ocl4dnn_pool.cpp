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
#include "common.hpp"
#include "ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"

#ifdef HAVE_OPENCL
namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNPool<Dtype>::OCL4DNNPool(OCL4DNNPoolConfig config)
{
    int dims = config.in_shape.size();
    int spatial_dims = 2;

    channels_ = config.channels;
    pool_method_ = config.pool_method;
    avePoolPaddedArea = config.avePoolPaddedArea;

    for (int i = 0; i < spatial_dims; ++i)
    {
        kernel_shape_.push_back(i == 0 ? config.kernel.height : config.kernel.width);
        pad_.push_back(i == 0 ? config.pad.height : config.pad.width);
        stride_.push_back(i == 0 ? config.stride.height : config.stride.width);
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    pad_h_ = pad_[0];
    pad_w_ = pad_[1];
    height_ = im_in_shape_[0];
    width_ = im_in_shape_[1];
    pooled_height_ = im_out_shape_[0];
    pooled_width_ = im_out_shape_[1];

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
    size_t global[] = { 128 * 128 };
    size_t local[] = { 128 };

    // support 2D case
    switch (pool_method_)
    {
    case LIBDNN_POOLING_METHOD_MAX:
        {
            bool haveMask = !top_mask.empty();
            ocl::Kernel oclk_max_pool_forward(
                haveMask ? CL_KERNEL_SELECT("max_pool_forward_mask") : CL_KERNEL_SELECT("max_pool_forward"),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format("-D KERNEL_MAX_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_W=%d -D PAD_H=%d%s",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_w_, pad_h_,
                       haveMask ? " -D HAVE_MASK=1" : ""
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
                ocl::KernelArg::PtrWriteOnly(top),
                ocl::KernelArg::PtrWriteOnly(top_mask)
            );

            ret = oclk_max_pool_forward.run(1, global, local, false);
        }
        break;
    case LIBDNN_POOLING_METHOD_AVE:
        {
            CV_Assert(top_mask.empty());

            ocl::Kernel oclk_ave_pool_forward(CL_KERNEL_SELECT("ave_pool_forward"),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format("-D KERNEL_AVE_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d"
                       " -D PAD_W=%d -D PAD_H=%d%s",
                       kernel_w_, kernel_h_,
                       stride_w_, stride_h_,
                       pad_w_, pad_h_,
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

            ocl::Kernel oclk_sto_pool_forward(CL_KERNEL_SELECT("sto_pool_forward_test"),
                ocl::dnn::ocl4dnn_pooling_oclsrc,
                format("-D KERNEL_STO_POOL=1 -D KERNEL_W=%d -D KERNEL_H=%d"
                       " -D STRIDE_W=%d -D STRIDE_H=%d",
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
} // namespace ocl4dnn
}
}
#endif // HAVE_OPENCL
