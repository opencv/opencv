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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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
#include <vector>
#include "../include/common.hpp"
#include "../include/ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"

namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNSoftmax<Dtype>::OCL4DNNSoftmax(OCL4DNNSoftmaxConfig config)
{
    softmax_axis_ = config.axis;
    channels_ = config.channels;
    log_softmax_ = config.logsoftmax;
    use_half_ = config.use_half;

    inner_num_ = 1;
    outer_num_ = 1;
    count_ = 1;
    int32_t scale_sz = 1;
    for (int32_t i = softmax_axis_ + 1; i < config.in_shape.size(); i++)
        inner_num_ *= config.in_shape[i];
    use_slm_ = (config.in_shape[softmax_axis_] * inner_num_ + inner_num_ * 17) <= 8192;
    for (int32_t i = 0; i < softmax_axis_; i++)
        outer_num_ *= config.in_shape[i];
    count_ = inner_num_ + outer_num_;

    std::vector<int32_t> scale_dims = config.in_shape;
    scale_dims[softmax_axis_] = use_slm_ ? 1 : 17;
    for (int32_t i = 0; i < scale_dims.size(); i++)
        scale_sz *= scale_dims[i];

    scale_data_.create(1, scale_sz, CV_32FC1);
}

template<typename Dtype>
OCL4DNNSoftmax<Dtype>::~OCL4DNNSoftmax()
{
    scale_data_.release();
}

template<typename Dtype>
bool OCL4DNNSoftmax<Dtype>::Forward(const UMat& bottom, UMat& top)
{
    bool ret = false;
    bool intel_subgroup = ocl::Device::getDefault().intelSubgroupsSupport();
    if (intel_subgroup && inner_num_ < 128)
    {
        String opts = clOptionSupport("-cl-no-subgroup-ifp") ? " -cl-no-subgroup-ifp " : "";
        String kname;
        ocl::Kernel oclk_softmax_forward_kernel;

        if (log_softmax_) opts += " -DLOG_SOFTMAX ";
        if (use_slm_)
            kname = "softmax_forward_slm";
        else
            kname = "softmax_forward";

        kname += format("%s", (use_half_) ? "_half" : "_float");
        opts += format(" -D Dtype=%s -D DTYPE_MAX=%s", (use_half_) ? "half" : "float",
                       (use_half_) ? "HALF_MAX" : "FLT_MAX");
        if (!oclk_softmax_forward_kernel.create(kname.c_str(), ocl::dnn::softmax_loss_oclsrc, opts))
            return false;

        size_t global_size[] = { 256, (size_t)outer_num_, 1 };
        size_t local_size[] = { 256, 1, 1 };
        cl_uint argIdx = 0;

        if (use_slm_)
        {
            oclk_softmax_forward_kernel.set(argIdx++, outer_num_);
            oclk_softmax_forward_kernel.set(argIdx++, channels_);
            oclk_softmax_forward_kernel.set(argIdx++, inner_num_);
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(scale_data_));
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
            oclk_softmax_forward_kernel.set(argIdx++, NULL, channels_ * inner_num_* sizeof(Dtype));
            oclk_softmax_forward_kernel.set(argIdx++, NULL, inner_num_* sizeof(Dtype));
            oclk_softmax_forward_kernel.set(argIdx++, NULL, 16 * inner_num_* sizeof(Dtype));
        }
        else
        {
            oclk_softmax_forward_kernel.set(argIdx++, outer_num_);
            oclk_softmax_forward_kernel.set(argIdx++, channels_);
            oclk_softmax_forward_kernel.set(argIdx++, inner_num_);
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(scale_data_));
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            oclk_softmax_forward_kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
        }
        ret = oclk_softmax_forward_kernel.run_(3, global_size, local_size, false);
    }
    return ret;
}

template class OCL4DNNSoftmax<float>;

}}} // namespace cv::dnn::ocl4dnn
