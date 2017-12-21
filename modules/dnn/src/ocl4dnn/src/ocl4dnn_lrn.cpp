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
#include "common.hpp"
#include "ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"

#ifdef HAVE_OPENCL
namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNLRN<Dtype>::OCL4DNNLRN(OCL4DNNLRNConfig config)
{
    lrn_type_ = config.lrn_type;
    phase_test_ = config.phase_test;
    size_ = config.local_size;
    CHECK_EQ(size_ % 2, 1)<< "LRN only supports odd values for local_size";
    alpha_ = config.alpha;
    beta_ = config.beta;
    k_ = config.k;
    norm_by_size_ = config.norm_by_size;
    num_ = config.batch_size;
    channels_ = config.channels;
    height_ = config.height;
    width_ = config.width;
}

template<typename Dtype>
bool OCL4DNNLRN<Dtype>::Forward(const UMat& bottom, UMat& top)
{
    bool ret = true;

    if (!ocl::Device::getDefault().intelSubgroupsSupport())
        return false;

    switch (lrn_type_)
    {
    case LRNParameter_NormRegion_ACROSS_CHANNELS:
        ret = crossChannelForward(bottom, top);
        break;
    case LRNParameter_NormRegion_WITHIN_CHANNEL:
        //TODO
        //WithinChannelForward(bottom_data, top_data);
        ret = false;
        break;
    default:
        ret = false;
        LOG(FATAL)<< "Unknown normalization region.";
    }
    return ret;
}

template<typename Dtype>
bool OCL4DNNLRN<Dtype>::crossChannelForward(const UMat& bottom, UMat& top)
{
    CHECK_EQ(phase_test_, true) << "Only support forward inference.";

    cl_uint argIdx = 0;
    int32_t n_threads = num_ * height_ * width_;
    size_t global_work_size_[1] = {(size_t)n_threads};
    String opts = clOptionSupport("-cl-no-subgroup-ifp") ? " -cl-no-subgroup-ifp " : "";
    ocl::Kernel oclk_lrn_fill;
    if (!oclk_lrn_fill.create(CL_KERNEL_SELECT("lrn_full_no_scale"), ocl::dnn::ocl4dnn_lrn_oclsrc, opts))
        return false;

    oclk_lrn_fill.set(argIdx++, n_threads);
    oclk_lrn_fill.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
    oclk_lrn_fill.set(argIdx++, num_);
    oclk_lrn_fill.set(argIdx++, channels_);
    oclk_lrn_fill.set(argIdx++, height_);
    oclk_lrn_fill.set(argIdx++, width_);
    oclk_lrn_fill.set(argIdx++, size_);
    int size_norm_factor = norm_by_size_ ? size_ : 1;
    oclk_lrn_fill.set(argIdx++, alpha_ / size_norm_factor);
    oclk_lrn_fill.set(argIdx++, k_);
    oclk_lrn_fill.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
    oclk_lrn_fill.set(argIdx++, -beta_);

    return oclk_lrn_fill.run(1, global_work_size_, NULL, false);
}

template class OCL4DNNLRN<float>;
} // namespace ocl4dnn
}
}
#endif // HAVE_OPENCL
