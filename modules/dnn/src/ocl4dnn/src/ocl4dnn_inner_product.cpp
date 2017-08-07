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
#include "math_functions.hpp"

#ifdef HAVE_OPENCL
namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNInnerProduct<Dtype>::OCL4DNNInnerProduct(OCL4DNNInnerProductConfig config)
{
    bias_term_  = config.bias_term;
    transpose_  = config.transpose;
    N_ = num_output_ = config.num_output;
    M_ = config.M;
    K_ = config.K;
    phase_test_ = config.phase_test;
    image_copied_ = false;
    weight_image_ = NULL;

    // Set up the bias multiplier
    if (bias_term_)
    {
        int32_t flags = 0;
        flags = CL_MEM_READ_WRITE;
        allocateMemory((void **)&bias_multiplier_,  sizeof(Dtype) * M_, flags);
        ocl4dnnSet(0, M_, Dtype(1), (cl_mem) bias_multiplier_, 0);
    }
}

template<typename Dtype>
bool OCL4DNNInnerProduct<Dtype>::Forward(const Dtype* bottom_data,
                                         const Dtype* weight,
                                         const Dtype* bias,
                                         Dtype* top_data)
{
    if (M_ == 1)
    {
        ocl4dnnGEMV<Dtype>(0, CblasNoTrans, N_,
                           K_, (Dtype) 1., (cl_mem) weight, 0,
                           (cl_mem) bottom_data, 0, (Dtype) 0.,
                           (cl_mem) top_data, 0);
        if (bias_term_)
            ocl4dnnAXPY<Dtype>(0, N_,
                               1,
                               (cl_mem) bias, 0,
                               (cl_mem) top_data, 0);
    }
    else
    {
        return false;

        /*
        size_t max_image_size = std::min(ocl::Device::getDefault().image2DMaxWidth(),
                                         ocl::Device::getDefault().image2DMaxHeight());
        if (M_ <= max_image_size &&
            N_ <= max_image_size &&
            K_ <= max_image_size &&
            std::is_same<Dtype, float>::value &&
            ocl::Device::getDefault().intelSubgroupsSupport())
        {

            if (phase_test_ == false || image_copied_ == false)
            {
                int height = !transpose_ ? N_ : K_;
                int width = !transpose_ ? K_ : N_;
                int padded_height = !transpose_ ? height : (height + ((height & 7) ? 1 : 0));
                int padded_width = !transpose_ ? width : (width + ((width & 7) ? 1 : 0));
                ocl4dnnGEMMCopyBufferToImage(0,
                                             (cl_mem *)&weight_image_, (cl_mem) weight, 0,
                                             false, !transpose_,
                                             true, padded_height, padded_width,
                                             height, width, (int)0, NULL, NULL);
                image_copied_ = true;
            }
            ocl4dnnGEMM<Dtype>(0, CblasNoTrans,
                               transpose_ ? CblasNoTrans : CblasTrans,
                               M_, N_, K_, (Dtype) 1.,
                               (cl_mem) bottom_data, 0, (cl_mem) weight_image_, 0,
                               (Dtype) 0., (cl_mem) top_data, 0, false, true);
        } else
            ocl4dnnGEMM<Dtype>(0, CblasNoTrans,
                               transpose_ ? CblasNoTrans : CblasTrans,
                               M_, N_, K_, (Dtype) 1.,
                               (cl_mem) bottom_data, 0, (cl_mem) weight, 0,
                               (Dtype) 0., (cl_mem) top_data, 0, false, false);

        if (bias_term_)
            ocl4dnnGEMM<Dtype>(0, CblasNoTrans,
                               CblasNoTrans, M_, N_, 1, (Dtype) 1.,
                               (cl_mem) bias_multiplier_, 0,
                               (cl_mem) bias, 0,
                               (Dtype) 1., (cl_mem) top_data, 0, false, false);
        */
    }
    return true;
}

template class OCL4DNNInnerProduct<float>;
} // namespace ocl4dnn
}
}
#endif // HAVE_OPENCL
