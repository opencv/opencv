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
#include "../include/common.hpp"
#include "../include/ocl4dnn.hpp"
#include "../include/math_functions.hpp"

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
    use_half_ = config.use_half;
}

template<typename Dtype>
OCL4DNNInnerProduct<Dtype>::~OCL4DNNInnerProduct()
{
}

template<typename Dtype>
bool OCL4DNNInnerProduct<Dtype>::Forward(const UMat& bottom,
                                         const UMat& weight,
                                         const UMat& bias,
                                         UMat& top)
{
    bool ret;

    if (M_ == 1)
    {
        ret = ocl4dnnGEMV<Dtype>(CblasNoTrans, N_, K_, (Dtype) 1.,
                                 weight, 0, bottom, 0, (Dtype) 0., top, 0);

        if (bias_term_ && ret)
            ret = ocl4dnnAXPY<Dtype>(N_, 1, bias, 0, top, 0);

        return ret;
    }
    else
    {
        ret = false;
        size_t max_image_size = std::min(ocl::Device::getDefault().image2DMaxWidth(),
                                         ocl::Device::getDefault().image2DMaxHeight());
        if (M_ <= max_image_size &&
            N_ <= max_image_size &&
            K_ <= max_image_size &&
            ocl::Device::getDefault().intelSubgroupsSupport())
        {
            ret = ocl4dnnGEMMCommon<Dtype>(transpose_ ? CblasNoTrans : CblasTrans,
                                           M_, N_, K_, bottom, weight, UMat(), top,
                                           max_image_size);
        }

        if (use_half_ && bias_term_)
        {
            UMat biasOneMat = UMat::ones(M_, 1, CV_32F);
            UMat newbias, tmpTop;

            convertFp16(bias, newbias);
            convertFp16(top, tmpTop);
            cv::gemm(biasOneMat, newbias, 1, tmpTop, 1, tmpTop, 0);
            convertFp16(tmpTop, top);
        }

        return ret;
    }
}

template class OCL4DNNInnerProduct<float>;

}}} // namespace cv::dnn::ocl4dnn
