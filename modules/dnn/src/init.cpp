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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

namespace cv
{
namespace dnn
{

struct AutoInitializer
{
    bool status;

    AutoInitializer() : status(false)
    {
        initModule();
    }
};

static AutoInitializer init;

void initModule()
{
    if (init.status)
        return;

    REG_RUNTIME_LAYER_CLASS(Slice,          SliceLayer);
    REG_RUNTIME_LAYER_CLASS(Split,          SplitLayer);
    REG_RUNTIME_LAYER_CLASS(Concat,         ConcatLayer);
    REG_RUNTIME_LAYER_CLASS(Reshape,        ReshapeLayer);
    REG_RUNTIME_LAYER_CLASS(Flatten,        FlattenLayer);

    REG_RUNTIME_LAYER_CLASS(Convolution,    ConvolutionLayer);
    REG_RUNTIME_LAYER_CLASS(Deconvolution,  DeconvolutionLayer);
    REG_RUNTIME_LAYER_CLASS(Pooling,        PoolingLayer);
    REG_RUNTIME_LAYER_CLASS(LRN,            LRNLayer);
    REG_RUNTIME_LAYER_CLASS(InnerProduct,   InnerProductLayer);
    REG_RUNTIME_LAYER_CLASS(Softmax,        SoftmaxLayer);
    REG_RUNTIME_LAYER_CLASS(MVN,            MVNLayer);

    REG_RUNTIME_LAYER_CLASS(ReLU,           ReLULayer);
    REG_RUNTIME_LAYER_CLASS(ChannelsPReLU,  ChannelsPReLULayer);
    REG_RUNTIME_LAYER_CLASS(Sigmoid,        SigmoidLayer);
    REG_RUNTIME_LAYER_CLASS(TanH,           TanHLayer);
    REG_RUNTIME_LAYER_CLASS(BNLL,           BNLLLayer);
    REG_RUNTIME_LAYER_CLASS(AbsVal,         AbsLayer);
    REG_RUNTIME_LAYER_CLASS(Power,          PowerLayer);
    REG_RUNTIME_LAYER_CLASS(BatchNorm,      BatchNormLayer);
    REG_RUNTIME_LAYER_CLASS(MaxUnpool,      MaxUnpoolLayer);
    REG_RUNTIME_LAYER_CLASS(Dropout,        BlankLayer);
    REG_RUNTIME_LAYER_CLASS(Identity,       BlankLayer);

    REG_RUNTIME_LAYER_CLASS(Crop,           CropLayer);
    REG_RUNTIME_LAYER_CLASS(Eltwise,        EltwiseLayer);
    REG_RUNTIME_LAYER_CLASS(Permute,        PermuteLayer);
    REG_RUNTIME_LAYER_CLASS(PriorBox,       PriorBoxLayer);
    REG_RUNTIME_LAYER_CLASS(DetectionOutput, DetectionOutputLayer);
    REG_RUNTIME_LAYER_CLASS(NormalizeBBox,  NormalizeBBoxLayer);
    REG_RUNTIME_LAYER_CLASS(Normalize,      NormalizeBBoxLayer);
    REG_RUNTIME_LAYER_CLASS(Shift,          ShiftLayer);
    REG_RUNTIME_LAYER_CLASS(Padding,        PaddingLayer);
    REG_RUNTIME_LAYER_CLASS(Scale,          ScaleLayer);

    init.status = true;
}

}
}
