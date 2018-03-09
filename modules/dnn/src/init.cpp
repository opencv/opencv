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
#include <opencv2/dnn/layer.details.hpp>

#include <google/protobuf/stubs/common.h>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

static Mutex* __initialization_mutex = NULL;
Mutex& getInitializationMutex()
{
    if (__initialization_mutex == NULL)
        __initialization_mutex = new Mutex();
    return *__initialization_mutex;
}
// force initialization (single-threaded environment)
Mutex* __initialization_mutex_initializer = &getInitializationMutex();

namespace {
using namespace google::protobuf;
class ProtobufShutdown {
public:
    bool initialized;
    ProtobufShutdown() : initialized(true) {}
    ~ProtobufShutdown()
    {
        initialized = false;
        google::protobuf::ShutdownProtobufLibrary();
    }
};
} // namespace

void initializeLayerFactory()
{
    CV_TRACE_FUNCTION();

    static ProtobufShutdown protobufShutdown; (void)protobufShutdown;

    CV_DNN_REGISTER_LAYER_CLASS(Slice,          SliceLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Split,          SplitLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Concat,         ConcatLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reshape,        ReshapeLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Flatten,        FlattenLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ResizeNearestNeighbor, ResizeNearestNeighborLayer);

    CV_DNN_REGISTER_LAYER_CLASS(Convolution,    ConvolutionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Deconvolution,  DeconvolutionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Pooling,        PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ROIPooling,     PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PSROIPooling,   PoolingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(LRN,            LRNLayer);
    CV_DNN_REGISTER_LAYER_CLASS(InnerProduct,   InnerProductLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Softmax,        SoftmaxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(MVN,            MVNLayer);

    CV_DNN_REGISTER_LAYER_CLASS(ReLU,           ReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(ReLU6,          ReLU6Layer);
    CV_DNN_REGISTER_LAYER_CLASS(ChannelsPReLU,  ChannelsPReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(PReLU,          ChannelsPReLULayer);
    CV_DNN_REGISTER_LAYER_CLASS(Sigmoid,        SigmoidLayer);
    CV_DNN_REGISTER_LAYER_CLASS(TanH,           TanHLayer);
    CV_DNN_REGISTER_LAYER_CLASS(ELU,            ELULayer);
    CV_DNN_REGISTER_LAYER_CLASS(BNLL,           BNLLLayer);
    CV_DNN_REGISTER_LAYER_CLASS(AbsVal,         AbsLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Power,          PowerLayer);
    CV_DNN_REGISTER_LAYER_CLASS(BatchNorm,      BatchNormLayer);
    CV_DNN_REGISTER_LAYER_CLASS(MaxUnpool,      MaxUnpoolLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Dropout,        BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Identity,       BlankLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Silence,        BlankLayer);

    CV_DNN_REGISTER_LAYER_CLASS(Crop,           CropLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Eltwise,        EltwiseLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Permute,        PermuteLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PriorBox,       PriorBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(PriorBoxClustered, PriorBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Reorg,          ReorgLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Region,         RegionLayer);
    CV_DNN_REGISTER_LAYER_CLASS(DetectionOutput, DetectionOutputLayer);
    CV_DNN_REGISTER_LAYER_CLASS(NormalizeBBox,  NormalizeBBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Normalize,      NormalizeBBoxLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Shift,          ShiftLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Padding,        PaddingLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Proposal,       ProposalLayer);
    CV_DNN_REGISTER_LAYER_CLASS(Scale,          ScaleLayer);

    CV_DNN_REGISTER_LAYER_CLASS(LSTM,           LSTMLayer);
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
