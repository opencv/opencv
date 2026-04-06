// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_LAYERS_CONV2_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_CONV2_COMMON_HPP__

#include <opencv2/dnn/all_layers.hpp>
#include <array>

namespace cv
{
namespace dnn
{
CV__DNN_INLINE_NS_BEGIN

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
MatShape convInferShape(const MatShape& inpshape, const MatShape& wshape,
                        const std::vector<int>& kernel_shape, int ngroups,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        AutoPadding auto_pad, bool ceil_mode);

enum FastActivation {
    FAST_ACTIV_NONE=0,
    FAST_ACTIV_RELU,
    FAST_ACTIV_LEAKY_RELU,
    FAST_ACTIV_PRELU,
    FAST_ACTIV_CLIP
};

std::string fastActivationToString(FastActivation fastActivation);

struct ConvState
{
    enum { MAX_CONV_DIMS = 3 };
    bool depthwise = true;
    int ngroups, nspatialdims;
    int kshape[MAX_CONV_DIMS];
    int strides[MAX_CONV_DIMS];
    int dilations[MAX_CONV_DIMS];
    int pads[MAX_CONV_DIMS*2];
    MatShape inpshape, outshape;
    MatShape wshape; // (ngroups, Kblk, ksize, C1Max, C0*K0) in the case of non-depthwise convolution
    int inner[MAX_CONV_DIMS*2];
    std::vector<int> coordtab;
    std::vector<int> ofstab;

    FastActivation fastActivation = FAST_ACTIV_NONE;
    ActivationFunc activation = nullptr;
    std::vector<float> activParams;

    std::ostream& dump(std::ostream& strm);
    bool sameShape(const ConvState& cs) const;

    void initConv(const MatShape& inpShape,
                  const MatShape& wshape,
                  const MatShape& outShape,
                  int ngroups,
                  const std::vector<int>& strides,
                  const std::vector<int>& dilations,
                  const std::vector<int>& pads,
                  AutoPadding autoPad, bool ceilMode,
                  FastActivation fastActivation,
                  ActivationFunc activationFunc,
                  const std::vector<float>& activParams);

    // initializes the structure of parameters for 1D/2D/3D
    // depth-wise convolution, max pooling or average pooling
    void initPooling(const MatShape& inpshape, const MatShape& outshape,
                     const std::vector<int>& kernel_shape,
                     const std::vector<int>& strides,
                     const std::vector<int>& dilations,
                     const std::vector<int>& pads,
                     AutoPadding auto_pad, bool ceil_mode);

    // internal-use method to initialize coordtab and ofstab.
    // it's called from initConv and initPooling
    void initOfs();
};

AutoPadding getAutoPadding(const LayerParams& params);

typedef void (*ConvFunc)(const void* inp, const void* residual, void* out,
                         const ConvState& cs, const void* weights,
                         const float* scale, const float* bias);

ConvFunc getConvFunc(int depth, int C0);
ConvFunc getDepthwiseConvFunc(int depth);

void repackDepthwiseConvWeights(const Mat& weights, Mat& Wpack, int outtype, int C0);
void repackConvWeights(const Mat& weights, Mat& Wpack, int outtype, int ngroups, int C0);

CV__DNN_INLINE_NS_END
}
}

#endif
