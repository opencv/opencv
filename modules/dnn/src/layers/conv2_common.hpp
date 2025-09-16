// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_LAYERS_CONV2_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_CONV2_COMMON_HPP__

#include <opencv2/dnn.hpp>
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
    FAST_ACTIV_CLIP
};

typedef void (*activation_func_t)(const void* input, void* output,
                                  size_t len, const float* params);

struct ConvState
{
    enum { MAX_CONV_DIMS = 3 };
    int ngroups, nspatialdims;
    int kshape[MAX_CONV_DIMS];
    int strides[MAX_CONV_DIMS];
    int dilations[MAX_CONV_DIMS];
    int pads[MAX_CONV_DIMS*2];
    MatShape inpshape, outshape;
    int inner[MAX_CONV_DIMS*2];
    std::vector<int> coordtab;
    std::vector<int> ofstab;

    FastActivation fastActivation;
    enum {ACTIV_MAX_PARAMS = 16};
    float activParams[ACTIV_MAX_PARAMS];
    activation_func_t activation;

    std::ostream& dump(std::ostream& strm);
    bool sameShape(const ConvState& cs) const;
};

AutoPadding getAutoPadding(const LayerParams& params);

void initConvState(const MatShape& inpshape,
                   const MatShape& wshape,
                   const MatShape& outshape,
                   const Ptr<Layer>& activ,
                   int ngroups,
                   const std::vector<int>& strides,
                   const std::vector<int>& dilations,
                   const std::vector<int>& pads,
                   AutoPadding auto_pad, bool ceil_mode,
                   ConvState& cs);

// initializes the structure of parameters for 1D/2D/3D
// depth-wise convolution, max pooling or average pooling
void initPoolingState(const MatShape& inpshape, const MatShape& outshape,
                      const std::vector<int>& kernel_shape,
                      const std::vector<int>& strides,
                      const std::vector<int>& dilations,
                      const std::vector<int>& pads,
                      AutoPadding auto_pad, bool ceil_mode,
                      ConvState& cs);

typedef void (*ConvFunc)(const void* inp, const void* residual, void* out,
                         const ConvState& cs, const void* weights,
                         const float* scale, const float* bias,
                         const int32_t* ofs, const int32_t* ofsofs);

typedef void (*DepthwiseConvFunc)(const void* inp, const void* residual,
                                  void* out, const ConvState& cs,
                                  const void* weights,
                                  const float* scale,
                                  const float* bias);

DepthwiseConvFunc getDepthwiseConvFunc(int depth);
ConvFunc getConvFunc(int depth);

void repackDepthwiseConvWeights(const void* inpw, int inptype,
                                void* outw, int outtype,
                                const MatShape& wsize, int C0);

CV__DNN_INLINE_NS_END

}
}

#endif
