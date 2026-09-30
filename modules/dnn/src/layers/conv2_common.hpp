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

    // The 1x1 fast paths map output coordinates straight onto input ones, which only holds without padding.
    bool hasPadding() const
    {
        for (int i = 0; i < MAX_CONV_DIMS; i++)
            if (pads[i] != 0 || pads[i + MAX_CONV_DIMS] != 0)
                return true;
        return false;
    }

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

    void initOfs();

    void initDeconv(const MatShape& inpShape,
                    const MatShape& wshape,
                    const MatShape& outShape,
                    int ngroups,
                    const std::vector<int>& strides,
                    const std::vector<int>& dilations,
                    const std::vector<int>& pads);
};

AutoPadding getAutoPadding(const LayerParams& params);

typedef void (*ConvFunc)(const void* inp, const void* residual, void* out,
                         const ConvState& cs, const void* weights,
                         const float* scale, const float* bias);

ConvFunc getConvFunc(int depth, int C0);
ConvFunc getDepthwiseConvFunc(int depth);

/** @brief Where a weight lands in the (ngroups, Kblk, ksize, C1Max, C0*K0) packed buffer.

repackConvWeights() writes through this and anything editing packed weights afterwards reads
through it, so the layout has one definition. Hands back element indices rather than typed
pointers, since the same layout carries int8. Integer math only: this header is compiled at
several CPU baselines.
*/
struct ConvWeightPack
{
    int Kg, Cg, K0, Kblk, ksize, C1Max, C0;

    //! @p wshape0 is the plain (K, Cg, spatial...) filter. ConvTranspose swaps that
    //! convention, so it needs its own factory rather than this one.
    static ConvWeightPack forConv(const MatShape& wshape0, const MatShape& wpackShape,
                                  int ngroups, int C0)
    {
        ConvWeightPack p;
        p.Kg = wshape0[0]/ngroups;
        p.Cg = wshape0[1];
        p.K0 = p.C0 = C0;
        p.Kblk = wpackShape[1];
        p.ksize = wpackShape[2];
        p.C1Max = wpackShape[3];
        return p;
    }

    size_t tapStride() const { return (size_t)C1Max*C0*K0; }

    //! Output channel @p k, input channel @p c within its group. Tap i adds i*tapStride().
    size_t offset(int k, int c) const
    {
        int g = k/Kg, kin = k - g*Kg;
        int kblk = kin/K0, k0 = kin & (K0 - 1);
        int ch = ((g*Cg) & (C0 - 1)) + c;
        return ((((size_t)(g*Kblk + kblk)*ksize)*C1Max + ch/C0)*C0 + (ch & (C0 - 1)))*K0 + k0;
    }
};

void repackDepthwiseConvWeights(const Mat& weights, Mat& Wpack, int outtype, int C0);
void repackConvWeights(const Mat& weights, Mat& Wpack, int outtype, int ngroups, int C0);

MatShape deconvInferShape(const MatShape& inpShape, const MatShape& wshape,
                          const std::vector<int>& kernelShape, int ngroups,
                          const std::vector<int>& strides,
                          const std::vector<int>& dilations,
                          const std::vector<int>& pads,
                          const std::vector<int>& adjustPads,
                          AutoPadding autoPad);

typedef void (*DeconvFunc)(const void* inp, const void* residual, void* out,
                           const ConvState& cs, const void* weights,
                           const float* scale, const float* bias);

DeconvFunc getDeconvFunc(int depth);
void repackDeconvWeights(const Mat& weights, Mat& Wpack, int outtype, int ngroups, int C0);

CV__DNN_INLINE_NS_END
}
}

#endif
