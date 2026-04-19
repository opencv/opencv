// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include <math.h>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

std::string fastActivationToString(FastActivation fastActivation)
{
    switch (fastActivation) {
    case FAST_ACTIV_NONE: return "None";
    case FAST_ACTIV_RELU: return "ReLU";
    case FAST_ACTIV_LEAKY_RELU: return "LeakyReLU";
    case FAST_ACTIV_PRELU: return "PReLU";
    case FAST_ACTIV_CLIP: return "Clip";
    default: return format("unknown(%d)", int(fastActivation));
    }
}

AutoPadding getAutoPadding(const LayerParams& params)
{
    std::string auto_pad = params.get<std::string>("auto_pad", "NOTSET");
    std::string pad_mode = params.get<std::string>("pad_mode", "");
    if (pad_mode == "SAME")
        return AUTO_PAD_SAME_UPPER;
    if (pad_mode == "VALID")
        return AUTO_PAD_VALID;
    if (auto_pad == "NOTSET")
        return AUTO_PAD_NONE;
    if (auto_pad == "SAME_UPPER")
        return AUTO_PAD_SAME_UPPER;
    if (auto_pad == "SAME_LOWER")
        return AUTO_PAD_SAME_LOWER;
    if (auto_pad != "VALID") {
        CV_Error_(Error::StsBadArg, ("invalid auto_pad value '%s'", auto_pad.c_str()));
    }
    return AUTO_PAD_VALID;
}

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
MatShape convInferShape(const MatShape& inpShape, const MatShape& wshape,
                        const std::vector<int>& kernelShape, int ngroups,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        AutoPadding autoPad, bool ceilMode)
{
    bool blockLayout = true;
    int ndims = inpShape.dims;
    size_t nspatialdims = (size_t)(ndims - 2 - int(blockLayout));
    MatShape outshape = inpShape;
    int kshape[MatShape::MAX_DIMS];

    if (!kernelShape.empty()) {
        size_t kshape_size = kernelShape.size();
        CV_Assert(kshape_size == nspatialdims || kshape_size == nspatialdims+2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = kernelShape[kshape_size - nspatialdims + i];
    } else {
        CV_Assert(!wshape.empty() && wshape.dims == nspatialdims + 2);
        for (size_t i = 0; i < nspatialdims; i++)
            kshape[i] = wshape[wshape.dims - nspatialdims + i];
    }

    if (ngroups == 0 || wshape.empty()) {
        outshape[1] = inpShape[1];
    } else if (blockLayout) {
        int C0 = inpShape[ndims-1];
        outshape[1] = (wshape[0] + C0 - 1)/C0;
    } else {
        outshape[1] = wshape[0];
    }

    CV_Assert(strides.empty() || strides.size() == nspatialdims);
    CV_Assert(dilations.empty() || dilations.size() == nspatialdims);
    CV_Assert(autoPad == AUTO_PAD_NONE || pads.empty());
    CV_Assert(pads.empty() || pads.size() == nspatialdims*2);

    for (size_t i = 0; i < nspatialdims; i++) {
        int inpsz = inpShape[i+2], k_i = kshape[i];
        int stride = strides.empty() ? 1 : strides[i];
        int dilation = dilations.empty() ? 1 : dilations[i];
        int outsz;
        if (autoPad == AUTO_PAD_NONE || autoPad == AUTO_PAD_VALID) {
            int pad = 0;
            if (!pads.empty()) {
                pad = pads[i] + pads[i + nspatialdims];
            }
            outsz = (inpsz + pad - 1 - dilation * (k_i - 1) + (ceilMode ? stride - 1 : 0)) / stride + 1;
        } else {
            if (ceilMode)
                outsz = (inpsz + stride - 1)/stride;
            else
                outsz = (inpsz - 1)/stride + 1;
        }
        outshape[i + 2] = outsz;
    }

    if (blockLayout) {
        outshape.C = ngroups == 0 || wshape.empty() ? inpShape.C : wshape[0];
    } else {
        outshape.C = 0;
    }

    return outshape;
}

static inline void getPadding(const std::vector<int>& pads,
                              int dim, int nspatialdims, AutoPadding autoPad,
                              int ksize, int& pad0, int& pad1)
{
    CV_Assert(0 <= dim && dim < nspatialdims);

    if (autoPad == AUTO_PAD_NONE || autoPad == AUTO_PAD_VALID) {
        if (!pads.empty()) {
            pad0 = pads[dim];
            pad1 = pads[dim + nspatialdims];
        } else {
            pad0 = pad1 = 0;
        }
    } else {
        CV_Assert(autoPad == AUTO_PAD_SAME_LOWER || autoPad == AUTO_PAD_SAME_UPPER);
        pad0 = pad1 = ksize/2;
        if (pad0*2 == ksize) {
            pad0 -= autoPad == AUTO_PAD_SAME_UPPER;
            pad1 -= autoPad == AUTO_PAD_SAME_LOWER;
        }
    }
}

bool ConvState::sameShape(const ConvState& cs) const
{
    for (int i = 0; i < ConvState::MAX_CONV_DIMS; i++) {
        if (kshape[i] != cs.kshape[i] ||
            strides[i] != cs.strides[i] ||
            dilations[i] != cs.dilations[i] ||
            pads[i] != cs.pads[i] ||
            pads[i + MAX_CONV_DIMS] != cs.pads[i + MAX_CONV_DIMS]) {
            return false;
        }
    }
    return inpshape == cs.inpshape && outshape == cs.outshape;
}

static MatShape getWpackShape(const MatShape& wshape, int ngroups, int C0)
{
    CV_Assert(wshape.dims >= 3);
    int K = wshape[0], Cg = wshape[1];
    int ksize = int(wshape.total())/(K*Cg);
    CV_Assert_N(K % ngroups == 0);
    int Kg = K / ngroups, K0 = C0;
    int Kblk = (Kg + K0 - 1)/K0;
    int C1Max = 0;
    for (int g = 0; g < ngroups; ++g) {
        int c_start = g * Cg;
        int c00 = c_start & (C0 - 1);
        int cblocks = (c00 + Cg + C0 - 1)/C0;
        C1Max = std::max(C1Max, cblocks);
    }
    return MatShape({ngroups, Kblk, ksize, C1Max, C0*K0}, DATA_LAYOUT_UNKNOWN);
}

void ConvState::initConv(const MatShape& inpshape_,
                         const MatShape& wshape_,
                         const MatShape& outshape_,
                         int ngroups_,
                         const std::vector<int>& strides_,
                         const std::vector<int>& dilations_,
                         const std::vector<int>& pads_,
                         AutoPadding autoPad, bool ceilMode,
                         FastActivation fastActivation_,
                         ActivationFunc activationFunc_,
                         const std::vector<float>& activParams_)
{
    nspatialdims = wshape_.dims - 2;
    CV_Assert(0 < nspatialdims && nspatialdims <= ConvState::MAX_CONV_DIMS);
    CV_Assert(strides_.empty() || (strides_.size() == size_t(nspatialdims)));
    CV_Assert(dilations_.empty() || (dilations_.size() == size_t(nspatialdims)));
    CV_Assert(pads_.empty() || (pads_.size() == size_t(nspatialdims*2)));
    CV_Assert(inpshape_.dims == outshape_.dims);
    CV_Assert(inpshape_.dims == nspatialdims + 2 + int(inpshape_.layout == DATA_LAYOUT_BLOCK));
    CV_Assert_N(inpshape_.layout == outshape_.layout,
                inpshape_[0] == outshape_[0]);

    inpshape = inpshape_;
    outshape = outshape_;
    ngroups = ngroups_;

    int C = inpshape.channels();
    int K = outshape.channels();

    depthwise = ngroups == C && ngroups == K;

    if (inpshape.layout == DATA_LAYOUT_BLOCK) {
        CV_Assert(inpshape.back() == outshape.back());
    }

    fastActivation = fastActivation_;
    activation = activationFunc_;
    activParams = activParams_;

    CV_Assert(wshape_[0] > 0 && wshape_[1] > 0);
    for (int i = 0; i < MAX_CONV_DIMS; i++) {
        kshape[i] = strides[i] = dilations[i] = 1;
        pads[i] = pads[i + MAX_CONV_DIMS] = 0;
        inner[i] = 0;
        inner[i + MAX_CONV_DIMS] = 1;
    }

    for (int i = 0; i < nspatialdims; i++) {
        int j = i + (MAX_CONV_DIMS - nspatialdims);
        kshape[j] = wshape_[i+2];
        CV_Assert(kshape[j] > 0);

        strides[j] = strides_.empty() ? 1 : strides_[i];
        dilations[j] = dilations_.empty() ? 1 : dilations_[i];

        CV_Assert(strides[j] > 0);
        CV_Assert(dilations[j] > 0);

        int pad0, pad1;
        getPadding(pads_, i, nspatialdims, autoPad, kshape[j], pad0, pad1);
        CV_Assert_N(pad0 >= 0, pad1 >= 0);
        pads[j] = pad0;
        pads[j + MAX_CONV_DIMS] = pad1;

        int inner0 = (pad0 + strides[j] - 1)/strides[j];
        int inner1 = (inpshape[i+2] - (kshape[j] - 1)*dilations[j] + pad0)/strides[j];
        inner1 += inner1*strides[j] - pad0 + (kshape[j] - 1)*dilations[j] < inpshape[i+2];
        inner1 = std::min(inner1, outshape[i+2]);
        if (inner0 >= inner1) {
            inner0 = inner1 = outshape[i+2];
        }
        inner[j] = inner0;
        inner[j + MAX_CONV_DIMS] = inner1;
    }

    initOfs();
    if (!depthwise && inpshape.layout == DATA_LAYOUT_BLOCK) {
        int C0 = inpshape.back();
        wshape = getWpackShape(wshape_, ngroups, C0);

        CV_Assert(wshape.dims == 5);
    }
}

void repackConvWeights(const Mat& weights, Mat& Wpack, int outtype, int ngroups, int C0_)
{
    CV_Assert(weights.isContinuous());
    CV_Assert_N(weights.type() == CV_32F, outtype == CV_32F);
    CV_Assert(ngroups > 0);
    CV_Assert((C0_ & (C0_ - 1)) == 0 && C0_ >= 4);

    MatShape wshape = weights.shape();
    CV_Assert(wshape.dims >= 3);

    int K = wshape[0];
    CV_Assert(K % ngroups == 0);

    if (!Wpack.isContinuous()) {
        Wpack.release();
    }
    MatShape wpackShape = getWpackShape(weights.shape(), ngroups, C0_);
    Wpack.create(wpackShape, CV_32F);
    Wpack.setZero();

    parallel_for_(Range(0, K), [&](const Range& range) {
        int Cg = wshape[1], Kg = K / ngroups;
        int ksize = wpackShape[2], Kblk = wpackShape[1], C1Max = wpackShape[3];
        int C0 = C0_, K0 = C0;
        const float* wdata = weights.ptr<float>();
        float* Wpackdata = Wpack.ptr<float>();

        for (int k = range.start; k < range.end; ++k) {
            int g = k / Kg;
            int kin = k - g * Kg;
            int kblk = kin / K0;
            int k0   = kin & (K0 - 1);

            int c_start = g * Cg;
            int c00 = c_start & (C0 - 1);

            for (int c = 0; c < Cg; ++c) {
                int ch = c00 + c;
                int c1  = ch / C0;
                int c0  = ch & (C0 - 1);

                const float* wptr = wdata + ((k * Cg + c) * ksize);
                float* wpackptr = Wpackdata + (((g * Kblk + kblk) * ksize * C1Max + c1) * C0 + c0)*K0 + k0;
                for (int i = 0; i < ksize; ++i) {
                    wpackptr[i*(C1Max*C0*K0)] = wptr[i];
                }
            }
        }
    });
}


void ConvState::initPooling(const MatShape& inpshape_,
                            const MatShape& outshape_,
                            const std::vector<int>& kshape_,
                            const std::vector<int>& strides_,
                            const std::vector<int>& dilations_,
                            const std::vector<int>& pads_,
                            AutoPadding autoPad, bool ceilMode)
{
    nspatialdims = int(kshape_.size());
    CV_Assert(0 < nspatialdims && nspatialdims <= ConvState::MAX_CONV_DIMS);
    CV_Assert(strides_.empty() || (strides_.size() == size_t(nspatialdims)));
    CV_Assert(dilations_.empty() || (dilations_.size() == size_t(nspatialdims)));
    CV_Assert(pads_.empty() || (pads_.size() == size_t(nspatialdims*2)));
    CV_Assert(inpshape_.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(inpshape_.dims == nspatialdims + 3);

    int C = inpshape_.C;
    inpshape = inpshape_;
    outshape = outshape_;
    ngroups = C;
    depthwise = true;

    for (int i = 0; i < MAX_CONV_DIMS; i++) {
        kshape[i] = strides[i] = dilations[i] = 1;
        pads[i] = pads[i + MAX_CONV_DIMS] = 0;
        inner[i] = 0;
        inner[i + MAX_CONV_DIMS] = 1;
    }

    for (int i = 0; i < nspatialdims; i++) {
        int j = i + (MAX_CONV_DIMS - nspatialdims);
        kshape[j] = kshape_[i];
        CV_Assert(kshape[j] > 0);

        strides[j] = strides_.empty() ? 1 : strides_[i];
        dilations[j] = dilations_.empty() ? 1 : dilations_[i];

        CV_Assert(strides[j] > 0);
        CV_Assert(dilations[j] > 0);

        int pad0, pad1;
        getPadding(pads_, i, nspatialdims, autoPad, kshape[j], pad0, pad1);
        CV_Assert_N(pad0 >= 0, pad1 >= 0);
        pads[j] = pad0;
        pads[j + MAX_CONV_DIMS] = pad1;

        int inner0 = (pad0 + strides[j] - 1)/strides[j];
        int inner1 = (inpshape[i+2] - (kshape[j] - 1)*dilations[j] + pad0)/strides[j];
        inner1 += inner1*strides[j] - pad0 + (kshape[j] - 1)*dilations[j] < inpshape[i+2];
        inner1 = std::min(inner1, outshape[i+2]);
        if (inner0 >= inner1) {
            inner0 = inner1 = outshape[i+2];
        }
        inner[j] = inner0;
        inner[j + MAX_CONV_DIMS] = inner1;
    }
    initOfs();
}

void ConvState::initOfs()
{
    CV_Assert(MAX_CONV_DIMS == 3);
    int sdims = nspatialdims;
    int KD = kshape[0], KH = kshape[1], KW = kshape[2];
    int DZ = dilations[0], DY = dilations[1], DX = dilations[2];
    int Hi = sdims > 1 ? inpshape[sdims] : 1;
    int Wi = inpshape[sdims + 1];
    int ksize = KD*KH*KW;
    int C0 = inpshape.back();
    coordtab.resize(ksize*MAX_CONV_DIMS);
    ofstab.resize(ksize);
    for (int z = 0, k = 0; z < KD; z++) {
        int dz = z*DZ;
        for (int y = 0; y < KH; y++) {
            int dy = y*DY;
            for (int x = 0; x < KW; x++, k++) {
                int dx = x*DX;
                coordtab[k*MAX_CONV_DIMS] = dz;
                coordtab[k*MAX_CONV_DIMS + 1] = dy;
                coordtab[k*MAX_CONV_DIMS + 2] = dx;
                ofstab[k] = ((dz*Hi + dy)*Wi + dx)*C0;
            }
        }
    }
}

CV__DNN_INLINE_NS_END
}}
