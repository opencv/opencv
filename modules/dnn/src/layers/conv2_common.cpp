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

AutoPadding getAutoPadding(const LayerParams& params)
{
    std::string auto_pad = params.get<std::string>("auto_pad", "NOTSET");
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

void ConvState::initConv(const MatShape& inpshape_,
                         const MatShape& wshape_,
                         const MatShape& outshape_,
                         int ngroups_,
                         const std::vector<int>& strides_,
                         const std::vector<int>& dilations_,
                         const std::vector<int>& pads_,
                         AutoPadding autoPad, bool ceilMode,
                         FastActivation fastActivation_,
                         const float* activParams_,
                         size_t nactivParams_)
{
    nspatialdims = wshape_.dims - 2;
    CV_Assert(0 < nspatialdims && nspatialdims <= ConvState::MAX_CONV_DIMS);
    CV_Assert(strides_.empty() || (strides_.size() == size_t(nspatialdims)));
    CV_Assert(dilations_.empty() || (dilations_.size() == size_t(nspatialdims)));
    CV_Assert(pads_.empty() || (pads_.size() == size_t(nspatialdims*2)));
    CV_Assert(inpshape_.dims == outshape_.dims);
    CV_Assert(nspatialdims == inpshape_.dims - 3);
    CV_Assert(inpshape_[1] % ngroups_ == 0);
    CV_Assert(outshape_[1] % ngroups_ == 0);
    CV_Assert(inpshape_[0] == outshape_[0]);
    CV_Assert(inpshape_.back() == outshape_.back());

    inpshape = inpshape_;
    outshape = outshape_;
    ngroups = ngroups_;

    fastActivation = fastActivation_;
    activation = nullptr;
    memset(activParams, 0, sizeof(activParams));
    if (activParams_ && nactivParams_ > 0) {
        CV_Assert(nactivParams_ <= (size_t)MAX_ACTIV_PARAMS);
        for (size_t i = 0; i < nactivParams_; i++)
            activParams[i] = activParams_[i];
    }

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
        getPadding(pads_, i, nspatialdims, autoPad, kshape[i], pad0, pad1);
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
    
    int C = inpshape.layout == DATA_LAYOUT_BLOCK ? inpshape.C :
        inpshape.layout == DATA_LAYOUT_NHWC ? inpshape.back() : inpshape[1];
    bool depthwise = ngroups == C;
    if (depthwise) {
        initOfs();
    }
}

void initConvTables(const ConvState& cs,
                    std::vector<int32_t>& inpofs_,
                    std::vector<int32_t>& ofsofs_)
{
    int sdims = cs.nspatialdims;
    CV_Assert(sdims + 3 == cs.inpshape.dims);
    int Dk = cs.kshape[0], Hk = cs.kshape[1], Wk = cs.kshape[2];
    int DZ = cs.dilations[0], DY = cs.dilations[1], DX = cs.dilations[2];
    int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
    int pad_z0 = cs.pads[0], pad_y0 = cs.pads[1], pad_x0 = cs.pads[2];
    int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
    int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
    int Wi = cs.inpshape[sdims+1];
    int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
    int H = sdims > 1 ? cs.outshape[sdims] : 1;
    int W = cs.outshape[sdims + 1];
    int C0 = cs.inpshape.back(), C1 = cs.inpshape[1];
    int ngroups = cs.ngroups, C1g = C1/ngroups;
    int inner_z0 = cs.inner[0], inner_y0 = cs.inner[1], inner_x0 = cs.inner[2];
    int inner_z1 = cs.inner[3], inner_y1 = cs.inner[4], inner_x1 = cs.inner[5];

    ofsofs_.resize(D*H*W*2);

    int ofs_blocksize = C1g*Dk*Hk*Wk;
    bool have_inner = inner_z0 < inner_z1 && inner_y0 < inner_y1 && inner_x0 < inner_x1;

    inpofs_.resize(ofs_blocksize);
    
    if (have_inner) {
        for (int c = 0, k = 0; c < C1g; c++) {
            for (int dz = 0; dz < Dk; dz++) {
                int zi = dz*DZ;
                for (int dy = 0; dy < Hk; dy++) {
                    int yi = dy*DY;
                    for (int dx = 0; dx < Wk; dx++, k++) {
                        int xi = dx*DX;
                        int ofs = (((c*Di + zi)*Hi + yi)*Wi + xi)*C0;
                        inpofs_[k] = (int32_t)ofs;
                    }
                }
            }
        }
    }

    int32_t* ofsofs = ofsofs_.data();
    int64_t curr_block = 1;

    bool prev_z_inside = false;
    for (int z0 = 0; z0 < D; z0++) {
        int zi_ = z0*SZ - pad_z0;
        bool z_inside = inner_z0 <= z0 && z0 < inner_z1;
        bool prev_y_inside = false;

        for (int y0 = 0; y0 < H; y0++) {
            int yi_ = y0*SY - pad_y0;
            bool y_inside = inner_y0 <= y0 && y0 < inner_y1;
            bool prev_x_inside = false;

            for (int x0 = 0; x0 < W; x0++) {
                int xi_ = x0*SX - pad_x0;
                bool x_inside = inner_x0 <= x0 && x0 < inner_x1;
                int idx = ((z0*H + y0)*W + x0)*2;

                if (x_inside && y_inside && z_inside) {
                    ofsofs[idx] = (int32_t)(((zi_*Hi + yi_)*Wi + xi_)*C0);
                    ofsofs[idx + 1] = 0;
                } else if (z_inside && prev_z_inside) {
                    ofsofs[idx] = ofsofs[idx - W*H*2] + Wi*Hi*SZ*C0;
                    ofsofs[idx + 1] = ofsofs[idx - W*H*2 + 1];
                } else if (y_inside && prev_y_inside) {
                    ofsofs[idx] = ofsofs[idx - W*2] + Wi*SY*C0;
                    ofsofs[idx + 1] = ofsofs[idx - W*2 + 1];
                } else if (x_inside && prev_x_inside) {
                    ofsofs[idx] = ofsofs[idx - 2] + SX*C0;
                    ofsofs[idx + 1] = ofsofs[idx - 1];
                } else {
                    int64_t curr_ofs = curr_block*ofs_blocksize;
                    ofsofs[idx] = 0;
                    ofsofs[idx + 1] = int(curr_ofs);
                    curr_block++;
                    inpofs_.resize(curr_ofs + ofs_blocksize);
                    int32_t* inpofs = &inpofs_[curr_ofs];

                    int firstofs = -1;
                    for (int c = 0, k = 0; c < C1g; c++) {
                        for (int dz = 0; dz < Dk; dz++) {
                            int zi = zi_ + dz*DZ;
                            bool zi_inside = 0 <= zi && zi < Di;

                            for (int dy = 0; dy < Hk; dy++) {
                                int yi = yi_ + dy*DY;
                                bool yi_inside = 0 <= yi && yi < Hi;

                                for (int dx = 0; dx < Wk; dx++, k++) {
                                    int xi = xi_ + dx*DX;
                                    bool xi_inside = 0 <= xi && xi < Wi;

                                    if (zi_inside && yi_inside && xi_inside) {
                                        int ofs = (((c*Di + zi)*Hi + yi)*Wi + xi)*C0;
                                        if (firstofs < 0)
                                            ofsofs[idx] = firstofs = ofs;
                                        inpofs[k] = (int32_t)(ofs - firstofs);
                                    } else {
                                        inpofs[k] = INT_MIN/2;
                                    }
                                }
                            }
                        }
                    }
                }
                prev_x_inside = x_inside;
            }
            prev_y_inside = y_inside;
        }
        prev_z_inside = z_inside;
    }
}

template<typename _InpT, typename _OutT> void
repackConvWeights_(const _InpT* inpw_, _OutT* outw_,
                   size_t inp_step_c, size_t inp_step_k, int ksize,
                   int C0, int K0, int curr_C0, int curr_K0)
{
    const _InpT* inpw = inpw_;
    _OutT* outw = outw_;
    for (int xy = 0; xy < ksize; xy++, inpw++, outw += C0*K0) {
        for (int c0 = 0; c0 < curr_C0; c0++) {
            for (int k0 = 0; k0 < curr_K0; k0++) {
                outw[c0*K0 + k0] = _OutT(inpw[inp_step_k*k0 + inp_step_c*c0]);
            }
        }
    }
}


// K x (C/ngroups) x Dk x Hk x Wk => K1 x C1/ngroups x Dk x Hk x Wk x C0 x K0,
// where K0 == C0
void repackConvWeights(const void* inpw__, int inptype_,
                       void* outw__, int outtype_,
                       const MatShape& wshape, int C0_)
{
    CV_Assert(inptype_ == CV_32F || inptype_ == CV_16F);
    CV_Assert(outtype_ == CV_32F || outtype_ == CV_16F);

    int K1 = (wshape[0] + C0_ - 1)/C0_;
    parallel_for_(Range(0, K1), [&](const Range& r) {
        int inptype = inptype_, outtype = outtype_;
        size_t inp_esz = CV_ELEM_SIZE(inptype);
        size_t out_esz = CV_ELEM_SIZE(outtype);
        int C0 = C0_, K0 = C0_;
        int K = wshape[0], Cg = wshape[1];
        int C1g = (Cg + C0 - 1)/C0;
        int Hk = wshape[2], Wk = wshape[3];
        int ksize = Hk*Wk;
        size_t inp_step_c = ksize, inp_step_k = Cg*ksize;
        size_t out_microplane_size = ksize*C0*K0*out_esz;

        for (int k1 = r.start; k1 < r.end; k1++) {
            int curr_K0 = std::min(K - k1*K0, K0);
            for (int c1g = 0; c1g < C1g; c1g++) {
                uint8_t* inpw_ = (uint8_t*)inpw__ + (k1*K0*inp_step_k + c1g*C0*inp_step_c)*inp_esz;
                uint8_t* outw_ = (uint8_t*)outw__ + (k1*C1g + c1g)*out_microplane_size;
                int curr_C0 = std::min(Cg - c1g*C0, C0);
                if (curr_K0 != K0 || curr_C0 != C0)
                    memset(outw_, 0, out_microplane_size);

                if (inptype == CV_32F && outtype == CV_32F)
                    repackConvWeights_((const float*)inpw_, (float*)outw_, inp_step_c,
                                        inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_32F && outtype == CV_16F)
                    repackConvWeights_((const float*)inpw_, (hfloat*)outw_, inp_step_c,
                                        inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_32F)
                    repackConvWeights_((const hfloat*)inpw_, (float*)outw_, inp_step_c,
                                        inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_16F)
                    repackConvWeights_((const hfloat*)inpw_, (hfloat*)outw_, inp_step_c,
                                        inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else break;
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
        getPadding(pads_, i, nspatialdims, autoPad, kshape[i], pad0, pad1);
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
