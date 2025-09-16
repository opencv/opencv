// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace dnn
{

/*
    Convolution layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Conv.html

    Opset's 1 to 22 are covered.
*/

static void initConv2DTables(const ConvState& cs,
                             std::vector<int32_t>& ofsbuf_,
                             std::vector<int32_t>& ofsofs_)
{
    int Hk_ = cs.kshape[0], Wk_ = cs.kshape[1];
    int DY_ = cs.dilations[0], DX_ = cs.dilations[1];
    int pad_y0 = cs.pads[0], pad_x0 = cs.pads[1];
    int Hi_ = cs.inpshape[2], Wi_ = cs.inpshape[3], H = cs.outshape[2], W = cs.outshape[3];
    int C0_ = cs.inpshape.back(), C1_ = cs.inpshape[1];
    int ngroups = cs.ngroups, C1g = C1_/ngroups;
    int inner_y0 = cs.inner[0], inner_y1 = cs.inner[1];
    int inner_x0 = cs.inner[cs.nspatialdims], inner_x1 = cs.inner[cs.nspatialdims + 1];

    ofsofs_.resize(H*W*2);

    int ofs_blocksize = C1g*Hk_*Wk_;
    bool have_inner = inner_y0 < inner_y1 && inner_x0 < inner_x1;

    int nblocks = have_inner ? 1 + (inner_y0 + (H - inner_y1))*W +
        (inner_y1 - inner_y0)*(inner_x0 + W - inner_x1) : W*H;

    ofsbuf_.resize(ofs_blocksize*nblocks);
    int32_t* ofsbuf = ofsbuf_.data();

    if (have_inner) {
        for (int c = 0, k = 0; c < C1g; c++) {
            for (int dy = 0; dy < Hk_; dy++) {
                int yi = dy*DY_;
                for (int dx = 0; dx < Wk_; dx++, k++) {
                    int xi = dx*DX_;
                    ofsbuf[k] = (int32_t)(((c*Hi_ + yi)*Wi_ + xi)*C0_);
                }
            }
        }
    }

    parallel_for_(Range(0, H), [&](const Range& r) {
        int C0 = C0_;
        int Hk = Hk_, Wk = Wk_;
        int Hi = Hi_, Wi = Wi_;
        int SY = cs.strides[0], SX = cs.strides[1];
        int DY = cs.dilations[0], DX = cs.dilations[1];
        uint8_t* mask = mask_.data();
        int32_t* ofs0 = ofs0_.data();
        int32_t** ofsptrs = ofsptrs_.data();
        int64_t curr_block = 1;
        if (have_inner) {
            curr_block += std::min(r.start, inner_y0)*W;
            curr_block += std::min(std::max(r.start - inner_y0, 0),
                                   inner_y1 - inner_y0)*(inner_x0 + W - inner_x1);
            curr_block += std::max(r.start - inner_y1, 0)*W;
        } else {
            curr_block = r.start*W;
        }
        for (int y0 = r.start; y0 < r.end; y0++) {
            int yi_ = y0*SY - pad_y0;
            bool y_inside = inner_y0 <= y0 && y0 < inner_y1;

            for (int x0 = 0; x0 < W; x0++) {
                int xi_ = x0*SX - pad_x0;
                bool x_inside = inner_x0 <= x0 && x0 < inner_x1;
                uint8_t m = (uint8_t)(y_inside & x_inside);

                mask[y0*W + x0] = m;

                if (m) {
                    ofs0[y0*W + x0] = (int32_t)((yi_*Wi + xi_)*C0);
                    ofsptrs[y0*W + x0] = ofsbuf;
                } else {
                    ofs0[y0*W + x0] = 0;
                    int32_t* ofsptr = ofsbuf + curr_block*ofs_blocksize;
                    ofsptrs[y0*W + x0] = ofsptr;
                    curr_block++;

                    for (int c = 0, k = 0; c < C1g; c++) {
                        for (int dy = 0; dy < Hk; dy++) {
                            int yi = yi_ + dy*DY;
                            bool yi_inside = 0 <= yi && yi < Hi;

                            for (int dx = 0; dx < Wk; dx++, k++) {
                                int xi = xi_ + dx*DX;
                                bool xi_inside = 0 <= xi && xi < Wi;
                                ofsptr[k] = (yi_inside & xi_inside) ?
                                    (int32_t)(((c*Hi + yi)*Wi + xi)*C0) : INT_MIN/2;
                            }
                        }
                    }
                }
            }
        }
    });
}

template<typename _InpT, typename _OutT> void
repackConv2DWeights_(const _InpT* inpw_, _OutT* outw_,
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


// K x (C/ngroups) x Hk x Wk => K1 x C1/ngroups x Hk x Wk x C0 x K0,
// where K0 == C0
static void repackConv2DWeights(const void* inpw__, int inptype_,
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
                    repackConv2DWeights_((const float*)inpw_, (float*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_32F && outtype == CV_16F)
                    repackConv2DWeights_((const float*)inpw_, (hfloat*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_32F)
                    repackConv2DWeights_((const hfloat*)inpw_, (float*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_16F)
                    repackConv2DWeights_((const hfloat*)inpw_, (hfloat*)outw_, inp_step_c,
                                         inp_step_k, ksize, C0, K0, curr_C0, curr_K0);
                else break;
            }
        }
    });
}

static void conv2d_32f(const void* inp__, const void* residual__, void* out__,
                       const ConvState& cs, const void* weights__,
                       const float* scale__, const float* bias__,
                       const int32_t* ofs0__, const int32_t** ofsptrs__,
                       const uint8_t* mask__)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);

    int NK1 = cs.outshape[0]*cs.outshape[1];

    parallel_for_(Range(0, NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        const int32_t* ofs0_ = ofs0__;
        const int32_t** ofsptrs_ = ofsptrs__;
        constexpr int BLOCK_SIZE = 10;
        int nk0 = r.start, nk1 = r.end;
        int C0 = C0_, K0 = C0;
        int Hi = cs.inpshape[2], Wi = cs.inpshape[3];
        int H = cs.outshape[2], W = cs.outshape[3];
        int iplanesize = Hi*Wi;
        int planesize = H*W;
        int Hk = cs.kshape[0], Wk = cs.kshape[1];
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*Hk*Wk*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        const int32_t* ofsptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int nk = nk0; nk < nk1; nk++) {
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            for (int xy0 = 0; xy0 < planesize; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                            resptr += (resptr ? K0*BLOCK_SIZE : 0)) {
                int j = 0, blocksize = std::min(planesize - xy0, BLOCK_SIZE);

                for (; j < blocksize; j++) {
                    inptrs[j] = inp0 + ofs0_[xy0 + j];
                    ofsptrs[j] = ofsptrs_[xy0 + j];
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    const int32_t* last_ofsptr = ofsptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++) {
                        inptrs[j] = last_inptr;
                        ofsptrs[j] = last_ofsptr;
                    }
                }

                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int32_t ofs_ij = ofsptrs[j][i];
                        const float* x = &inptrs[j][std::max(ofs_ij, 0)];
                        float mij = (float)(ofs_ij >= 0);
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0]*mij;
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}

static void conv2d_1x1_32f(const void* inp__, const void* residual__, void* out__,
                           const ConvState& cs, const void* weights__,
                           const float* scale__, const float* bias__,
                           const int32_t*, const int32_t**, const uint8_t*)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);
    CV_Assert(cs.kshape[0] == 1 && cs.kshape[1] == 1);
    CV_Assert(cs.outshape.back() == cs.inpshape.back());
    CV_Assert(cs.pads[0] == 0 && cs.pads[1] == 0 &&
              cs.pads[cs.nspatialdims] == 0 && cs.pads[cs.nspatialdims+1] == 0);

    int NK1 = cs.outshape[0]*cs.outshape[1];

    parallel_for_(Range(0, NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        constexpr int BLOCK_SIZE = 10;
        int nk0 = r.start, nk1 = r.end;
        //int nlanes = nlanes_;
        int C0 = C0_, K0 = C0;
        int Hi = cs.inpshape[2], Wi = cs.inpshape[3];
        int H0 = cs.outshape[2], W0 = cs.outshape[3];
        int iplanesize = Hi*Wi;
        int planesize = H0*W0;
        int SY = cs.strides[0], SX = cs.strides[1];
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;
        bool S1 = SY == 1 && SX == 1;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int nk = nk0; nk < nk1; nk++) {
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            int yiWi = 0, xi = 0;
            for (int xy0 = 0; xy0 < W0*H0; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                               resptr += (resptr ? K0*BLOCK_SIZE : 0))
            {
                int j = 0, blocksize = std::min(W0*H0 - xy0, BLOCK_SIZE);

                if (S1) {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (xy0 + j)*C0;
                    }
                } else {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (yiWi + xi)*C0;
                        if ((xi += SX) >= Wi) {
                            yiWi += Wi*SY;
                            xi = 0;
                        }
                    }
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++)
                        inptrs[j] = last_inptr;
                }

                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    int ofs_ij = i*iplanesize*C0;
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        const float* x = &inptrs[j][ofs_ij];
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0];
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}


typedef void (*conv2d_func_t)(const void* inp, const void* residual, void* out,
                              const ConvState& cs, const void* weights,
                              const float* scale, const float* bias,
                              const int32_t* ofs0, const int32_t** ofsptrs,
                              const uint8_t* mask);

class Conv2LayerImpl : public Conv2Layer
{
public:
    Conv2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        ceil_mode = params.get<bool>("ceil_mode", false);
        strides = params.getVector<int>("strides");
        dilations = params.getVector<int>("dilations");
        pads = params.getVector<int>("pads");
        ngroups = params.get<int>("group", 1);
        fused_batch_norm = false;
        add_residual = false;
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "ngroups: " << ngroups << ",\n";

        /*prindent(strm, indent);
        strm << "ksizes: [";
        for (int k = 0; k < wshape0.ndims; k++)
            strm << (k > 0 ? ", " : "") << wshape0.size[k];
        strm << "],\n";*/

        prindent(strm, indent);
        strm << "dilations: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pads: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << pads[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "strides: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << strides[k];
        strm << "],\n";

        if (fused_batch_norm) {
            prindent(strm, indent);
            strm << "batch_norm: true,\n";
        }

        if (add_residual) {
            prindent(strm, indent);
            strm << "add_residual: true,\n";
        }

        if (activ) {
            prindent(strm, indent);
            strm << "activation: " << activ->name << ",\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void setWeights(InputArray weights_arr, InputArray bias_arr,
                            int C0, int accuracy) CV_OVERRIDE
    {
        Mat weights_ = weights_arr.getMat();
        Mat bias_ = bias_arr.getMat();
        CV_Assert(!weights_.empty());
        int wtype0 = weights_.type();
        CV_Assert(wtype0 == CV_32F || wtype0 == CV_16F || wtype0 == CV_16BF);
        CV_Assert(accuracy == -1 || accuracy == CV_32F);
        int wtype = accuracy < 0 ? CV_32F : accuracy;

        wshape0 = weights_.shape();
        MatShape wshape1 = wshape0;
        bool depthwise = ngroups == wshape0[0] && wshape0[1] == 1;

        if (depthwise) {
            wshape1.layout = DATA_LAYOUT_BLOCK;
            wshape1.C = wshape1[0];
            wshape1[0] = (wshape1[0] + C0 - 1)/C0;
            for (int i = 2; i < wshape1.dims; i++)
                wshape1[i-1] = wshape1[i];
            wshape1[wshape1.dims-1] = C0;
            weights.fit(wshape1, wtype);

            repackDepthwiseConvWeights(weights_.data, wtype0, weights.data, wtype, wshape0, C0);
        } else {
            wshape1.dims += 2;
            wshape1[wshape1.dims-1] = wshape1[wshape1.dims-2] = C0;
            wshape1[0] = (wshape1[0] + C0 - 1)/C0;
            wshape1[1] = (wshape1[1] + C0 - 1)/C0;
            weights.fit(wshape1, wtype);

            repackConv2DWeights(weights_.data, wtype0, weights.data, wtype, wshape0, C0);
        }

        if (!bias_.empty()) {
            CV_Assert(bias_.isContinuous() && bias_.total() == wshape0[0]);
            bias_.convertTo(bias, CV_32F);
        }
    }

    void fuseBatchNormWeights(const BatchNorm2Layer* bn)
    {
        Mat bn_scale, bn_bias;
        bn->getScaleBias(bn_scale, bn_bias);

        CV_Assert(bn_scale.isContinuous() && bn_bias.isContinuous());
        CV_Assert(bn_scale.type() == CV_32F && bn_bias.type() == CV_32F);
        CV_Assert(bn_scale.total() == bn_bias.total());
        int K = (int)bn_scale.total();
        CV_Assert(bias.empty() || (bias.type() == CV_32F && bias.total() == (size_t)K));
        const float* bias_data = bias.data ? bias.ptr<float>() : nullptr;

        fused_scale.fit(1, &K, CV_32F);
        fused_bias.fit(1, &K, CV_32F);

        const float* bn_scale_data = bn_scale.ptr<float>();
        const float* bn_bias_data = bn_bias.ptr<float>();
        float* fused_scale_data = fused_scale.ptr<float>();
        float* fused_bias_data = fused_bias.ptr<float>();

        // (sum(x*w) + bias)*bn_scale + bn_bias => sum(x*w)*fused_scale + fused_bias,
        // where fused_scale = bn_scale and fused_bias = bias*bn_scale + bn_bias.
        for (size_t i = 0; i < K; i++) {
            fused_scale_data[i] = bn_scale_data[i];
            fused_bias_data[i] = (bias_data ? bn_scale_data[i]*bias_data[i] : 0.f) + bn_bias_data[i];
        }
    }

    virtual bool fuseBatchNorm(const Ptr<Layer>& bnlayer) override
    {
        BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(bnlayer.get());
        if (fused_batch_norm || !bn || bn->inputs.size() > 1)
            return false;
        fuseBatchNormWeights(bn);
        fused_batch_norm = true;
        return true;
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1 || fused_batch_norm || add_residual)
            return false;
        Mat new_bias = arr.getMat();
        CV_Assert(new_bias.isContinuous() && new_bias.dims == 1);
        if (new_bias.type() != CV_32F) {
            Mat temp;
            new_bias.convertTo(temp, CV_32F);
            new_bias = temp;
            CV_Assert(new_bias.type() == CV_32F);
        }
        if (!bias.empty()) {
            CV_Assert(bias.shape() == new_bias.shape());
            add(bias, new_bias, bias);
        } else {
            new_bias.copyTo(bias);
        }
        return true;
    }

    /*virtual bool fuseActivation(const Ptr<layer>& activ) override
    {
        ElemwiseOp* activ_ptr = dynamic_cast<ElemwiseOp*>(op.get());
        if (activ || activ_ptr->maxNumInputs() != 1 || !activ_ptr || !activ_ptr->getActivation(CV_32F))
            return false;
        activ = op;
        return true;
    }*/

    virtual int64_t getFLOPS(const std::vector<MatShape>& inputs,
                             const std::vector<MatShape>& outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        MatShape inpshape = inputs[0], wshape = inputs.size() > 1 ? inputs[1] : wshape0;
        int C = inpshape[1]*inpshape.back();
        size_t ksize = wshape.total();
        return (int64_t)((inputs[0].total()/C)*ksize/ngroups);
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs == 1);

        outtypes.assign(1, inferType(inptypes[0]));
        temptypes.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, convInferShape(inpshapes[0], wshape, empty_kernel_shape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode));
        tempshapes.clear();
        return true;
    }

    void getLayouts(const std::vector<DataLayout>& actualInputs,
                    std::vector<DataLayout>& desiredInputs,
                    const int requiredOutputs,
                    std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u && requiredOutputs == 1u);
        desiredInputs = actualInputs;
        desiredInputs[0] = DATA_LAYOUT_BLOCK;
        for (size_t i = 1; i < ninputs; i++)
            desiredInputs[i] = DATA_LAYOUT_UNKNOWN;
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        int ninputs = (int)inputs_arr.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = inputs_arr.getMat(0);
        Mat residual;
        const void* resptr = nullptr;
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        if (add_residual) {
            residual = inputs_arr.getMat(ninputs-1);
            resptr = residual.data;
            ninputs--;
        }

        bool dynamic_weights = ninputs > 1;
        if (dynamic_weights) {
            setWeights(inputs_arr.getMat(1), ninputs > 2 ? inputs_arr.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = convInferShape(inpshape, wshape0, empty_kernel_shape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = inferType(inptype);
        int outkind = outputs_arr.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

        if (add_residual) {
            CV_Assert(outshape == residual.shape());
            CV_Assert(outtype == residual.type());
        }

        int nspatialdims = inpshape.dims - 3;
        CV_Assert(wshape0.dims == nspatialdims+2);

        initConvState(inpshape, wshape0, outshape, activ, ngroups,
                      strides, dilations, pads, auto_pad, ceil_mode, cs);
        bool conv1x1 = cs.kshape[0]*cs.kshape[1]*cs.kshape[2] == 1;
        bool depthwise = ngroups == cs.inpshape.C;

        const float* scale_data = nullptr;
        const float* bias_data = bias.ptr<float>();

        if (fused_batch_norm) {
            scale_data = fused_scale.ptr<float>();
            bias_data = fused_bias.ptr<float>();
        }

        std::vector<Mat>* outs = nullptr;
        std::vector<UMat>* uouts = nullptr;
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            outs = &outputs_arr.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &outputs_arr.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        const void* inptr = inp.data;
        void* outptr = out.data;
        const void* wptr = weights.data;

        if (depthwise) {
            DepthwiseConvFunc func = getDepthwiseConvFunc(inptype);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data);
        } else {
            if (!conv1x1 && (ofs0.empty() || !cs.sameShape(prev_cs))) {
                initConv2DTables(cs, ofsbuf, ofsofs);
                prev_cs = cs;
            }

            ConvFunc func = getConvFunc(inptype);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data,
                 ofsbuf.data(), ofsofs.data());
        }

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamic_weights) {
            // to keep memory footprint low in the case of
            // very rare situation of dynamic convolution weights,
            // we release temporarily allocated and reordered copy of the weights
            weights.release();
        }
    }

    std::vector<int> empty_kernel_shape;
    Ptr<Layer> activ, batchNorm;
    Mat weights, bias, fused_scale, fused_bias;
    MatShape wshape0;
    ConvState cs, prev_cs;
    std::vector<int32_t> ofsbuf;
    std::vector<int32_t> ofs0;
    std::vector<int32_t> ofsofs;
    std::vector<uint8_t> mask;
    bool fused_batch_norm;
};

Ptr<Conv2Layer> Conv2Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Layer>(new Conv2LayerImpl(params));
}

}}
