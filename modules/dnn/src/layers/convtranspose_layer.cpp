// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "cpu_kernels/mlas_gemm.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <opencv2/core/utils/configuration.private.hpp>

namespace cv
{
namespace dnn
{

/*
    ONNX Det operator
    Spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    Supported opsets: 1-22
*/

class ConvTranspose2LayerImpl : public ConvTranspose2Layer
{
public:
    ConvTranspose2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        adjust_pads = params.getVector<int>("adj");
        // ONNX output_shape (spatial only); fixes the output, paddings derived from it.
        explicit_out_shape = params.getVector<int>("output_shape_spatial");
        ngroups = params.get<int>("group", 1);
    }

    void applyExplicitOutShape(MatShape& outshape) const
    {
        if (explicit_out_shape.empty())
            return;
        int nsd = outshape.dims - 2 - int(outshape.layout == DATA_LAYOUT_BLOCK);
        CV_CheckEQ((int)explicit_out_shape.size(), nsd, "output_shape must cover all spatial dims");
        for (int i = 0; i < nsd; i++)
            outshape[i + 2] = explicit_out_shape[i];
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "group: " << ngroups << ",\n";

        if (!wshape0.empty()) {
            prindent(strm, indent);
            strm << "ksize: [";
            for (int k = 0; k < wshape0.dims; k++)
                strm << (k > 0 ? ", " : "") << wshape0[k];
            strm << "],\n";
        }

        prindent(strm, indent);
        strm << "stride: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << strides[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "dilation: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pad: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << pads[k];
        strm << "],\n";

        if (!adjust_pads.empty()) {
            prindent(strm, indent);
            strm << "adj: [";
            for (size_t k = 0; k < adjust_pads.size(); k++)
                strm << (k > 0 ? ", " : "") << adjust_pads[k];
            strm << "],\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void setWeights(InputArray weights_arr, InputArray bias_arr,
                            int C0, int /*accuracy*/) CV_OVERRIDE
    {
        Mat rawWeights = weights_arr.getMat();
        Mat rawBias = bias_arr.getMat();
        CV_Assert(!rawWeights.empty());

        wshape0 = rawWeights.shape();

        Mat wfloat;
        if (rawWeights.type() != CV_32F)
            rawWeights.convertTo(wfloat, CV_32F);
        else
            wfloat = rawWeights;
        packGemmWeights(wfloat, C0);
        weights.release();
        if (gemmWeights.empty())
            repackDeconvWeights(wfloat, weights, CV_32F, ngroups, C0);

        if (!rawBias.empty())
            rawBias.convertTo(bias, CV_32F);
    }

    // ConvTranspose as GEMM + col2im; B = [Cg x (taps * KC)] per group and output channel chunk.
    // With one group the output channels are padded to whole C0 blocks.
    void packGemmWeights(const Mat& w, int C0)
    {
        gemmWeights.release();
        gemmCg = gemmKg = gemmKgp = gemmKC = gemmChunks = gemmKsize = gemmC0 = 0;
        gemmChunkStep = 0;
        // CI's Deconvolution tests all take this path on an MLAS build; flip to false to
        // exercise the fallback deconvBlock32f kernel instead.
        static const bool useGemm =
            utils::getConfigurationParameterBool("OPENCV_DNN_CONVTRANSPOSE_GEMM", true);
        if (!useGemm || !mlasAvailable() || !w.isContinuous() || w.type() != CV_32F ||
            w.dims < 3 || w.size[0] % ngroups != 0 || C0 % 4 != 0)
            return;
        const int Kg = w.size[1], Cg = w.size[0] / ngroups;
        const int ksize = (int)(w.total() / ((size_t)w.size[0] * Kg));
        const int Kgp = ngroups == 1 ? (int)alignSize(Kg, C0) : Kg;
        // Overlapping taps limit how thin forwardGemm's bands can get, so parallelize over output
        // channel chunks of about 128 GEMM columns instead.
        const int s0 = strides.empty() ? 1 : strides[0], d0 = dilations.empty() ? 1 : dilations[0];
        const bool overlap = (w.size[2] - 1) * d0 >= s0;
        const int KC = !overlap || Kgp % C0 != 0 ? Kgp :
                       std::min(Kgp, (int)alignSize((128 + ksize - 1) / ksize, C0));
        const int nchunks = (Kgp + KC - 1) / KC;

        const size_t step = alignSize(mlasSgemmPackBSize(false, false, ksize * KC, Cg), 64);
        if (step == 0 || step * ngroups * nchunks > (size_t)INT_MAX)
            return;
        gemmWeights.create(1, (int)(step * ngroups * nchunks), CV_8U);

        Mat B(Cg, ksize * KC, CV_32F);
        const float* wdata = w.ptr<float>();
        for (int g = 0; g < ngroups; g++) {
            for (int ch = 0; ch < nchunks; ch++) {
                const int k0 = ch * KC, kn = std::min(KC, Kgp - k0);
                B.setTo(Scalar(0));
                for (int c = 0; c < Cg; c++) {
                    float* brow = B.ptr<float>(c);
                    for (int k = 0; k < kn && k0 + k < Kg; k++) {
                        const float* wk = wdata + ((size_t)(g * Cg + c) * Kg + k0 + k) * ksize;
                        for (int t = 0; t < ksize; t++)
                            brow[t * kn + k] = wk[t];
                    }
                }
                if (!mlasSgemmPackB(false, false, ksize * kn, Cg, B.ptr<float>(), ksize * KC,
                                    gemmWeights.data + step * (g * nchunks + ch))) {
                    gemmWeights.release();
                    return;
                }
            }
        }
        gemmChunkStep = step;
        gemmCg = Cg;
        gemmKg = Kg;
        gemmKgp = Kgp;
        gemmKC = KC;
        gemmChunks = nchunks;
        gemmKsize = ksize;
        gemmC0 = C0;
    }

    bool gemmCoversShape(const MatShape& inpshape, const MatShape& outshape) const
    {
        return !gemmWeights.empty() && inpshape.back() == gemmC0 && outshape.back() == gemmC0 &&
               inpshape.C == gemmCg * ngroups && outshape.C == gemmKg * ngroups;
    }

    // With an empty input no tap reaches any output pixel, so the output is the bias alone.
    void fillWithBias(Mat& out) const
    {
        CV_Assert(out.type() == CV_32F);
        const MatShape& shape = out.shape();
        const int N = shape[0], K1 = shape[1], C0 = shape.back(), K = shape.C;
        const size_t plane = out.total() / ((size_t)N * K1 * C0);
        const int nbias = bias.empty() ? 0 : std::min((int)bias.total(), K);
        std::vector<float> block((size_t)K1 * C0, 0.f);
        if (nbias > 0)
            std::copy(bias.ptr<float>(), bias.ptr<float>() + nbias, block.begin());
        float* data = out.ptr<float>();
        for (int n = 0; n < N; n++)
            for (int k1 = 0; k1 < K1; k1++) {
                const float* b = block.data() + (size_t)k1 * C0;
                float* dst = data + ((size_t)n * K1 + k1) * plane * C0;
                for (size_t p = 0; p < plane; p++, dst += C0)
                    std::copy(b, b + C0, dst);
            }
    }

    static int floorDiv(int a, int b) { return a >= 0 ? a / b : -((b - 1 - a) / b); }

    void forwardGemm(const Mat& inp, Mat& out)
    {
        enum { MAX_DIMS = ConvState::MAX_CONV_DIMS };
        const MatShape& inpshape = inp.shape();
        const MatShape& outshape = out.shape();
        const int sdims = cs.nspatialdims, C0 = gemmC0;
        const int N = inpshape[0], C1 = inpshape[1], K1 = outshape[1], K = outshape.C;
        const int Cg = gemmCg, Kg = gemmKg, Kgp = gemmKgp, KC = gemmKC, nchunks = gemmChunks;
        const int ksize = gemmKsize;
        const bool blocked = Kgp % C0 == 0;

        int idims[MAX_DIMS], odims[MAX_DIMS], strd[MAX_DIMS];
        int ispatial = 1, ospatial = 1;
        // Every output pixel gets exactly one tap, so it can be written instead of accumulated.
        bool singleTap = blocked;
        for (int i = 0; i < sdims; i++) {
            const int j = MAX_DIMS - sdims + i;
            idims[i] = inpshape[2 + i];
            odims[i] = outshape[2 + i];
            strd[i] = cs.strides[j];
            ispatial *= idims[i];
            ospatial *= odims[i];
            singleTap = singleTap && cs.kshape[j] == strd[i] &&
                        (cs.kshape[j] == 1 || cs.dilations[j] == 1) && cs.pads[j] == 0 &&
                        cs.pads[j + MAX_DIMS] == 0 && odims[i] == idims[i] * strd[i];
        }

        // Tap t moves input coordinate x along dim i to x * strd[i] + tapofs[t * sdims + i];
        // tapflat[t] is the same move in flat output positions.
        int ostep[MAX_DIMS], minofs[MAX_DIMS], maxofs[MAX_DIMS];
        for (int i = sdims - 1, step = 1; i >= 0; i--) {
            ostep[i] = step;
            step *= odims[i];
            minofs[i] = INT_MAX;
            maxofs[i] = INT_MIN;
        }
        std::vector<int> tapofs((size_t)ksize * sdims), tapflat(ksize, 0);
        for (int t = 0; t < ksize; t++) {
            int tmp = t;
            for (int i = sdims - 1; i >= 0; i--) {
                const int j = MAX_DIMS - sdims + i;
                const int ofs = (tmp % cs.kshape[j]) * cs.dilations[j] - cs.pads[j];
                tmp /= cs.kshape[j];
                tapofs[t * sdims + i] = ofs;
                tapflat[t] += ofs * ostep[i];
                minofs[i] = std::min(minofs[i], ofs);
                maxofs[i] = std::max(maxofs[i], ofs);
            }
        }
        const int minofs0 = minofs[0], maxofs0 = maxofs[0];

        std::vector<float> biasbuf((size_t)K1 * C0, 0.f);
        if (!bias.empty())
            std::copy(bias.ptr<float>(), bias.ptr<float>() + std::min((int)bias.total(), K),
                      biasbuf.begin());

        // A task owns a band of output rows in one channel chunk, so col2im needs no locking.
        // Bands are at least 4x the tap overlap, so at most 1/4 of input rows run the GEMM twice.
        const int O0 = odims[0], I0 = idims[0], S0 = strd[0], span0 = maxofs0 - minofs0;
        const int iinner = ispatial / I0, oinner = ospatial / O0;
        const int ntasks0 = N * ngroups * nchunks;
        const int wantBands = std::max(1, (getNumThreads() * 4 + ntasks0 - 1) / ntasks0);
        // roundUp, not alignSize: the stride need not be a power of 2
        const int minRows = span0 < S0 ? S0 : roundUp(4 * span0, S0);
        const int bandRows = std::max(minRows, roundUp((O0 + wantBands - 1) / wantBands, S0));
        const int nbands = (O0 + bandRows - 1) / bandRows;
        const int rows = std::max(16, std::min(256, (1 << 17) / (ksize * KC)));

        const float* inpdata = inp.ptr<float>();
        float* outdata = out.ptr<float>();
        const float* biasdata = biasbuf.data();

        // Groups vary fastest, so groups sharing a C0 block mostly stay on one thread.
        parallel_for_(Range(0, ntasks0 * nbands), [&](const Range& range) {
            AutoBuffer<float> buf((size_t)rows * (Cg + ksize * KC));
            float* A = buf.data();
            float* Cm = A + (size_t)rows * Cg;
            AutoBuffer<size_t> ofsbuf((size_t)Cg + K1 * C0);
            size_t* cofs = ofsbuf.data();
            size_t* kofs = cofs + Cg;
            int x[MAX_DIMS];

            for (int task = range.start; task < range.end; task++) {
                const int g = task % ngroups, ch = (task / ngroups) % nchunks;
                const int band = (task / (ngroups * nchunks)) % nbands;
                const int n = task / (ngroups * nchunks * nbands);
                const int o0 = band * bandRows, o1 = std::min(o0 + bandRows, O0);
                const int cfirst = g * Cg, kfirst = g * Kg + ch * KC;
                const int kn = std::min(KC, Kgp - ch * KC), ncols = ksize * kn;
                const float* inp_n = inpdata + (size_t)n * C1 * ispatial * C0;
                float* out_n = outdata + (size_t)n * K1 * ospatial * C0;
                // Offsets of each input channel and output channel (or block) at spatial pos 0.
                for (int c = 0; c < Cg; c++)
                    cofs[c] = (size_t)((cfirst + c) / C0) * ispatial * C0 + (cfirst + c) % C0;
                const int nk = blocked ? kn / C0 : (g == ngroups - 1 ? K1 * C0 - kfirst : Kg);
                for (int k = 0; k < nk; k++) {
                    const int kk = blocked ? kfirst + k * C0 : kfirst + k;
                    kofs[k] = (size_t)(kk / C0) * ospatial * C0 + kk % C0;
                }

                // Start from the bias; the padding lanes of the last block end up zero.
                for (int opos = o0 * oinner; !singleTap && opos < o1 * oinner; opos++) {
                    float* out_p = out_n + (size_t)opos * C0;
                    if (blocked) {
                        for (int b = 0; b < nk; b++) {
                            const float* bb = biasdata + kfirst + b * C0;
                            std::copy(bb, bb + C0, out_p + kofs[b]);
                        }
                    } else {
                        for (int k = 0; k < nk; k++)
                            out_p[kofs[k]] = biasdata[kfirst + k];
                    }
                }

                // Input rows whose taps reach [o0, o1) along the first spatial dim.
                const int ilo = std::max(0, -floorDiv(maxofs0 - o0, S0));
                const int ihi = std::min(I0 - 1, floorDiv(o1 - 1 - minofs0, S0));
                const int pend = (ihi + 1) * iinner;

                for (int p0 = ilo * iinner; p0 < pend; p0 += rows) {
                    const int M = std::min(rows, pend - p0);
                    for (int m = 0; m < M; m++) {
                        float* arow = A + (size_t)m * Cg;
                        const float* inp_p = inp_n + (size_t)(p0 + m) * C0;
                        if (cfirst % C0 == 0) {
                            for (int c = 0; c < Cg; c += C0) {
                                const float* src = inp_p + cofs[c];
                                std::copy(src, src + std::min(C0, Cg - c), arow + c);
                            }
                        } else {
                            for (int c = 0; c < Cg; c++)
                                arow[c] = inp_p[cofs[c]];
                        }
                    }

                    bool ok = mlasSgemmPacked(false, false, M, ncols, Cg, 1.f, A, Cg,
                                              gemmWeights.data + gemmChunkStep * (g * nchunks + ch),
                                              0.f, Cm, ncols);
                    CV_Assert(ok);

                    // col2im: add each (pixel, tap) row of Cm to the output pixel it lands on.
                    auto emit = [&](int opos, const float* src) {
                        float* out_p = out_n + (size_t)opos * C0;
                        if (singleTap || blocked) {
                            // In a block, C0 is a multiple of 4.
                            for (int b = 0; b < nk; b++) {
                                float* dst = out_p + kofs[b];
                                const float* s = src + b * C0;
                                const float* acc = singleTap ? biasdata + kfirst + b * C0 : dst;
                                for (int j = 0; j < C0; j += 4) {
#if CV_SIMD128
                                    v_store(dst + j, v_add(v_load(s + j), v_load(acc + j)));
#else
                                    for (int l = j; l < j + 4; l++)
                                        dst[l] = s[l] + acc[l];
#endif
                                }
                            }
                        } else {
                            for (int k = 0; k < Kg; k++)
                                out_p[kofs[k]] += src[k];
                        }
                    };
                    for (int m = 0; m < M; m++) {
                        int tmp = p0 + m, base = 0;
                        bool interior = true;
                        for (int i = sdims - 1; i >= 0; i--) {
                            x[i] = tmp % idims[i];
                            tmp /= idims[i];
                            const int o = x[i] * strd[i];
                            base += o * ostep[i];
                            interior = interior && o + minofs[i] >= (i == 0 ? o0 : 0) &&
                                       o + maxofs[i] < (i == 0 ? o1 : odims[i]);
                        }
                        const float* crow = Cm + (size_t)m * ncols;
                        if (interior) {
                            for (int t = 0; t < ksize; t++)
                                emit(base + tapflat[t], crow + t * kn);
                            continue;
                        }
                        for (int t = 0; t < ksize; t++) {
                            const int* ofs = &tapofs[t * sdims];
                            int o = x[0] * S0 + ofs[0];
                            if (o < o0 || o >= o1)
                                continue;
                            int opos = o;
                            int i = 1;
                            for (; i < sdims; i++) {
                                o = x[i] * strd[i] + ofs[i];
                                if (o < 0 || o >= odims[i])
                                    break;
                                opos = opos * odims[i] + o;
                            }
                            if (i == sdims)
                                emit(opos, crow + t * kn);
                        }
                    }
                }
            }
        });
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1)
            return false;
        Mat new_bias = arr.getMat();
        CV_Assert(new_bias.isContinuous() && new_bias.dims == 1);
        if (new_bias.type() != CV_32F) {
            Mat temp;
            new_bias.convertTo(temp, CV_32F);
            new_bias = temp;
        }
        if (!bias.empty()) {
            CV_Assert(bias.shape() == new_bias.shape());
            add(bias, new_bias, bias);
        } else {
            new_bias.copyTo(bias);
        }
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape>& inputs,
                             const std::vector<MatShape>& outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        MatShape wshape = inputs.size() > 1 ? inputs[1] : wshape0;
        int K = wshape[1] * ngroups;
        size_t ksize = wshape.total() / (wshape[0] * wshape[1]);
        int C = inputs[0][1] * inputs[0].back();
        return (int64_t)((inputs[0].total() / C) * ksize * K);
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs >= 1);
        outtypes.assign(1, inferType(inptypes[0]));
        temptypes.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape>& outshapes,
                                 std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, deconvInferShape(inpshapes[0], wshape, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad));
        applyExplicitOutShape(outshapes[0]);
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
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
        return getNetImpl(this)->defaultC0;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = input_arrs.getMat(0);
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        bool dynamicWeights = false;
        for (int i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                dynamicWeights = true;
        }
        if (dynamicWeights || (weights.empty() && gemmWeights.empty())) {
            setWeights(input_arrs.getMat(1), ninputs > 2 ? input_arrs.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = deconvInferShape(inpshape, wshape0, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad);
        applyExplicitOutShape(outshape);

        // compute actual pads for SAME/VALID auto-padding, or from an explicit output_shape
        int nsd = inpshape.dims - 3;
        std::vector<int> pads_resolved = pads;
        if (auto_pad != AUTO_PAD_NONE || !explicit_out_shape.empty()) {
            pads_resolved.resize(nsd * 2, 0);
            for (int i = 0; i < nsd; i++) {
                int inpsz   = inpshape[2 + i];
                int outsz   = outshape[2 + i];
                int adj_i   = adjust_pads.empty() ? 0 : adjust_pads[i];
                int stride  = strides.empty() ? 1 : strides[i];
                int dil     = dilations.empty() ? 1 : dilations[i];
                int ki      = wshape0[2 + i];
                int total   = (inpsz - 1) * stride + dil * (ki - 1) + 1 + adj_i - outsz;
                total = std::max(total, 0);
                // ONNX splits total_padding evenly; odd pixel goes last for SAME_UPPER.
                int pb = (auto_pad == AUTO_PAD_SAME_UPPER) ? total / 2 : total - total / 2;
                pads_resolved[i]       = pb;
                pads_resolved[nsd + i] = total - pb;
            }
        }

        int outtype = inferType(inptype);

        if (inpshape != prevInpshape) {
            cs.initDeconv(inpshape, wshape0, outshape, ngroups,
                          strides, dilations, pads_resolved);
            prevInpshape = inpshape;
        }

        int outkind = output_arrs.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

        std::vector<Mat>* outs = nullptr;
        std::vector<UMat>* uouts = nullptr;
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            outs = &output_arrs.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &output_arrs.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        if (out.total() == 0) {
            // nothing to compute; the kernels below assume non-empty batch and spatial dims
        } else if (inp.total() == 0) {
            fillWithBias(out);
        } else if (inptype == CV_32F && gemmCoversShape(inpshape, outshape)) {
            forwardGemm(inp, out);
        } else {
            DeconvFunc func = getDeconvFunc(inptype);
            CV_Assert(func != nullptr);
            CV_Assert(!weights.empty());
            const float* bias_data = bias.empty() ? nullptr : bias.ptr<float>();
            func(inp.data, nullptr, out.data, cs, weights.data, nullptr, bias_data);
        }

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamicWeights) {
            weights.release();
            gemmWeights.release();
        }
    }

    std::vector<int> emptyKernelShape;
    std::vector<int> explicit_out_shape;
    Mat weights, bias;
    Mat gemmWeights;  // MLAS-packed B per (group, channel chunk), empty without MLAS
    size_t gemmChunkStep = 0;
    int gemmCg = 0, gemmKg = 0, gemmKgp = 0, gemmKC = 0, gemmChunks = 0, gemmKsize = 0, gemmC0 = 0;
    MatShape wshape0, prevInpshape;
    ConvState cs;
};

Ptr<ConvTranspose2Layer> ConvTranspose2Layer::create(const LayerParams& params)
{
    return Ptr<ConvTranspose2Layer>(new ConvTranspose2LayerImpl(params));
}

}}
