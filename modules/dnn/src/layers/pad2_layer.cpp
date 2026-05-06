// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv
{
namespace dnn
{

static constexpr int PAD_MAX_DIMS = 5;

/*
    Padding layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Pad.html

    Opset's 1 to 23 are covered.
*/

// out must be pre-allocated
// pads[] should contains as many elements as inp.dims*2
static void pad(const Mat& inp, const std::vector<int>& pads_, int mode_, const Mat& value, Mat& out)
{
    int inptype = inp.type();
    MatShape inpshape_ = inp.shape();
    MatShape outshape_ = out.shape();
    double buf = 0;
    Mat vbuf(1, 1, inptype, &buf);

    int inpshape[PAD_MAX_DIMS];
    int outshape[PAD_MAX_DIMS];
    int pads[PAD_MAX_DIMS*2];
    int64_t inpstep[PAD_MAX_DIMS];
    int64_t outstep[PAD_MAX_DIMS];
    std::vector<int> tab[PAD_MAX_DIMS];

    int ndims = inp.dims, delta = PAD_MAX_DIMS - ndims;
    int64_t esz = inp.elemSize();

    CV_Assert(inp.isContinuous());
    CV_Assert(out.isContinuous());
    CV_Assert(inp.type() == out.type());
    CV_Assert(esz == 1 || esz == 2 || esz == 4 || esz == 8);
    CV_Assert(inp.dims == out.dims);
    CV_Assert(inp.dims <= PAD_MAX_DIMS);

    if (!value.empty()) {
        CV_Assert(value.dims <= 2 && value.total() == 1 && value.channels() == 1);
        tensorToScalar(value, inptype, &buf);
    }

    for (int i = 0; i < PAD_MAX_DIMS; i++) {
        inpshape[i] = outshape[i] = 1;
        pads[i] = pads[i + PAD_MAX_DIMS] = 0;
    }

    for (int i = 0; i < ndims; i++) {
        inpshape[i+delta] = inpshape_[i];
        outshape[i+delta] = outshape_[i];
        pads[i+delta] = pads_[i];
        pads[i+delta + PAD_MAX_DIMS] = pads_[i + ndims];

        // initialize lookup table along the corresponding axis
        int inpsz_i = inpshape_[i];
        int outsz_i = outshape_[i];
        tab[i+delta].resize(outsz_i);
        int* tab_i = tab[i+delta].data();
        int before = pads_[i];
        for (int j = 0; j < outsz_i; j++)
            tab_i[j] = borderInterpolate(j - before, inpsz_i, mode_);
    }

    for (int i = PAD_MAX_DIMS-1; i >= 0; i--) {
        if (i == PAD_MAX_DIMS-1)
            inpstep[i] = outstep[i] = 1;
        else {
            inpstep[i] = inpstep[i+1]*inpshape[i+1];
            outstep[i] = outstep[i+1]*outshape[i+1];
        }
    }

    int nplanes = outshape[0]*outshape[1]*outshape[2];

    CV_Assert(!tab[4].empty());

    #undef IMPL_PAD
    #define IMPL_PAD(T) \
    parallel_for_(Range(0, nplanes), [&](const Range& r) { \
        int mode = mode_; \
        int sz1 = outshape[1], sz2 = outshape[2], sz3 = outshape[3], sz4 = outshape[4]; \
        const int* tab0 = tab[0].data(); \
        const int* tab1 = tab[1].data(); \
        const int* tab2 = tab[2].data(); \
        const int* tab3 = tab[3].data(); \
        const int* tab4 = tab[4].data(); \
        const T* inpdata0 = (const T*)inp.data; \
        T val0 = *reinterpret_cast<T*>(vbuf.data); \
        T* outdata0 = (T*)out.data; \
        int p0 = pads[PAD_MAX_DIMS-1], p1 = pads[PAD_MAX_DIMS*2-1]; \
        int p0_ = std::max(p0, 0), p1_ = std::max(p1, 0); \
        for (int plane = r.start; plane < r.end; plane++) { \
            int plane_ = plane; \
            int i2 = plane_ % sz2; \
            plane_ /= sz2; \
            int i1 = plane_ % sz1; \
            int i0 = plane_ / sz1; \
            int ii0 = tab0 ? tab0[i0] : i0; \
            int ii1 = tab1 ? tab1[i1] : i1; \
            int ii2 = tab2 ? tab2[i2] : i2; \
            for (int i3 = 0; i3 < sz3; i3++) { \
                int ii3 = tab3 ? tab3[i3] : i3; \
                T* outdata = outdata0 + i0*outstep[0] + i1*outstep[1] + i2*outstep[2] + i3*outstep[3]; \
                int i4 = 0; \
                if ((ii0|ii1|ii2|ii3) < 0) { \
                    for (; i4 < sz4; i4++) \
                        outdata[i4] = val0; \
                    continue; \
                } \
                const T* inpdata = inpdata0 + ii0*inpstep[0] + ii1*inpstep[1] + ii2*inpstep[2] + ii3*inpstep[3]; \
                if (mode == BORDER_CONSTANT) {\
                    for (; i4 < p0_; i4++) \
                        outdata[i4] = val0; \
                } else { \
                    for (; i4 < p0_; i4++) \
                        outdata[i4] = inpdata[tab4[i4]]; \
                } \
                for (; i4 < sz4 - p1_; i4++) \
                    outdata[i4] = inpdata[i4 - p0]; \
                if (mode == BORDER_CONSTANT) { \
                    for (; i4 < sz4; i4++) \
                        outdata[i4] = val0; \
                } else { \
                    for (; i4 < sz4; i4++) \
                        outdata[i4] = inpdata[tab4[i4]]; \
                } \
            } \
        } \
    })

    if (esz == 1) {
        IMPL_PAD(uint8_t);
    } else if (esz == 2) {
        IMPL_PAD(uint16_t);
    } else if (esz == 4) {
        IMPL_PAD(uint32_t);
    } else {
        CV_Assert(esz == 8);
        IMPL_PAD(uint64_t);
    }
}

// Specialized semantic-padding path for block tensors with shape [N, C1, H, W, C0].
// It handles semantic channel-axis padding directly in blocked representation.
static void pad_block5d_semantic(const Mat& inp,
                                 const std::vector<int>& semanticPads,
                                 int mode_,
                                 const Mat& value,
                                 DataLayout origLayout,
                                 Mat& out)
{
    CV_Assert(inp.dims == 5 && out.dims == 5);
    CV_Assert(inp.shape().layout == DATA_LAYOUT_BLOCK && out.shape().layout == DATA_LAYOUT_BLOCK);
    CV_Assert(semanticPads.size() == 8);
    CV_Assert(mode_ == BORDER_CONSTANT);

    int inptype = inp.type();
    CV_Assert(inp.type() == out.type());
    CV_Assert(inp.isContinuous() && out.isContinuous());

    double buf = 0;
    Mat vbuf(1, 1, inptype, &buf);
    if (!value.empty()) {
        CV_Assert(value.dims <= 2 && value.total() == 1 && value.channels() == 1);
        tensorToScalar(value, inptype, &buf);
    }

    const int C0 = inp.shape()[4];
    const int semantic_Cin = inp.shape().C;
    CV_Assert(C0 > 0);

    // Physical layout is always [N, C1, H, W, C0] for BLOCK;
    // semantic channel axis depends on original model layout.
    const int semanticChannelAxis = (origLayout == DATA_LAYOUT_NCHW) ? 1 : 3;
    const int semanticHAxis = (origLayout == DATA_LAYOUT_NCHW) ? 2 : 1;
    const int semanticWAxis = (origLayout == DATA_LAYOUT_NCHW) ? 3 : 2;
    const int cPadBefore = semanticPads[semanticChannelAxis];

    const int N = inp.size[0];
    const int Hin = inp.size[2];
    const int Win = inp.size[3];

    const int Nout = out.size[0];
    const int C1out = out.size[1];
    const int Hout = out.size[2];
    const int Wout = out.size[3];
    const int phys_Cout = C1out * C0;

    std::vector<int> tabN(Nout), tabH(Hout), tabW(Wout), tabC(phys_Cout);
    for (int i = 0; i < Nout; i++)
        tabN[i] = borderInterpolate(i - semanticPads[0], N, mode_);
    for (int i = 0; i < Hout; i++)
        tabH[i] = borderInterpolate(i - semanticPads[semanticHAxis], Hin, mode_);
    for (int i = 0; i < Wout; i++)
        tabW[i] = borderInterpolate(i - semanticPads[semanticWAxis], Win, mode_);
    for (int i = 0; i < phys_Cout; i++)
        tabC[i] = borderInterpolate(i - cPadBefore, semantic_Cin, mode_);

    std::vector<int> blockL(C1out), blockR(C1out), blockInStart(C1out);
    std::vector<int> seg0OutOff(C1out), seg0Len(C1out), seg0InC1(C1out), seg0InC0(C1out);
    std::vector<int> seg1OutOff(C1out), seg1Len(C1out), seg1InC1(C1out);
    for (int oc1 = 0; oc1 < C1out; oc1++) {
        int base = oc1 * C0;
        int l = 0;
        while (l < C0 && tabC[base + l] < 0)
            l++;
        int r = C0;
        while (r > l && tabC[base + r - 1] < 0)
            r--;
        blockL[oc1] = l;
        blockR[oc1] = r;
        blockInStart[oc1] = (l < r) ? tabC[base + l] : 0;

        int len = r - l;
        if (len > 0) {
            int in_c = blockInStart[oc1];
            int in_c1 = in_c / C0;
            int in_c0 = in_c - in_c1 * C0;
            int len0 = std::min(len, C0 - in_c0);
            seg0OutOff[oc1] = l;
            seg0Len[oc1] = len0;
            seg0InC1[oc1] = in_c1;
            seg0InC0[oc1] = in_c0;

            int len1 = len - len0;
            seg1OutOff[oc1] = l + len0;
            seg1Len[oc1] = len1;
            seg1InC1[oc1] = in_c1 + 1;
        } else {
            seg0OutOff[oc1] = 0;
            seg0Len[oc1] = 0;
            seg0InC1[oc1] = 0;
            seg0InC0[oc1] = 0;
            seg1OutOff[oc1] = 0;
            seg1Len[oc1] = 0;
            seg1InC1[oc1] = 0;
        }
    }

    const size_t inStep0 = inp.step.p[0] / inp.elemSize();
    const size_t inStep1 = inp.step.p[1] / inp.elemSize();
    const size_t inStep2 = inp.step.p[2] / inp.elemSize();
    const size_t inStep3 = inp.step.p[3] / inp.elemSize();

    const size_t outStep0 = out.step.p[0] / out.elemSize();
    const size_t outStep1 = out.step.p[1] / out.elemSize();
    const size_t outStep2 = out.step.p[2] / out.elemSize();
    const size_t outStep3 = out.step.p[3] / out.elemSize();

    int wL = 0;
    while (wL < Wout && tabW[wL] < 0)
        wL++;
    int wR = Wout;
    while (wR > wL && tabW[wR - 1] < 0)
        wR--;

    const int nrows = Nout * C1out * Hout;

#undef IMPL_PAD_BLOCK5D
#define IMPL_PAD_BLOCK5D(T) \
    parallel_for_(Range(0, nrows), [&](const Range& r) { \
        const T* inpdata0 = reinterpret_cast<const T*>(inp.data); \
        T* outdata0 = reinterpret_cast<T*>(out.data); \
        T val0 = *reinterpret_cast<T*>(vbuf.data); \
        bool val_is_zero = (val0 == (T)0); \
        for (int row = r.start; row < r.end; row++) { \
            int t = row; \
            int oh = t % Hout; \
            t /= Hout; \
            int oc1 = t % C1out; \
            int on = t / C1out; \
            T* outrow = outdata0 + on*outStep0 + oc1*outStep1 + oh*outStep2; \
            int in_n = tabN[on]; \
            int in_h = tabH[oh]; \
            if ((in_n | in_h) < 0) { \
                for (int ow = 0; ow < Wout; ow++) { \
                    T* outptr = outrow + ow*outStep3; \
                    if (val_is_zero) \
                        memset(outptr, 0, (size_t)C0 * sizeof(T)); \
                    else { \
                        for (int oc0 = 0; oc0 < C0; oc0++) outptr[oc0] = val0; \
                    } \
                } \
                continue; \
            } \
            int l = blockL[oc1]; \
            int rr = blockR[oc1]; \
            int len0 = seg0Len[oc1]; \
            int len1 = seg1Len[oc1]; \
            for (int ow = 0; ow < wL; ow++) { \
                T* outptr = outrow + ow*outStep3; \
                if (val_is_zero) \
                    memset(outptr, 0, (size_t)C0 * sizeof(T)); \
                else { \
                    for (int oc0 = 0; oc0 < C0; oc0++) outptr[oc0] = val0; \
                } \
            } \
            for (int ow = wL; ow < wR; ow++) { \
                T* outptr = outrow + ow*outStep3; \
                if (l > 0) { \
                    if (val_is_zero) \
                        memset(outptr, 0, (size_t)l * sizeof(T)); \
                    else { \
                        for (int oc0 = 0; oc0 < l; oc0++) \
                            outptr[oc0] = val0; \
                    } \
                } \
                if (l < rr) { \
                    int in_w = tabW[ow]; \
                    const T* inbase = inpdata0 + in_n*inStep0 + seg0InC1[oc1]*inStep1 + in_h*inStep2 + in_w*inStep3; \
                    if (len0 > 0) { \
                        const T* src0 = inbase + seg0InC0[oc1]; \
                        memcpy(outptr + seg0OutOff[oc1], src0, (size_t)len0 * sizeof(T)); \
                    } \
                    if (len1 > 0) { \
                        const T* src1 = inbase + inStep1; \
                        memcpy(outptr + seg1OutOff[oc1], src1, (size_t)len1 * sizeof(T)); \
                    } \
                } \
                if (rr < C0) { \
                    if (val_is_zero) \
                        memset(outptr + rr, 0, (size_t)(C0 - rr) * sizeof(T)); \
                    else { \
                        for (int oc0 = rr; oc0 < C0; oc0++) \
                            outptr[oc0] = val0; \
                    } \
                } \
            } \
            for (int ow = wR; ow < Wout; ow++) { \
                T* outptr = outrow + ow*outStep3; \
                if (val_is_zero) \
                    memset(outptr, 0, (size_t)C0 * sizeof(T)); \
                else { \
                    for (int oc0 = 0; oc0 < C0; oc0++) outptr[oc0] = val0; \
                } \
            } \
        } \
    })

    if (inp.elemSize() == 1) {
        IMPL_PAD_BLOCK5D(uint8_t);
    } else if (inp.elemSize() == 2) {
        IMPL_PAD_BLOCK5D(uint16_t);
    } else if (inp.elemSize() == 4) {
        IMPL_PAD_BLOCK5D(uint32_t);
    } else {
        CV_Assert(inp.elemSize() == 8);
        IMPL_PAD_BLOCK5D(uint64_t);
    }
}

static DataLayout getOriginalLayout(const Layer* layer)
{
    Net::Impl* netimpl_ = getNetImpl(layer);
    DataLayout layout = netimpl_ ? netimpl_->originalLayout : DATA_LAYOUT_NCHW;
    CV_Assert(layout == DATA_LAYOUT_NCHW || layout == DATA_LAYOUT_NHWC);
    return layout;
}

static int semanticToPhysicalAxis(int semanticAxis, int semanticNdims, DataLayout origLayout)
{
    if (origLayout == DATA_LAYOUT_NCHW)
        return semanticAxis;
    CV_Assert(origLayout == DATA_LAYOUT_NHWC);
    if (semanticAxis == 0)
        return 0;
    if (semanticAxis == semanticNdims - 1)
        return 1;
    return semanticAxis + 1;
}

static bool hasNonZeroChannelPad(const std::vector<int>& semanticPads,
                                 int semanticNdims,
                                 DataLayout origLayout)
{
    if ((int)semanticPads.size() != semanticNdims * 2)
        return false;
    int cAxis = origLayout == DATA_LAYOUT_NCHW ? 1 : semanticNdims - 1;
    if (!(0 <= cAxis && cAxis < semanticNdims))
        return false;
    return semanticPads[cAxis] != 0 || semanticPads[cAxis + semanticNdims] != 0;
}

static bool canUseBlockWithChannelPad(const MatShape& inpshape,
                                      const std::vector<int>& semanticPads,
                                      int mode,
                                      DataLayout origLayout)
{
    if (mode != BORDER_CONSTANT)
        return false;
    if ((int)semanticPads.size() != 8)
        return false;

    int cAxis = origLayout == DATA_LAYOUT_NCHW ? 1 : 3;
    CV_Assert(cAxis >= 0);

    int cBefore = semanticPads[cAxis];
    int cAfter = semanticPads[cAxis + 4];

    int semantic_Cin = 0;
    if (inpshape.layout == DATA_LAYOUT_BLOCK && inpshape.dims == 5) {
        semantic_Cin = inpshape.C;
    } else if (inpshape.dims == 4) {
        semantic_Cin = (origLayout == DATA_LAYOUT_NCHW) ? inpshape[1] : inpshape[3];
    } else {
        return false;
    }

    int semantic_Cout = semantic_Cin + cBefore + cAfter;
    return semantic_Cout >= 0;
}

static MatShape getOutShapeBlockSemantic(const MatShape& inpshape,
                                         const std::vector<int>& semanticPads,
                                         DataLayout origLayout)
{
    CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(inpshape.dims == 5);
    CV_Assert(semanticPads.size() == 8);

    const int semanticNdims = 4;
    const int cAxis = origLayout == DATA_LAYOUT_NCHW ? 1 : 3;
    const int C0 = inpshape[4];

    MatShape outshape = inpshape;
    for (int i = 0; i < semanticNdims; i++) {
        int inSz = (i == cAxis) ? inpshape.C : inpshape[semanticToPhysicalAxis(i, semanticNdims, origLayout)];
        int outSz = inSz + semanticPads[i] + semanticPads[i + semanticNdims];
        CV_Assert(outSz >= 0);

        if (i == cAxis) {
            outshape[1] = (outSz + C0 - 1) / C0;
        } else {
            int physAxis = semanticToPhysicalAxis(i, semanticNdims, origLayout);
            outshape[physAxis] = outSz;
        }
    }
    outshape[4] = C0;
    outshape.C = inpshape.C + semanticPads[cAxis] + semanticPads[cAxis + semanticNdims];
    return outshape;
}

static bool mapPadsToInputLayout(const MatShape& inpshape,
                                 const std::vector<int>& semanticPads,
                                 std::vector<int>& pads,
                                 DataLayout origLayout)
{
    int ndims = inpshape.dims;
    DataLayout layout = inpshape.layout;

    if (layout != DATA_LAYOUT_BLOCK) {
        if ((int)semanticPads.size() != ndims*2)
            return false;
        pads = semanticPads;
        return true;
    }

    if (ndims < 2 || ndims > PAD_MAX_DIMS)
        return false;

    int semanticNdims = ndims - 1;
    if ((int)semanticPads.size() != semanticNdims*2)
        return false;

    int cAxis = origLayout == DATA_LAYOUT_NCHW ? 1 : semanticNdims - 1;

    if (semanticPads[cAxis] != 0 || semanticPads[cAxis + semanticNdims] != 0)
        return false;

    pads.assign(ndims*2, 0);

    for (int i = 0; i < semanticNdims; i++) {
        int j = i;
        if (origLayout == DATA_LAYOUT_NHWC) {
            if (i == 0)
                j = 0;
            else if (i == semanticNdims - 1)
                j = 1;
            else
                j = i + 1;
        }
        pads[j] = semanticPads[i];
        pads[j + ndims] = semanticPads[i + semanticNdims];
    }

    pads[ndims - 1] = 0;
    pads[ndims*2 - 1] = 0;
    return true;
}

class Pad2LayerImpl CV_FINAL : public Pad2Layer
{
public:
    std::vector<int> pads0;
    float value0 = 0.f;
    int mode = BORDER_CONSTANT;

    Pad2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        std::vector<int> pads0_ = params.getVector<int>("paddings");
        // [TODO] remove this transposition after the original transposition is removed from onnx importer 2
        if (!pads0_.empty()) {
            int i, ndims = (int)(pads0_.size()/2);
            pads0.resize(ndims*2);
            for (i = 0; i < ndims; i++) {
                pads0[i] = pads0_[i*2];
                pads0[i + ndims] = pads0_[i*2+1];
            }
        }
        std::string strmode = params.get<std::string>("mode", "constant");
        if (strmode == "constant")
            mode = BORDER_CONSTANT;
        else if (strmode == "reflect")
            mode = BORDER_REFLECT101;
        else if (strmode == "edge")
            mode = BORDER_REPLICATE;
        else if (strmode == "wrap")
            mode = BORDER_WRAP;
        else {
            CV_Error_(Error::StsNotImplemented, ("mode '%s' is not supported", strmode.c_str()));
        }
        value0 = params.get<float>("value", 0.f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        CV_Assert(netimpl_);
        size_t ninputs = this->inputs.size();
        CV_Assert(1 <= ninputs && ninputs <= 4);
        return (ninputs >= 2 && !netimpl_->isConstArg(this->inputs[1])) ||
               (ninputs >= 4 && !netimpl_->isConstArg(this->inputs[3]));
    }

    bool getConstSemanticPads(std::vector<int>& semanticPads) const
    {
        size_t ninputs = this->inputs.size();
        CV_Assert(1 <= ninputs && ninputs <= 4);

        if (ninputs == 1) {
            semanticPads = pads0;
            return !semanticPads.empty();
        }

        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_ || !netimpl_->isConstArg(this->inputs[1]))
            return false;

        Mat padsTensor = netimpl_->argTensor(this->inputs[1]);
        Mat axesTensor;
        if (ninputs >= 4) {
            if (!netimpl_->isConstArg(this->inputs[3]))
                return false;
            axesTensor = netimpl_->argTensor(this->inputs[3]);
        }

        int ndims = netimpl_->argData(this->inputs[0]).shape.dims;
        if (ndims <= 0) {
            if (axesTensor.empty())
                ndims = int(padsTensor.total()/2);
            else
                return false;
        }

        if (ndims < 1 || ndims > PAD_MAX_DIMS)
            return false;

        getPads(ndims, padsTensor, axesTensor, semanticPads);
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(1 <= ninputs && ninputs <= 4);

        desiredInputs.assign(ninputs, DATA_LAYOUT_UNKNOWN);
        desiredInputs[0] = actualInputs[0];
        outputs.assign(requiredOutputs, actualInputs[0]);

        if (actualInputs[0] != DATA_LAYOUT_BLOCK)
            return 0;

        DataLayout origLayout = getOriginalLayout(this);
        std::vector<int> semanticPads;
        bool canUseBlock = getConstSemanticPads(semanticPads);
        if (canUseBlock) {
            int semanticNdims = int(semanticPads.size() / 2);
            canUseBlock = (semanticNdims >= 1 && semanticNdims + 1 <= PAD_MAX_DIMS);
            if (canUseBlock) {
                bool hasChannelPad = hasNonZeroChannelPad(semanticPads, semanticNdims, origLayout);
                if (hasChannelPad) {
                    // Channel padding in semantic space requires special block path.
                    // 4D semantic tensors, constant mode.
                    canUseBlock = (mode == BORDER_CONSTANT && semanticNdims == 4);
                } else {
                    // Non-channel semantic padding can be represented by per-axis
                    // physical pads in block layout.
                    canUseBlock = true;
                }
            }
        } else {
            canUseBlock = false;
        }

        if (!canUseBlock) {
            desiredInputs[0] = origLayout;
            outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);
        } else {
            desiredInputs[0] = DATA_LAYOUT_BLOCK;
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        }
        return 0;
    }

    void getPads(int ndims, const Mat& pads_, const Mat& axes_, std::vector<int>& pads) const
    {
        int atype = axes_.type(), ptype = pads_.type();
        CV_Assert(ndims <= PAD_MAX_DIMS);

        const int32_t* adata_i32 = nullptr;
        const int64_t* adata_i64 = nullptr;
        const int32_t* pdata_i32 = nullptr;
        const int64_t* pdata_i64 = nullptr;

        bool axismask[PAD_MAX_DIMS];
        int naxes = !axes_.empty() ? (int)axes_.total() : ndims;

        CV_Assert(pads_.dims == 1);
        CV_Assert(ptype == CV_32S || ptype == CV_64S);

        if (ptype == CV_32S)
            pdata_i32 = reinterpret_cast<const int32_t*>(pads_.data);
        else
            pdata_i64 = reinterpret_cast<const int64_t*>(pads_.data);

        if (!axes_.empty()) {
            CV_Assert(axes_.dims == 1);
            CV_Assert(atype == CV_32S || atype == CV_64S);
            CV_Assert(pads_.total() == axes_.total()*2);
            CV_Assert(axes_.total() <= (size_t)ndims);

            if (atype == CV_32S)
                adata_i32 = reinterpret_cast<const int32_t*>(axes_.data);
            else
                adata_i64 = reinterpret_cast<const int64_t*>(axes_.data);
        } else {
            CV_Assert(pads_.total() == (size_t)ndims*2);
        }

        pads.resize(ndims*2);

        for (int i = 0; i < ndims; i++) {
            pads[i] = pads[i+ndims] = 0;
            axismask[i] = false;
        }

        for (int i = 0; i < naxes; i++) {
            int a = adata_i32 ? (int)adata_i32[i] : adata_i64 ? (int)adata_i64[i] : i;
            a = normalize_axis(a, ndims);
            if (axismask[a]) {
                CV_Error_(Error::StsBadArg, ("duplicate axis %d in Pad", a));
            }
            axismask[a] = true;
            int p0 = pdata_i32 ? (int)pdata_i32[i] : pdata_i64 ? (int)pdata_i64[i] : 0;
            int p1 = pdata_i32 ? (int)pdata_i32[i+naxes] : pdata_i64 ? (int)pdata_i64[i+naxes] : 0;
            pads[a] = p0;
            pads[a+ndims] = p1;
            // p0, p1 can be positive, zero or even negative, according to ONNX specification.
            // so we don't put any checks here.
        }
    }

    MatShape getOutShape(const MatShape& inpshape, const std::vector<int>& pads) const
    {
        MatShape outshape = inpshape;
        int ndims = inpshape.dims;
        for (int i = 0; i < ndims; i++) {
            outshape[i] += pads[i] + pads[i+ndims];
            CV_Assert(outshape[i] >= 0);
        }
        return outshape;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!dynamicOutputShapes());

        size_t ninputs = inputs.size();
        CV_Assert(1 <= ninputs && ninputs <= 4);
        Net::Impl* netimpl_ = getNetImpl(this);

        std::vector<int> semanticPads;
        std::vector<int> padsbuf;
        DataLayout origLayout = getOriginalLayout(this);

        if (ninputs >= 2) {
            int ndims = inputs[0].layout == DATA_LAYOUT_BLOCK ? inputs[0].dims - 1 : inputs[0].dims;
            Mat padsTensor = netimpl_->argTensor(this->inputs[1]);
            Mat axesTensor;
            if (ninputs >= 4)
                axesTensor = netimpl_->argTensor(this->inputs[3]);
            getPads(ndims, padsTensor, axesTensor, semanticPads);
        } else {
            semanticPads = pads0;
        }

        bool mapped = mapPadsToInputLayout(inputs[0], semanticPads, padsbuf, origLayout);
        bool useBlockSemantic = !mapped && canUseBlockWithChannelPad(inputs[0], semanticPads, mode, origLayout);
        if (mapped) {
            outputs.assign(1, getOutShape(inputs[0], padsbuf));
        } else if (useBlockSemantic) {
            outputs.assign(1, getOutShapeBlockSemantic(inputs[0], semanticPads, origLayout));
        } else {
            CV_Error(Error::StsInternal, "Pad2: unexpected block layout with unsupported padding");
        }

        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(1 <= ninputs && ninputs <= 4);
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(1 <= ninputs && ninputs <= 4);

        Mat inp = inputs_arr.getMat(0);
        Mat value(1, 1, CV_32F, &value0);
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        std::vector<int> semanticPads;
        std::vector<int> padsbuf;
        DataLayout origLayout = getOriginalLayout(this);

        if (ninputs >= 2) {
            int ndims = inpshape.layout == DATA_LAYOUT_BLOCK ? inpshape.dims - 1 : inpshape.dims;
            Mat padsTensor = inputs_arr.getMat(1);
            Mat axesTensor;
            if (ninputs >= 4)
                axesTensor = inputs_arr.getMat(3);
            getPads(ndims, padsTensor, axesTensor, semanticPads);
            if (ninputs >= 3)
                value = inputs_arr.getMat(2);
        } else {
            semanticPads = pads0;
        }

        bool mapped = mapPadsToInputLayout(inpshape, semanticPads, padsbuf, origLayout);
        bool useBlockSemantic = !mapped && canUseBlockWithChannelPad(inpshape, semanticPads, mode, origLayout);
        MatShape outshape;
        if (mapped)
            outshape = getOutShape(inpshape, padsbuf);
        else if (useBlockSemantic)
            outshape = getOutShapeBlockSemantic(inpshape, semanticPads, origLayout);
        else
            CV_Error(Error::StsInternal, "Pad2: unexpected block layout with unsupported padding");

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            if (mapped)
                pad(inp, padsbuf, mode, value, outs[0]);
            else
                pad_block5d_semantic(inp, semanticPads, mode, value, origLayout, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            if (mapped)
                pad(inp, padsbuf, mode, value, temp);
            else
                pad_block5d_semantic(inp, semanticPads, mode, value, origLayout, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<Pad2Layer> Pad2Layer::create(const LayerParams& params)
{
    return Ptr<Pad2Layer>(new Pad2LayerImpl(params));
}

}
}
