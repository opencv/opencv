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

        std::vector<int> padsbuf;
        const std::vector<int>* pads = &pads0;

        if (ninputs >= 2) {
            int ndims = inputs[0].dims;
            Mat padsTensor = netimpl_->argTensor(this->inputs[1]);
            Mat axesTensor;
            if (ninputs >= 4)
                axesTensor = netimpl_->argTensor(this->inputs[3]);
            getPads(ndims, padsTensor, axesTensor, padsbuf);
            pads = &padsbuf;
        }

        outputs.assign(1, getOutShape(inputs[0], *pads));
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
        std::vector<int> padsbuf;
        const std::vector<int>* pads = &pads0;

        if (ninputs >= 2) {
            int ndims = inp.dims;
            Mat padsTensor = inputs_arr.getMat(1);
            Mat axesTensor;
            if (ninputs >= 4)
                axesTensor = inputs_arr.getMat(3);
            getPads(ndims, padsTensor, axesTensor, padsbuf);
            pads = &padsbuf;
            if (ninputs >= 3)
                value = inputs_arr.getMat(2);
        }

        MatShape inpshape = inp.shape();
        MatShape outshape = getOutShape(inpshape, *pads);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            pad(inp, *pads, mode, value, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            pad(inp, *pads, mode, value, temp);
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
