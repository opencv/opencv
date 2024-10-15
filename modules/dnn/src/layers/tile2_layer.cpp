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

static constexpr int TILE_MAX_DIMS = 6;

/*
    Tile layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Tile.html

    Opset's 1 to 13 are covered.
*/

// out must be pre-allocated
// repeats_[] should contains as many elements as inp.dims (== out.dims)
static void tile(const Mat& inp, const int* repeats_, Mat& out)
{
    MatShape inpshape_ = inp.shape();
    MatShape outshape_ = out.shape();
    const uchar* inpdata0 = inp.data;
    uchar* outdata0_ = out.data;

    int inpshape[TILE_MAX_DIMS];
    int outshape[TILE_MAX_DIMS];
    int repeats[TILE_MAX_DIMS];
    int64_t inpstep[TILE_MAX_DIMS];
    int64_t outstep[TILE_MAX_DIMS];

    int ndims = inp.dims, delta = TILE_MAX_DIMS - ndims;
    int64_t esz = inp.elemSize();
    int64_t total_size = 1, total_repeats = 1;

    CV_Assert(inp.isContinuous());
    CV_Assert(out.isContinuous());
    CV_Assert(inp.type() == out.type());
    CV_Assert(esz == 1 || esz == 2 || esz == 4 || esz == 8);
    CV_Assert(inp.dims == out.dims);
    CV_Assert(inp.dims <= TILE_MAX_DIMS);

    for (int i = 0; i < TILE_MAX_DIMS; i++) {
        inpshape[i] = outshape[i] = repeats[i] = 1;
    }

    for (int i = 0; i < ndims; i++) {
        inpshape[i + delta] = inpshape_[i];
        outshape[i + delta] = outshape_[i];
        repeats[i + delta] = repeats_[i];

        CV_Assert(inpshape_[i]*repeats_[i] == outshape_[i]);

        total_size *= outshape_[i];
        total_repeats *= repeats_[i];
    }

    for (int i = TILE_MAX_DIMS-1; i >= 0; i--) {
        if (i == TILE_MAX_DIMS-1)
            inpstep[i] = outstep[i] = 1;
        else {
            inpstep[i] = inpstep[i+1]*inpshape[i+1];
            outstep[i] = outstep[i+1]*outshape[i+1];
        }
    }

    int ntasks = 8;
    if (ntasks > total_repeats)
        ntasks = (int)total_repeats;
    if (total_size < 1000000)
        ntasks = 1;

    //parallel_for_(Range(0, ntasks), [&](const Range& r)
    //Range r(0, ntasks);
    {
        int sz0 = inpshape[0], sz1 = inpshape[1], sz2 = inpshape[2];
        int sz3 = inpshape[3], sz4 = inpshape[4], sz5 = inpshape[5];

        int64_t outstep_prelast = outstep[TILE_MAX_DIMS-2];
        //int64_t j0 = r.start*total_repeats/ntasks, j1 = r.end*total_repeats/ntasks;

        //for (int64_t j = j0; j < j1; j++)
        for (int64_t j = 0; j < total_repeats; j++)
        {
            // convert raw tile index into n-dim tile index.
            // but we don't need this nd-index itself, we just need the
            // offset of the tile in the output tensor
            int64_t j_ = j, rawofs = 0;
            for (int k = TILE_MAX_DIMS-1; k >= 0; k--) {
                int r = repeats[k];
                int64_t q = j_ / r;
                rawofs += (j_ - q*r)*inpshape[k]*outstep[k];
                j_ = q;
            }

            #undef IMPL_COPY_TILE
            #define IMPL_COPY_TILE(T) \
                T* inpdata = (T*)inpdata0; \
                T* outdata0 = (T*)outdata0_ + rawofs; \
                for (int i0 = 0; i0 < sz0; i0++) { \
                for (int i1 = 0; i1 < sz1; i1++) { \
                for (int i2 = 0; i2 < sz2; i2++) { \
                for (int i3 = 0; i3 < sz3; i3++) { \
                    T* outdata = outdata0 + i0*outstep[0] + i1*outstep[1] + i2*outstep[2] + i3*outstep[3]; \
                    for (int i4 = 0; i4 < sz4; i4++, outdata += outstep_prelast, inpdata += sz5) { \
                        for (int i5 = 0; i5 < sz5; i5++) \
                            outdata[i5] = inpdata[i5]; \
                    } \
                }}}}

            if (esz == 1) {
                IMPL_COPY_TILE(uint8_t)
            } else if (esz == 2) {
                IMPL_COPY_TILE(uint16_t)
            } else if (esz == 4) {
                IMPL_COPY_TILE(uint32_t)
            } else {
                IMPL_COPY_TILE(uint64_t)
            }
        }
    }
    //, ntasks);
}

class Tile2LayerImpl CV_FINAL : public Tile2Layer
{
public:
    Tile2LayerImpl(const LayerParams& params)
    {
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
        CV_Assert(ninputs == 2 || ninputs == 3);
        return  !netimpl_->isConstArg(this->inputs[1]) ||
                (ninputs == 3 && !netimpl_->isConstArg(this->inputs[2]));
    }

    void getRepeats(const Mat& repeats_, const Mat& axes_, int ndims, int* repeats) const
    {
        int atype = axes_.type(), rtype = repeats_.type();
        CV_Assert(ndims <= TILE_MAX_DIMS);

        const int32_t* adata_i32 = nullptr;
        const int64_t* adata_i64 = nullptr;
        const int32_t* rdata_i32 = nullptr;
        const int64_t* rdata_i64 = nullptr;

        bool axismask[TILE_MAX_DIMS];

        CV_Assert(repeats_.dims == 1);
        CV_Assert(rtype == CV_32S || rtype == CV_64S);

        if (rtype == CV_32S)
            rdata_i32 = reinterpret_cast<const int32_t*>(repeats_.data);
        else
            rdata_i64 = reinterpret_cast<const int64_t*>(repeats_.data);

        if (!axes_.empty()) {
            CV_Assert(axes_.dims == 1);
            CV_Assert(atype == CV_32S || atype == CV_64S);
            CV_Assert(repeats_.total() == axes_.total());
            CV_Assert(axes_.total() <= (size_t)ndims);

            if (atype == CV_32S)
                adata_i32 = reinterpret_cast<const int32_t*>(axes_.data);
            else
                adata_i64 = reinterpret_cast<const int64_t*>(axes_.data);
        } else {
            CV_Assert(repeats_.total() == (size_t)ndims);
        }

        for (int i = 0; i < ndims; i++) {
            repeats[i] = 1;
            axismask[i] = false;
        }

        int nrepeats = (int)repeats_.total();
        for (int i = 0; i < nrepeats; i++) {
            int a = adata_i32 ? (int)adata_i32[i] : adata_i64 ? (int)adata_i64[i] : i;
            a = normalize_axis(a, ndims);
            if (axismask[a]) {
                CV_Error_(Error::StsBadArg, ("duplicate axis %d in Tile", a));
            }
            axismask[a] = true;
            int r = rdata_i32 ? (int)rdata_i32[i] : rdata_i64 ? (int)rdata_i64[i] : 1;
            repeats[a] = r;
        }
    }

    MatShape getOutShape(const MatShape& inpshape, const int* repeats) const
    {
        MatShape outshape = inpshape;
        for (int i = 0; i < outshape.dims; i++)
            outshape[i] *= repeats[i];
        return outshape;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(!dynamicOutputShapes());

        size_t ninputs = inputs.size();
        CV_Assert(ninputs == (size_t)2 || ninputs == (size_t)3);
        Net::Impl* netimpl_ = getNetImpl(this);

        int repeats[TILE_MAX_DIMS];

        Mat repeatsTensor = netimpl_->argTensor(this->inputs[1]);
        Mat axesTensor;
        if (ninputs > 2)
            axesTensor = netimpl_->argTensor(this->inputs[2]);

        int ndims = inputs[0].dims;
        getRepeats(repeatsTensor, axesTensor, ndims, repeats);

        outputs.assign(1, getOutShape(inputs[0], repeats));
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
        CV_Assert(ninputs == (size_t)2 || ninputs == (size_t)3);
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
        CV_Assert(ninputs == 2 || ninputs == 3);

        Mat inp = inputs_arr.getMat(0);
        Mat repeatsTensor = inputs_arr.getMat(1);
        Mat axesTensor;
        int repeats[TILE_MAX_DIMS];
        int inptype = inp.type();
        int ndims = inp.dims;

        if (ninputs > 2)
            axesTensor = inputs_arr.getMat(2);

        getRepeats(repeatsTensor, axesTensor, ndims, repeats);
        MatShape outshape = getOutShape(inp.shape(), repeats);

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            tile(inp, repeats, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            tile(inp, repeats, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<Tile2Layer> Tile2Layer::create(const LayerParams& params)
{
    return Ptr<Tile2Layer>(new Tile2LayerImpl(params));
}

}
}
