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

/*
    Concat layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Concat.html

    Opset's 1 to 13 are covered.
*/

// out must be pre-allocated
static void concat(const std::vector<Mat>& inps, Mat& out, int axis)
{
    CV_Assert(out.isContinuous());

    MatShape outShape = out.shape();
    int ndims = outShape.dims, nslices = 1;
    size_t esz = out.elemSize();
    size_t sliceSize = esz;
    size_t totalSize = 0;
    size_t outStep = 0;
    int ninputs = (int)inps.size();
    for (int i = ndims-1; i > axis; i--)
        sliceSize *= outShape[i];
    outStep = sliceSize*outShape[axis];
    for (int i = 0; i < axis; i++)
        nslices *= outShape[i];
    for (int i = 0; i < ninputs; i++) {
        CV_Assert(inps[i].isContinuous());
        totalSize += inps[i].total()*esz;
    }

    // Precompute per-input destination offset and per-slice size.
    std::vector<size_t> dstOffset(ninputs);
    std::vector<size_t> sliceSize_k_vec(ninputs);
    {
        size_t acc = 0;
        for (int k = 0; k < ninputs; k++) {
            int sz_a = inps[k].size[axis];
            dstOffset[k] = acc;
            sliceSize_k_vec[k] = sliceSize * sz_a;
            acc += sliceSize_k_vec[k];
        }
    }
    const size_t CHUNK = 64 * 1024;

    // Precompute per-input chunk counts and a prefix sum for fast index decode.
    std::vector<int> chunkOff(ninputs + 1, 0);
    for (int k = 0; k < ninputs; k++)
        chunkOff[k + 1] = chunkOff[k] + (int)((sliceSize_k_vec[k] + CHUNK - 1) / CHUNK);
    int chunksPerSlice = chunkOff[ninputs];
    int totalChunks = chunksPerSlice * nslices;

    if (totalSize > CHUNK && totalChunks > 0) {
        parallel_for_(Range(0, totalChunks), [&](const Range& r) {
            for (int c = r.start; c < r.end; c++) {
                int s = c / chunksPerSlice;
                int local = c % chunksPerSlice;
                int k = 0;
                while (local >= chunkOff[k + 1]) k++;
                int chunkInK = local - chunkOff[k];
                size_t byteStart = (size_t)chunkInK * CHUNK;
                size_t byteEnd = std::min(byteStart + CHUNK, sliceSize_k_vec[k]);

                const uchar* inptr_k = inps[k].data;
                uchar* outptr = out.data + dstOffset[k];
                memcpy(outptr + (size_t)s * outStep + byteStart,
                       inptr_k + (size_t)s * sliceSize_k_vec[k] + byteStart,
                       byteEnd - byteStart);
            }
        });
    } else {
        for (int k = 0; k < ninputs; k++) {
            const uchar* inptr_k = inps[k].data;
            uchar* outptr = out.data + dstOffset[k];
            size_t sliceSize_k = sliceSize_k_vec[k];
            for (int s = 0; s < nslices; s++)
                memcpy(outptr + (size_t)s * outStep, inptr_k + (size_t)s * sliceSize_k, sliceSize_k);
        }
    }
}

class Concat2LayerImpl CV_FINAL : public Concat2Layer
{
public:
    Concat2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const std::vector<MatShape>& inpShapes) const
    {
        size_t ninputs = inpShapes.size();
        CV_Assert(ninputs == inputs.size());

        const MatShape& inpShape0 = inpShapes[0];
        int inpDims = inpShape0.dims;
        int axis_ = normalize_axis(axis, inpDims);
        CV_Assert(0 <= axis_ && axis_ < inpDims);
        MatShape outShape = inpShape0;
        outShape[axis_] = 0;

        for (size_t i = 0; i < ninputs; i++) {
            const MatShape& inpShape_i = inpShapes[i];
            CV_Assert(inpShape_i.dims == inpDims);
            for (int j = 0; j < inpDims; j++) {
                if (j == axis_) {
                    outShape[j] += inpShape_i[j];
                    continue;
                }
                CV_Assert(inpShape0[j] == inpShape_i[j]);
            }
        }

        // BLOCK axis=1: per-input c-block sum overcounts on misaligned inputs
        // (YOLOX 80+4+1: 12 vs correct ceil(85/8)=11).
        if (outShape.layout == DATA_LAYOUT_BLOCK && axis_ == 1) {
            int total_C = 0;
            for (size_t i = 0; i < ninputs; i++) total_C += inpShapes[i].C;
            int C0 = outShape[outShape.dims - 1];
            outShape[1] = (total_C + C0 - 1) / C0;
            outShape.C = total_C;
        }

        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, getOutShape(inputs));
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
        CV_Assert(ninputs > 0);
        for (size_t i = 1; i < ninputs; i++) {
            CV_Assert(inputs[i] == inputs[0]);
        }
        outputs.assign(requiredOutputs, inputs[0]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout defaultLayout = netimpl_->originalLayout;
        const size_t ninputs = actualInputs.size();
        desiredInputs = actualInputs;
        outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);

        bool allBlock = ninputs > 0;
        for (size_t i = 0; i < ninputs; ++i)
            if (actualInputs[i] != DATA_LAYOUT_BLOCK) { allBlock = false; break; }

        // BLOCK layout on the channel axis would expose inner-block padding as real channels; let TransformLayout repack instead.
        const bool canKeepBlock = allBlock && axis >= 0;

        if (canKeepBlock) {
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        } else {
            for (size_t i = 0; i < ninputs; ++i)
                if (actualInputs[i] == DATA_LAYOUT_BLOCK) desiredInputs[i] = defaultLayout;
        }
        return outputs[0] == DATA_LAYOUT_BLOCK ? netimpl_->defaultC0 : 0;
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

        CV_Assert(ninputs > 0);

        std::vector<MatShape> inpShapes(ninputs);
        int inpType = inputs_arr.type(0);

        for (int i = 0; i < ninputs; i++) {
            inpShapes[i] = inputs_arr.shape(i);
            CV_Assert(inputs_arr.type(i) == inpType);
        }

        MatShape outShape = getOutShape(inpShapes);
        int outKind = outputs_arr.kind();
        int axis_ = normalize_axis(axis, inpShapes[0].dims);

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat> inps;
            inputs_arr.getMatVector(inps);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inps, outs[0], axis_);
        } else {
            // [TODO] more efficient OpenCL implementation
            std::vector<Mat> inps;
            inputs_arr.getMatVector(inps);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inps, temp, axis_);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const std::vector<Mat>& inps, Mat& out, int axis_)
    {
        // Byte-level BLOCK concat needs every input's C to be a multiple of
        // C0; misaligned inputs (YOLOX 80+4+1) route through NCHW.
        if (axis_ == 1 && out.size.layout == DATA_LAYOUT_BLOCK) {
            const int C0 = out.size[out.dims - 1];
            bool allAligned = true;
            for (const auto& inp : inps) {
                if (inp.size.layout != DATA_LAYOUT_BLOCK || (inp.size.channels() % C0) != 0) {
                    allAligned = false;
                    break;
                }
            }
            if (!allAligned) {
                runOpBlockAxis1Misaligned(inps, out);
                return;
            }
        }
        concat(inps, out, axis_);
    }

    // Fallback for axis=1 BLOCK concat when inputs aren't C0-aligned.
    void runOpBlockAxis1Misaligned(const std::vector<Mat>& inps, Mat& out)
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout origLayout = netimpl_->originalLayout;
        const int C0 = out.size[out.dims - 1];

        std::vector<Mat> nchwInps(inps.size());
        for (size_t i = 0; i < inps.size(); i++) {
            transformLayout(inps[i], nchwInps[i], origLayout, origLayout, C0);
        }

        // NCHW output shape: drop the trailing C0 dim, use logical C from out.size.C.
        const int dims = out.dims;
        std::vector<int> nchwDims(dims - 1);
        nchwDims[0] = out.size[0];
        nchwDims[1] = out.size.C;
        for (int d = 2; d < dims - 1; d++) nchwDims[d] = out.size[d];
        MatShape nchwShape(nchwDims, origLayout, out.size.C);
        Mat nchwOut;
        nchwOut.fit(nchwShape, out.type());

        concat(nchwInps, nchwOut, /*axis=*/1);
        transformLayout(nchwOut, out, DATA_LAYOUT_BLOCK, origLayout, C0);
    }
};

Ptr<Concat2Layer> Concat2Layer::create(const LayerParams& params)
{
    return Ptr<Concat2Layer>(new Concat2LayerImpl(params));
}

}
}
