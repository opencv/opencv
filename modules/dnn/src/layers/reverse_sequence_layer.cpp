// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ONNX ReverseSequence operator
// Spec: https://onnx.ai/onnx/operators/onnx__ReverseSequence.html
// Supported opsets: ai.onnx opset 10 and newer. Opset 28 only relaxed the type
// constraint from all_tensor_types() to all_tensor_types_ir4(), semantics are unchanged.
// NOTE: time_axis/batch_axis must be 0 or 1 and must differ; input rank must be >= 2.

#include "../precomp.hpp"

namespace cv {
namespace dnn {
class ReverseSequenceLayerImpl CV_FINAL : public ReverseSequenceLayer
{
public:
    explicit ReverseSequenceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        time_axis = params.get<int>("time_axis", 0);
        batch_axis = params.get<int>("batch_axis", 1);
        CV_CheckTrue(time_axis==0 || time_axis==1, "ReverseSequence: time axis must be 0 or 1");
        CV_CheckTrue(batch_axis==0 || batch_axis==1, "ReverseSequence: batch axis must be 0 or 1");
        CV_CheckTrue(time_axis!=batch_axis, "ReverseSequence: batch_axis and time_axis must differ");
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size()==2);
        CV_Assert(inputs[0].size()>=2);
        CV_Assert(inputs[1].size()==1);
        CV_Assert(requiredOutputs==1);
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        CV_Assert(inputs.size()==2);
        const int t = inputs[0];
        CV_CheckType(
            t,
            t==CV_16F || t==CV_32F || t==CV_64F|| t==CV_16BF ||
            t==CV_8U || t==CV_16U || t==CV_32U || t==CV_64U ||
            t==CV_8S || t==CV_16S || t==CV_32S || t==CV_64S ||
            t==CV_Bool,
            "ReverseSequence: unsupported input type");
        CV_CheckType(inputs[1], inputs[1]==CV_32S||inputs[1]==CV_64S,
                     "ReverseSequence: sequence_lens must be int64 or int32");
        outputs.assign(requiredOutputs, MatType(t));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size()==2);
        CV_Assert(!outputs.empty());

        const Mat& X = inputs[0];
        const Mat& lens = inputs[1];
        Mat& Y = outputs[0];

        const int rank = X.dims;
        CV_Assert(rank>=2);
        CV_Assert(X.isContinuous() && Y.isContinuous());

        const int batchAxis = batch_axis, timeAxis = time_axis;
        std::vector<size_t> stride(rank, 1);
        for (int i = rank - 2; i >= 0; --i)
        {
            stride[i] = stride[i + 1] * X.size[i + 1];
        }

        const uchar* src = X.ptr();
        uchar* dst = Y.ptr();
        const int nBatch = X.size[batchAxis];
        const int nTimeStep = X.size[timeAxis];
        const size_t elemSize = X.elemSize();
        const size_t block_size = stride[max(timeAxis, batchAxis)] * elemSize;
        const size_t timeStepSize = stride[timeAxis] * elemSize;
        const size_t batchStepSize = stride[batchAxis] * elemSize;

        CV_Assert(lens.dims==1);
        CV_Assert(lens.total()==(size_t)nBatch);

        parallel_for_(Range(0, nBatch), [&](const Range& r)
        {
            for (int b = r.start; b < r.end; ++b)
            {
                const int64_t reverseT = seqLenAt(lens, b);
                const size_t batchOffset = b * batchStepSize;
                CV_CheckTrue(0<=reverseT && reverseT <= nTimeStep, "ReverseSequence: sequence lens out of range");
                for (int64_t t = 0; t < reverseT; ++t)
                {
                    memcpy(dst + batchOffset + (reverseT - 1 - t) * timeStepSize,
                           src + batchOffset + t * timeStepSize, block_size);
                }
                for (int64_t t = reverseT; t < nTimeStep; ++t)
                {
                    memcpy(dst + batchOffset + t * timeStepSize,
                           src + batchOffset + t * timeStepSize, block_size);
                }
            }
        });
    }

    static int64_t seqLenAt(const Mat& lens, const int i)
    {
        if (lens.depth() == CV_32S)
        {
            return (int64_t) lens.ptr<int32_t>()[i];
        }
        if (lens.depth() == CV_64S)
        {
            return lens.ptr<int64_t>()[i];
        }

        CV_Error_(Error::StsNotImplemented,
                  ("ReverseSequence: unsupported sequence_lens depth %d (expected int64)", lens.depth()));

        return 0;
    }

private:
    int time_axis, batch_axis;
};

Ptr<ReverseSequenceLayer> ReverseSequenceLayer::create(const LayerParams& params)
{
    return Ptr<ReverseSequenceLayer>(new ReverseSequenceLayerImpl(params));
}
}
}
