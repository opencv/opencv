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
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size()==2);
        CV_Assert(inputs[0].size()>=2);
        CV_Assert(inputs[1].size()==1);
        CV_CheckEQ(inputs[1][0], inputs[0][batch_axis],
                   "ReverseSequence: sequence_lens must contain one length per batch element");
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size()==2);
        CV_CheckType(inputs[1], inputs[1]==CV_32S||inputs[1]==CV_64S,
                     "ReverseSequence: sequence_lens must be int64 or int32");
        outputs.assign(max(1, requiredOutputs), inputs[0]);
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
        const int nBatch = X.size[batchAxis], nTimeStep = X.size[timeAxis];

        std::vector<int64_t> seqLen(nBatch);
        const int lensDepth = lens.depth();
        for (int b = 0; b < nBatch; ++b)
        {
            seqLen[b] = lensDepth == CV_32S ? (int64_t) lens.ptr<int32_t>()[b] : lens.ptr<int64_t>()[b];
            CV_CheckTrue(seqLen[b] >= 0 && seqLen[b] <= (int64_t) nTimeStep,
                         "ReverseSequence: sequence_lens values must be in [0, nTimeStep]");
        }

        const uchar* src = X.ptr();
        uchar* dst = Y.ptr();
        const size_t batchStep = X.step[batchAxis];
        const size_t timeStep = X.step[timeAxis];
        const size_t blockSize = X.step[1];
        const int nTasks = nBatch * nTimeStep;

        parallel_for_(Range(0, nTasks), [&](const Range& r)
        {
            for (int i = r.start; i < r.end; ++i)
            {
                const int b = i / nTimeStep, t = i % nTimeStep;
                const int L = seqLen[b];
                const size_t dstT = (t < L ? L - 1 - t : t);
                memcpy(dst + (size_t) b * batchStep + dstT * timeStep,
                       src + (size_t) b * batchStep + (size_t) t * timeStep,
                       blockSize);
            }
        });
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
