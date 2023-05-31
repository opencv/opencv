// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"

#include <algorithm>
#include <stdlib.h>

namespace cv
{
namespace dnn
{

class SoftMaxLayerInt8Impl CV_FINAL : public SoftmaxLayerInt8
{
public:
    float input_sc;
    int input_zp;

    SoftMaxLayerInt8Impl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 1);
        logSoftMax = params.get<bool>("log_softmax", false);
        coerced_2d = params.get<bool>("coerced_2d", false);

        input_sc = params.get<float>("input_scale");
        input_zp = params.get<int>("input_zeropoint");

        output_sc = params.get<float>("scales");
        output_zp = params.get<int>("zeropoints");

        if (blobs.empty()) // if no lookUpTable is found
        {
            Mat lookUpTable(1, 256, CV_32F);
            float* table = lookUpTable.ptr<float>();
            for (int i = -128; i < 128; i++)
            {
                float x = input_sc * (i - 127); // ensures exp(x) is always between (0, 1)
                table[i + 128] = std::exp(x);
            }
            blobs.push_back(lookUpTable);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        MatShape shape = inputs[0];
        internals.assign(1, shape);
        return inplace;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        auto src = inputs[0];
        auto dims_src = src.dims;
        auto shape_src = shape(src);
        axis = normalize_axis(axis, dims_src);

        if (!coerced_2d) {
            is_transpose_needed = (axis == dims_src - 1) ? false : true;
            if (is_transpose_needed) {
                permutation.resize(dims_src);
                std::iota(permutation.begin(), permutation.end(), 0);
                permutation[axis] = dims_src - 1;
                permutation[dims_src - 1] = axis;

                transposed_shape.resize(dims_src);
                std::transform(permutation.begin(), permutation.end(), transposed_shape.begin(), [&shape_src](int axis) { return shape_src[axis]; });
                N = std::accumulate(transposed_shape.begin(), transposed_shape.end() - 1, 1, std::multiplies<int>());
                D = transposed_shape.back();

                return;
            }
        }

        N = src.total(0, axis);
        D = src.total(axis);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
            (backendId == DNN_BACKEND_TIMVX && haveTimVX());
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Ptr<DequantizeLayer> dequantize_layer = top.dynamicCast<DequantizeLayer>();
        return !dequantize_layer.empty() && preferableTarget != DNN_TARGET_OPENCL_FP16;
    }

   virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index, output_index;

        // input Tensor
        CV_Assert(inputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        const Mat &src = inputWrapper->getMat();

        // convert axis from OpenCV NCHW toTimVX WHCN.
        int tvAxis = src.dims - 1 - normalize_axis(axis, src.dims);
        if(tvAxis < 0)
            tvAxis = 0; // default value is 0.

        if (inputWrapper->isTensor())
        {
            input_index = tvGraph->getTensorIndex(inputWrapper->getTensor());
            if (input_index == -1)
            {
                // Copy To New inputWrapper
                Mat tmp = inputWrapper->getMat();
                inputWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tmp));
            }
        }

        if (!inputWrapper->isTensor())
        {
            Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, input_sc, input_zp));
            inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT, tvInputQuant);
            input_index = tvGraph->addWrapper(inputWrapper);
        }
        inputsIndex.push_back(input_index);

        // output tensor
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Mat dstMat = outputWrapper->getMat();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        Ptr<tim::vx::Tensor> outputTensor;

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            if (dstMat.type() == CV_32F)
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT);
            else
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            if (dstMat.type() == CV_32F)
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT);
            else
                outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvSoftmax;

        if (logSoftMax)
        {
            tvSoftmax = graph->CreateOperation<tim::vx::ops::LogSoftmax>(tvAxis);

        }
        else
        {
            tvSoftmax = graph->CreateOperation<tim::vx::ops::Softmax>(1.0f, tvAxis);
        }

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvSoftmax, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

    template <bool with_log>
    class SoftmaxInt8Invoker : public ParallelLoopBody {
    public:
        const Mat& src_;
        Mat& dst_;

        const Mat& lookup_table_;

        int N_;
        int D_;

        float y_scale_;
        int y_zero_point_;

        int threads;
        int cost_per_thread;

        SoftmaxInt8Invoker(const Mat& src, Mat& dst, const Mat& lookup_table, int N, int D, float y_scale, int y_zero_point)
            : src_(src), dst_(dst), lookup_table_(lookup_table), N_(N), D_(D), y_scale_(1.f / y_scale), y_zero_point_(y_zero_point) {
            threads = N_;
            cost_per_thread = D_;
        }

        static void run(const Mat& src, Mat& dst, const Mat& lookup_table, int N, int D, float y_scale, int y_zero_point) {
            CV_Assert(src.isContinuous());
            CV_Assert(dst.isContinuous());
            CV_CheckTypeEQ(src.type(), CV_8S, "DNN/SoftmaxInt8: type of input must be int8");
            CV_CheckTypeEQ(dst.type(), CV_8S, "DNN/SoftmaxInt8: type of output must be int8");

            SoftmaxInt8Invoker p(src, dst, lookup_table, N, D, y_scale, y_zero_point);

            double nstripes = ((size_t)p.threads * p.cost_per_thread) * (1 / 1024.0);
            parallel_for_(Range(0, p.threads), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE {
            int start = r.start;
            int end = r.end;

            const int8_t* p_src = src_.ptr<int8_t>();
            int8_t* p_dst = dst_.ptr<int8_t>();
            const float* table = lookup_table_.ptr<float>();

            for (int i = start; i < end; ++i) {
                const int8_t* x = p_src + i * D_;
                int8_t* y = p_dst + i * D_;

                float vsum = 0;
                for (int j = 0; j < D_; ++j) {
                    const uint8_t idx = uint8_t((*x++) + 128);
                    vsum += table[idx];
                }

                // FIXME: avoid divide by vsum==0

                x = p_src + i * D_;
                if (with_log) {
                    for (int j = 0; j < D_; ++j) {
                        const uint8_t idx = uint8_t((*x++) + 128);
                        const float v = table[idx];
                        *y++ = saturate_cast<int8_t>(std::nearbyintf(y_scale_ * std::log(v / vsum)) + y_zero_point_);
                    }
                } else {
                    for (int j = 0; j < D_; ++j) {
                        const uint8_t idx = uint8_t((*x++) + 128);
                        const float v = table[idx];
                        *y++ = saturate_cast<int8_t>(std::nearbyintf(y_scale_ * v / vsum) + y_zero_point_);
                    }
                }
            }
        }
    };

    template <bool with_log>
    class SoftmaxInt8OutputFloatInvoker : public ParallelLoopBody {
    public:
        const Mat& src_;
        Mat& dst_;

        const Mat& lookup_table_;

        int N_;
        int D_;

        int threads;
        int cost_per_thread;

        SoftmaxInt8OutputFloatInvoker(const Mat& src, Mat& dst, const Mat& lookup_table, int N, int D)
            : src_(src), dst_(dst), lookup_table_(lookup_table), N_(N), D_(D) {
            threads = N_;
            cost_per_thread = D_;
        }

        static void run(const Mat& src, Mat& dst, const Mat& lookup_table, int N, int D) {
            CV_Assert(src.isContinuous());
            CV_Assert(dst.isContinuous());
            CV_CheckTypeEQ(src.type(), CV_8S, "DNN/SoftmaxInt8: type of input must be int8");
            CV_CheckTypeEQ(dst.type(), CV_32F, "DNN/SoftmaxInt8: type of input must be float32 since Dequantization is fused");

            SoftmaxInt8OutputFloatInvoker p(src, dst, lookup_table, N, D);

            double nstripes = ((size_t)p.threads * p.cost_per_thread) * (1 / 1024.0);
            parallel_for_(Range(0, p.threads), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE {
            int start = r.start;
            int end = r.end;

            const int8_t* p_src = src_.ptr<int8_t>();
            float* p_dst = dst_.ptr<float>();
            const float* table = lookup_table_.ptr<float>();

            for (int i = start; i < end; ++i) {
                const int8_t* x = p_src + i * D_;
                float* y = p_dst + i * D_;

                float vsum = 0;
                for (int j = 0; j < D_; ++j) {
                    const uint8_t idx = uint8_t((*x++) + 128);
                    vsum += table[idx];
                }

                // FIXME: avoid divide by vsum==0

                x = p_src + i * D_;
                if (with_log) {
                    for (int j = 0; j < D_; ++j) {
                        const uint8_t idx = uint8_t((*x++) + 128);
                        const float v = table[idx];
                        *y++ = std::log(v / vsum);
                    }
                } else {
                    for (int j = 0; j < D_; ++j) {
                        const uint8_t idx = uint8_t((*x++) + 128);
                        const float v = table[idx];
                        *y++ = v / vsum;
                    }
                }
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat src, dst;

        if (!coerced_2d && is_transpose_needed) {
            transposeND(inputs[0], permutation, src);
            dst = Mat::zeros(transposed_shape.size(), transposed_shape.data(), outputs[0].type());
        } else {
            src = inputs[0];
            dst = outputs[0];
        }

        switch (dst.type()) {
            case CV_8S: {
                if (logSoftMax) {
                    SoftmaxInt8Invoker<true>::run(src, dst, blobs[0], N, D, output_sc, output_zp);
                } else {
                    SoftmaxInt8Invoker<false>::run(src, dst, blobs[0], N, D, output_sc, output_zp);
                }
            } break;
            case CV_32F: {
                if (logSoftMax) {
                    SoftmaxInt8OutputFloatInvoker<true>::run(src, dst, blobs[0], N, D);
                } else {
                    SoftmaxInt8OutputFloatInvoker<false>::run(src, dst, blobs[0], N, D);
                }
            } break;
            default: CV_Error(cv::Error::BadDepth, "DNN/SoftmaxInt8: Unsupported output type");
        }

        if (!coerced_2d && is_transpose_needed) {
            transposeND(dst, permutation, outputs[0]);
        }
    }

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        int64 flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 4*total(inputs[i]);
        }

        return flops;
    }

private:
    int axis;
    int N;
    int D;
    bool coerced_2d;
    bool is_transpose_needed;
    std::vector<int> permutation;
    std::vector<int> transposed_shape;
};

Ptr<SoftmaxLayerInt8> SoftmaxLayerInt8::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayerInt8>(new SoftMaxLayerInt8Impl(params));
}

}
}
