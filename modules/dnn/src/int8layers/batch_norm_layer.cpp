// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class BatchNormLayerInt8Impl CV_FINAL : public BatchNormLayerInt8
{
public:
    Mat origin_weights, origin_bias;
    Mat weights_, bias_;
    mutable int dims;

    BatchNormLayerInt8Impl(const LayerParams& params)
        : dims(-1)
    {
        setParamsFrom(params);
        useGlobalStats = params.get<bool>("use_global_stats", true);
        input_sc = params.get<float>("input_scale");
        input_zp = params.get<int>("input_zeropoint");
        output_sc = params.get<float>("scales");
        output_zp = params.get<int>("zeropoints");

        CV_Assert(blobs.size() == 2);
        size_t n = blobs[0].total();
        CV_Assert(blobs[1].total() == n &&
                  blobs[0].isContinuous() && blobs[1].isContinuous() &&
                  blobs[0].type() == CV_32F && blobs[1].type() == CV_32F);

        origin_weights = blobs[0];
        origin_bias = blobs[1];
    }

    virtual void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
        origin_weights.convertTo(weights_, CV_32F, input_sc/output_sc);
        addWeighted(origin_bias, 1.0/output_sc, weights_, -input_zp, output_zp, bias_, CV_32F);
    }

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = origin_weights;
        shift = origin_bias;
    }

    void getScaleZeropoint(float& scale, int& zeropoint) const CV_OVERRIDE
    {
        scale = output_sc;
        zeropoint = output_zp;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Mat w_, b_;
        top->getScaleShift(w_, b_);
        if (w_.empty() && b_.empty())
            return false;

        const int numChannels = weights_.total();
        const int numFusedWeights = w_.total();
        const int numFusedBias = b_.total();

        if ((numFusedWeights != numChannels && numFusedWeights != 1 && !w_.empty()) ||
            (numFusedBias != numChannels && numFusedBias != 1 && !b_.empty()))
            return false;

        float new_sc;
        int new_zp;
        top->getScaleZeropoint(new_sc, new_zp);

        Mat w = numFusedWeights == 1 ? Mat(1, numChannels, CV_32F, Scalar(w_.at<float>(0))) :
                (w_.empty() ? Mat::ones(1, numChannels, CV_32F) : w_.reshape(1, 1));

        Mat b = numFusedBias == 1 ? Mat(1, numChannels, CV_32F, Scalar(b_.at<float>(0))) :
                (b_.empty() ? Mat::zeros(1, numChannels, CV_32F) : b_.reshape(1, 1));

        weights_ = Mat(); bias_ = Mat();
        multiply(origin_weights, w, weights_, input_sc/new_sc, CV_32F);
        multiply(origin_bias, w, bias_);
        add(bias_, b, bias_);
        addWeighted(bias_, 1.0/new_sc, weights_, -input_zp, new_zp, bias_, CV_32F);
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        dims = inputs[0].size();
        if (!useGlobalStats && inputs[0][0] != 1)
            CV_Error(Error::StsNotImplemented, "Batch normalization in training mode with batch size > 1");
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_TIMVX && haveTimVX())
        {
            return true;
        }

        return backendId == DNN_BACKEND_OPENCV;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            return activ_int8->blobs.empty();
        }
        return false;
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

        const int numChannels = (int)origin_bias.total();
        Mat tvGamma = origin_weights.reshape(1, numChannels);
        Mat tvBeta = origin_bias.reshape(1, numChannels);

        std::vector<int> inputsIndex;
        std::vector<int> outputsIndex;

        Mat tvMean = Mat::zeros(1, numChannels, CV_32F);
        tvMean = tvMean.reshape(1, numChannels);
        Mat tvVar = Mat::ones(1, numChannels, CV_32F);
        tvVar = tvVar.reshape(1, numChannels);

        CV_Assert(inputsWrapper.size() == 1);
        if (outputsWrapper.size() > 1)
            return Ptr<BackendNode>();

        Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, input_sc, input_zp));

        // input Tensor
        auto inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Mat tmpInput = inputWrapper->getMat();

        if (tmpInput.dims != 4)  // Only support 4 dim input.
            return Ptr<BackendNode>();

        int input_index = -1, mean_index = -1, var_index = -1, gamma_index = -1, beta_index = -1, output_index = -1;

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
            inputWrapper->createTensor(graph,tim::vx::TensorAttribute::INPUT, tvInputQuant);
            input_index = tvGraph->addWrapper(inputWrapper);
        }
        inputsIndex.push_back(input_index);

        // Mean tensor
        Ptr<TimVXBackendWrapper> meanWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvMean));
        Ptr<tim::vx::Quantization> meanQuant;
        meanWrapper->createTensor(graph, tim::vx::TensorAttribute::CONSTANT);
        mean_index = tvGraph->addWrapper(meanWrapper);
        inputsIndex.push_back(mean_index);

        // Var tensor
        Ptr<TimVXBackendWrapper> varWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvVar));
        varWrapper->createTensor(graph,tim::vx::TensorAttribute::CONSTANT);
        var_index = tvGraph->addWrapper(varWrapper);
        inputsIndex.push_back(var_index);

        // Gamma tensor
        Ptr<TimVXBackendWrapper> gammaWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvGamma));
        gammaWrapper->createTensor(graph,tim::vx::TensorAttribute::CONSTANT);
        gamma_index = tvGraph->addWrapper(gammaWrapper);
        inputsIndex.push_back(gamma_index);

        // Beta tensor
        Ptr<TimVXBackendWrapper> betaWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tvBeta));
        betaWrapper->createTensor(graph,tim::vx::TensorAttribute::CONSTANT);
        beta_index = tvGraph->addWrapper(betaWrapper);
        inputsIndex.push_back(beta_index);

        // Output tensor
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }

        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvBatchNorm = graph->CreateOperation<tim::vx::ops::BatchNorm>(0.f);

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvBatchNorm, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(blobs.size() == 2);
        CV_Assert(inputs.size() == 1);

        Mat &inpBlob = inputs[0];
        int planeSize = 1;
        for (size_t i = 2; i < inpBlob.dims; i++) {
            planeSize *= inpBlob.size[i];
        }

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &outBlob = outputs[ii];

            for(int num = 0; num < outBlob.size[0]; num++)
            {
                for (int n = 0; n < outBlob.size[1]; n++)
                {
                    float w = weights_.at<float>(n);
                    float b = bias_.at<float>(n);
                    Mat inpBlobPlane(1, planeSize, CV_8S, inpBlob.ptr<int8_t>(num, n));
                    Mat outBlobPlane(1, planeSize, CV_8S, outBlob.ptr<int8_t>(num, n));
                    inpBlobPlane.convertTo(outBlobPlane, CV_8S, w, b);
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 3*total(inputs[i]);
        }
        return flops;
    }

private:
    bool useGlobalStats;
};

Ptr<BatchNormLayerInt8> BatchNormLayerInt8::create(const LayerParams& params)
{
    return Ptr<BatchNormLayerInt8>(new BatchNormLayerInt8Impl(params));
}

}  // namespace dnn
}  // namespace cv
