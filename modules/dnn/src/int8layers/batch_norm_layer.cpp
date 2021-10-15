// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
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
