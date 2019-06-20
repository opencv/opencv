// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Instance Normalization layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

class InstanceNormLayerImpl CV_FINAL : public InstanceNormLayer
{
public:
    Mat weights_, bias_;
    UMat umat_weight, umat_bias;
    mutable int dims;
    float epsilon_;


    InstanceNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 2);

        hasWeights = params.get<bool>("has_weight", false);
        hasBias = params.get<bool>("has_bias", false);
        epsilon_ = params.get<float>("eps", 1E-5);

        size_t n = blobs[0].total();
        CV_Assert(blobs[1].total() == n &&
                  blobs[0].isContinuous() && blobs[1].isContinuous() &&
                  blobs[0].type() == CV_32F && blobs[1].type() == CV_32F);

        const int weightsBlobIndex = 0;
        const int biasBlobIndex = 1;

        if( hasWeights )
        {
            CV_Assert((size_t)weightsBlobIndex < blobs.size());
            const Mat& w = blobs[weightsBlobIndex];
            CV_Assert(w.isContinuous() && w.type() == CV_32F && w.total() == (size_t)n);
        }

        if( hasBias )
        {
            CV_Assert((size_t)biasBlobIndex < blobs.size());
            const Mat& b = blobs[weightsBlobIndex];
            CV_Assert(b.isContinuous() && b.type() == CV_32F && b.total() == (size_t)n);
        }

        const float* weightsData = hasWeights ? blobs[weightsBlobIndex].ptr<float>() : 0;
        const float* biasData = hasBias ? blobs[biasBlobIndex].ptr<float>() : 0;

        weights_.create(1, (int)n, CV_32F);
        bias_.create(1, (int)n, CV_32F);

        float* dstWeightsData = weights_.ptr<float>();
        float* dstBiasData = bias_.ptr<float>();

        for (size_t i = 0; i < n; ++i)
        {
            dstWeightsData[i] = (hasWeights ? weightsData[i] : 1.0f);
            dstBiasData[i] = (hasBias ? biasData[i] : 0.0f);
        }
    }

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = weights_;
        shift = bias_;
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
        return (backendId == DNN_BACKEND_OPENCV);
    }


    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(blobs.size() >= 2);
        CV_Assert(inputs.size() == 1);

        Mat &inpBlob = inputs[0];
        int planeSize = 1;
        for (size_t i = 2; i < inpBlob.dims; i++) {
            planeSize *= inpBlob.size[i];
        }

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &outBlob = outputs[ii];

            for(int n = 0; n < outBlob.size[0]; n++)
            {
                //Iterate over channels
                for (int c = 0; c < outBlob.size[1]; c++)
                {
                    float w = weights_.at<float>(c);
                    float b = bias_.at<float>(c);
                    Mat inpBlobPlane(1, planeSize, CV_32F, inpBlob.ptr<float>(n, c));
                    Mat outBlobPlane(1, planeSize, CV_32F, outBlob.ptr<float>(n, c));

                    Mat mean(1, planeSize,CV_32F), stdv(1, planeSize,CV_32F);

                    meanStdDev(inpBlobPlane, mean, stdv); // std and mean per channel
                    subtract(inpBlobPlane, mean, outBlobPlane); //Y = X - Mean
                    add(stdv, epsilon_, stdv); // stdv = variance + epsilon
                    sqrt(stdv, stdv); // stdv = sqrt(variance + epsilon)
                    add(stdv, b, stdv); // stdv = sqrt(variance + epsilon) + B
                    multiply(outBlobPlane, w, outBlobPlane); //Y = W * (X - mean)
                    divide(outBlobPlane, stdv, outBlobPlane); //Y /= stdv

                }
            }
        }
    }

    void forwardSlice(const float* srcptr, float* dstptr, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        return;
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

Ptr<InstanceNormLayer> InstanceNormLayer::create(const LayerParams& params)
{
    return Ptr<InstanceNormLayer>(new InstanceNormLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
