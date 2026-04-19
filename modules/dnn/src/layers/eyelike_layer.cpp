// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

// ONNX EyeLike operator
// Spec: https://onnx.ai/onnx/operators/onnx__EyeLike.html
// Supported opsets: 9-22

class EyeLikeLayerImpl CV_FINAL : public EyeLikeLayer
{
public:
    int k; // diagonal offset
    int outputDtype; // -1 means use input dtype

    EyeLikeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        k = params.get<int>("k", 0);
        outputDtype = params.get<int>("dtype", -1);
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
        CV_Assert(inputs.size() == 1);
        const MatShape& in = inputs[0];
        CV_Assert(in.size() == 2);
        outputs.assign(1, in);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                const int requiredOutputs,
                const int requiredInternals,
                std::vector<MatType>& outputs,
                std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        int t = (outputDtype >= 0) ? outputDtype : inputs[0];
        outputs.assign(requiredOutputs, MatType(t));
        internals.assign(requiredInternals, MatType(t));
    }

    void forward(InputArrayOfArrays inputs_arr,
             OutputArrayOfArrays outputs_arr,
             OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 1);
        const Mat& X = inputs[0];
        CV_Assert(X.dims == 2);

        int rows = X.size[0];
        int cols = X.size[1];

        int outType = (outputDtype >= 0) ? outputDtype : X.type();
        outputs[0].create(rows, cols, outType);
        Mat& Y = outputs[0];
        Y.setTo(Scalar::all(0));

        // Set ones on the k-th diagonal: Y[i, i+k] = 1
        int iStart = (k >= 0) ? 0 : -k;
        int jStart = (k >= 0) ? k : 0;
        int diagLen = std::min(rows - iStart, cols - jStart);

        switch (outType)
        {
        case CV_32F:  fillDiag<float>   (Y, iStart, jStart, diagLen, 1.0f);         break;
        case CV_64F:  fillDiag<double>  (Y, iStart, jStart, diagLen, 1.0);          break;
        case CV_32S:  fillDiag<int32_t> (Y, iStart, jStart, diagLen, 1);            break;
        case CV_64S:  fillDiag<int64_t> (Y, iStart, jStart, diagLen, 1);            break;
        case CV_8U:   fillDiag<uint8_t> (Y, iStart, jStart, diagLen, 1);            break;
        case CV_8S:   fillDiag<int8_t>  (Y, iStart, jStart, diagLen, 1);            break;
        case CV_16U:  fillDiag<uint16_t>(Y, iStart, jStart, diagLen, 1);            break;
        case CV_16S:  fillDiag<int16_t> (Y, iStart, jStart, diagLen, 1);            break;
        case CV_16F:  fillDiag<hfloat>  (Y, iStart, jStart, diagLen, hfloat(1.0f)); break;
        case CV_16BF: fillDiag<bfloat>  (Y, iStart, jStart, diagLen, bfloat(1.0f)); break;
        case CV_Bool: fillDiag<bool>    (Y, iStart, jStart, diagLen, true);         break;
        default:
            CV_Error(Error::BadDepth, "Unsupported output depth for EyeLikeLayer");
        }
    }

private:
    template<typename T>
    static void fillDiag(Mat& Y, int iStart, int jStart, int diagLen, T one)
    {
        for (int d = 0; d < diagLen; ++d)
            Y.at<T>(iStart + d, jStart + d) = one;
    }
};

Ptr<EyeLikeLayer> EyeLikeLayer::create(const LayerParams& params)
{
    return Ptr<EyeLikeLayer>(new EyeLikeLayerImpl(params));
}

}}
