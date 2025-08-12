// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

class DetLayerImpl CV_FINAL : public DetLayer
{
public:
    DetLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
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
        CV_Assert(in.size() >= 2);

        int n0 = in[in.size() - 2];
        int n1 = in[in.size() - 1];
        CV_Assert(n0 == -1 || n1 == -1 || n0 == n1);

        MatShape out;
        if (in.size() > 2)
            out.assign(in.begin(), in.end() - 2);
        else
            out = MatShape({1});

        outputs.assign(1, out);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                const int requiredOutputs,
                const int requiredInternals,
                std::vector<MatType>& outputs,
                std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        int t = inputs[0];

        CV_Assert(t == CV_32F || t == CV_64F);
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

        CV_Assert(X.dims >= 2);
        int n = X.size[X.dims - 1];
        int m = X.size[X.dims - 2];
        CV_Assert(n == m);

        size_t batch = X.total() / (X.size[X.dims - 2] * X.size[X.dims - 1]);

        int outDims;
        std::vector<int> outSizes;
        if (X.dims > 2)
        {
            outDims = X.dims - 2;
            outSizes.assign(X.size.p, X.size.p + outDims);
        }
        else
        {
            outDims = 1;
            outSizes = {1};
        }
        outputs[0].create(outDims, outSizes.data(), X.type());

        const int type = X.type();
        const size_t elemSz = X.elemSize();
        const size_t matStrideBytes = (size_t)n * (size_t)m * elemSz;

        const uchar* base = X.data;
        uchar* outp = outputs[0].ptr();

        cv::parallel_for_(cv::Range(0, static_cast<int>(batch)), [&](const cv::Range& r){
            for (int bi = r.start; bi < r.end; ++bi)
            {
                size_t b = static_cast<size_t>(bi);
                const uchar* p = base + b * matStrideBytes;
                Mat A(m, n, type, const_cast<uchar*>(p));

                double det = cv::determinant(A);
                if (type == CV_32F)
                    reinterpret_cast<float*>(outp)[b] = static_cast<float>(det);
                else // CV_64F
                    reinterpret_cast<double*>(outp)[b] = det;
            }
        });
    }
};

Ptr<DetLayer> DetLayer::create(const LayerParams& params)
{
    return Ptr<DetLayer>(new DetLayerImpl(params));
}

}}
