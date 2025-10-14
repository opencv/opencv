// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <atomic>

namespace cv { namespace dnn {

/*
    OneHot layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__OneHot.html
    Supported opsets: 9-11
*/

template<typename T>
static inline T getScalarAsT(const Mat& m, int idx)
{
    CV_Assert(idx >= 0 && idx < (int)m.total());
    Mat flat = m.reshape(1, (int)m.total());
    T value = T();
    tensorToScalar(flat.row(idx), DataType<T>::type, &value);
    return value;
}

class OneHotLayerImpl CV_FINAL : public OneHotLayer
{
public:
    int axis;

    OneHotLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", -1);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return false;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        const MatShape& idxShape = inputs[0];
        CV_Assert(!idxShape.empty());

        try
        {
            if ((int)this->inputs.size() >= 2)
            {
                Net::Impl* netimpl_ = getNetImpl(this);
                Mat depthTensor = netimpl_->argTensor(this->inputs[1]);
                if (!depthTensor.empty() && depthTensor.total() == 1)
                {
                    int64_t depth64 = 0;
                    tensorToScalar(depthTensor, CV_64S, &depth64);
                    if (depth64 > 0)
                    {
                        MatShape outShape = idxShape;
                        int insAxis = normalize_axis(axis, (int)outShape.size() + 1);
                        outShape.insert(outShape.begin() + insAxis, (int)depth64);
                        outputs.assign(1, outShape);
                        internals.clear();
                        return true;
                    }
                }
            }
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_DEBUG(NULL, "OneHot: getMemoryShapes dynamic fallback (cv::Exception): " << e.what());
        }

        outputs.assign(1, MatShape());
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        int outType = CV_32F;
        if (inputs.size() >= 3 && inputs[2] >= 0)
            outType = inputs[2];
        outputs.assign(requiredOutputs, outType);
        internals.assign(requiredInternals, MatType(-1));
    }

private:
    template<typename Tout>
    void runKernel(const Mat& indices, const Mat& depthMat, const Mat& values, Mat& Y)
    {
        int64_t depth64 = 0;
        tensorToScalar(depthMat, CV_64S, &depth64);
        CV_CheckGT(depth64, 0, "OneHot: depth must be > 0");
        int depth = (int)depth64;

        MatShape outShape = shape(indices);
        int ndims = (int)outShape.size();
        int insAxis = normalize_axis(axis, ndims + 1);
        outShape.insert(outShape.begin() + insAxis, depth);

        if (Y.empty() || Y.type() != DataType<Tout>::type || Y.shape() != outShape)
            Y.create(outShape, DataType<Tout>::type);

        Tout offVal = (Tout)0;
        Tout onVal  = (Tout)1;
        if (!values.empty())
        {
            CV_Assert(values.total() == 2);
            offVal = getScalarAsT<Tout>(values, 0);
            onVal  = getScalarAsT<Tout>(values, 1);
        }

        Y.setTo(Scalar::all(offVal));

        Mat idx64;
        if (indices.depth() == CV_64S) idx64 = indices;
        else indices.convertTo(idx64, CV_64S);
        const int64_t* idxPtr64 = idx64.ptr<int64_t>();

        std::vector<size_t> strides(outShape.size());
        size_t stride = 1;
        for (int k = (int)outShape.size() - 1; k >= 0; --k) {
            strides[k] = stride;
            stride *= (size_t)outShape[k];
        }

        auto rankIdx = shape(indices);
        size_t totalIdx = indices.total();
        Tout* outPtr = Y.ptr<Tout>();

        std::vector<size_t> mappedStrides(rankIdx.size());
        for (int d = 0; d < (int)rankIdx.size(); ++d) {
            int outDim = d < insAxis ? d : d + 1;
            mappedStrides[d] = strides[outDim];
        }

        std::atomic<int64_t> numSet(0), numNegWrapped(0), numOOR(0);

        parallel_for_(Range(0, (int)totalIdx), [&](const Range& r){
            int64_t localSet = 0, localNeg = 0, localOOR = 0;
            for (int i = r.start; i < r.end; ++i) {
                int rem = i;
                size_t base = 0;
                for (int d = (int)rankIdx.size() - 1; d >= 0; --d) {
                    int dim = rankIdx[d];
                    int coord = rem % dim;
                    rem /= dim;
                    base += (size_t)coord * mappedStrides[d];
                }
                int64_t idxVal = idxPtr64[i];
                bool wasOutOfRange = (idxVal < 0 || idxVal >= depth);
                if (idxVal < 0) { localNeg++; }
                if (wasOutOfRange) { localOOR++; }
                int64_t wrapped = idxVal % depth;
                if (wrapped < 0) wrapped += depth;
                size_t pos = base + strides[insAxis] * (size_t)wrapped;
                outPtr[pos] = onVal;
                localSet++;
            }
            numSet += localSet;
            numNegWrapped += localNeg;
            numOOR += localOOR;
        });
    }

public:
    void forward(InputArrayOfArrays in, OutputArrayOfArrays out, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs; in.getMatVector(inputs); out.getMatVector(outputs);
        CV_Assert(inputs.size() >= 2);
        CV_Assert(outputs.size() == 1);

        const Mat& indices = inputs[0];
        const Mat& depthMat = inputs[1];
        Mat values; if (inputs.size() >= 3) values = inputs[2];
        Mat& Y = outputs[0];

        int outDepth = values.empty() ? CV_32F : values.depth();

        MatShape outShape = shape(indices);
        int ndims = (int)outShape.size();
        int insAxis = normalize_axis(axis, ndims + 1);

        MatShape finalShape;
        if (depthMat.total() == 1)
        {
            int64_t depth64 = 0;
            tensorToScalar(depthMat, CV_64S, &depth64);
            if (depth64 > 0)
            {
                outShape.insert(outShape.begin() + insAxis, (int)depth64);
                finalShape = outShape;
            }
        }
        if (!finalShape.empty())
        {
            outputs[0].fit(finalShape, outDepth);
        }

        switch (outDepth) {
            case CV_8U:   runKernel<uchar>(indices, depthMat, values, Y); break;
            case CV_8S:   runKernel<schar>(indices, depthMat, values, Y); break;
            case CV_16U:  runKernel<uint16_t>(indices, depthMat, values, Y); break;
            case CV_16S:  runKernel<int16_t>(indices, depthMat, values, Y); break;
            case CV_32S:  runKernel<int>(indices, depthMat, values, Y); break;
            case CV_64S:  runKernel<int64_t>(indices, depthMat, values, Y); break;
            case CV_16F:  runKernel<hfloat>(indices, depthMat, values, Y); break;
            case CV_16BF: runKernel<bfloat>(indices, depthMat, values, Y); break;
            case CV_32F:  runKernel<float>(indices, depthMat, values, Y); break;
            case CV_64F:  runKernel<double>(indices, depthMat, values, Y); break;
            default: CV_Error(Error::BadDepth, "OneHot: unsupported output type");
        }
    }
};

Ptr<OneHotLayer> OneHotLayer::create(const LayerParams& params)
{
    return makePtr<OneHotLayerImpl>(params);
}

}} // namespace cv::dnn
