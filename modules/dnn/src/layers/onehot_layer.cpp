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

class OneHotLayerImpl CV_FINAL : public OneHotLayer
{
public:
    OneHotLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", -1);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

private:
    bool getConstPositiveDepth(int& depthOut) const
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_ || !netimpl_->isConstArg(this->inputs[1]))
            return false;

        try
        {
            Mat depthTensor = netimpl_->argTensor(this->inputs[1]);
            if (!depthTensor.empty() && depthTensor.total() == 1)
            {
                int64_t depth64 = 0;
                tensorToScalar(depthTensor, CV_64S, &depth64);
                if (depth64 > 0)
                {
                    depthOut = (int)depth64;
                    return true;
                }
            }
        }
        catch (const cv::Exception& e)
        {
            CV_Error_(Error::StsError, ("OneHot: failed to access constant depth: %s", e.what()));
        }
        return false;
    }

    bool tryInferStaticOutputShape(const MatShape& idxShape, MatShape& outShape) const
    {
        outShape.clear();
        int depth = 0;
        if (getConstPositiveDepth(depth))
        {
            outShape = idxShape;
            int insAxis = normalize_axis(axis, (int)outShape.size() + 1);
            outShape.insert(outShape.begin() + insAxis, depth);
            return true;
        }
        return false;
    }

public:
    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        return !(netimpl_ && netimpl_->isConstArg(this->inputs[1]));
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        const MatShape& idxShape = inputs[0];
        CV_Assert(!idxShape.empty());

        MatShape outShape;
        if (tryInferStaticOutputShape(idxShape, outShape))
        {
            outputs.assign(1, outShape);
            internals.clear();
            return true;
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
        CV_Assert(inputs.size() == 3);
        int outType = (inputs[2] >= 0) ? inputs[2] : CV_32F;
        outputs.assign(requiredOutputs, outType);
        internals.assign(requiredInternals, MatType(-1));
    }

private:
    template<typename Tout>
    void runKernel(const Mat& indices, int depth, const Mat& values, Mat& Y)
    {
        CV_CheckGT(depth, 0, "OneHot: depth must be > 0");

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
            Mat flat = values.reshape(1, (int)values.total());
            offVal = tensorToScalar<Tout>(flat.row(0));
            onVal  = tensorToScalar<Tout>(flat.row(1));
        }

        Y.setTo(Scalar::all(offVal));

        Mat idx64;
        if (indices.depth() == CV_64S) idx64 = indices;
        else indices.convertTo(idx64, CV_64S);
        const int64_t* idxPtr64 = idx64.ptr<int64_t>();
        size_t inTotal = indices.total();
        size_t outStep = 1;
        for (int k = insAxis + 1; k < (int)outShape.size(); ++k)
            outStep *= (size_t)outShape[k];

        Tout* outPtr = Y.ptr<Tout>();

        parallel_for_(Range(0, (int)inTotal), [&](const Range& r){
            const int64_t* localIdxPtr64 = idxPtr64;
            Tout* localOutPtr = outPtr;
            const size_t localOutStep = outStep;
            const int localDepth = depth;
            const Tout localOnVal = onVal;
            for (int pos = r.start; pos < r.end; ++pos) {
                int64_t idxVal = localIdxPtr64[pos];
                int64_t wrapped = idxVal % localDepth;
                if (wrapped < 0) wrapped += localDepth;
                size_t hi = (size_t)pos / localOutStep;
                size_t lo = (size_t)pos % localOutStep;
                size_t base = hi * ((size_t)localDepth * localOutStep) + lo;
                localOutPtr[base + (size_t)wrapped * localOutStep] = localOnVal;
            }
        });
    }

public:
    void forward(InputArrayOfArrays in, OutputArrayOfArrays out, OutputArrayOfArrays) CV_OVERRIDE
    {
        int inKind = in.kind();
        int outKind = out.kind();
        CV_Assert(in.size().area() == 3);

        Mat indices, depthMat, values;
        if (inKind == _InputArray::STD_VECTOR_MAT)
        {
            indices = in.getMat(0);
            depthMat = in.getMat(1);
            values = in.getMat(2);
        }
        else if (inKind == _InputArray::STD_VECTOR_UMAT)
        {
            indices = in.getUMat(0).getMat(ACCESS_READ);
            depthMat = in.getUMat(1).getMat(ACCESS_READ);
            values = in.getUMat(2).getMat(ACCESS_READ);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "OneHot: unsupported input type");
        }

        std::vector<Mat>* out_mats = nullptr;
        std::vector<UMat>* out_umats = nullptr;
        Mat Y;
        if (outKind == _InputArray::STD_VECTOR_MAT)
        {
            out_mats = &out.getMatVecRef();
            out_mats->resize(1);
        }
        else if (outKind == _InputArray::STD_VECTOR_UMAT)
        {
            out_umats = &out.getUMatVecRef();
            out_umats->resize(1);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "OneHot: unsupported output type");
        }

        int outDepth = values.empty() ? CV_32F : values.depth();

        MatShape outShape = shape(indices);
        int ndims = (int)outShape.size();
        int insAxis = normalize_axis(axis, ndims + 1);

        CV_Assert(depthMat.total() == 1);
        int64_t depth64 = 0;
        tensorToScalar(depthMat, CV_64S, &depth64);
        CV_CheckGT(depth64, 0, "OneHot: depth must be > 0");
        MatShape finalShape;
        outShape.insert(outShape.begin() + insAxis, (int)depth64);
        finalShape = outShape;
        if (!finalShape.empty())
        {
            if (outKind == _InputArray::STD_VECTOR_MAT)
            {
                out_mats->at(0).fit(finalShape, outDepth);
                Y = out_mats->at(0);
            }
            else
            {
                out_umats->at(0).fit(finalShape, outDepth);
                Y = Mat(finalShape, outDepth);
            }
        }

        switch (outDepth) {
            case CV_8U:   runKernel<uchar>(indices, (int)depth64, values, Y); break;
            case CV_8S:   runKernel<schar>(indices, (int)depth64, values, Y); break;
            case CV_16U:  runKernel<uint16_t>(indices, (int)depth64, values, Y); break;
            case CV_16S:  runKernel<int16_t>(indices, (int)depth64, values, Y); break;
            case CV_32S:  runKernel<int>(indices, (int)depth64, values, Y); break;
            case CV_64S:  runKernel<int64_t>(indices, (int)depth64, values, Y); break;
            case CV_16F:  runKernel<hfloat>(indices, (int)depth64, values, Y); break;
            case CV_16BF: runKernel<bfloat>(indices, (int)depth64, values, Y); break;
            case CV_32F:  runKernel<float>(indices, (int)depth64, values, Y); break;
            case CV_64F:  runKernel<double>(indices, (int)depth64, values, Y); break;
            default: CV_Error(Error::BadDepth, "OneHot: unsupported output type");
        }

        if (outKind == _InputArray::STD_VECTOR_UMAT)
        {
            Y.copyTo(out_umats->at(0));
        }
    }
};

Ptr<OneHotLayer> OneHotLayer::create(const LayerParams& params)
{
    return makePtr<OneHotLayerImpl>(params);
}

}} // namespace cv::dnn
