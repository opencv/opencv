// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

// ONNX NonZero operator
// Spec: https://onnx.ai/onnx/operators/onnx__NonZero.html
// Supported opsets: 9-18 (no attributes)

namespace {

template <typename T, typename WT = T>
static inline bool isNonZero(const T v)
{
    return (WT)v != (WT)0;
}
}

class NonZeroLayerImpl CV_FINAL : public NonZeroLayer
{
public:
    NonZeroLayerImpl(const LayerParams& params) { setParamsFrom(params); }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs, std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        const MatShape& in = inputs[0];
        CV_Assert(!in.empty());

        int rank = (int)in.size();
        MatShape out = shape(rank, -1);
        outputs.assign(1, out);
        return false;
    }

    void getTypes(const std::vector<MatType>& /*inputs*/, const int requiredOutputs,
                  const int requiredInternals, std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, CV_64S);
    }

    void forward(InputArrayOfArrays in_arr, OutputArrayOfArrays out_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_Assert(in_arr.size().area() == 1);
        const Mat X = in_arr.getMat(0);
        CV_Assert(X.data != nullptr);

        const int rank = X.dims;
        std::vector<int> dims(rank), strides(rank);
        for (int i = 0; i < rank; ++i) dims[i] = X.size[i];
        if (rank == 0) {
            const bool nz = X.type() == CV_32F ? isNonZero<float>(X.at<float>(0))
                          : X.type() == CV_64F ? isNonZero<double>(X.at<double>(0))
                          : X.type() == CV_32S ? (X.at<int>(0) != 0)
                          : X.type() == CV_16F ? isNonZero<hfloat, float>(X.at<hfloat>(0))
                          : X.type() == CV_16BF ? isNonZero<bfloat, float>(X.at<bfloat>(0))
                          : X.at<uchar>(0) != 0;
            MatShape outShape = shape(0, nz ? 1 : 0);
            auto kind = out_arr.kind();
            if (kind == _InputArray::STD_VECTOR_MAT) {
                std::vector<Mat>& outs_m = out_arr.getMatVecRef();
                outs_m.resize(1);
                outs_m[0].fit(outShape, CV_64S);
            } else if (kind == _InputArray::STD_VECTOR_UMAT) {
                std::vector<UMat>& outs_u = out_arr.getUMatVecRef();
                outs_u.resize(1);
                outs_u[0].fit(outShape, CV_64S);
            } else {
                CV_Error(Error::StsNotImplemented, "NonZero: unsupported OutputArrayOfArrays kind");
            }
            return;
        }

        strides.back() = 1;
        for (int i = rank - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * dims[i + 1];

        size_t total = X.total();
        size_t nnz = 0;

        const int depth = CV_MAT_DEPTH(X.type());
        switch (depth)
        {
        case CV_32F: {
            const float* p = X.ptr<float>();
            for (size_t i = 0; i < total; ++i) nnz += isNonZero<float>(p[i]);
            break; }
        case CV_64F: {
            const double* p = X.ptr<double>();
            for (size_t i = 0; i < total; ++i) nnz += isNonZero<double>(p[i]);
            break; }
        case CV_32S: {
            const int* p = X.ptr<int>();
            for (size_t i = 0; i < total; ++i) nnz += (p[i] != 0);
            break; }
        case CV_16F: {
            const hfloat* p = X.ptr<hfloat>();
            for (size_t i = 0; i < total; ++i) nnz += isNonZero<hfloat, float>(p[i]);
            break; }
        case CV_16BF: {
            const bfloat* p = X.ptr<bfloat>();
            for (size_t i = 0; i < total; ++i) nnz += isNonZero<bfloat, float>(p[i]);
            break; }
        case CV_8U:
        case CV_Bool: {
            const uchar* p = X.ptr<uchar>();
            for (size_t i = 0; i < total; ++i) nnz += (p[i] != 0);
            break; }
        default:
            CV_Error_(Error::StsError, ("NonZero: Unsupported input depth=%d", depth));
        }

        MatShape outShape = shape(rank, (int)nnz);
        auto kind = out_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs_m = out_arr.getMatVecRef();
            outs_m.resize(1);
            outs_m[0].fit(outShape, CV_64S);
            Mat& Y = outs_m[0];
            if (nnz == 0) return;

            int64* y = Y.ptr<int64>();
            size_t col = 0;

            auto emit_idx = [&](size_t lin) {
                size_t r = lin;
                for (int d = 0; d < rank; ++d)
                {
                    int idx = static_cast<int>(r / strides[d]);
                    y[d * (size_t)nnz + col] = static_cast<int64>(idx);
                    r -= (size_t)idx * (size_t)strides[d];
                }
                ++col;
            };

            switch (depth)
            {
            case CV_32F: { const float* p = X.ptr<float>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<float>(p[i])) emit_idx(i); } break; }
            case CV_64F: { const double* p = X.ptr<double>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<double>(p[i])) emit_idx(i); } break; }
            case CV_32S: { const int* p = X.ptr<int>();
                for (size_t i = 0; i < total; ++i) { if (p[i] != 0) emit_idx(i); } break; }
            case CV_16F: { const hfloat* p = X.ptr<hfloat>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<hfloat, float>(p[i])) emit_idx(i); } break; }
            case CV_16BF:{ const bfloat* p = X.ptr<bfloat>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<bfloat, float>(p[i])) emit_idx(i); } break; }
            case CV_8U:
            case CV_Bool: { const uchar* p = X.ptr<uchar>();
                for (size_t i = 0; i < total; ++i) { if (p[i] != 0) emit_idx(i); } break; }
            default: break;
            }
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs_u = out_arr.getUMatVecRef();
            outs_u.resize(1);
            outs_u[0].fit(outShape, CV_64S);
            if (nnz == 0) return;

            Mat Y(outShape, CV_64S);
            int64* y = Y.ptr<int64>();
            size_t col = 0;

            auto emit_idx = [&](size_t lin) {
                size_t r = lin;
                for (int d = 0; d < rank; ++d)
                {
                    int idx = static_cast<int>(r / strides[d]);
                    y[d * (size_t)nnz + col] = static_cast<int64>(idx);
                    r -= (size_t)idx * (size_t)strides[d];
                }
                ++col;
            };

            switch (depth)
            {
            case CV_32F: { const float* p = X.ptr<float>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<float>(p[i])) emit_idx(i); } break; }
            case CV_64F: { const double* p = X.ptr<double>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<double>(p[i])) emit_idx(i); } break; }
            case CV_32S: { const int* p = X.ptr<int>();
                for (size_t i = 0; i < total; ++i) { if (p[i] != 0) emit_idx(i); } break; }
            case CV_16F: { const hfloat* p = X.ptr<hfloat>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<hfloat, float>(p[i])) emit_idx(i); } break; }
            case CV_16BF:{ const bfloat* p = X.ptr<bfloat>();
                for (size_t i = 0; i < total; ++i) { if (isNonZero<bfloat, float>(p[i])) emit_idx(i); } break; }
            case CV_8U:
            case CV_Bool: { const uchar* p = X.ptr<uchar>();
                for (size_t i = 0; i < total; ++i) { if (p[i] != 0) emit_idx(i); } break; }
            default: break;
            }
            Y.copyTo(outs_u[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "NonZero: unsupported OutputArrayOfArrays kind");
        }
    }
};

Ptr<NonZeroLayer> NonZeroLayer::create(const LayerParams& params)
{
    return makePtr<NonZeroLayerImpl>(params);
}

}} // namespace cv::dnn
