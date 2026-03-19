// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <cstdint>
#include <array>

namespace cv {
namespace dnn {

// ONNX NonZero operator
// Spec: https://onnx.ai/onnx/operators/onnx__NonZero.html
// Supported opsets: 9-18 (no attributes)

namespace {
static inline bool isNonZeroF16(uint16_t val)
{
    return (val & 0x7fff) != 0;
}

template <typename T, typename PT>
static size_t computeNonZeroCountsPerStripe(const T* data, size_t total, int nstripes, std::vector<size_t>& nzcounts, PT isNonZero)
{
    nzcounts.assign(nstripes, 0);
    if (total == 0)
        return 0;
    cv::parallel_for_(cv::Range(0, nstripes), [&](const cv::Range& range) {
        const size_t i0 = total * (size_t)range.start / (size_t)nstripes;
        const size_t i1 = total * (size_t)range.end   / (size_t)nstripes;
        size_t nz = 0;
        for (size_t i = i0; i < i1; ++i)
            nz += isNonZero(data[i]);
        nzcounts[(size_t)range.start] = nz;
    });

    size_t total_nz = 0;
    for (size_t c : nzcounts) total_nz += c;
    return total_nz;
}

template <typename T>
static inline size_t computeNonZeroCountsPerStripe(const T* data, size_t total, int nstripes, std::vector<size_t>& nzcounts)
{
    return computeNonZeroCountsPerStripe<T>(data, total, nstripes, nzcounts, [](T v){ return v != (T)0; });
}

template <typename T, typename PT>
static void emitIndicesStripes(const T* data,
                               size_t total,
                               int nstripes,
                               int rank,
                               const std::vector<int>& dims,
                               const std::vector<int>& strides,
                               const std::vector<size_t>& nzstart,
                               int64* y,
                               PT isNonZero)
{
    const size_t nnz_total = nzstart.back();
    if (rank == 0) {
        return;
    }
    const int lastDimSize = dims[rank - 1];

    cv::parallel_for_(cv::Range(0, nstripes), [&](const cv::Range& range) {
        size_t i = total * (size_t)range.start / (size_t)nstripes;
        const size_t i1 = total * (size_t)range.end / (size_t)nstripes;
        size_t col = nzstart[(size_t)range.start];

        std::array<int, CV_MAX_DIM> coord;
        std::fill_n(coord.begin(), rank, 0);
        if (rank > 0) coord[rank - 1] = lastDimSize - 1;

        for (; i < i1; ++i)
        {
            if (rank > 0 && ++coord[rank - 1] >= lastDimSize)
            {
                size_t idxLinear = i;
                for (int d = rank - 1; d >= 0; --d)
                {
                    coord[d] = static_cast<int>(idxLinear % (size_t)dims[d]);
                    idxLinear /= (size_t)dims[d];
                }
            }
            if (isNonZero(data[i]))
            {
                for (int d = 0; d < rank; ++d)
                    y[(size_t)d * nnz_total + col] = (int64)coord[d];
                ++col;
            }
        }
    });
}

template <typename T>
static void emitIndicesStripes(const T* data,
                               size_t total,
                               int nstripes,
                               int rank,
                               const std::vector<int>& dims,
                               const std::vector<int>& strides,
                               const std::vector<size_t>& nzstart,
                               int64* y)
{
    emitIndicesStripes<T>(data, total, nstripes, rank, dims, strides, nzstart, y, [](T v){ return v != (T)0; });
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

        if (rank > 0) {
            strides.back() = 1;
            for (int i = rank - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * dims[i + 1];
        }

        size_t total = X.total();
        const int nstripes = 16;

        const int depth = CV_MAT_DEPTH(X.type());
        std::vector<size_t> nzcounts;
        std::vector<size_t> nzstart;
        size_t nnz = 0;
        switch (depth)
        {
        case CV_Bool:
        case CV_8U:
        case CV_8S:  { nnz = computeNonZeroCountsPerStripe<uchar>(X.ptr<uchar>(), total, nstripes, nzcounts); break; }
        case CV_16U:
        case CV_16S: { nnz = computeNonZeroCountsPerStripe<uint16_t>(X.ptr<uint16_t>(), total, nstripes, nzcounts); break; }
        case CV_32U:
        case CV_32S: { nnz = computeNonZeroCountsPerStripe<uint32_t>(X.ptr<uint32_t>(), total, nstripes, nzcounts); break; }
        case CV_64U:
        case CV_64S: { nnz = computeNonZeroCountsPerStripe<uint64_t>(X.ptr<uint64_t>(), total, nstripes, nzcounts); break; }
        case CV_32F: { nnz = computeNonZeroCountsPerStripe<float>(X.ptr<float>(), total, nstripes, nzcounts); break; }
        case CV_64F: { nnz = computeNonZeroCountsPerStripe<double>(X.ptr<double>(), total, nstripes, nzcounts); break; }
        case CV_16F:
        case CV_16BF:{ nnz = computeNonZeroCountsPerStripe<uint16_t>(X.ptr<uint16_t>(), total, nstripes, nzcounts, [](uint16_t v){ return isNonZeroF16(v); }); break; }
        default:
            CV_Error_(Error::StsError, ("NonZero: Unsupported input depth=%d", depth));
        }

        nzstart.assign(nzcounts.size() + 1, 0);
        for (size_t i = 1; i <= nzcounts.size(); ++i)
            nzstart[i] = nzstart[i - 1] + nzcounts[i - 1];

        MatShape outShape = shape(rank, (int)nnz);
        auto kind = out_arr.kind();
        std::vector<Mat>* out_mats = nullptr;
        std::vector<UMat>* out_umats = nullptr;
        Mat Y;
        if (kind == _InputArray::STD_VECTOR_MAT) {
            out_mats = &out_arr.getMatVecRef();
            out_mats->resize(1);
            out_mats->at(0).fit(outShape, CV_64S);
            Y = out_mats->at(0);
        } else {
            CV_Assert(kind == _InputArray::STD_VECTOR_UMAT);
            out_umats = &out_arr.getUMatVecRef();
            out_umats->resize(1);
            out_umats->at(0).fit(outShape, CV_64S);
            Y = Mat(outShape, CV_64S);
        }

        if (nnz == 0) return;

        int64* y = Y.ptr<int64>();

        switch (depth)
        {
        case CV_Bool:
        case CV_8U:
        case CV_8S:  { emitIndicesStripes<uchar>(X.ptr<uchar>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_16U:
        case CV_16S: { emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_32U:
        case CV_32S: { emitIndicesStripes<uint32_t>(X.ptr<uint32_t>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_64U:
        case CV_64S: { emitIndicesStripes<uint64_t>(X.ptr<uint64_t>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_32F: { emitIndicesStripes<float>(X.ptr<float>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_64F: { emitIndicesStripes<double>(X.ptr<double>(), total, nstripes, rank, dims, strides, nzstart, y); break; }
        case CV_16F:
        case CV_16BF:{ emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, nstripes, rank, dims, strides, nzstart, y, [](uint16_t v){ return isNonZeroF16(v); }); break; }
        default: break;
        }

        if (kind == _InputArray::STD_VECTOR_UMAT) {
            Y.copyTo(out_umats->at(0));
        }
    }
};

Ptr<NonZeroLayer> NonZeroLayer::create(const LayerParams& params)
{
    return makePtr<NonZeroLayerImpl>(params);
}

}} // namespace cv::dnn
