// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the
// top-level directory of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <cstdint>

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
static size_t computeNonZeroCountsPerStripe(const T* data, size_t total, std::vector<size_t>& nzcounts, PT isNonZero)
{
    const int nstripes = 16;
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
static inline size_t computeNonZeroCountsPerStripe(const T* data, size_t total, std::vector<size_t>& nzcounts)
{
    return computeNonZeroCountsPerStripe<T>(data, total, nzcounts, [](T v){ return v != (T)0; });
}

template <typename T, typename PT>
static void emitIndicesStripes(const T* data,
                               size_t total,
                               int rank,
                               const std::vector<int>& dims,
                               const std::vector<int>& strides,
                               const std::vector<size_t>& nzstart,
                               int64* y,
                               PT isNonZero)
{
    const int nstripes = 16;
    const size_t nnz_total = nzstart.back();
    if (rank == 0) {
        return;
    }
    const int lastDimSize = dims[rank - 1];

    cv::parallel_for_(cv::Range(0, nstripes), [&](const cv::Range& range) {
        size_t i = total * (size_t)range.start / (size_t)nstripes;
        const size_t i1 = total * (size_t)range.end   / (size_t)nstripes;
        size_t col = nzstart[(size_t)range.start];

        std::vector<int> coord(rank, 0);
        if (i < i1)
        {
            size_t r = i;
            for (int d = 0; d < rank; ++d)
            {
                const int sd = strides[d];
                const int idx = static_cast<int>(r / (size_t)sd);
                coord[d] = idx;
                r -= (size_t)idx * (size_t)sd;
            }
        }

        while (i < i1)
        {
            const int innerStart = coord[rank - 1];
            const size_t innerAvail = (size_t)(lastDimSize - innerStart);
            const size_t chunk = std::min(innerAvail, i1 - i);

            for (size_t dj = 0; dj < chunk; ++dj)
            {
                const size_t idxLinear = i + dj;
                if (isNonZero(data[idxLinear]))
                {
                    for (int d = 0; d < rank - 1; ++d)
                        y[(size_t)d * nnz_total + col] = (int64)coord[d];
                    y[(size_t)(rank - 1) * nnz_total + col] = (int64)(innerStart + (int)dj);
                    ++col;
                }
            }

            i += chunk;
            if (i >= i1) break;

            coord[rank - 1] = 0;
            if (rank > 1)
            {
                int d = rank - 2;
                for (; d >= 0; --d)
                {
                    coord[d] += 1;
                    if (coord[d] < dims[d])
                        break;
                    coord[d] = 0;
                }
            }
        }
    });
}

template <typename T>
static void emitIndicesStripes(const T* data,
                               size_t total,
                               int rank,
                               const std::vector<int>& dims,
                               const std::vector<int>& strides,
                               const std::vector<size_t>& nzstart,
                               int64* y)
{
    emitIndicesStripes<T>(data, total, rank, dims, strides, nzstart, y, [](T v){ return v != (T)0; });
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

        const int depth = CV_MAT_DEPTH(X.type());
        std::vector<size_t> nzcounts;
        std::vector<size_t> nzstart;
        size_t nnz = 0;
        switch (depth)
        {
        case CV_8U:  { nnz = computeNonZeroCountsPerStripe<uchar>(X.ptr<uchar>(), total, nzcounts); break; }
        case CV_8S:  { nnz = computeNonZeroCountsPerStripe<schar>(X.ptr<schar>(), total, nzcounts); break; }
        case CV_16U: { nnz = computeNonZeroCountsPerStripe<ushort>(X.ptr<ushort>(), total, nzcounts); break; }
        case CV_16S: { nnz = computeNonZeroCountsPerStripe<short>(X.ptr<short>(), total, nzcounts); break; }
        case CV_32U: { nnz = computeNonZeroCountsPerStripe<uint32_t>(X.ptr<uint32_t>(), total, nzcounts); break; }
        case CV_32S: { nnz = computeNonZeroCountsPerStripe<int>(X.ptr<int>(), total, nzcounts); break; }
        case CV_64U: { nnz = computeNonZeroCountsPerStripe<uint64_t>(X.ptr<uint64_t>(), total, nzcounts); break; }
        case CV_64S: { nnz = computeNonZeroCountsPerStripe<int64>(X.ptr<int64>(), total, nzcounts); break; }
        case CV_32F: { nnz = computeNonZeroCountsPerStripe<float>(X.ptr<float>(), total, nzcounts); break; }
        case CV_64F: { nnz = computeNonZeroCountsPerStripe<double>(X.ptr<double>(), total, nzcounts); break; }
        case CV_16F: { nnz = computeNonZeroCountsPerStripe<uint16_t>(X.ptr<uint16_t>(), total, nzcounts, [](uint16_t v){ return isNonZeroF16(v); }); break; }
        case CV_16BF:{ nnz = computeNonZeroCountsPerStripe<uint16_t>(X.ptr<uint16_t>(), total, nzcounts, [](uint16_t v){ return isNonZeroF16(v); }); break; }
        case CV_Bool:{ nnz = computeNonZeroCountsPerStripe<uchar>(X.ptr<uchar>(), total, nzcounts); break; }
        default:
            CV_Error_(Error::StsError, ("NonZero: Unsupported input depth=%d", depth));
        }

        nzstart.assign(nzcounts.size() + 1, 0);
        for (size_t i = 1; i <= nzcounts.size(); ++i)
            nzstart[i] = nzstart[i - 1] + nzcounts[i - 1];

        MatShape outShape = shape(rank, (int)nnz);
        auto kind = out_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs_m = out_arr.getMatVecRef();
            outs_m.resize(1);
            outs_m[0].fit(outShape, CV_64S);
            Mat& Y = outs_m[0];
            if (nnz == 0) return;

            int64* y = Y.ptr<int64>();

            switch (depth)
            {
            case CV_8U:  { emitIndicesStripes<uchar>(X.ptr<uchar>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_8S:  { emitIndicesStripes<schar>(X.ptr<schar>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16U: { emitIndicesStripes<ushort>(X.ptr<ushort>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16S: { emitIndicesStripes<short>(X.ptr<short>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32U: { emitIndicesStripes<uint32_t>(X.ptr<uint32_t>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32S: { emitIndicesStripes<int>(X.ptr<int>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64U: { emitIndicesStripes<uint64_t>(X.ptr<uint64_t>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64S: { emitIndicesStripes<int64>(X.ptr<int64>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32F: { emitIndicesStripes<float>(X.ptr<float>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64F: { emitIndicesStripes<double>(X.ptr<double>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16F: { emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, rank, dims, strides, nzstart, y, [](uint16_t v){ return isNonZeroF16(v); }); break; }
            case CV_16BF:{ emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, rank, dims, strides, nzstart, y, [](uint16_t v){ return isNonZeroF16(v); }); break; }
            case CV_Bool:{ emitIndicesStripes<uchar>(X.ptr<uchar>(), total, rank, dims, strides, nzstart, y); break; }
            default: break;
            }
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs_u = out_arr.getUMatVecRef();
            outs_u.resize(1);
            outs_u[0].fit(outShape, CV_64S);
            if (nnz == 0) return;

            Mat Y(outShape, CV_64S);
            int64* y = Y.ptr<int64>();

            switch (depth)
            {
            case CV_8U:  { emitIndicesStripes<uchar>(X.ptr<uchar>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_8S:  { emitIndicesStripes<schar>(X.ptr<schar>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16U: { emitIndicesStripes<ushort>(X.ptr<ushort>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16S: { emitIndicesStripes<short>(X.ptr<short>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32U: { emitIndicesStripes<uint32_t>(X.ptr<uint32_t>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32S: { emitIndicesStripes<int>(X.ptr<int>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64U: { emitIndicesStripes<uint64_t>(X.ptr<uint64_t>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64S: { emitIndicesStripes<int64>(X.ptr<int64>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_32F: { emitIndicesStripes<float>(X.ptr<float>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_64F: { emitIndicesStripes<double>(X.ptr<double>(), total, rank, dims, strides, nzstart, y); break; }
            case CV_16F: { emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, rank, dims, strides, nzstart, y, [](uint16_t v){ return isNonZeroF16(v); }); break; }
            case CV_16BF:{ emitIndicesStripes<uint16_t>(X.ptr<uint16_t>(), total, rank, dims, strides, nzstart, y, [](uint16_t v){ return isNonZeroF16(v); }); break; }
            case CV_Bool:{ emitIndicesStripes<uchar>(X.ptr<uchar>(), total, rank, dims, strides, nzstart, y); break; }
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
