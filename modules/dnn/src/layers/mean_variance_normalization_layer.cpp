// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

class MeanVarianceNormalizationLayerImpl CV_FINAL : public MeanVarianceNormalizationLayer
{
public:
    explicit MeanVarianceNormalizationLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axes = params.has("axes") ? params.getVector<int>("axes") : std::vector<int>{0, 2, 3};
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>&) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)1, "MeanVarianceNormalization expects one input");
        CV_Assert(requiredOutputs == 0 || requiredOutputs == 1);
        normalizeAxes(inputs[0].size());
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>&) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)1, "MeanVarianceNormalization expects one input");
        CV_CheckType(inputs[0],
                     inputs[0] == CV_16F || inputs[0] == CV_16BF ||
                     inputs[0] == CV_32F || inputs[0] == CV_64F,
                     "MeanVarianceNormalization supports floating-point inputs only");
        outputs.assign(requiredOutputs, inputs[0]);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_CheckEQ(inputs.size(), (size_t)1, "MeanVarianceNormalization expects one input");
        CV_CheckEQ(outputs.size(), (size_t)1, "MeanVarianceNormalization expects one output");

        const Mat& src = inputs[0];
        Mat& dst = outputs[0];
        const std::vector<int> normalizedAxes = normalizeAxes(src.dims);
        if (src.empty())
            return;

        Mat src64;
        src.convertTo(src64, CV_64F);

        std::vector<bool> reduced(src.dims, false);
        for (int axis : normalizedAxes)
            reduced[axis] = true;

        std::vector<size_t> inputStrides(src.dims);
        std::vector<size_t> groupStrides(src.dims, 0);
        size_t inputStride = 1;
        size_t groupCount = 1;
        for (int i = src.dims - 1; i >= 0; --i)
        {
            inputStrides[i] = inputStride;
            inputStride *= static_cast<size_t>(src.size[i]);
            if (!reduced[i])
            {
                groupStrides[i] = groupCount;
                groupCount *= static_cast<size_t>(src.size[i]);
            }
        }

        std::vector<double> sums(groupCount, 0.0), squaredSums(groupCount, 0.0);
        std::vector<size_t> counts(groupCount, 0);
        const double* srcData = src64.ptr<double>();
        for (size_t offset = 0; offset < src.total(); ++offset)
        {
            size_t group = 0;
            for (int axis = 0; axis < src.dims; ++axis)
            {
                if (!reduced[axis])
                {
                    const size_t coordinate = (offset / inputStrides[axis]) % src.size[axis];
                    group += coordinate * groupStrides[axis];
                }
            }
            const double value = srcData[offset];
            sums[group] += value;
            squaredSums[group] += value * value;
            ++counts[group];
        }

        std::vector<double> means(groupCount), inverseStdDevs(groupCount);
        for (size_t group = 0; group < groupCount; ++group)
        {
            means[group] = sums[group] / counts[group];
            const double variance = squaredSums[group] / counts[group] - means[group] * means[group];
            inverseStdDevs[group] = 1.0 / (std::sqrt(variance) + 1e-9);
        }

        Mat dst64(src.dims, src.size.p, CV_64F);
        double* dstData = dst64.ptr<double>();
        for (size_t offset = 0; offset < src.total(); ++offset)
        {
            size_t group = 0;
            for (int axis = 0; axis < src.dims; ++axis)
            {
                if (!reduced[axis])
                {
                    const size_t coordinate = (offset / inputStrides[axis]) % src.size[axis];
                    group += coordinate * groupStrides[axis];
                }
            }
            dstData[offset] = (srcData[offset] - means[group]) * inverseStdDevs[group];
        }
        dst64.convertTo(dst, dst.depth());
    }

    static Ptr<MeanVarianceNormalizationLayer> create(const LayerParams& params)
    {
        return makePtr<MeanVarianceNormalizationLayerImpl>(params);
    }

private:
    std::vector<int> axes;

    std::vector<int> normalizeAxes(size_t rank) const
    {
        std::vector<int> normalized;
        std::vector<bool> seen(rank, false);
        std::vector<int> axesToReduce = axes;
        if (axesToReduce.empty())
        {
            axesToReduce.resize(rank);
            for (size_t i = 0; i < rank; ++i)
                axesToReduce[i] = static_cast<int>(i);
        }

        for (int axis : axesToReduce)
        {
            CV_Assert(axis >= -static_cast<int>(rank) && axis < static_cast<int>(rank));
            if (axis < 0)
                axis += static_cast<int>(rank);
            CV_Assert(!seen[axis]);
            seen[axis] = true;
            normalized.push_back(axis);
        }
        return normalized;
    }
};

Ptr<MeanVarianceNormalizationLayer> MeanVarianceNormalizationLayer::create(const LayerParams& params)
{
    return MeanVarianceNormalizationLayerImpl::create(params);
}

}} // namespace cv::dnn
