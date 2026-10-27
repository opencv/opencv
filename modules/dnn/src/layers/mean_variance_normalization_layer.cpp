// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

/*
    MeanVarianceNormalization layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html

    Opsets 9 and 13 are covered.
*/

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

        std::vector<bool> reduced(src.dims, false);
        for (int axis : normalizedAxes)
            reduced[axis] = true;

        std::vector<size_t> groupStrides(src.dims, 0);
        size_t groupCount = 1;
        for (int i = src.dims - 1; i >= 0; --i)
        {
            if (!reduced[i])
            {
                groupStrides[i] = groupCount;
                groupCount *= static_cast<size_t>(src.size[i]);
            }
        }

        dst.create(src.dims, src.size.p, src.type());
        switch (src.depth())
        {
        case CV_16F: forwardTyped<hfloat>(src, dst, reduced, groupStrides, groupCount); break;
        case CV_16BF: forwardTyped<bfloat>(src, dst, reduced, groupStrides, groupCount); break;
        case CV_32F: forwardTyped<float>(src, dst, reduced, groupStrides, groupCount); break;
        case CV_64F: forwardTyped<double>(src, dst, reduced, groupStrides, groupCount); break;
        default: CV_Error(Error::StsUnsupportedFormat, "Unsupported input type");
        }
    }

    static Ptr<MeanVarianceNormalizationLayer> create(const LayerParams& params)
    {
        return makePtr<MeanVarianceNormalizationLayerImpl>(params);
    }

private:
    std::vector<int> axes;

    template<typename T>
    static void forwardTyped(const Mat& src, Mat& dst, const std::vector<bool>& reduced,
                             const std::vector<size_t>& groupStrides, size_t groupCount)
    {
        size_t reductionCount = 1;
        for (int axis = src.dims - 1; axis >= 0; --axis)
        {
            if (reduced[axis]){
                reductionCount *= static_cast<size_t>(src.size[axis]);
            }
        }

        std::vector<double> referenceValues(groupCount), meanOffsets(groupCount);
        std::vector<double> squaredDeviations(groupCount, 0.0);
        std::vector<int> coordinates(src.dims, 0);
        size_t group = 0;

        for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex)
        {
            for (size_t i = 0; i < reductionCount; ++i)
            {
                const double value = static_cast<double>(src.ptr<T>(coordinates.data())[0]);
                if (i == 0){
                    referenceValues[group] = value;
                }

                const double delta = (value - referenceValues[group]) - meanOffsets[group];
                meanOffsets[group] += delta / (i + 1);
                const double centeredDelta = (value - referenceValues[group]) - meanOffsets[group];
                squaredDeviations[group] += delta * centeredDelta;

                for (int axis = src.dims - 1; axis >= 0; --axis)
                {
                    if (!reduced[axis]){
                        continue;
                    }
                    if (++coordinates[axis] < src.size[axis])
                    {
                        break;
                    }
                    coordinates[axis] = 0;
                }
            }

            if (groupIndex + 1 == groupCount)
                break;
            for (int axis = src.dims - 1; axis >= 0; --axis)
            {
                if (reduced[axis]){
                    continue;
                }
                if (++coordinates[axis] < src.size[axis])
                {
                    group += groupStrides[axis];
                    break;
                }
                coordinates[axis] = 0;
                group -= (static_cast<size_t>(src.size[axis]) - 1) * groupStrides[axis];
            }
        }

        std::vector<double> inverseStdDevs(groupCount);
        for (size_t i = 0; i < groupCount; ++i)
        {
            const double variance = squaredDeviations[i] / reductionCount;
            inverseStdDevs[i] = 1.0 / (std::sqrt(variance) + 1e-9);
        }

        std::fill(coordinates.begin(), coordinates.end(), 0);
        group = 0;
        for (size_t i = 0; i < src.total(); ++i)
        {
            T* output = dst.ptr<T>(coordinates.data());
            const double centeredValue =
                (static_cast<double>(src.ptr<T>(coordinates.data())[0]) - referenceValues[group]) - meanOffsets[group];
            output[0] = saturate_cast<T>(centeredValue * inverseStdDevs[group]);

            for (int axis = src.dims - 1; axis >= 0; --axis)
            {
                if (++coordinates[axis] < src.size[axis])
                {
                    if (!reduced[axis]){
                        group += groupStrides[axis];
                    }
                    break;
                }
                coordinates[axis] = 0;
                if (!reduced[axis])
                    group -= (static_cast<size_t>(src.size[axis]) - 1) * groupStrides[axis];
            }
        }
    }

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
