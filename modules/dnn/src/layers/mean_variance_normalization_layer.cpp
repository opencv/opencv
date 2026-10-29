// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <climits>

namespace cv {
namespace dnn {

// Conservative heuristic: use parallel_for_ for tensors with at least 32K elements
// to avoid its scheduling overhead on small inputs. This cutoff is not architecture-tuned.
static const size_t MVN_PARALLEL_MIN_TOTAL = 32 * 1024;

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
        std::vector<size_t> srcSteps(src.dims);
        for (int axis = 0; axis < src.dims; ++axis)
            srcSteps[axis] = src.step.p[axis] / sizeof(T);
        const T* srcData = src.ptr<T>();

        int statsSuffixStart = src.dims;
        size_t statsSuffixSize = 1;
        for (int axis = src.dims - 1; axis >= 0 && reduced[axis]; --axis)
        {
            if ((axis == src.dims - 1 && srcSteps[axis] != 1) ||
                (axis < src.dims - 1 &&
                 srcSteps[axis] != srcSteps[axis + 1] * static_cast<size_t>(src.size[axis + 1])))
                break;
            statsSuffixStart = axis;
            statsSuffixSize *= static_cast<size_t>(src.size[axis]);
        }
        const size_t outerReductionCount = reductionCount / statsSuffixSize;

        const bool parallelGroups = src.total() >= MVN_PARALLEL_MIN_TOTAL &&
                                   groupCount > 1 &&
                                   groupCount <= static_cast<size_t>(INT_MAX);
        StatisticsInvoker<T> statisticsInvoker(
            src, reduced, groupStrides, srcSteps, srcData,
            referenceValues, meanOffsets, squaredDeviations,
            statsSuffixStart, statsSuffixSize, outerReductionCount, reductionCount);
        if (parallelGroups)
            parallel_for_(Range(0, static_cast<int>(groupCount)), statisticsInvoker);
        else
            statisticsInvoker.run(0, groupCount);

        std::vector<double> inverseStdDevs(groupCount);
        for (size_t i = 0; i < groupCount; ++i)
        {
            const double variance = squaredDeviations[i] / reductionCount;
            inverseStdDevs[i] = 1.0 / (std::sqrt(variance) + 1e-9);
        }

        std::vector<size_t> dstSteps(dst.dims);
        for (int axis = 0; axis < dst.dims; ++axis)
            dstSteps[axis] = dst.step.p[axis] / sizeof(T);
        T* dstData = dst.ptr<T>();

        int outputSuffixStart = src.dims;
        size_t outputSuffixSize = 1;
        for (int axis = src.dims - 1; axis >= 0 && reduced[axis]; --axis)
        {
            if ((axis == src.dims - 1 && (srcSteps[axis] != 1 || dstSteps[axis] != 1)) ||
                (axis < src.dims - 1 &&
                 (srcSteps[axis] != srcSteps[axis + 1] * static_cast<size_t>(src.size[axis + 1]) ||
                  dstSteps[axis] != dstSteps[axis + 1] * static_cast<size_t>(dst.size[axis + 1]))))
                break;
            outputSuffixStart = axis;
            outputSuffixSize *= static_cast<size_t>(src.size[axis]);
        }
        const size_t outerOutputCount = reductionCount / outputSuffixSize;

        NormalizeGroupsInvoker<T> normalizeGroups(
            src, dst, reduced, groupStrides, srcSteps, dstSteps, srcData, dstData,
            referenceValues, meanOffsets, inverseStdDevs,
            outputSuffixStart, outputSuffixSize, outerOutputCount, reductionCount);
        if (parallelGroups)
            parallel_for_(Range(0, static_cast<int>(groupCount)), normalizeGroups);
        else
            normalizeGroups.run(0, groupCount);
    }

    template<typename T>
    static void computeStatisticsGroup(size_t groupIndex, std::vector<int>& coordinates,
                                       const Mat& src, const std::vector<bool>& reduced,
                                       const std::vector<size_t>& groupStrides,
                                       const std::vector<size_t>& srcSteps, const T* srcData,
                                       std::vector<double>& referenceValues,
                                       std::vector<double>& meanOffsets,
                                       std::vector<double>& squaredDeviations,
                                       int statsSuffixStart, size_t statsSuffixSize,
                                       size_t outerReductionCount, size_t reductionCount)
    {
        size_t groupOffset = 0;
        for (int axis = src.dims - 1; axis >= 0; --axis)
        {
            if (reduced[axis])
                continue;
            coordinates[axis] =
                static_cast<int>((groupIndex / groupStrides[axis]) % src.size[axis]);
            groupOffset += static_cast<size_t>(coordinates[axis]) * srcSteps[axis];
        }

        if (statsSuffixStart < src.dims)
        {
            size_t offset = groupOffset;
            for (size_t outer = 0; outer < outerReductionCount; ++outer)
            {
                for (size_t inner = 0; inner < statsSuffixSize; ++inner)
                {
                    const size_t i = outer * statsSuffixSize + inner;
                    const double value = static_cast<double>(srcData[offset + inner]);
                    if (i == 0)
                        referenceValues[groupIndex] = value;

                    const double delta = (value - referenceValues[groupIndex]) - meanOffsets[groupIndex];
                    meanOffsets[groupIndex] += delta / (i + 1);
                    const double centeredDelta =
                        (value - referenceValues[groupIndex]) - meanOffsets[groupIndex];
                    squaredDeviations[groupIndex] += delta * centeredDelta;
                }

                for (int axis = statsSuffixStart - 1; axis >= 0; --axis)
                {
                    if (!reduced[axis])
                        continue;
                    if (++coordinates[axis] < src.size[axis])
                    {
                        offset += srcSteps[axis];
                        break;
                    }
                    coordinates[axis] = 0;
                    offset -= static_cast<size_t>(src.size[axis] - 1) * srcSteps[axis];
                }
            }
            return;
        }

        size_t offset = groupOffset;
        for (size_t i = 0; i < reductionCount; ++i)
        {
            const double value = static_cast<double>(srcData[offset]);
            if (i == 0)
                referenceValues[groupIndex] = value;

            const double delta = (value - referenceValues[groupIndex]) - meanOffsets[groupIndex];
            meanOffsets[groupIndex] += delta / (i + 1);
            const double centeredDelta =
                (value - referenceValues[groupIndex]) - meanOffsets[groupIndex];
            squaredDeviations[groupIndex] += delta * centeredDelta;

            for (int axis = src.dims - 1; axis >= 0; --axis)
            {
                if (!reduced[axis])
                    continue;
                if (++coordinates[axis] < src.size[axis])
                {
                    offset += srcSteps[axis];
                    break;
                }
                coordinates[axis] = 0;
                offset -= static_cast<size_t>(src.size[axis] - 1) * srcSteps[axis];
            }
        }
    }

    template<typename T>
    class StatisticsInvoker : public ParallelLoopBody
    {
    public:
        StatisticsInvoker(const Mat& src, const std::vector<bool>& reduced,
                          const std::vector<size_t>& groupStrides,
                          const std::vector<size_t>& srcSteps, const T* srcData,
                          std::vector<double>& referenceValues,
                          std::vector<double>& meanOffsets,
                          std::vector<double>& squaredDeviations,
                          int statsSuffixStart, size_t statsSuffixSize,
                          size_t outerReductionCount, size_t reductionCount)
            : src_(src), reduced_(reduced), groupStrides_(groupStrides), srcSteps_(srcSteps),
              srcData_(srcData), referenceValues_(referenceValues), meanOffsets_(meanOffsets),
              squaredDeviations_(squaredDeviations), statsSuffixStart_(statsSuffixStart),
              statsSuffixSize_(statsSuffixSize), outerReductionCount_(outerReductionCount),
              reductionCount_(reductionCount)
        {
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            run(static_cast<size_t>(range.start), static_cast<size_t>(range.end));
        }

        void run(size_t start, size_t end) const
        {
            std::vector<int> coordinates(src_.dims, 0);
            for (size_t groupIndex = start; groupIndex < end; ++groupIndex)
            {
                MeanVarianceNormalizationLayerImpl::computeStatisticsGroup<T>(
                    groupIndex, coordinates, src_, reduced_, groupStrides_,
                    srcSteps_, srcData_, referenceValues_, meanOffsets_, squaredDeviations_,
                    statsSuffixStart_, statsSuffixSize_, outerReductionCount_, reductionCount_);
            }
        }

    private:
        const Mat& src_;
        const std::vector<bool>& reduced_;
        const std::vector<size_t>& groupStrides_;
        const std::vector<size_t>& srcSteps_;
        const T* srcData_;
        std::vector<double>& referenceValues_;
        std::vector<double>& meanOffsets_;
        std::vector<double>& squaredDeviations_;
        int statsSuffixStart_;
        size_t statsSuffixSize_;
        size_t outerReductionCount_;
        size_t reductionCount_;
    };

    template<typename T>
    static void normalizeGroup(size_t groupIndex, std::vector<int>& coordinates,
                               const Mat& src, const Mat& dst, const std::vector<bool>& reduced,
                               const std::vector<size_t>& groupStrides,
                               const std::vector<size_t>& srcSteps,
                               const std::vector<size_t>& dstSteps,
                               const T* srcData, T* dstData,
                               const std::vector<double>& referenceValues,
                               const std::vector<double>& meanOffsets,
                               const std::vector<double>& inverseStdDevs,
                               int outputSuffixStart, size_t outputSuffixSize,
                               size_t outerOutputCount, size_t reductionCount)
    {
        size_t srcBase = 0, dstBase = 0;
        for (int axis = src.dims - 1; axis >= 0; --axis)
        {
            if (reduced[axis])
                continue;
            coordinates[axis] =
                static_cast<int>((groupIndex / groupStrides[axis]) % src.size[axis]);
            srcBase += static_cast<size_t>(coordinates[axis]) * srcSteps[axis];
            dstBase += static_cast<size_t>(coordinates[axis]) * dstSteps[axis];
        }

        if (outputSuffixStart < src.dims)
        {
            size_t srcOffset = srcBase, dstOffset = dstBase;
            for (size_t outer = 0; outer < outerOutputCount; ++outer)
            {
                for (size_t inner = 0; inner < outputSuffixSize; ++inner)
                {
                    const double centeredValue =
                        (static_cast<double>(srcData[srcOffset + inner]) - referenceValues[groupIndex]) -
                        meanOffsets[groupIndex];
                    dstData[dstOffset + inner] =
                        saturate_cast<T>(centeredValue * inverseStdDevs[groupIndex]);
                }

                for (int axis = outputSuffixStart - 1; axis >= 0; --axis)
                {
                    if (!reduced[axis])
                        continue;
                    if (++coordinates[axis] < src.size[axis])
                    {
                        srcOffset += srcSteps[axis];
                        dstOffset += dstSteps[axis];
                        break;
                    }
                    coordinates[axis] = 0;
                    srcOffset -= static_cast<size_t>(src.size[axis] - 1) * srcSteps[axis];
                    dstOffset -= static_cast<size_t>(dst.size[axis] - 1) * dstSteps[axis];
                }
            }
            return;
        }

        size_t srcOffset = srcBase, dstOffset = dstBase;
        for (size_t i = 0; i < reductionCount; ++i)
        {
            const double centeredValue =
                (static_cast<double>(srcData[srcOffset]) - referenceValues[groupIndex]) -
                meanOffsets[groupIndex];
            dstData[dstOffset] = saturate_cast<T>(centeredValue * inverseStdDevs[groupIndex]);

            for (int axis = src.dims - 1; axis >= 0; --axis)
            {
                if (!reduced[axis])
                    continue;
                if (++coordinates[axis] < src.size[axis])
                {
                    srcOffset += srcSteps[axis];
                    dstOffset += dstSteps[axis];
                    break;
                }
                coordinates[axis] = 0;
                srcOffset -= static_cast<size_t>(src.size[axis] - 1) * srcSteps[axis];
                dstOffset -= static_cast<size_t>(dst.size[axis] - 1) * dstSteps[axis];
            }
        }
    }

    template<typename T>
    class NormalizeGroupsInvoker : public ParallelLoopBody
    {
    public:
        NormalizeGroupsInvoker(const Mat& src, Mat& dst, const std::vector<bool>& reduced,
                               const std::vector<size_t>& groupStrides,
                               const std::vector<size_t>& srcSteps,
                               const std::vector<size_t>& dstSteps,
                               const T* srcData, T* dstData,
                               const std::vector<double>& referenceValues,
                               const std::vector<double>& meanOffsets,
                               const std::vector<double>& inverseStdDevs,
                               int outputSuffixStart, size_t outputSuffixSize,
                               size_t outerOutputCount, size_t reductionCount)
            : src_(src), dst_(dst), reduced_(reduced), groupStrides_(groupStrides),
              srcSteps_(srcSteps), dstSteps_(dstSteps), srcData_(srcData), dstData_(dstData),
              referenceValues_(referenceValues), meanOffsets_(meanOffsets),
              inverseStdDevs_(inverseStdDevs), outputSuffixStart_(outputSuffixStart),
              outputSuffixSize_(outputSuffixSize), outerOutputCount_(outerOutputCount),
              reductionCount_(reductionCount)
        {
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            run(static_cast<size_t>(range.start), static_cast<size_t>(range.end));
        }

        void run(size_t start, size_t end) const
        {
            std::vector<int> coordinates(src_.dims, 0);
            for (size_t groupIndex = start; groupIndex < end; ++groupIndex)
            {
                MeanVarianceNormalizationLayerImpl::normalizeGroup<T>(
                    groupIndex, coordinates, src_, dst_, reduced_,
                    groupStrides_, srcSteps_, dstSteps_, srcData_, dstData_,
                    referenceValues_, meanOffsets_, inverseStdDevs_,
                    outputSuffixStart_, outputSuffixSize_, outerOutputCount_, reductionCount_);
            }
        }

    private:
        const Mat& src_;
        Mat& dst_;
        const std::vector<bool>& reduced_;
        const std::vector<size_t>& groupStrides_;
        const std::vector<size_t>& srcSteps_;
        const std::vector<size_t>& dstSteps_;
        const T* srcData_;
        T* dstData_;
        const std::vector<double>& referenceValues_;
        const std::vector<double>& meanOffsets_;
        const std::vector<double>& inverseStdDevs_;
        int outputSuffixStart_;
        size_t outputSuffixSize_;
        size_t outerOutputCount_;
        size_t reductionCount_;
    };

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
