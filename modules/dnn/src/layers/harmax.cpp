#include <inttypes.h>
 #include <opencv2/dnn/shape_utils.hpp>
 #include "../precomp.hpp"
 #include "layers_common.hpp"

namespace cv
{
namespace dnn
{

class LayerHardmaxImpl CV_FINAL : public HardmaxLayer
{
public:
    int axis;
    LayerHardmaxImpl(const LayerParams& params)
    {
        axis = params.get<int>("axis", -1);
    }


    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        for (auto input : inputs)
        {
            if (preferableTarget == DNN_TARGET_OPENCL_FP16)
                CV_CheckType(input, input == CV_16F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S || input == CV_Bool, "");
            else
                CV_CheckType(input, input == CV_32F || input == CV_8S || input == CV_8U || input == CV_32S || input == CV_64S || input == CV_Bool, "");
        }

        outputs.assign(requiredOutputs, inputs[0]);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 1ull, "Hardmax: one input is expected");
        outputs.resize(1);
        outputs[0] = inputs[0];
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat src = inputs[0];
        Mat dst = outputs[0];

        int dims = src.dims;
        axis = normalize_axis(axis, dims);
        MatShape shape(src.size.p, src.size.p + src.dims);

        // Calculate strides for efficient iteration
        std::vector<size_t> strides(src.dims);
        size_t total = 1;
        for (int i = src.dims - 1; i >= 0; --i)
        {
            strides[i] = total;
            total *= shape[i];
        }

        // Prepare output
        dst.create(shape, src.type());
        dst = Scalar(0);

        // Iterate over all elements except the axis dimension
        std::vector<int> indices(src.dims, 0);
        size_t count = total / shape[axis];

        switch (src.depth())
        {
            case CV_8U: hardmaxApply<uchar>(src, dst, axis); break;
            case CV_8S: hardmaxApply<schar>(src, dst, axis); break;
            case CV_16U: hardmaxApply<ushort>(src, dst, axis); break;
            case CV_16S: hardmaxApply<short>(src, dst, axis); break;
            case CV_32S: hardmaxApply<int>(src, dst, axis); break;
            case CV_32F: hardmaxApply<float>(src, dst, axis); break;
            case CV_64F: hardmaxApply<double>(src, dst, axis); break;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Unsupported input data type");
        }
    }

    template <typename T>
    void hardmaxApply(const cv::Mat& src, cv::Mat& dst, const int axis)
    {
        const auto *src_ptr = src.ptr<T>();
        auto *dst_ptr = dst.ptr<T>();

        const size_t outer_size = src.total(0, axis);
        const auto mid_size = static_cast<size_t>(src.size[axis]);
        const size_t inner_size = src.total(axis + 1);

        const size_t outer_step = src.total(axis);
        const size_t dst_step = dst.total(axis);

        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            const size_t outer_offset = outer * outer_step;
            const size_t dst_offset = outer * dst_step;

            for (size_t inner = 0; inner < inner_size; ++inner)
            {
                T max_val = std::numeric_limits<T>::lowest();
                size_t max_idx = 0;

                // Find max along the reduction axis
                for (size_t mid = 0; mid < mid_size; ++mid)
                {
                    const size_t src_idx = outer_offset + mid * inner_size + inner;
                    if (src_ptr[src_idx] > max_val)
                    {
                        max_val = src_ptr[src_idx];
                        max_idx = mid;
                    }
                }

                // Set 1 for max, 0 for others
                for (size_t mid = 0; mid < mid_size; ++mid)
                {
                    const size_t dst_idx = dst_offset + mid * inner_size + inner;
                    dst_ptr[dst_idx] = (mid == max_idx) ? 1 : 0;
                }
            }
        }
    }

};


Ptr<HardmaxLayer> HardmaxLayer::create(const LayerParams& params)
{
    return Ptr<HardmaxLayer>(new LayerHardmaxImpl(params));
}

}}
