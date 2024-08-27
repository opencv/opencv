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
            case CV_8U:  hardmaxImpl<uchar>(src, dst, shape, strides, indices, count, axis); break;
            case CV_8S:  hardmaxImpl<schar>(src, dst, shape, strides, indices, count, axis); break;
            case CV_16U: hardmaxImpl<ushort>(src, dst, shape, strides, indices, count, axis); break;
            case CV_16S: hardmaxImpl<short>(src, dst, shape, strides, indices, count, axis); break;
            case CV_32S: hardmaxImpl<int>(src, dst, shape, strides, indices, count, axis); break;
            case CV_32F: hardmaxImpl<float>(src, dst, shape, strides, indices, count, axis); break;
            case CV_64F: hardmaxImpl<double>(src, dst, shape, strides, indices, count, axis); break;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Unsupported input data type");
        }
    }

    template<typename T>
    void hardmaxImpl(const Mat& src, Mat& dst, const MatShape& shape, const std::vector<size_t>& strides,
                     std::vector<int>& indices, size_t count, int axis)
    {
        for (size_t i = 0; i < count; ++i)
        {
            // Find max element along the axis
            T max_val = std::numeric_limits<T>::lowest();
            int max_idx = -1;
            for (int j = 0; j < shape[axis]; ++j)
            {
                indices[axis] = j;
                size_t offset = 0;
                for (int k = 0; k < src.dims; ++k)
                    offset += indices[k] * strides[k];

                T val = src.ptr<T>()[offset];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = j;
                }
            }

            // Set the max element to 1, others to 0
            indices[axis] = max_idx;
            size_t offset = 0;
            for (int k = 0; k < src.dims; ++k)
                offset += indices[k] * strides[k];

            dst.ptr<T>()[offset] = 1;

            // Update indices for the next iteration
            for (int j = src.dims - 1; j >= 0; --j)
            {
                if (j == axis) continue;
                if (++indices[j] < shape[j]) break;
                indices[j] = 0;
            }
        }
    }


};


Ptr<HardmaxLayer> HardmaxLayer::create(const LayerParams& params)
{
    return Ptr<HardmaxLayer>(new LayerHardmaxImpl(params));
}

}}
