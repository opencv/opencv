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


    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
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
        std::cout << "input shape: " << src.size << std::endl;
        std::cout << "input type: " << src.type() << std::endl;

        std::cout << "output shape: " << dst.size << std::endl;
        std::cout << "output type: " << dst.type() << std::endl;

        int dims = src.dims;
        int real_axis = normalize_axis(axis, dims);
        std::cout << "axis: " << axis << std::endl;
        std::cout << "dims: " << dims << std::endl;
        std::cout << "actual_axis: " << real_axis << std::endl;
        MatShape shape(src.size.p, src.size.p + src.dims);

        // print all elements of src
        for (int i = 0; i < src.total(); ++i)
        {
            std::cout << src.ptr<float>()[i] << " ";
        }
        std::cout << std::endl;

        // Calculate strides for efficient iteration
        std::vector<size_t> strides(src.dims);
        size_t total = 1;
        for (int i = src.dims - 1; i >= 0; --i)
        {
            strides[i] = total;
            total *= shape[i];
        }
        std::cout << "strides: " << strides << std::endl;
        std::cout << "shape: " << shape << std::endl;
        std::cout << "total: " << total << std::endl;

        // Prepare output
        dst.create(shape, src.type());
        dst = Scalar(0);



        // Iterate over all elements except the axis dimension
        std::vector<int> indices(src.dims, 0);
        size_t count = total / shape[real_axis];
        for (size_t i = 0; i < count; ++i)
        {
            // Find max element along the axis
            float max_val = -std::numeric_limits<float>::max();
            int max_idx = -1;
            for (int j = 0; j < shape[real_axis]; ++j)
            {
                indices[real_axis] = j;
                size_t offset = 0;
                for (int k = 0; k < src.dims; ++k)
                    offset += indices[k] * strides[k];
                float val = src.ptr<float>()[offset];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = j;
                }
            }

            // Set the max element to 1, others to 0
            indices[real_axis] = max_idx;
            size_t offset = 0;
            for (int k = 0; k < src.dims; ++k)
                offset += indices[k] * strides[k];

            dst.ptr<float>()[offset] = 1.0f;

            // Update indices for the next iteration
            for (int j = src.dims - 1; j >= 0; --j)
            {
                if (j == real_axis) continue;
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
