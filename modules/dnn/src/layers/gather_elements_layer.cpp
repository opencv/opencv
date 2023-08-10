// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class GatherElementsLayerImpl CV_FINAL : public GatherElementsLayer
{
public:
    GatherElementsLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 2ull, "GatherElements: requires two inputs ");
        outputs.assign(1, inputs[1]); // shape of output is same as indices
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        const Mat& indices = inputs[1];
        Mat& out = outputs[0];

        typeDispatch(outputs[0].type(), data, indices, out);
    }

    template<typename T, typename Functor>
    void forward_impl(const Functor& rd, const Mat& data, const Mat& indices,  Mat& out)
    {

        const int ndims = data.dims;
        CV_Assert(ndims >= 1);
        CV_Assert(axis >= -1 * ndims && axis <= (ndims - 1));

        CV_CheckEQ(data.dims, indices.dims, "GatherElements: input and indices have to be of same rank.");

        const int* shape = data.size.p;
        const size_t* step = data.step.p;

        const int* ind_shape = indices.size.p;
        const size_t* ind_step = indices.step.p;

        size_t inp_offset = 0;
        size_t ind_offset = 0;
        const T* p_index = indices.ptr<const T>();
        const T* p_data = data.ptr<const T>();
        T* p_out = out.ptr<T>();

        size_t total = indices.total();

        int j, offset_at_idx, index;
        size_t t, idx;
        for (size_t i = 0; i < total; i++)
        {
            t = i;
            inp_offset = 0;
            ind_offset = 0;
            int offset_at_axis = 0;
            for (j = ndims - 1; j >= 0; j--)
            {
                idx = t / ind_shape[j];
                offset_at_idx = (int)(t - idx * ind_shape[j]);
                ind_offset += offset_at_idx * ind_step[j];
                inp_offset += offset_at_idx * step[j];
                t = idx;
                if (j == axis)
                {
                    offset_at_axis = offset_at_idx * step[j];
                }
            }
            ind_offset /= sizeof(T);

            // get index and overwrite current indices
            const T* tmp_p_index = p_index + ind_offset;
            index = (int)(*tmp_p_index);
            CV_Assert(index < shape[axis] && index > -shape[axis]);

            inp_offset = inp_offset - offset_at_axis + ((index + shape[axis]) % shape[axis]) * step[axis];
            inp_offset /= sizeof(T);

            const T* tmp_p_data = p_data + inp_offset;
            T* tmp_p_out = p_out + ind_offset;
            *tmp_p_out = rd(*tmp_p_out, *tmp_p_data);
        }
    }

    template<typename... Args>
    inline void typeDispatch(const int type, Args&&... args)
    {
        switch (type)
        {
            case CV_8U:
                reductionDispatch<uint8_t>(std::forward<Args>(args)...);
                break;
            case CV_32S:
                reductionDispatch<int32_t>(std::forward<Args>(args)...);
                break;
            case CV_32F:
                reductionDispatch<float>(std::forward<Args>(args)...);
                break;
            default:
                CV_Error(cv::Error::BadDepth, "Unsupported type.");
        };
    }

    template<typename T, typename... Args>
    inline void reductionDispatch(Args&&... args)
    {
        auto rd = [](const T& a, const T& b) { return b; };
        forward_impl<T>(rd, std::forward<Args>(args)...);
    }


private:
    int axis;
};

Ptr<GatherElementsLayer> GatherElementsLayer::create(const LayerParams& params)
{
    return makePtr<GatherElementsLayerImpl>(params);
}

}} // namespace cv::dnn
