// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <algorithm> // for std::max & std::min

namespace cv { namespace dnn {

class ScatterLayerImpl CV_FINAL : public ScatterLayer
{
public:
    enum class REDUCTION
    {
        NONE = 1,
        ADD,
        MUL,
        MAX,
        MIN
    } reduction;

    ScatterLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 0);
        String reduction_name = toLowerCase(params.get<String>("reduction", "none"));
        if (reduction_name == "none")
            reduction = REDUCTION::NONE;
        else if (reduction_name == "add")
            reduction = REDUCTION::ADD;
        else if (reduction_name == "mul")
            reduction = REDUCTION::MUL;
        else if (reduction_name == "max")
            reduction = REDUCTION::MAX;
        else if (reduction_name == "min")
            reduction = REDUCTION::MIN;
        else
            CV_Error(cv::Error::StsBadArg, "Unkown reduction \"" + reduction_name + "\"");
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
        CV_CheckEQ(inputs.size(), 3ull, "Scatter: require three inputs.");
        CV_CheckEQ(inputs[0].size(), inputs[1].size(), "Scatter: input data should have the same ndim with indices.");
        CV_CheckEQ(inputs[0].size(), inputs[2].size(), "Scatter: input data should have the same ndim with updates.");
        for (size_t i = 0; i < inputs[0].size(); i++)
        {
            CV_CheckGE(inputs[0][i], inputs[1][i], "Scatter: each dim of input data should be greater than (or equal to) indices'.");
            CV_CheckEQ(inputs[1][i], inputs[2][i], "Scatter: each dim of indices should be equal to updates'.");
        }
        outputs.assign(1, inputs[0]);
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
        const Mat& updates = inputs[2];
        Mat& out = outputs[0];

        typeDispatch(outputs[0].type(), data, indices, updates, out);
    }

    template<typename T, typename Functor>
    void forward_impl(const Functor& rd, const Mat& data, const Mat& indices, const Mat& updates, Mat& out)
    {
        data.copyTo(out);

        const int ndims = data.dims;
        const int* shape = data.size.p;
        const size_t* step = data.step.p;

        const int* ind_shape = indices.size.p;
        const size_t* ind_step = indices.step.p;

        size_t inp_offset = 0;
        size_t ind_offset = 0;
        const T* p_index = indices.ptr<const T>();
        const T* p_update = updates.ptr<const T>();
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

            const T* tmp_p_update = p_update + ind_offset;
            T* tmp_p_out = p_out + inp_offset;
            *tmp_p_out = rd(*tmp_p_out, *tmp_p_update);
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
        switch (reduction)
        {
            case REDUCTION::NONE:
            {
                auto rd = [](const T& a, const T& b) { return b; }; // a from input data, b from updates
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::ADD:
            {
                auto rd = [](const T& a, const T& b) { return a + b; };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MUL:
            {
                auto rd = [](const T& a, const T& b) { return a * b; };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MAX:
            {
                auto rd = [](const T& a, const T& b) { return std::max(a, b); };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            case REDUCTION::MIN:
            {
                auto rd = [](const T& a, const T& b) { return std::min(a, b); };
                forward_impl<T>(rd, std::forward<Args>(args)...);
                break;
            }
            default:
                CV_Error(Error::StsBadArg, "Unsupported reduction.");
        };
    }

private:
    // Attributes
    int axis;
};

Ptr<ScatterLayer> ScatterLayer::create(const LayerParams& params)
{
    return makePtr<ScatterLayerImpl>(params);
}

}} // namespace cv::dnn
