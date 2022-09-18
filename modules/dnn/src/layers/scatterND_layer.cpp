// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <algorithm> // for std::max & std::min

namespace cv { namespace dnn {

class ScatterNDLayerImpl CV_FINAL : public ScatterNDLayer
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

    ScatterNDLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

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
        CV_CheckEQ(inputs.size(), 3ull, "ScatterND: require three inputs.");

        size_t r = inputs[0].size(), q = inputs[1].size(), p = inputs[2].size(), k = inputs[1].back();
        CV_CheckEQ(r + q - inputs[1].back() - 1, p, "ScatterND: updates should have rank of data.dims + indices.dims - indices.size[-1] - 1");
        CV_CheckLE(k, r, "ScatterND: indices.shape[-1] must be less than (or equal to) the rank of input data.");

        for (int i = 0; i < q - 1; i++) // np.ndindex(indices.shape[-1])
        {
            CV_CheckEQ(inputs[2][i], inputs[1][i], "ScatterND: updates.shape[0 : rank(indices)-1] must equal to indices.shape[0 : rank(indices)-1].");
        }
        for (int i = q - 1, j = k, m = 0; i + m < p; m++)
        {
            CV_CheckEQ(inputs[2][i + m], inputs[0][j + m], "ScatterND: updates.shape[rank(indices)-1 : ] must equal to data[indices.shape[-1] : rank(data)-1].");
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

    // NOTE: This impl does not check whether indices have duplicate entries.
    //       The last duplicate entry will overwrite the previous.
    template<typename T, typename Functor>
    void forward_impl(const Functor& rd, const Mat& data, const Mat& indices, const Mat& updates, Mat& out)
    {
        data.copyTo(out);

        const int* shape = data.size.p;
        const size_t* step = data.step.p;

        const int ind_ndims = indices.dims;
        const int* ind_shape = indices.size.p;
        const T* p_indices = indices.ptr<const T>();

        const int upd_ndims = updates.dims;
        const int* upd_shape = updates.size.p;
        const T* p_updates = updates.ptr<const T>();

        T* p_out = out.ptr<T>();

        int k = ind_shape[ind_ndims - 1]; // last dim of indices
        size_t total = (size_t)(indices.total() / k);

        size_t updates_size = 1;
        for (int i = ind_ndims - 1; i < upd_ndims; i++)
            updates_size *= upd_shape[i];

        size_t inp_start_offset = 0;
        size_t ind_start_offset = 0;
        size_t upd_start_offset = 0;
        for (size_t i = 0; i < total; i++, ind_start_offset += k, upd_start_offset += updates_size)
        {
            const T* tmp_p_indices = p_indices + ind_start_offset;
            inp_start_offset = 0;
            for (int j = 0; j < k; j++)
            {
                CV_Assert(tmp_p_indices[j] < shape[j] && tmp_p_indices[j] > -shape[j]);
                inp_start_offset += (((int)tmp_p_indices[j] + shape[j]) % shape[j]) * step[j];
            }
            inp_start_offset /= sizeof(T);

            const T* tmp_p_updates = p_updates + upd_start_offset;
            T* tmp_p_out = p_out + inp_start_offset;
            for (int j = 0; j < updates_size; j++)
                tmp_p_out[j] = rd(tmp_p_out[j], tmp_p_updates[j]);
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
};

Ptr<ScatterNDLayer> ScatterNDLayer::create(const LayerParams& params)
{
    return makePtr<ScatterNDLayerImpl>(params);
}

}} // namespace cv::dnn
