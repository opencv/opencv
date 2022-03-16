// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class GatherLayerImpl CV_FINAL : public GatherLayer
{
public:
    GatherLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        m_axis = params.get<int>("axis", 0);
        m_real_ndims = params.get<int>("real_ndims", -1);
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
        CV_CheckEQ(inputs.size(), 2ull, "");
        MatShape inpShape = inputs[0];
        const int axis = normalize_axis(m_axis, inpShape);

        inpShape.erase(inpShape.begin() + axis);
        auto end = m_real_ndims == -1 ? inputs[1].end() : inputs[1].begin() + m_real_ndims;
        inpShape.insert(inpShape.begin() + axis, inputs[1].begin(), end);

        outputs.assign(1, inpShape);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& inp = inputs[0];
        const Mat& indices = inputs[1];
        Mat& out = outputs[0];

        const int axis = normalize_axis(m_axis, shape(inp));

        const size_t outer_size = axis == 0 ? inp.total() : inp.step1(axis - 1);
        const size_t outer_dims = inp.total() / outer_size;
        const size_t inner_size = inp.step1(axis);

        const float* idx = indices.ptr<const float>(); // TODO: change type to integer in the future.
        const char* src = inp.ptr<const char>();
        char* dst = out.ptr<char>();

        const size_t es = inp.elemSize1();
        for (size_t i = 0; i < outer_dims; ++i)
        {
            const size_t src_offset = i * outer_size;
            for (size_t j = 0 ; j < indices.total(); ++j)
            {
                const size_t index = (static_cast<int>(idx[j]) + inp.size[axis]) % inp.size[axis];
                const size_t new_offset = src_offset + index * inp.step1(axis);
                std::memcpy(dst, src + new_offset * es, inner_size * es);
                dst += inner_size * es;
            }
        }
    }

private:
    // The axis to gather along
    int m_axis;
    int m_real_ndims;
};

Ptr<GatherLayer> GatherLayer::create(const LayerParams& params)
{
    return makePtr<GatherLayerImpl>(params);
}

}}  // namespace cv::dnn
