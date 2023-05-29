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

        // FP16 fallback is not needed as we handle FP16 below

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_CheckEQ(inputs.size(), (size_t)2, "");
        CV_CheckEQ(outputs.size(), (size_t)1, "");

        const Mat& inp = inputs[0];

        int indicesType = inputs[1].type();
        CV_CheckType(indicesType, indicesType == CV_32FC1 || indicesType == CV_16SC1, "");
        Mat indices32S;
        if (indicesType == CV_16S/*FP16*/)
        {
            Mat indicesF32;
            convertFp16(inputs[1], indicesF32);
            indicesF32.convertTo(indices32S, CV_32S);
        }
        else
        {
            inputs[1].convertTo(indices32S, CV_32S);
        }
        const size_t indices_total = indices32S.total();
        indices32S = indices32S.reshape(1, indices_total);

        Mat& out = outputs[0];

        CV_CheckTypeEQ(inp.type(), out.type(), "");
        CV_CheckTypeEQ(indices32S.type(), CV_32SC1, "");

        const int axis = normalize_axis(m_axis, shape(inp));

        // FIXIT: why should we work with non-normalized input? it should be handled in importer or layers's output generator
        const int axis_size = (int)inp.size[axis];
        for (size_t j = 0 ; j < indices_total; ++j)
        {
            int& idx = indices32S.at<int>(j);
            idx = normalize_axis(idx, axis_size);  // validate and normalize indices
        }

        const size_t outer_size = axis == 0 ? inp.total() : inp.step1(axis - 1);
        const size_t outer_dims = inp.total() / outer_size;
        const size_t inner_size = inp.step1(axis);

        const int* idx = indices32S.ptr<int>();
        const char* src = inp.ptr<const char>();
        char* dst = out.ptr<char>();
        CV_CheckEQ(out.total(), outer_dims * indices_total * inner_size, "");

        const size_t es = inp.elemSize1();
        // TODO: optimize through switch (inner_size * es)
        const size_t inner_bytes = inner_size * es;
        for (size_t i = 0; i < outer_dims; ++i)
        {
            const size_t src_offset = i * outer_size;
            for (size_t j = 0 ; j < indices_total; ++j)
            {
                const int index = idx[j];
                CV_DbgCheck(index, index >= 0 && index < axis_size, "");
                const size_t new_offset = src_offset + index * inner_size;
                std::memcpy(dst, src + new_offset * es, inner_bytes);
                dst += inner_bytes;
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
