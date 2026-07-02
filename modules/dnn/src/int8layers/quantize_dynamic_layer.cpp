// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <cstring>
#include <algorithm>

static_assert(sizeof(float) == 4, "float must be 4 bytes for int8 encoding");

namespace cv
{
namespace dnn
{

// Encode a float value as 4 raw bytes into a CV_8S Mat of shape (1, 4).
// This is used to pass float scale through the blob pipeline that only supports CV_8S dtype.
static inline void encodeFloatToInt8Mat(float value, Mat& dst)
{
    CV_Assert(dst.type() == CV_8S && dst.total() == 4);
    std::memcpy(dst.ptr(), &value, sizeof(float));
}

// Decode a float value from a CV_8S Mat of shape (1, 4).
static inline float decodeFloatFromInt8Mat(const Mat& src)
{
    CV_Assert(src.type() == CV_8S && src.total() == 4);
    float value;
    std::memcpy(&value, src.ptr(), sizeof(float));
    return value;
}

// Dynamic Quantize: compute scale/zp at runtime from activation min/max
class QuantizeDynamicLayerImpl CV_FINAL : public QuantizeDynamicLayer
{
public:
    int axis;

    QuantizeDynamicLayerImpl(const LayerParams& params)
    {
        axis = params.get<int>("axis", 1);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(3);
        outputs[0] = inputs[0];          // quantized INT8 data (same shape as input)
        outputs[1] = MatShape({1, 4});   // scale: float encoded as 4 x int8 raw bytes
        outputs[2] = MatShape({1, 1});   // zeropoint (int8, dtype matches CV_8S)
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // ONNX DynamicQuantizeLinear spec: quantize to uint8 range [0, 255]
        const int qmin = 0, qmax = 255;

        // Dynamically compute scale and zeropoint from activation range
        double rmin, rmax;
        cv::minMaxIdx(inputs[0], &rmin, &rmax);
        rmin = std::min(rmin, 0.0);
        rmax = std::max(rmax, 0.0);

        float sc = (float)((rmax == rmin) ? 1.0 : (rmax - rmin) / (qmax - qmin));
        // zp in uint8 space
        int zp_uint8 = saturate_cast<uchar>(cvRound(-rmin / sc));

        scales.resize(1); scales[0] = sc;
        zeropoints.resize(1); zeropoints[0] = zp_uint8;

        // output[0]: quantize using uint8 math, then subtract 128 to store as CV_8S
        // This matches getMatFromTensor's convention: int8_value = uint8_value - 128
        const float* inp = inputs[0].ptr<float>();
        int8_t* out = outputs[0].ptr<int8_t>();
        size_t total = inputs[0].total();
        for (size_t i = 0; i < total; i++)
        {
            // Quantize to uint8 range with round-to-nearest ties-to-even.
            int y_uint8 = saturate_cast<uchar>(cvRound(inp[i] / sc) + zp_uint8);
            // Convert to int8 for CV_8S storage
            out[i] = (int8_t)(y_uint8 - 128);
        }

        // output[1]: scale encoded as 4 raw bytes in CV_8S blob (avoids dtype mismatch)
        // output[2]: zeropoint as int8 (CV_8S dtype matches, no encoding needed)
        encodeFloatToInt8Mat(sc, outputs[1]);
        outputs[2].at<int8_t>(0) = static_cast<int8_t>(zp_uint8 - 128);
    }
};

// Dynamic Dequantize: reads scale/zp from input tensors (produced by QuantizeDynamic)
class DequantizeDynamicLayerImpl CV_FINAL : public DequantizeDynamicLayer
{
public:
    DequantizeDynamicLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // inputs[0] = INT8 data, inputs[1] = encoded scale (1x4 bytes in CV_8S), inputs[2] = zeropoint (1x1 int8)
        CV_Check(inputs.size(), inputs.size() >= 1 && inputs.size() <= 3,
                 "Number of inputs must be between 1 and 3 inclusive.");
        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        float sc = 1.0f;
        int zp_int8 = -128;  // default corresponds to zp_uint8=0 in stored int8 space (uint8_value - 128)

        if (inputs.size() > 1)
        {
            // Decode scale from 4 raw bytes in CV_8S blob
            sc = decodeFloatFromInt8Mat(inputs[1]);
            if (inputs.size() > 2)
                zp_int8 = static_cast<int>(inputs[2].at<int8_t>(0));
        }
        else if (!scales.empty())
        {
            sc = scales[0];
            // zeropoints stores uint8 value; convert to int8 space
            zp_int8 = zeropoints.empty() ? -128 : (zeropoints[0] - 128);
        }

        // Dequantize: y_int8 = uint8_value - 128, zp_int8 = zp_uint8 - 128
        // x = (y_int8 - zp_int8) * sc
        inputs[0].convertTo(outputs[0], CV_32F, sc, -sc * zp_int8);
    }
};

Ptr<QuantizeDynamicLayer> QuantizeDynamicLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeDynamicLayer>(new QuantizeDynamicLayerImpl(params));
}

Ptr<DequantizeDynamicLayer> DequantizeDynamicLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeDynamicLayer>(new DequantizeDynamicLayerImpl(params));
}

}
}
