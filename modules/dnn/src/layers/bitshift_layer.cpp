// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <type_traits>

// ONNX reference: BitShift operator
// Spec: https://onnx.ai/onnx/operators/onnx__BitShift.html
// Supported opsets: ai.onnx opset 11 and newer; opset 28 added the signed types.
// NOTE: Broadcasting is NOT fully supported. Only two cases are handled:
// 1) array (input[0]) shifted by array (input[1]) of the SAME SHAPE
// 2) array (input[0]) shifted by a SCALAR (0-D) shift amount

namespace cv {
namespace dnn {

template<typename T>
static inline T doShift(T inputVal, T shiftVal, int direction, int bitWidth)
{
    typedef typename std::make_unsigned<T>::type U;
    const U uval = (U)inputVal;
    // A negative signed shift reinterprets as a huge unsigned, so one test rejects both the
    // negative and the too-wide amount.
    const uint64_t n = (uint64_t)(U)shiftVal;
    const bool negative = std::is_signed<T>::value && (uval >> (bitWidth - 1)) != 0;

    // Out of range leaves only the sign fill behind.
    if (n >= (uint64_t)bitWidth)
        return (direction && negative) ? T(-1) : T(0);
    // Left shifts travel through the unsigned domain: bits reaching the sign bit are kept
    // and bits past it are dropped, where a signed overflow would be undefined.
    if (!direction)
        return (T)(U)(uval << n);
    U res = (U)(uval >> n);
    if (negative && n != 0)
        res |= (U)((U)~(U)0 << (bitWidth - n));  // arithmetic shift: vacated bits take the sign
    return (T)res;
}

template<typename T, int CvTypeConst, int BitWidth>
void runBitShift(const Mat& input, const Mat& shift, Mat& output, int direction)
{
    output.create(input.size, input.type());
    const size_t numElements = input.total();

    const T* inputPtr = input.ptr<T>();
    T* outputPtr = output.ptr<T>();

    if (shift.total() == 1)
    {
        T shiftScalar = 0;
        tensorToScalar(shift, CvTypeConst, &shiftScalar);
        parallel_for_(Range(0, (int)numElements), [&](const Range& r){
            for (int i = r.start; i < r.end; ++i)
                outputPtr[i] = doShift<T>(inputPtr[i], shiftScalar, direction, BitWidth);
        });
    }
    else
    {
        CV_Assert(shift.size == input.size);
        CV_Assert(shift.type() == CvTypeConst);
        const T* shiftPtr = shift.ptr<T>();
        parallel_for_(Range(0, (int)numElements), [&](const Range& r){
            for (int i = r.start; i < r.end; ++i)
                outputPtr[i] = doShift<T>(inputPtr[i], shiftPtr[i], direction, BitWidth);
        });
    }
}

class BitShiftLayerImpl CV_FINAL : public BitShiftLayer
{
    int direction_; // 0=LEFT, 1=RIGHT

public:
    BitShiftLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        direction_ = params.get<int>("direction", 0);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int requiredOutputs,
                         std::vector<MatShape>& outputs, std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        bool isScalarShift = inputs[1].empty() || total(inputs[1]) == 1;
        CV_Assert(isScalarShift || inputs[1] == inputs[0]);
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& in, const int reqOut, const int reqInt,
                  std::vector<MatType>& out, std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(in.size() >= 2);
        int t = in[0];
        CV_Assert(t == CV_8U || t == CV_16U || t == CV_32U || t == CV_64U ||
                  t == CV_8S || t == CV_16S || t == CV_32S || t == CV_64S);
        CV_Assert(in[1] == in[0]);
        out.assign(reqOut, MatType(t));
        internals.assign(reqInt, MatType(t));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() >= 2);
        const Mat& input = inputs[0];
        const Mat& shift = inputs[1];
        Mat& output = outputs[0];

        const int depth = input.depth();
        if (depth == CV_8U)
        {
            runBitShift<uint8_t, CV_8U, 8>(input, shift, output, direction_);
        }
        else if (depth == CV_16U)
        {
            runBitShift<uint16_t, CV_16U, 16>(input, shift, output, direction_);
        }
        else if (depth == CV_32U)
        {
            runBitShift<uint32_t, CV_32U, 32>(input, shift, output, direction_);
        }
        else if (depth == CV_64U)
        {
            runBitShift<uint64_t, CV_64U, 64>(input, shift, output, direction_);
        }
        else if (depth == CV_8S)
        {
            runBitShift<int8_t, CV_8S, 8>(input, shift, output, direction_);
        }
        else if (depth == CV_16S)
        {
            runBitShift<int16_t, CV_16S, 16>(input, shift, output, direction_);
        }
        else if (depth == CV_32S)
        {
            runBitShift<int32_t, CV_32S, 32>(input, shift, output, direction_);
        }
        else if (depth == CV_64S)
        {
            runBitShift<int64_t, CV_64S, 64>(input, shift, output, direction_);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "BitShift: unsupported depth");
        }
    }
};

Ptr<BitShiftLayer> BitShiftLayer::create(const LayerParams& params)
{
    return Ptr<BitShiftLayer>(new BitShiftLayerImpl(params));
}

}}
