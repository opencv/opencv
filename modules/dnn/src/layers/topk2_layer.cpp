// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

/*
    TopK layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__TopK.html

    Opset’s 1, 10 and 11 are covered.
*/

namespace {

template<typename T, typename WT = T>
class ComparatorGreater {
public:
    ComparatorGreater(const T* data, size_t step)
        : data_(data), step_(step) {}


    bool operator()(const size_t lhs_idx, const size_t rhs_idx) {
        WT lhs = static_cast<WT>(*(data_ + lhs_idx * step_));
        WT rhs = static_cast<WT>(*(data_ + rhs_idx * step_));
        return (lhs > rhs || (lhs == rhs && lhs_idx < rhs_idx));
    }

private:
    const T* data_;
    size_t step_;
};

template<typename T, typename WT = T>
class ComparatorLess {
public:
    ComparatorLess(const T* data, size_t step)
        : data_(data), step_(step) {}

    bool operator()(const size_t lhs_idx, const size_t rhs_idx) {
        WT lhs = static_cast<WT>(*(data_ + lhs_idx * step_));
        WT rhs = static_cast<WT>(*(data_ + rhs_idx * step_));
        return (lhs < rhs || (lhs == rhs && lhs_idx < rhs_idx));
    }

private:
    const T* data_;
    size_t step_;
};
}

class TopK2LayerImpl CV_FINAL : public TopK2Layer
{
public:
    int axis;
    bool largest;
    bool sorted;
    int K;
    bool dynamicK;

    TopK2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis    = params.get<int>("axis", -1);
        largest = params.get<int>("largest", 1) == 1;
        sorted  = params.get<int>("sorted", 1) == 1;
        CV_CheckTrue(sorted, "TopK2: sorted == false is not supported");
        if (params.has("k")) {
            K = params.get<int>("k");
            CV_CheckGT(K, 0, "TopK2: K needs to be a positive integer");
            dynamicK = false;
        } else {
            dynamicK = true;
            K = 0;
        }
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return dynamicK;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        const MatShape& inShape = inputs[0];
        int inputDims = static_cast<int>(inShape.size());
        int a = normalize_axis(axis, inputDims);
        CV_Assert(a >= 0 && a < inputDims);
        CV_CheckLT(K, inShape[a] + 1, "TopK2: K is out of range");

        MatShape outShape = inShape;
        outShape[a] = K;
        outputs.assign(2, outShape);
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.resize(2);
        outputs[0] = inputs.front();
        outputs[1] = CV_64S;
    }

private:
    template<typename T, typename WT = T>
    void FindTopK(const Mat &input, Mat &output_vals, Mat &output_idxs, int normalized_axis)
    {
        const auto inShape = shape(input);
        size_t outer = std::accumulate(inShape.begin(), inShape.begin() + normalized_axis, 1, std::multiplies<int>());
        size_t inner = std::accumulate(inShape.begin() + normalized_axis + 1, inShape.end(), 1, std::multiplies<int>());
        int dimAxis = inShape[normalized_axis];

        auto worker = [&](const Range &r) {
            const T* inPtr = input.ptr<T>();
            T* valPtr = output_vals.ptr<T>();
            int64_t* idxPtr = output_idxs.ptr<int64_t>();
            AutoBuffer<size_t> indices(dimAxis);
            size_t* idxBuf = indices.data();
            for (size_t b = r.start; b < r.end; ++b) {
                for (size_t j = 0; j < inner; ++j) {
                    size_t offset = b * dimAxis * inner + j;
                    if (largest){
                        ComparatorGreater<T,WT> cmp(inPtr + offset, inner);
                        std::iota(idxBuf, idxBuf + dimAxis, 0);
                        std::partial_sort(idxBuf, idxBuf + K, idxBuf + dimAxis, cmp);
                    }
                    else{
                        ComparatorLess<T,WT>    cmp(inPtr + offset, inner);
                        std::iota(idxBuf, idxBuf + dimAxis, 0);
                        std::partial_sort(idxBuf, idxBuf + K, idxBuf + dimAxis, cmp);
                    }

                    for (int i = 0; i < K; ++i) {
                        size_t src = idxBuf[i];
                        valPtr[b * K * inner + i * inner + j] = inPtr[offset + src * inner];
                        idxPtr[b * K * inner + i * inner + j] = static_cast<int64_t>(src);
                    }
                }
            }
        };
        parallel_for_(Range(0, static_cast<int>(outer)), worker);
    }

public:
    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto &input = inputs.front();
        auto &output_value = outputs.front();
        auto &output_index = outputs.back();

        // Normalize axis to handle negative values
        int normalized_axis = normalize_axis(axis, input.dims);

        if (dynamicK) {
            CV_Assert(inputs.size() == 2);
            int64_t kVal = inputs[1].at<int64_t>(0);
            CV_CheckGT(kVal, 0, "TopK2: dynamic K must be > 0");
            CV_CheckLT(kVal, input.size[normalized_axis] + 1, "TopK2: dynamic K is out of range");
            K = static_cast<int>(kVal);

            MatShape outShape = shape(input);
            outShape[normalized_axis] = K;

            auto kind = outputs_arr.kind();
            if (kind == _InputArray::STD_VECTOR_MAT) {
                std::vector<Mat>& outs = outputs_arr.getMatVecRef();
                CV_Assert(outs.size() == 2);
                outs[0].fit(outShape, input.type());
                outs[1].fit(outShape, CV_64S);
                output_value = outs[0];
                output_index = outs[1];
            } else if (kind == _InputArray::STD_VECTOR_UMAT) {
                std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
                CV_Assert(uouts.size() == 2);
                uouts[0].fit(outShape, input.type());
                uouts[1].fit(outShape, CV_64S);
                output_value = uouts[0].getMat(ACCESS_WRITE);
                output_index = uouts[1].getMat(ACCESS_WRITE);
            } else {
                // Fallback – recreate directly on provided Mat refs.
                output_value.create(outShape, input.type());
                output_index.create(outShape, CV_64S);
            }
        }
        switch (input.depth()) {
            case CV_8U:   FindTopK<uint8_t          >(input, output_value, output_index, normalized_axis); break;
            case CV_8S:   FindTopK<int8_t           >(input, output_value, output_index, normalized_axis); break;
            case CV_16U:  FindTopK<uint16_t         >(input, output_value, output_index, normalized_axis); break;
            case CV_16S:  FindTopK<int16_t          >(input, output_value, output_index, normalized_axis); break;
            case CV_16F:  FindTopK<hfloat,   float   >(input, output_value, output_index, normalized_axis); break;
            case CV_16BF: FindTopK<bfloat, float   >(input, output_value, output_index, normalized_axis); break;
            case CV_32U:  FindTopK<uint32_t         >(input, output_value, output_index, normalized_axis); break;
            case CV_32S:  FindTopK<int32_t          >(input, output_value, output_index, normalized_axis); break;
            case CV_32F:  FindTopK<float            >(input, output_value, output_index, normalized_axis); break;
            case CV_64U:  FindTopK<uint64_t         >(input, output_value, output_index, normalized_axis); break;
            case CV_64S:  FindTopK<int64_t          >(input, output_value, output_index, normalized_axis); break;
            case CV_64F:  FindTopK<double           >(input, output_value, output_index, normalized_axis); break;
            default: CV_Error(Error::BadDepth, "Unsupported input data type");
        }
    }
};

Ptr<TopK2Layer> TopK2Layer::create(const LayerParams& params)
{
    return makePtr<TopK2LayerImpl>(params);
}

}} // namespace cv::dnn
