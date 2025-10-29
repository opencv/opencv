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
    using value_type = std::pair<T, WT>;
    bool operator()(const value_type& a, const value_type& b) const {
        return (a.second > b.second || (a.second == b.second && a.first < b.first));
    }
};

template<typename T, typename WT = T>
class ComparatorLess {
public:
    using value_type = std::pair<T, WT>;
    bool operator()(const value_type& a, const value_type& b) const {
        return (a.second < b.second || (a.second == b.second && a.first < b.first));
    }
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
        CV_Assert(K > 0);
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
    void FindTopK(const Mat &input, Mat &output_vals, Mat &output_idxs, int normalized_axis, int kVal)
    {
        const auto inShape = shape(input);
        size_t outer = std::accumulate(inShape.begin(), inShape.begin() + normalized_axis, 1, std::multiplies<int>());
        size_t inner = std::accumulate(inShape.begin() + normalized_axis + 1, inShape.end(), 1, std::multiplies<int>());
        int dimAxis = inShape[normalized_axis];

        auto worker = [&](const Range &r) {
            const T* inPtr = input.ptr<T>();
            T* valPtr = output_vals.ptr<T>();
            int64_t* idxPtr = output_idxs.ptr<int64_t>();

            std::vector<std::pair<uint32_t, WT>> sortbuf(dimAxis);
            for (size_t b = r.start; b < r.end; ++b) {
                for (size_t j = 0; j < inner; ++j) {
                    size_t offset = b * dimAxis * inner + j;
                    for (uint32_t u = 0; u < (uint32_t)dimAxis; ++u) {
                        sortbuf[u].first  = u;
                        sortbuf[u].second = WT(inPtr[offset + u * inner]);
                    }
                    if (largest){
                        ComparatorGreater<uint32_t,WT> cmp;
                        std::partial_sort(sortbuf.begin(), sortbuf.begin() + kVal, sortbuf.end(), cmp);
                    }
                    else{
                        ComparatorLess<uint32_t,WT> cmp;
                        std::partial_sort(sortbuf.begin(), sortbuf.begin() + kVal, sortbuf.end(), cmp);
                    }
                    for (int i = 0; i < kVal; ++i) {
                        auto &p = sortbuf[i];
                        valPtr[b * kVal * inner + i * inner + j] = T(p.second);
                        idxPtr[b * kVal * inner + i * inner + j] = p.first;
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
        Mat output_value, output_index;

        // Normalize axis to handle negative values
        int normalized_axis = normalize_axis(axis, input.dims);

        int kVal = K;
        if (dynamicK) {
            CV_Assert(inputs.size() == 2);
            CV_Assert(inputs[1].type() == CV_64S);
            CV_Assert(inputs[1].total() == 1);
            int64_t kTemp = inputs[1].at<int64_t>(0);
            CV_CheckGT(kTemp, 0, "TopK2: dynamic K must be > 0");
            CV_CheckLT(kTemp, input.size[normalized_axis] + 1, "TopK2: dynamic K is out of range");
            kVal = static_cast<int>(kTemp);
        }

        MatShape outShape = shape(input);
        outShape[normalized_axis] = kVal;

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
            CV_Error(cv::Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }

        switch (input.depth()) {
            case CV_8U:   FindTopK<uint8_t          >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_8S:   FindTopK<int8_t           >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_16U:  FindTopK<uint16_t         >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_16S:  FindTopK<int16_t          >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_16F:  FindTopK<hfloat, float    >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_16BF: FindTopK<bfloat, float    >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_32U:  FindTopK<uint32_t         >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_32S:  FindTopK<int32_t          >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_32F:  FindTopK<float            >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_64U:  FindTopK<uint64_t         >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_64S:  FindTopK<int64_t          >(input, output_value, output_index, normalized_axis, kVal); break;
            case CV_64F:  FindTopK<double           >(input, output_value, output_index, normalized_axis, kVal); break;
            default: CV_Error(Error::BadDepth, "Unsupported input data type");
        }
    }
};

Ptr<TopK2Layer> TopK2Layer::create(const LayerParams& params)
{
    return makePtr<TopK2LayerImpl>(params);
}

}} // namespace cv::dnn
