#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class GatherNDLayerImpl CV_FINAL : public GatherNDLayer
{
public:
    GatherNDLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        batch_dims = params.get<int>("batch_dims", 0);
    }

    void getTypes(const std::vector<MatType>& inputs,
              const int requiredOutputs,
              const int requiredInternals,
              std::vector<MatType>& outputs,
              std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);

        MatType dataType = inputs[0];
        MatType indicesType = inputs[1];

        // Check that indices are always integer type
        CV_CheckType(indicesType, indicesType == CV_32S || indicesType == CV_64S,
                     "GatherND: indices must be CV_32S or CV_64S");

        if (preferableTarget == DNN_TARGET_OPENCL_FP16)
        {
            CV_CheckType(dataType, dataType == CV_16F || dataType == CV_8S || dataType == CV_8U ||
                                   dataType == CV_32S || dataType == CV_64S,
                         "GatherND: unsupported data type for OpenCL FP16 target");
        }
        else
        {
            CV_CheckType(dataType, dataType == CV_32F || dataType == CV_8S || dataType == CV_8U ||
                                   dataType == CV_32S || dataType == CV_64S,
                         "GatherND: unsupported data type");
        }

        outputs.resize(1, dataType);
        internals.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 2ull, "GatherND: requires two inputs");
        const MatShape& data = inputs[0];
        const MatShape& indices = inputs[1];

        int r = data.size();
        int q = indices.size();
        int last_indices_dim = indices[q - 1];

        CV_CheckGE(r, 1, "GatherND: data rank must be >= 1");
        CV_CheckGE(q, 1, "GatherND: indices rank must be >= 1");
        CV_CheckLE(batch_dims, std::min(q, r), "GatherND: batch_dims must be <= min(q, r)");
        CV_CheckGE(last_indices_dim, 1, "GatherND: last dimension of indices must be >= 1");
        CV_CheckLE(last_indices_dim, r - batch_dims, "GatherND: last dimension of indices must be <= r - batch_dims");

        MatShape output_shape;
        for (int i = 0; i < q - 1; ++i)
            output_shape.push_back(indices[i]);
        for (int i = batch_dims + last_indices_dim; i < r; ++i)
            output_shape.push_back(data[i]);

        outputs.assign(1, output_shape);
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

        int dtype = data.depth();
        int itype = indices.depth();

        switch (itype) {
            case CV_32S:
            {
                switch (dtype) {
                    case CV_8U: forward_impl<int32_t, uchar>(data, indices, out); break;
                    case CV_8S: forward_impl<int32_t, schar>(data, indices, out); break;
                    case CV_32S: forward_impl<int32_t, int32_t>(data, indices, out); break;
                    case CV_16F: forward_impl<int32_t, float16_t>(data, indices, out); break;
                    case CV_32F: forward_impl<int32_t, float>(data, indices, out); break;
                    case CV_64F: forward_impl<int32_t, double>(data, indices, out); break;
                    default: CV_Error(Error::StsNotImplemented, "Unsupported data type");
                }
            }
            case CV_64S:
            {
                switch (dtype) {
                    case CV_8U: forward_impl<int64_t, uchar>(data, indices, out); break;
                    case CV_8S: forward_impl<int64_t, schar>(data, indices, out); break;
                    case CV_32S: forward_impl<int64_t, int32_t>(data, indices, out); break;
                    case CV_16F: forward_impl<int64_t, float16_t>(data, indices, out); break;
                    case CV_32F: forward_impl<int64_t, float>(data, indices, out); break;
                    case CV_64F: forward_impl<int64_t, double>(data, indices, out); break;
                    default: CV_Error(Error::StsNotImplemented, "Unsupported data type");
                }
            }
        }

    }

    template <typename iT, typename dT>
    void forward_impl(const Mat& data, const Mat& indices, Mat& out)
    {
        CV_Assert(out.isContinuous());
        const iT* indices_ptr = indices.ptr<iT>();
        const dT* data_ptr = data.ptr<dT>();
        dT* out_ptr = out.ptr<dT>();

        int r = data.dims;
        int q = indices.dims;
        int last_indices_dim = indices.size[q - 1];

        std::vector<int> data_strides(r);
        data_strides[r - 1] = 1;
        for (int i = r - 2; i >= 0; --i)
            data_strides[i] = data_strides[i + 1] * data.size[i + 1];

        std::vector<int> indices_strides(q);
        indices_strides[q - 1] = 1;
        for (int i = q - 2; i >= 0; --i)
            indices_strides[i] = indices_strides[i + 1] * indices.size[i + 1];

        const int outer_size = indices.total() / last_indices_dim;
        const int inner_size = out.total() / outer_size;

        for (size_t i = 0; i < outer_size; ++i)
        {
            std::vector<int> sliced_indices(indices_ptr + i * last_indices_dim, indices_ptr + (i + 1) * last_indices_dim);

            size_t offset = 0;
            for (size_t j = 0; j < last_indices_dim; ++j)
            {
                offset += sliced_indices[j] * data_strides[batch_dims + j];
            }

            if (batch_dims > 0)
                offset += data_strides[batch_dims - 1] * i;

            // copy data from data to out
            for (size_t j = 0; j < inner_size; ++j)
            {
                out_ptr[i * inner_size + j] = data_ptr[offset + j];
            }
        }
    }

private:
    int batch_dims;
};

Ptr<GatherNDLayer> GatherNDLayer::create(const LayerParams& params)
{
    return Ptr<GatherNDLayer>(new GatherNDLayerImpl(params));
}

}} // namespace cv::dnn