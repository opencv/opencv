// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <numeric>

namespace cv {
namespace dnn {

// ONNX Col2Im operator
// Spec: https://onnx.ai/onnx/operators/onnx__Col2Im.html
// Supported opsets: 18-24

class Col2ImLayerImpl CV_FINAL : public Col2ImLayer
{
public:
    Col2ImLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        if (params.has("dilations"))
        {
            DictValue val = params.get("dilations");
            dilations_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                dilations_[i] = val.get<int>(i);
        }
        else if (params.has("dilation"))
        {
            DictValue val = params.get("dilation");
            dilations_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                dilations_[i] = val.get<int>(i);
        }

        if (params.has("pads"))
        {
            DictValue val = params.get("pads");
            pads_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                pads_[i] = val.get<int>(i);
        }
        else if (params.has("pad"))
        {
            DictValue val = params.get("pad");
            pads_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                pads_[i] = val.get<int>(i);
        }

        if (params.has("strides"))
        {
            DictValue val = params.get("strides");
            strides_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                strides_[i] = val.get<int>(i);
        }
        else if (params.has("stride"))
        {
            DictValue val = params.get("stride");
            strides_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                strides_[i] = val.get<int>(i);
        }
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        outputs.assign(1, shape(0));
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        outputs.assign(std::max(1, requiredOutputs), inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 3);

        const Mat& data = inputs[0];
        const Mat& image_shape_mat = inputs[1];
        const Mat& block_shape_mat = inputs[2];

        CV_CheckDepthEQ(data.depth(), CV_32F, "Col2Im currently supports CV_32F");
        CV_Assert(data.isContinuous());
        CV_Assert(image_shape_mat.isContinuous());
        CV_Assert(block_shape_mat.isContinuous());
        CV_Assert(data.dims == 3);

        const int N = data.size[0];
        const int C_x_block = data.size[1];
        const int L = data.size[2];

        const int D = (int)image_shape_mat.total();
        CV_Assert(D >= 1);
        CV_Assert((int)block_shape_mat.total() == D);

        AutoBuffer<int, 4> image_shape(D);
        AutoBuffer<int, 4> block_shape(D);
        size_t block_prod = 1;

        if (image_shape_mat.depth() == CV_64S)
        {
            const int64_t* ptr = image_shape_mat.ptr<int64_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_Assert(ptr[d] > 0);
                image_shape[d] = (int)ptr[d];
            }
        }
        else if (image_shape_mat.depth() == CV_32S)
        {
            const int32_t* ptr = image_shape_mat.ptr<int32_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_Assert(ptr[d] > 0);
                image_shape[d] = (int)ptr[d];
            }
        }
        else
        {
            CV_Error(Error::StsBadArg, "Col2Im: image_shape must be int32 or int64");
        }

        if (block_shape_mat.depth() == CV_64S)
        {
            const int64_t* ptr = block_shape_mat.ptr<int64_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_Assert(ptr[d] > 0);
                block_shape[d] = (int)ptr[d];
                block_prod *= (size_t)ptr[d];
            }
        }
        else if (block_shape_mat.depth() == CV_32S)
        {
            const int32_t* ptr = block_shape_mat.ptr<int32_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_Assert(ptr[d] > 0);
                block_shape[d] = (int)ptr[d];
                block_prod *= (size_t)ptr[d];
            }
        }
        else
        {
            CV_Error(Error::StsBadArg, "Col2Im: block_shape must be int32 or int64");
        }

        CV_Assert(block_prod > 0);
        CV_Assert(C_x_block % block_prod == 0);
        const int C = (int)(C_x_block / block_prod);

        AutoBuffer<int, 4> dilations(D);
        for (int d = 0; d < D; ++d)
            dilations[d] = (d < (int)dilations_.size()) ? dilations_[d] : 1;

        AutoBuffer<int, 4> strides(D);
        for (int d = 0; d < D; ++d)
            strides[d] = (d < (int)strides_.size()) ? strides_[d] : 1;

        AutoBuffer<int, 8> pads(2 * D);
        for (int d = 0; d < 2 * D; ++d)
            pads[d] = (d < (int)pads_.size()) ? pads_[d] : 0;

        AutoBuffer<int, 4> out_spatial_dims(D);
        size_t expected_L = 1;
        for (int d = 0; d < D; ++d)
        {
            int dim_padded = image_shape[d] + pads[d] + pads[d + D];
            int eff_kernel = (block_shape[d] - 1) * dilations[d] + 1;
            int l_d = (dim_padded - eff_kernel) / strides[d] + 1;
            out_spatial_dims[d] = l_d;
            expected_L *= (size_t)std::max(0, l_d);
        }
        CV_CheckEQ((size_t)L, expected_L, "Col2Im: input spatial length L does not match calculated blocks");

        MatShape outShape;
        outShape.reserve(2 + D);
        outShape.push_back(N);
        outShape.push_back(C);
        size_t spatial_size = 1;
        for (int d = 0; d < D; ++d)
        {
            outShape.push_back(image_shape[d]);
            spatial_size *= (size_t)image_shape[d];
        }

        auto kind = outputs_arr.kind();
        std::vector<Mat> tmpOuts;
        std::vector<Mat>* pOuts = nullptr;

        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            pOuts = &outputs_arr.getMatVecRef();
            CV_Assert(!pOuts->empty());
            (*pOuts)[0].fit(outShape, data.type());
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            tmpOuts.resize(uouts.size());
            tmpOuts[0].fit(outShape, data.type());
            pOuts = &tmpOuts;
        }
        else
        {
            CV_Error(Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }

        Mat& out = (*pOuts)[0];
        out.setTo(0);

        if (N == 0 || C == 0 || L == 0 || spatial_size == 0)
        {
            if (kind == _InputArray::STD_VECTOR_UMAT)
            {
                std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
                uouts[0].fit(outShape, data.type());
                tmpOuts[0].copyTo(uouts[0]);
            }
            return;
        }

        AutoBuffer<int, 4> block_steps(D);
        size_t b_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            block_steps[d] = (int)b_step;
            b_step *= (size_t)block_shape[d];
        }

        AutoBuffer<int, 4> out_steps(D);
        size_t o_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            out_steps[d] = (int)o_step;
            o_step *= (size_t)out_spatial_dims[d];
        }

        AutoBuffer<size_t, 4> img_steps(D);
        size_t im_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            img_steps[d] = im_step;
            im_step *= (size_t)image_shape[d];
        }

        std::vector<int64_t> precomp_offsets((size_t)L * block_prod, -1);
        AutoBuffer<int, 4> l_coords(D);
        for (int l = 0; l < L; ++l)
        {
            size_t rem_l = (size_t)l;
            for (int d = 0; d < D; ++d)
            {
                l_coords[d] = (int)(rem_l / (size_t)out_steps[d]);
                rem_l %= (size_t)out_steps[d];
            }

            int64_t* offsets_l = &precomp_offsets[(size_t)l * block_prod];
            for (size_t k = 0; k < block_prod; ++k)
            {
                bool valid = true;
                size_t offset = 0;
                size_t rem_k = k;
                for (int d = 0; d < D; ++d)
                {
                    int k_d = (int)(rem_k / (size_t)block_steps[d]);
                    rem_k %= (size_t)block_steps[d];

                    int pos = l_coords[d] * strides[d] + k_d * dilations[d] - pads[d];
                    if (pos < 0 || pos >= image_shape[d])
                    {
                        valid = false;
                        break;
                    }
                    offset += (size_t)pos * img_steps[d];
                }
                if (valid)
                    offsets_l[k] = (int64_t)offset;
            }
        }

        const int totalNC = N * C;
        CV_Assert(totalNC > 0);
        const size_t input_channel_stride = block_prod * (size_t)L;

        parallel_for_(Range(0, totalNC), [&](const Range& r) {
            for (int nc = r.start; nc < r.end; ++nc)
            {
                const float* src_nc = data.ptr<float>() + (size_t)nc * input_channel_stride;
                float* dst_nc = out.ptr<float>() + (size_t)nc * spatial_size;

                for (int l = 0; l < L; ++l)
                {
                    const int64_t* offsets_l = &precomp_offsets[(size_t)l * block_prod];
                    for (size_t k = 0; k < block_prod; ++k)
                    {
                        int64_t off = offsets_l[k];
                        if (off >= 0)
                        {
                            dst_nc[off] += src_nc[k * (size_t)L + (size_t)l];
                        }
                    }
                }
            }
        });

        if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            uouts[0].fit(outShape, data.type());
            tmpOuts[0].copyTo(uouts[0]);
        }
    }

private:
    std::vector<int> dilations_;
    std::vector<int> pads_;
    std::vector<int> strides_;
};

Ptr<Col2ImLayer> Col2ImLayer::create(const LayerParams& params)
{
    return makePtr<Col2ImLayerImpl>(params);
}

} // namespace dnn
} // namespace cv

