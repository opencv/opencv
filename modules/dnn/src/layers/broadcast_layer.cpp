// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../broadcast_common.hpp"
#include <opencv2/core/ocl.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/broadcast.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

#include <numeric>

namespace cv { namespace dnn {

class BroadcastLayerImpl CV_FINAL : public BroadcastLayer
{
public:

    BroadcastLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return (backendId == DNN_BACKEND_OPENCV && (preferableTarget == DNN_TARGET_CPU
                                                   || preferableTarget == DNN_TARGET_OPENCL
                                                   || preferableTarget == DNN_TARGET_OPENCL_FP16))
               || (backendId == DNN_BACKEND_CUDA);
    }

    static MatShape findCommonShape(std::vector<MatShape> shapes)
    {
        CV_Assert(!shapes.empty());
        const size_t dim = std::max_element(shapes.begin(), shapes.end(),
                                            [](const MatShape& a, const MatShape& b)
                                                  { return a.size() < b.size(); })->size();

        for (auto& shape : shapes)
        {
            shape.insert(shape.begin(), dim - shape.size(), 1);
        }

        MatShape outShape(dim, 1);
        for (size_t i = 0; i < dim; ++i)
        {
            for (const auto& shape : shapes)
            {
                if (shape[i] != outShape[i])
                {
                    CV_Assert(shape[i] == 1 || outShape[i] == 1);
                    outShape[i] = std::max(outShape[i], shape[i]);
                }
            }
        }

        return outShape;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        MatShape outShape = findCommonShape(inputs);
        outputs.assign(inputs.size(), outShape);
        return false;
    }

    void cacheIndices(const std::vector<Mat>& shapes, const MatShape& outShape)
    {
        m_outShape = outShape;
        m_cache.clear();
        m_cache.resize(shapes.size());
        for (size_t j = 0; j < shapes.size(); ++j)
        {
            InputCache& cache = m_cache[j];
            const auto inputShape = shape(shapes[j]);
            cache.shape_prods = std::vector<size_t>(inputShape.begin(), inputShape.end());
            cache.shape_prods.insert(cache.shape_prods.begin(), outShape.size() - cache.shape_prods.size(), 1);
            for (size_t i = 0; i < cache.shape_prods.size(); ++i)
            {
                if (outShape[i] != cache.shape_prods[i])
                {
                    cache.broadcast_dims.push_back(i);
                }
            }
            std::reverse(cache.broadcast_dims.begin(), cache.broadcast_dims.end());
            cache.shape_prods.insert(cache.shape_prods.begin(), 1);
            std::partial_sum(cache.shape_prods.begin(), cache.shape_prods.end(),
                             cache.shape_prods.begin(), std::multiplies<size_t>());
        }
        m_prods = std::vector<size_t>(outShape.begin(), outShape.end());
        m_prods.push_back(1);
        std::partial_sum(m_prods.rbegin(), m_prods.rend(), m_prods.rbegin(), std::multiplies<size_t>());
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        MatShape outShape = shape(outputs[0]);
        cacheIndices(inputs, outShape);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        auto q = static_cast<cl_command_queue>(ocl::Queue::getDefault().ptr());
        cl_int retval = CL_SUCCESS;
        size_t elSize = inputs[0].elemSize();

        auto copier = [&retval, &q, elSize](cl_mem src, size_t src_offset, cl_mem dst, size_t dst_offset,
                                            const size_t size, const size_t times) {
            for (size_t i = 0; i < times; ++i)
            {
                retval = clEnqueueCopyBuffer(q, src, dst, src_offset * elSize, dst_offset * elSize, size * elSize, 0, 0, 0);
                CV_Assert(retval == CL_SUCCESS);
                dst_offset += size;
            }
        };

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const UMat &input = inputs[i];
            UMat &output = outputs[i];

            CV_Assert(input.isContinuous() && output.isContinuous());
            auto input_ptr = static_cast<cl_mem>(input.handle(ACCESS_READ));
            auto output_ptr = static_cast<cl_mem>(output.handle(ACCESS_WRITE));

            broadcast(input_ptr, output_ptr, m_cache[i], m_prods, m_outShape, copier);
        }
        clFinish(q); // not sure

        return true;
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
            void *context_,
            const std::vector<Ptr<BackendWrapper>>& inputs,
            const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        return make_cuda_node<cuda4dnn::BroadcastOp>(preferableTarget, context->stream, m_cache, m_prods, m_outShape);
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   outputs_arr.isUMatVector(),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        static auto copier = [](const float* src, size_t src_offset, float* dst, size_t dst_offset,
                                const size_t size, const size_t times)
        {
            src += src_offset;
            dst += dst_offset;
            for (size_t i = 0; i < times; ++i)
            {
                memcpy(dst, src, size * sizeof(float));
                dst += size;
            }
        };

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const Mat &input = inputs[i];
            Mat &output = outputs[i];

            CV_Assert(input.isContinuous() && output.isContinuous());
            const float *input_ptr = input.ptr<float>();
            float *output_ptr = output.ptr<float>();

            broadcast(input_ptr, output_ptr, m_cache[i], m_prods, m_outShape, copier);
        }
    }

private:
    std::vector<InputCache> m_cache;
    std::vector<size_t> m_prods;
    MatShape m_outShape;
};

Ptr<BroadcastLayer> BroadcastLayer::create(const LayerParams& params)
{
    return Ptr<BroadcastLayer>(new BroadcastLayerImpl(params));
}

}}  // namespace cv::dnn
