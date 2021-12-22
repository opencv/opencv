// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"


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
        return backendId == DNN_BACKEND_OPENCV && (preferableTarget == DNN_TARGET_CPU
                                                   || preferableTarget == DNN_TARGET_OPENCL
                                                   || preferableTarget == DNN_TARGET_OPENCL_FP16);
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
        m_cache.clear();
        m_cache.resize(shapes.size(), InputCache(outShape.size()));
        for (size_t j = 0; j < shapes.size(); ++j)
        {
            InputCache& cache = m_cache[j];
            cache.alignedShape = shape(shapes[j]);
            cache.alignedShape.insert(cache.alignedShape.begin(), outShape.size() - cache.alignedShape.size(), 1);
            for (size_t i = 0; i < cache.alignedShape.size(); ++i)
            {
                if (outShape[i] != cache.alignedShape[i])
                {
                    cache.ids.push_back(i);
                    cache.idx_to[i] = cv::Range(0, 1);
                }
            }
        }
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        MatShape outShape = shape(outputs[0]);
        cacheIndices(inputs, outShape);
    }

    static bool advance(std::vector<size_t>& ids,
                        int& k,
                        std::vector<cv::Range>& idx_to,
                        std::vector<cv::Range>& idx_from,
                        const MatShape& out_shape)
    {
        for (int i = 0; i < k;)
        {
            size_t& d = ids[i];
            const int old_end = idx_to[d].end;
            int new_end = std::min(out_shape[d], old_end * 2);
            if (new_end > old_end)
            {
                idx_to[d] = cv::Range(idx_to[d].end, new_end);
                idx_from[d].end = idx_to[d].size();
                return true;
            }
            idx_to[d] = cv::Range::all();
            idx_from[d] = cv::Range::all();
            std::swap(d, ids[k - 1]);
            --k;
        }
        return false;
    }

    template <typename T>
    void forward_generic(const std::vector<T>& inputs, std::vector<T>& outputs)
    {
        MatShape outShape = shape(outputs[0]);

        std::vector<cv::Range> idx_to;
        std::vector<cv::Range> idx_from;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            const T& input = inputs[i];
            T& output = outputs[i];
            InputCache& cache = m_cache[i];

            idx_to = cache.idx_to;
            idx_from = idx_to;

            int k = static_cast<int>(cache.ids.size());
            input.reshape(1, cache.alignedShape.size(), cache.alignedShape.data()).copyTo(output(idx_to));
            while (advance(cache.ids, k, idx_to, idx_from, outShape))
            {
                output(idx_from).copyTo(output(idx_to));
            }
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        forward_generic(inputs, outputs);
        return false;
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

        forward_generic(inputs, outputs);
    }

private:
    struct InputCache
    {
        InputCache(size_t size) : idx_to(size, cv::Range::all()) {}

        std::vector<size_t> ids;
        std::vector<cv::Range> idx_to;
        MatShape alignedShape;
    };

    std::vector<InputCache> m_cache;
};

Ptr<BroadcastLayer> BroadcastLayer::create(const LayerParams& params)
{
    return Ptr<BroadcastLayer>(new BroadcastLayerImpl(params));
}

}}  // namespace cv::dnn
