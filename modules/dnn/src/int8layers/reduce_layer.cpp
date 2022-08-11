// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <algorithm>
#include <stdlib.h>
#include <numeric>

namespace cv
{
namespace dnn
{

class ReduceLayerInt8Impl CV_FINAL : public ReduceLayerInt8
{
public:
    ReduceLayerInt8Impl(const LayerParams& params)
    {
        // Set reduce type
        CV_Assert(params.has("reduce"));
        String typeString = toLowerCase(params.get<String>("reduce"));
        if (typeString == "max")
            reduceType = MAX;
        else if (typeString == "min")
            reduceType = MIN;
        else
            CV_Error(Error::StsBadArg, "Unknown reduce type \"" + typeString + "\"");

        // Set deleted dims
        CV_Assert(params.has("deleted_dims"));
        DictValue tempDims = params.get("deleted_dims");
        int i, n = tempDims.size();
        reduceDims.resize(n);
        for (i = 0; i < n; i++)
        {
            reduceDims[i] = tempDims.get<int>(i);
        }

        CV_Assert(params.has("target_dims"));
        tempDims = params.get("target_dims");
        n = tempDims.size();
        targetDims.resize(n);
        for (i = 0; i < n; i++)
        {
            targetDims[i] = tempDims.get<int>(i);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_OPENCV)
        {
            return true;
        }
        return false;
    }

    // reduceType == MIN
    struct ReduceOpMIN
    {
        int8_t apply(const int8_t* first, const int8_t* last)
        {
            return std::accumulate(first, last, *first,
                                   [](int8_t a, int8_t b)
                                   {
                                       return std::min(a, b);
                                   });
        }
    };

    // reduceType == MAX
    struct ReduceOpMAX
    {
        int8_t apply(const int8_t* first, const int8_t* last)
        {
            return std::accumulate(first, last, *first,
                                   [](int8_t a, int8_t b)
                                   {
                                       return std::max(a, b);
                                   });
        }
    };

    template<typename Func>
    class ReduceInvoker : public ParallelLoopBody
    {
    public:
        const Mat* src;
        Mat *dst;
        std::vector<size_t> reduceDims;
        int nstripes;
        int reduceType;
        Ptr<Func> func;

        ReduceInvoker() : src(0), dst(0), nstripes(0), reduceType(MAX), func(makePtr<Func>()) {}

        static void run(const Mat& src, Mat& dst, std::vector<size_t> reduceDims, int reduceType, int nstripes)
        {
            CV_Assert_N(src.isContinuous(), dst.isContinuous(), src.type() == CV_8S, src.type() == dst.type());

            ReduceInvoker<Func> p;

            p.src = &src;
            p.dst = &dst;

            p.reduceDims = reduceDims;
            p.nstripes = nstripes;
            p.reduceType = reduceType;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            size_t total = dst->total();
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, total);
            size_t totalDeleted = std::accumulate(reduceDims.begin(), reduceDims.end(), 1, std::multiplies<size_t>());

            int8_t *dstData = (int8_t *)dst->data;
            int8_t *srcData = (int8_t *)src->data;

            for (size_t ofs = stripeStart; ofs < stripeEnd;)
            {
                const int8_t* first = srcData + ofs * totalDeleted;
                const int8_t* last = srcData + (ofs + 1) * totalDeleted;

                dstData[ofs] = func->apply(first, last);
                ofs += 1;
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(inputs.size() == 1);
        const int nstripes = getNumThreads();

        switch (reduceType)
        {
            case MIN:
            {
                ReduceInvoker<ReduceOpMIN>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case MAX:
            {
                ReduceInvoker<ReduceOpMAX>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            default:
                CV_Error(Error::StsNotImplemented, "Not implemented");
                break;
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert( reduceDims.size() !=0 && targetDims.size() != 0 && inputs[0].size() >= reduceDims.size());

        // outShapeTmp can save the right number of `total(outShapeTmp)`. And the outShape is used as the final output shape.
        std::vector<int> outShapeTmp, outShape;
        outShape.assign(targetDims.begin(), targetDims.end());
        if (inputs[0].size() == reduceDims.size())
            outShapeTmp.push_back(1);
        else
        {
            for (int i = 0; i < inputs[0].size() - reduceDims.size(); i++)
            {
                outShapeTmp.push_back(inputs[0][i]);
            }
        }

        // Support dynamic shape of Batch size.
        // Note that: when there are multiple dynamic inputs, we will give an error.
        if (total(outShape) != total(outShapeTmp))
        {
            if (outShape[0] != outShapeTmp[0])
                outShape[0] = outShapeTmp[0];
        }

        CV_Assert(total(outShape) == total(outShapeTmp));
        outputs.assign(1, outShape);

        return false;
    }

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;
        size_t totalDeleted = std::accumulate(reduceDims.begin(), reduceDims.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(totalDeleted);
        }
        return flops;
    }
private:
    enum Type
    {
        MAX,
        MIN
    };
};

Ptr<ReduceLayerInt8> ReduceLayerInt8::create(const LayerParams& params)
{
    return Ptr<ReduceLayerInt8>(new ReduceLayerInt8Impl(params));
}

}
}
