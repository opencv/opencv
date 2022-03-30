// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "../op_cuda.hpp"
#include "../op_webnn.hpp"

#include <float.h>
#include <algorithm>
#include <numeric>
using std::max;
using std::min;

#include <opencv2/core/utils/logger.hpp>

namespace cv
{
namespace dnn
{

class ReduceLayerImpl CV_FINAL : public ReduceLayer
{
public:
    ReduceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        // set reduce type
        CV_Assert(params.has("reduce"));
        String typeString = toLowerCase(params.get<String>("reduce"));
        if (typeString == "max")
            reduceType= MAX;
        else if (typeString == "min")
            reduceType= MIN;
        else if (typeString == "ave")
            reduceType= AVE;
        else if (typeString == "sum")
            reduceType= SUM;
        else if (typeString == "sum_square")
            reduceType= SUM_SQUARE;
        else if (typeString == "l1")
            reduceType= L1;
        else if (typeString == "l2")
            reduceType= L2;
        else if (typeString == "log_sum")
            reduceType= LOG_SUM;
        else if (typeString == "log_sum_exp")
            reduceType= LOG_SUM_EXP;
        else if (typeString == "prod")
            reduceType= PROD;
        else
            CV_Error(Error::StsBadArg, "Unknown reduce type\"" + typeString + "\"");

        // set deleted dims
        CV_Assert(params.has("deleted_dims"));
        DictValue tempDims = params.get("deleted_dims");
        int i, n = tempDims.size();
        reduceDims.resize(n);
        for (i = 0; i < n; i++)
        {
            reduceDims[i] = tempDims.get<int>(i);
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
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, FLT_MAX,
                                   [](float a, float b)
                                   {
                                       return std::min(a, b);
                                   });
        }
    };

    // reduceType == MAX
    struct ReduceOpMAX
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, -FLT_MAX,
                                   [](float a, float b)
                                   {
                                       return std::max(a, b);
                                   });
        }
    };

    // reduceType == SUM
    struct ReduceOpSUM
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, 0.f);
        }
    };

    // reduceType == AVE
    struct ReduceOpAVE
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            float output = std::accumulate(first, last, 0.f);
            return output * ikarea;
        }
    };

    // reduceType == SUM_SQUARE
    struct ReduceOpSUM_SQUARE
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, 0.f,
                                   [](float a, float b)
                                   {
                                       return a + b * b;
                                   });
        }
    };

    // reduceType == L1
    struct ReduceOpL1
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, 0.f,
                                   [](float a, float b)
                                   {
                                       return a + std::abs(b);
                                   });
        }
    };

    // reduceType == L2
    struct ReduceOpL2
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            float output = std::accumulate(first, last, 0.f,
                                           [](float a, float b)
                                           {
                                               return a + b * b;
                                           });
            return std::sqrt(output);
        }
    };

    // reduceType == PROD
    struct ReduceOpPROD
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            return std::accumulate(first, last, 1.0f, std::multiplies<float>());
        }
    };

    // reduceType == LOG_SUM
    struct ReduceOpLOG_SUM
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            float output = std::accumulate(first, last, 0.0f);
            return std::log(output);
        }
    };

    // reduceType == LOG_SUM_EXP
    struct ReduceOpLOG_SUM_EXP
    {
        float apply(const float* first, const float* last, const float ikarea = 1.0f)
        {
            float output = std::accumulate(first, last, 0.0f,
                                           [](float a, float b)
                                           {
                                               return a + std::exp(b);
                                           });
            return std::log(output);
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
            CV_Assert_N( src.isContinuous(), dst.isContinuous(), src.type() == CV_32F, src.type() == dst.type());

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
            size_t stride_w = std::accumulate(reduceDims.begin(), reduceDims.end(), 1, std::multiplies<size_t>());

            float *dstData = (float *)dst->data;
            float *srcData = (float *)src->data;

            for (size_t ofs = stripeStart; ofs < stripeEnd;)
            {
                const float* first = srcData + ofs * stride_w;
                const float* last = srcData + (ofs + 1) * stride_w;

                if (ofs < stripeEnd)
                {
                    dstData[ofs] = func->apply(first, last, 1.0 / stride_w);
                    ofs += 1;
                }
            }
        }
    };

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(inputs.size() == 1 || (inputs.size() == 2 && reduceType== SUM));
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
            case AVE:
            {
                ReduceInvoker<ReduceOpAVE>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case SUM:
            {
                ReduceInvoker<ReduceOpSUM>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case L1:
            {
                ReduceInvoker<ReduceOpL1>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case L2:
            {
                ReduceInvoker<ReduceOpL2>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case SUM_SQUARE:
            {
                ReduceInvoker<ReduceOpSUM_SQUARE>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case PROD:
            {
                ReduceInvoker<ReduceOpPROD>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case LOG_SUM:
            {
                ReduceInvoker<ReduceOpLOG_SUM>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
                break;
            }
            case LOG_SUM_EXP:
            {
                ReduceInvoker<ReduceOpLOG_SUM_EXP>::run(inputs[0], outputs[0], reduceDims, reduceType, nstripes);
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
        CV_Assert(reduceDims.size() != 0 && inputs[0].size() >= reduceDims.size());

        std::vector<int> outShape;
        if (inputs[0].size() == reduceDims.size())
            outShape.push_back(1);
        else
        {
            for (int i = 0; i < inputs[0].size() - reduceDims.size(); i++)
            {
                outShape.push_back(inputs[0][i]);
            }
        }
        outputs.assign(1, outShape);

        return false;
    }

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        if (reduceType== MAX || reduceType== MIN)
        {
            return true;
        }
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(inputs); // suppress unused variable warning
        long flops = 0;
        size_t stride_w = std::accumulate(reduceDims.begin(), reduceDims.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(stride_w);
        }
        return flops;
    }
private:
    enum ReduceType
    {
        MAX,
        MIN,
        AVE,
        SUM,
        L1,
        L2,
        PROD,
        SUM_SQUARE,
        LOG_SUM,
        LOG_SUM_EXP
    };
};

Ptr<ReduceLayer> ReduceLayer::create(const LayerParams& params)
{
    return Ptr<ReduceLayer>(new ReduceLayerImpl(params));
}

}
}
