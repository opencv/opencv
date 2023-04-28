// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "../op_cuda.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

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

        CV_Assert(params.has("target_dims"));
        tempDims = params.get("target_dims");
        n = tempDims.size();
        targetDims.resize(n);
        for (i = 0; i < n; i++)
        {
            targetDims[i] = tempDims.get<int>(i);
        }

        // save original axes
        if (params.has("axes"))
        {
            DictValue tempAxes = params.get("axes");
            int axesNum = tempAxes.size();
            axes.resize(axesNum);
            for (int j = 0; j < axesNum; ++j)
            {
                axes[j] = tempAxes.get<int>(j);
            }
        }

        // save keepdims
        keepdims = params.get<int>("keepdims", 1) == 1;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_CANN
        if (backendId == DNN_BACKEND_CANN)
            return reduceType == ReduceType::MAX  || reduceType == ReduceType::MIN     ||
                   reduceType == ReduceType::AVE  || reduceType == ReduceType::SUM     ||
                   reduceType == ReduceType::PROD || reduceType == ReduceType::LOG_SUM ||
                   reduceType == ReduceType::LOG_SUM_EXP;
#endif
        return backendId == DNN_BACKEND_OPENCV;
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
        if (total(outShape) != total(outShapeTmp) && outShape[0] != outShapeTmp[0])
        {
                outShape[0] = outShapeTmp[0];
        }

        CV_Assert(total(outShape) == total(outShapeTmp));
        outputs.assign(1, outShape);

        return false;
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_CheckFalse(axes.empty(), "DNN/CANN: Reduce layers need axes to build CANN operators");

        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto desc_x = x->getTensorDesc();

        std::vector<int> axes_shape{(int)axes.size()};
        Mat axes_mat(axes_shape, CV_32SC1, &axes[0]);
        auto op_const_axes = std::make_shared<CannConstOp>(axes_mat.data, axes_mat.type(), axes_shape, cv::format("%s_axes", name.c_str()));
        auto desc_axes = op_const_axes->getTensorDesc();

        auto desc_y = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);

        std::shared_ptr<ge::Operator> reduce_op = nullptr;
        switch (reduceType)
        {
#define BUILD_CANN_REDUCE_OP(op_type, class_name, op_name)               \
            case op_type: {                                              \
                auto op = std::make_shared<ge::op::class_name>(op_name); \
                op->set_input_x_by_name(*op_x, x->name.c_str());         \
                op->set_input_axes(*(op_const_axes)->getOp());           \
                op->set_attr_keep_dims(keepdims);                        \
                op->update_input_desc_x(*desc_x);                        \
                op->update_input_desc_axes(*desc_axes);                  \
                op->update_output_desc_y(*desc_y);                       \
                reduce_op = op;                                          \
            } break;
            BUILD_CANN_REDUCE_OP(ReduceType::MAX,         ReduceMax,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::MIN,         ReduceMin,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::AVE,         ReduceMean,      name);
            BUILD_CANN_REDUCE_OP(ReduceType::SUM,         ReduceSum,       name);
            BUILD_CANN_REDUCE_OP(ReduceType::PROD,        ReduceProd,      name);
            BUILD_CANN_REDUCE_OP(ReduceType::LOG_SUM,     ReduceLogSum,    name);
            BUILD_CANN_REDUCE_OP(ReduceType::LOG_SUM_EXP, ReduceLogSumExp, name);
#undef BUILD_CANN_REDUCE_OP
            default: CV_Error(Error::StsNotImplemented, "Unsupported reduce operation");
        }

        return Ptr<BackendNode>(new CannBackendNode(reduce_op));
    }
#endif // HAVE_CANN

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

    std::vector<int> axes;
    bool keepdims;
};

Ptr<ReduceLayer> ReduceLayer::create(const LayerParams& params)
{
    return Ptr<ReduceLayer>(new ReduceLayerImpl(params));
}

}
}
