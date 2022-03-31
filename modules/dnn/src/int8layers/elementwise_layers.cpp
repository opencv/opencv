// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_timvx.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

class ActivationLayerInt8Impl CV_FINAL : public ActivationLayerInt8
{
public:
    int input_zp, output_zp;
    float input_sc, output_sc;
    float slope = 0.0f;

#ifdef HAVE_TIMVX
    tvActivationType tvActType;
#endif
    ActivationLayerInt8Impl(const LayerParams &params)
    {
        setParamsFrom(params);
        activationLUT = !blobs.empty() ? blobs[0] : Mat();

        input_zp = params.get<int>("input_zeropoint");
        input_sc = params.get<float>("input_scale");
        output_zp = params.get<int>("zeropoints");
        output_sc = params.get<float>("scales");

        if (params.has("slope"))
        {
            slope = params.get<float>("slope");
        }

#ifdef HAVE_TIMVX
        tvActType = getTimVXActType(type);
#endif

    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        if (backendId == DNN_BACKEND_TIMVX)
        {
            // TODO!: Leaky ReLU will be supported in future.
            if (tvActType == tvActReLU && slope != 0.f)
                return false;
            return tvActType != tvActNotSupported;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    class Activation : public cv::ParallelLoopBody
    {
    public:
        const Mat* src;
        const Mat* lut;
        Mat* dst;
        int nstripes;

        Activation() : src(0), lut(0), dst(0), nstripes(0){}

        static void run(const Mat& src, const Mat& lut, Mat& dst, int nstripes)
        {
            Activation p;

            p.src = &src;
            p.lut = &lut;
            p.dst = &dst;
            p.nstripes = nstripes;

            parallel_for_(Range(0, nstripes), p, nstripes);
        }

        void operator()(const Range &r) const CV_OVERRIDE
        {
            const int8_t* table = lut->ptr<int8_t>();
            int nsamples = 1, outCn = 1;
            size_t planeSize = 1;

            if (src->dims > 1)
            {
                nsamples = src->size[0];
                outCn = src->size[1];
            }
            else
                outCn = src->size[0];

            for (int i = 2; i < src->dims; ++i)
                planeSize *= src->size[i];

            size_t stripeSize = (planeSize + nstripes - 1)/nstripes;
            size_t stripeStart = r.start*stripeSize;
            size_t stripeEnd = std::min(r.end*stripeSize, planeSize);
            int len = (int)(stripeEnd - stripeStart);

            for( int i = 0; i < nsamples; i++ )
            {
                const int8_t* srcptr = src->ptr<int8_t>(i) + stripeStart;
                int8_t* dstptr = dst->ptr<int8_t>(i) + stripeStart;
                for( int cn = 0; cn < outCn; cn++, srcptr += planeSize, dstptr += planeSize )
                {
                    int i = 0;
#if CV_SIMD128
                    for( ; i <= len - 16; i += 16 )
                    {
                        v_int8x16 out(table[srcptr[i] + 128], table[srcptr[i+1] + 128], table[srcptr[i+2] + 128], table[srcptr[i+3] + 128],
                                      table[srcptr[i+4] + 128], table[srcptr[i+5] + 128], table[srcptr[i+6] + 128], table[srcptr[i+7] + 128],
                                      table[srcptr[i+8] + 128], table[srcptr[i+9] + 128], table[srcptr[i+10] + 128], table[srcptr[i+11] + 128],
                                      table[srcptr[i+12] + 128], table[srcptr[i+13] + 128], table[srcptr[i+14] + 128], table[srcptr[i+15] + 128]);
                        v_store(dstptr + i, out);
                    }
#endif
                    for( ; i < len; i++ )
                    {
                        dstptr[i] = table[srcptr[i] + 128];
                    }
                }
            }
        }
    };

    virtual Ptr<BackendNode> initTimVX(void* timVXInfo_,
                                       const std::vector<Ptr<BackendWrapper> > &inputsWrapper,
                                       const std::vector<Ptr<BackendWrapper> > &outputsWrapper,
                                       bool isLast) CV_OVERRIDE
    {
#ifdef HAVE_TIMVX
        // tvGraph Initialization.
        auto timVxInfo = reinterpret_cast<TimVXInfo *>(timVXInfo_);
        CV_Assert(timVxInfo);
        Ptr<TimVXGraph> tvGraph = timVxInfo->getGraph();
        CV_Assert(tvGraph);
        Ptr<tim::vx::Graph> graph = tvGraph->graph;

        std::vector<int> inputsIndex, outputsIndex;
        int input_index, output_index;
        CV_Assert(inputsWrapper.size() == 1);

        // input Tensor
        Ptr<TimVXBackendWrapper> inputWrapper = inputsWrapper[0].dynamicCast<TimVXBackendWrapper>();

        if (inputWrapper->isTensor())
        {
            input_index = tvGraph->getTensorIndex(inputWrapper->getTensor());
            if(input_index == -1)
            {
                // Copy To New inputWrapper
                Mat tmp = inputWrapper->getMat();
                inputWrapper = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(tmp));
            }
        }

        if (!inputWrapper->isTensor())
        {
            Ptr<tim::vx::Quantization> tvInputQuant = Ptr<tim::vx::Quantization>(
                    new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, input_sc, input_zp));
            inputWrapper->createTensor(graph, tim::vx::TensorAttribute::INPUT, tvInputQuant);
            input_index = tvGraph->addWrapper(inputWrapper);
        }

        inputsIndex.push_back(input_index);

        // output tensor
        CV_Assert(outputsWrapper.size() == 1);
        Ptr<TimVXBackendWrapper> outputWrapper = outputsWrapper[0].dynamicCast<TimVXBackendWrapper>();
        Ptr<tim::vx::Quantization> outputQuant = Ptr<tim::vx::Quantization>(
                new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, output_sc, output_zp));

        Ptr<tim::vx::Tensor> outputTensor;

        if (isLast)
        {
            auto shapeType = getShapeTypeFromMat(outputWrapper->getMat());

            // For Graph Output tensor, we need to set tensor shape before createTensor().
            outputWrapper->setTensorShape(shapeType);
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, outputQuant);
        }
        else
        {
            outputWrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, outputQuant);
        }
        output_index = tvGraph->addWrapper(outputWrapper);
        outputsIndex.push_back(output_index);

        std::shared_ptr<tim::vx::Operation> tvAct;

        switch(tvActType) {
            case tvActReLU:
            {
                if (slope != 0.f)
                    tvAct = graph->CreateOperation<tim::vx::ops::LeakyRelu>(slope);
                else
                    tvAct = graph->CreateOperation<tim::vx::ops::Relu>();
                break;
            }
            case tvActReLU6:
                tvAct = graph->CreateOperation<tim::vx::ops::Relu6>();
                break;
            case tvActTanH:
                tvAct = graph->CreateOperation<tim::vx::ops::Tanh>();
                break;
            case tvActSwish:
                tvAct = graph->CreateOperation<tim::vx::ops::Swish>();
                break;
            case tvActMish:
                tvAct = graph->CreateOperation<tim::vx::ops::Mish>();
                break;
            case tvActSigmoid:
                tvAct = graph->CreateOperation<tim::vx::ops::Sigmoid>();
                break;
            case tvActELU:
                tvAct = graph->CreateOperation<tim::vx::ops::Elu>();
                break;
            default:
                // TODO! check the default function.
                tvAct = graph->CreateOperation<tim::vx::ops::Relu>();
                break;
        }

        Ptr<TimVXBackendNode> tvBackendNode = new TimVXBackendNode(tvGraph, tvAct, inputsIndex, outputsIndex);

        return tvBackendNode;
#endif  // HAVE_TIMVX
        return Ptr<BackendNode>();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat &src = inputs[i];
            if (!activationLUT.empty())
            {
                const int nstripes = getNumThreads();
                Mat &dst = outputs[i];
                CV_Assert(src.size == dst.size && src.type() == dst.type() &&
                          src.isContinuous() && dst.isContinuous() && src.type() == CV_8S);

                Activation::run(src, activationLUT, dst, nstripes);
            }
            else
            {
                src.copyTo(outputs[i]);
            }
        }
    }

    void forwardSlice(const int8_t* src, const int8_t* lut, int8_t* dst, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        for( int cn = cn0; cn < cn1; cn++, src += planeSize, dst += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            for( ; i <= len - 16; i += 16 )
            {
                v_int8x16 out(lut[src[i] + 128], lut[src[i+1] + 128], lut[src[i+2] + 128], lut[src[i+3] + 128],
                              lut[src[i+4] + 128], lut[src[i+5] + 128], lut[src[i+6] + 128], lut[src[i+7] + 128],
                              lut[src[i+8] + 128], lut[src[i+9] + 128], lut[src[i+10] + 128], lut[src[i+11] + 128],
                              lut[src[i+12] + 128], lut[src[i+13] + 128], lut[src[i+14] + 128], lut[src[i+15] + 128]);
                v_store(dst + i, out);
            }
#endif
            for( ; i < len; i++ )
                dst[i] = lut[src[i] + 128];
        }
    }

    void forwardSlice(const int* src, const int* lut, int* dst, int len, size_t planeSize, int cn0, int cn1) const CV_OVERRIDE
    {
        for( int cn = cn0; cn < cn1; cn++, src += planeSize, dst += planeSize )
        {
            int i = 0;
#if CV_SIMD128
            for( ; i <= len - 16; i += 16 )
            {
                v_int32x4 out0(lut[src[i] + 128], lut[src[i+1] + 128], lut[src[i+2] + 128], lut[src[i+3] + 128]);
                v_int32x4 out1(lut[src[i+4] + 128], lut[src[i+5] + 128], lut[src[i+6] + 128], lut[src[i+7] + 128]);
                v_int32x4 out2(lut[src[i+8] + 128], lut[src[i+9] + 128], lut[src[i+10] + 128], lut[src[i+11] + 128]);
                v_int32x4 out3(lut[src[i+12] + 128], lut[src[i+13] + 128], lut[src[i+14] + 128], lut[src[i+15] + 128]);

                v_store(dst + i, out0);
                v_store(dst + i + 4, out1);
                v_store(dst + i + 8, out2);
                v_store(dst + i + 12, out3);
            }
#endif
            for( ; i < len; i++ )
                dst[i] = lut[src[i] + 128];
        }

    }

    Mat activationLUT;
};

Ptr<ActivationLayerInt8> ActivationLayerInt8::create(const LayerParams& params)
{
    return Ptr<ActivationLayerInt8>(new ActivationLayerInt8Impl(params));
}

}
}
