// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cann.hpp"

namespace cv { namespace dnn {

class LayerNormLayerImpl CV_FINAL : public LayerNormLayer
{
public:
    LayerNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        // standard attr
        axis = params.get<int>("axis", 0);
        epsilon = params.get<float>("epsilon", 1e-5);

        // opencv attr
        hasBias = params.get<bool>("hasBias", false);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CANN;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // check shapes of weight and bias if existed
        // inputs >= 2 (X and Weight are requested, bias is optional)
        CV_Check(inputs.size(), inputs.size() >= 2 && inputs.size() <= 3, "LayerNorm: require two (x, weight) or three (x, weight, bias) inputs");

        auto x_shape = inputs[0];
        int x_ndims = static_cast<int>(x_shape.size());

        auto w_shape = inputs[1];
        // if axis == last_dim, scale and b are both 1d tensor (represented as 2d mat nx1)
        int w_ndims = static_cast<int>(w_shape.size());
        w_ndims = (axis == x_ndims - 1 && w_ndims == 2) ? w_ndims - 1 : w_ndims;
        CV_CheckEQ(x_ndims - axis, w_ndims, "LayerNorm: shape of weight does not match with given axis and shape of input");
        for (int i = 0; i < w_ndims; ++i)
            CV_CheckEQ(x_shape[axis+i], w_shape[i], "LayerNorm: weight dimensions does not match with input dimensions");
        if (hasBias)
        {
            CV_CheckEQ(inputs.size(), (size_t)3, "");
            auto b_shape = inputs[2];
            CV_CheckEQ(w_shape.size(), b_shape.size(), "LayerNorm: shape of weight does not match with shape of bias");
            for (size_t i = 0; i < w_shape.size(); ++i)
                CV_CheckEQ(w_shape[i], b_shape[i], "LayerNorm: bias dimensions does not match with weight dimensions");
        }

        // only one output is needed; Mean & InvStdDev are not needed
        // in inference and should beomitted in onnx importer
        outputs.assign(1, inputs[0]);
        return false;
    }

    template<bool hasBias>
    class LayerNormInvoker : public ParallelLoopBody
    {
    public:
        const Mat& src;
        const float* scaleData;
        const float* biasData;
        Mat& dst;

        float epsilon;

        int total;
        int normSize;
        float invNormSize;

        LayerNormInvoker(const Mat& src_, const Mat& scale, const Mat* b, Mat& dst_, int axis, float epsilon_)
            : src(src_), scaleData(scale.ptr<float>()), biasData(nullptr), dst(dst_), epsilon(epsilon_)
        {
            if (hasBias)
            {
                CV_Assert(b != nullptr);
                CV_Assert(b->isContinuous());
                biasData = (const float*)b->ptr<float>();
            }

            auto dstShape = shape(dst);
            total = std::accumulate(dstShape.begin(), dstShape.begin() + axis, 1, std::multiplies<int>());
            normSize = std::accumulate(dstShape.begin() + axis, dstShape.end(), 1, std::multiplies<int>());
            invNormSize = 1.0f / normSize;
        }

        static void run(const Mat& src, const Mat& scale, const Mat* b, Mat& dst, int axis, float epsilon)
        {
            CV_Assert(src.isContinuous());
            CV_Assert(dst.isContinuous());
            CV_CheckTypeEQ(src.type(), CV_32F, "DNN/LayerNorm: only support float32");
            CV_CheckTypeEQ(src.type(), dst.type(), "");
            CV_Assert(scale.isContinuous());

            CV_CheckGE(epsilon, 0.0f, "");

            LayerNormInvoker p(src, scale, b, dst, axis, epsilon);

            double nstripes = ((size_t)p.total * p.normSize) * (1 / 1024.0);
            // double nstripes = ((size_t)p.total) * (1 / 1024.0);
            parallel_for_(Range(0, p.total), p, nstripes);
        }

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int stripeStart = r.start;
            int stripeEnd = r.end;

            const float* srcData = src.ptr<float>();
            float* dstData = dst.ptr<float>();

            for (int ofs = stripeStart; ofs < stripeEnd; ++ofs)
            {
                const float* first = srcData + ofs * normSize;
                float* dstFirst = dstData + ofs * normSize;

                float mean = 0;
                float meanSquare = 0;
                for (int h = 0; h < normSize; ++h)
                {
                    float v = first[h];
                    mean += v;
                    meanSquare += v * v;
                }
                mean *= invNormSize;
                meanSquare = std::sqrt(std::max(0.f, meanSquare * invNormSize - mean * mean) + epsilon);
                float invMeanSquare = 1.0f / meanSquare;
                for (int h = 0; h < normSize; ++h)
                {
                    float v = (first[h] - mean) * invMeanSquare * scaleData[h];
                    if (hasBias) {
                        v = v + biasData[h];
                    }
                    dstFirst[h] = v;
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

        if (hasBias) {
            LayerNormInvoker<true>::run(inputs[0], inputs[1], &inputs[2], outputs[0], axis, epsilon);
        } else {
            LayerNormInvoker<false>::run(inputs[0], inputs[1], nullptr, outputs[0], axis, epsilon);
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)3, "DNN/CANN: LayerNorm should have 3 inputs (x, gamma, beta)");

        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto gamma = inputs[1].dynamicCast<CannBackendWrapper>();
        auto beta = inputs[2].dynamicCast<CannBackendWrapper>();

        // create operator
        auto op = std::make_shared<ge::op::LayerNorm>(name);

        // set attrs
        op->set_attr_begin_norm_axis(axis);
        op->set_attr_begin_params_axis(axis);
        op->set_attr_epsilon(epsilon);

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto desc_x = x->getTensorDesc();
        op->update_input_desc_x(*desc_x);
        // set inputs : gamma
        auto op_gamma = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_gamma_by_name(*op_gamma, gamma->name.c_str());
        auto desc_gamma = x->getTensorDesc();
        op->update_input_desc_gamma(*desc_gamma);
        // set inputs : beta
        auto op_beta = nodes[2].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_beta_by_name(*op_beta, beta->name.c_str());
        auto desc_beta = x->getTensorDesc();
        op->update_input_desc_beta(*desc_beta);

        // set outputs
        auto desc_output_y = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*desc_output_y);
        auto desc_output_mean = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_mean(*desc_output_mean);
        auto desc_output_variance = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_variance(*desc_output_variance);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN
};

Ptr<LayerNormLayer> LayerNormLayer::create(const LayerParams& params)
{
    return makePtr<LayerNormLayerImpl>(params);
}

}} // cv::dnn
