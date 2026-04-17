// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "./cpu_kernels/fast_norm.hpp"
#include "../net_impl.hpp"
#include <opencv2/core/hal/intrin.hpp>

// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/group_norm.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// OpenCL backend
#ifdef HAVE_OPENCL
#include "../ocl4dnn/include/math_functions.hpp"
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv {
namespace dnn {

static void fastNormGroupBlock(const Mat& input, const Mat& scale, const Mat& bias,
                               Mat& output, float epsilon, size_t num_groups)
{
    CV_Assert(input.dims == 5 && output.dims == 5);
    CV_Assert(input.shape().layout == DATA_LAYOUT_BLOCK && output.shape().layout == DATA_LAYOUT_BLOCK);
    CV_Assert(input.type() == CV_32F && output.type() == CV_32F);
    CV_Assert(input.isContinuous() && output.isContinuous());

    const int N  = input.size[0];
    const int C1 = input.size[1];
    const int H  = input.size[2];
    const int W  = input.size[3];
    const int C0 = input.size[4];
    const int C  = input.shape().C;

    const float* scale_data = scale.ptr<float>();
    const float* bias_data  = bias.ptr<float>();

    const size_t inStep0 = input.step.p[0] / sizeof(float);
    const size_t inStep1 = input.step.p[1] / sizeof(float);
    const size_t inStep2 = input.step.p[2] / sizeof(float);
    const size_t inStep3 = input.step.p[3] / sizeof(float);

    const size_t outStep0 = output.step.p[0] / sizeof(float);
    const size_t outStep1 = output.step.p[1] / sizeof(float);
    const size_t outStep2 = output.step.p[2] / sizeof(float);
    const size_t outStep3 = output.step.p[3] / sizeof(float);

    const int channels_per_group = C / (int)num_groups;
    const size_t norm_size_per_group = (size_t)channels_per_group * (size_t)H * (size_t)W;
    const float inv_norm_size = 1.f / (float)norm_size_per_group;

#if CV_SIMD
    const int VEC_SZ = (int)v_float32::nlanes;
#endif

    parallel_for_(Range(0, N * (int)num_groups), [&](const Range& r) {
        const float* inptr = (const float*)input.data;
        float* outptr = (float*)output.data;

        AutoBuffer<float> buf(C0 * 2);
        float* alpha = buf.data();
        float* beta  = alpha + C0;

        for (int i = r.start; i < r.end; ++i) {
            int n = i / (int)num_groups;
            int g = i - n * (int)num_groups;
            int c_start = g * channels_per_group;
            int c_end   = c_start + channels_per_group;

            float group_sum = 0.f, group_sqsum = 0.f;

            for (int c = c_start; c < c_end; c++) {
                int c1 = c / C0;
                int c0 = c % C0;
                const float* inbase = inptr + n * inStep0 + c1 * inStep1;
                for (int h = 0; h < H; ++h) {
                    const float* inrow = inbase + h * inStep2;
                    for (int w = 0; w < W; ++w) {
                        float v = inrow[w * inStep3 + c0];
                        group_sum += v;
                        group_sqsum += v * v;
                    }
                }
            }

            float mean = group_sum * inv_norm_size;
            float var  = std::max(0.f, group_sqsum * inv_norm_size - mean * mean);
            float inv_stdev = 1.f / std::sqrt(var + epsilon);

            for (int c1_start = c_start / C0, c1_end_idx = (c_end - 1) / C0 + 1,
                     c1 = c1_start; c1 < c1_end_idx; ++c1) {
                int cbase = c1 * C0;
                int c0_lo = std::max(0, c_start - cbase);
                int c0_hi = std::min(C0, c_end - cbase);
                int validC0 = std::min(C0, std::max(0, C - cbase));

                for (int c0 = c0_lo; c0 < c0_hi; ++c0) {
                    alpha[c0] = scale_data[cbase + c0] * inv_stdev;
                    beta[c0]  = bias_data[cbase + c0] - alpha[c0] * mean;
                }

                const float* inbase  = inptr  + n * inStep0 + c1 * inStep1;
                float*       outbase = outptr + n * outStep0 + c1 * outStep1;

                int c0 = c0_lo;
#if CV_SIMD
                for (; c0 <= c0_hi - VEC_SZ; c0 += VEC_SZ) {
                    v_float32 va = v_load(alpha + c0);
                    v_float32 vb = v_load(beta + c0);
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w) {
                            v_float32 vin = v_load(inrow + w * inStep3 + c0);
                            v_store(outrow + w * outStep3 + c0, v_fma(vin, va, vb));
                        }
                    }
                }
#endif
                for (; c0 < c0_hi; ++c0) {
                    float a = alpha[c0], b = beta[c0];
                    for (int h = 0; h < H; ++h) {
                        const float* inrow  = inbase + h * inStep2;
                        float*       outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0] = inrow[w * inStep3 + c0] * a + b;
                    }
                }

                for (int c0_pad = std::max(c0_hi, validC0); c0_pad < C0; ++c0_pad)
                    for (int h = 0; h < H; ++h) {
                        float* outrow = outbase + h * outStep2;
                        for (int w = 0; w < W; ++w)
                            outrow[w * outStep3 + c0_pad] = 0.f;
                    }
            }
        }
    });
}

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GroupNormalization
class GroupNormLayerImpl CV_FINAL : public GroupNormLayer {
public:
    GroupNormLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", 1e-5);
        num_groups = params.get<int>("num_groups");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE {
        const auto &input = inputs[0];
        const auto &scale = inputs[1];
        const auto &bias = inputs[2];
        CV_CheckGE(input.size(), static_cast<size_t>(3), "DNN/GroupNorm: input dimension >= 3 is required");

        int C = input.layout == DATA_LAYOUT_BLOCK ? input.C : input[1];
        int scale_dim = std::accumulate(scale.begin(), scale.end(), 1, std::multiplies<int>());
        CV_CheckEQ(scale_dim, C, "DNN/InstanceNorm: scale must be a 1d tensor and match the channel of input");
        int bias_dim = std::accumulate(bias.begin(), bias.end(), 1, std::multiplies<int>());
        CV_CheckEQ(bias_dim, C, "DNN/InstanceNorm: bias must be a 1d tensor and match the channel of input");

        outputs.assign(1, inputs[0]);
        return false;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE {
        CV_Assert(!actualInputs.empty());
        desiredInputs = actualInputs;
        outputs.assign(requiredOutputs, actualInputs[0]);
        return 0;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto& input = inputs[0];
        const auto& scale = inputs[1];
        const auto& bias  = inputs[2];

        if (input.shape().layout == DATA_LAYOUT_BLOCK)
            fastNormGroupBlock(input, scale, bias, outputs[0], epsilon, num_groups);
        else if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        } else
            fastNormGroup(input, scale, bias, outputs[0], epsilon, num_groups);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        const auto &input = inputs[0], &scale = inputs[1], &bias = inputs[2];
        auto &output = outputs[0];

        const auto input_shape = shape(input);
        size_t N = input_shape[0], C = input_shape[1];
        size_t num_groups = this->num_groups;
        size_t channels_per_group = C / num_groups;
        size_t loops = N * num_groups, norm_size = static_cast<size_t>(total(input_shape, 2)) * channels_per_group;
        float inv_norm_size = 1.f / norm_size;

        // no fp16 support
        if (input.depth() == CV_16F) {
            return false;
        }

        String base_opts = format(" -DT=float -DT4=float4 -Dconvert_T=convert_float4");

        // Calculate mean
        UMat one = UMat::ones(norm_size, 1, CV_32F);
        UMat mean = UMat(loops, 1, CV_32F);
        UMat mean_square = UMat(loops, 1, CV_32F);
        UMat tmp = UMat(loops, norm_size, CV_32F);
        bool ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                               input, 0, one, 0, 0.f, mean, 0);
        if (!ret) {
            return false;
        }
        // Calculate mean_square
        int num_vector = (norm_size % 8 == 0) ? 8 : ((norm_size % 4 == 0) ? 4 : 1);
        size_t global[] = {loops, static_cast<size_t>(norm_size / num_vector)};
        String build_opt = format(" -DNUM=%d", num_vector) + base_opts;
        String mean_square_kernel_name = format("calc_mean%d", num_vector);
        ocl::Kernel mean_square_kernel(mean_square_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt + " -DKERNEL_MEAN");
        if (mean_square_kernel.empty()) {
            return false;
        }
        mean_square_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mean_square_kernel.set(1, (int)loops);
        mean_square_kernel.set(2, (int)norm_size);
        mean_square_kernel.set(3, ocl::KernelArg::PtrReadOnly(mean));
        mean_square_kernel.set(4, ocl::KernelArg::PtrWriteOnly(tmp));
        ret = mean_square_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }
        ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                          tmp, 0, one, 0, 0.f, mean_square, 0);
        if (!ret) {
            return false;
        }
        // Calculate group norm: output = scale * (x - mean) / sqrt(var + eps) + bias
        String mvn_group_kernel_name = format("mvn_group%d", num_vector);
        build_opt += " -DNORM_VARIANCE -DKERNEL_MVN_GROUP";
        ocl::Kernel mvn_group_kernel(mvn_group_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt);
        if (mvn_group_kernel.empty()) {
            return false;
        }
        mvn_group_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mvn_group_kernel.set(1, (int)loops);
        mvn_group_kernel.set(2, (int)norm_size);
        mvn_group_kernel.set(3, (float)epsilon);
        mvn_group_kernel.set(4, ocl::KernelArg::PtrReadOnly(mean));
        mvn_group_kernel.set(5, ocl::KernelArg::PtrReadOnly(mean_square));
        mvn_group_kernel.set(6, ocl::KernelArg::PtrReadOnly(scale));
        mvn_group_kernel.set(7, ocl::KernelArg::PtrReadOnly(bias));
        mvn_group_kernel.set(8, (int)C);
        mvn_group_kernel.set(9, (int)num_groups);
        mvn_group_kernel.set(10, (float)0.f);
        mvn_group_kernel.set(11, ocl::KernelArg::PtrWriteOnly(output));
        ret = mvn_group_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }

        return true;
        }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                          const std::vector<Ptr<BackendWrapper>>& inputs,
                          const std::vector<Ptr<BackendWrapper>>& outputs) override {
    auto context = reinterpret_cast<csl::CSLContext*>(context_);

    auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
    auto input_shape = input_wrapper->getShape();
    size_t N = input_shape[0];
    size_t num_groups = this->num_groups;
    size_t loops = N * num_groups;

    return make_cuda_node<cuda4dnn::GroupNormOp>(preferableTarget, std::move(context->stream), epsilon, loops, num_groups);
}
#endif // HAVE_CUDA

private:
    float epsilon;
    size_t num_groups;
};

Ptr<GroupNormLayer> GroupNormLayer::create(const LayerParams &params) {
    return Ptr<GroupNormLayer>(new GroupNormLayerImpl(params));
}

}} // cv::dnn
