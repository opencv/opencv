// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/softmax.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

static void gather(const Mat& data, const Mat& ind, Mat& out, int axis)
{
    // TODO
}


static void rotationKernel(
    float* data, const float* rotation_table,
    size_t seq_len, size_t d
)
{
    CV_Assert(d % 2 == 0);
    const size_t d_half = d / 2;

    double nstripes = double(seq_len) * d_half * (1.0/1024.0);

    auto fn = [&](const cv::Range& range)
    {
        for (int t = range.start; t < range.end; ++t)
        {
            float* out_ptr = data + size_t(t) * d;
            const float* table_ptr = rotation_table + size_t(t) * d;
            size_t i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            const size_t w = VTraits<v_float32>::vlanes();
            for (; i + w <= d_half; i += w)
            {
                v_float32 sin_v, cos_v, x_even, x_odd;
                v_load_deinterleave(table_ptr + 2*i, sin_v, cos_v);
                v_load_deinterleave(out_ptr    + 2*i, x_even, x_odd);

                v_float32 out_even = v_sub(v_mul(cos_v, x_even), v_mul(sin_v, x_odd));
                v_float32 out_odd  = v_add(v_mul(sin_v, x_even), v_mul(cos_v, x_odd));

                v_store_interleave(out_ptr + 2*i, out_even, out_odd);
            }
#endif
            // scalar tail
            for (; i < d_half; ++i)
            {
                float s  = table_ptr[2*i  ];
                float c  = table_ptr[2*i+1];
                float xe = out_ptr[2*i];
                float xo = out_ptr[2*i+1];
                out_ptr[2*i]   = xe * c - xo * s;
                out_ptr[2*i+1] = xo * c + xe * s;
            }
        }
    };

    // This will spin up threads and run fn over [0, seq_len)
    parallel_for_(cv::Range(0, int(seq_len)), fn, nstripes);
}

static void rotate(
    const float*data_in, float* data_out,
    const float*cos_cache, const float*sin_cache,
    const int seq_len, const int n_heads, const int dim_head,
    bool is_layout4d
){
    Range r(0, n_heads * seq_len);

    const int dhalf = dim_head / 2;
    for (int i = r.start; i < r.end; ++i)
    {
        const float* real = data_in + i * dim_head;
        const float *imag = data_in + i * dim_head + dhalf;
        float * out_real = data_out + i * dim_head;
        float * out_imag = data_out + i * dim_head + dhalf;

        int d = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const size_t w = VTraits<v_float32>::vlanes();
        for (; d + w <= dhalf; d += w)
        {
            // Alignment requirement: if CV_STRONG_ALIGNMENT=1 then passed pointer must be aligned (sizeof(lane type) should be enough
            // should be fullfilled by design
            v_float32 vreal = vx_load(real + d);
            v_float32 vimag = vx_load(imag + d);
            v_float32 vsin  = vx_load(sin_cache + d);
            v_float32 vcos  = vx_load(cos_cache + d);

            v_float32 vout_real = v_sub(v_mul(vcos, vreal), v_mul(vsin, vimag));
            v_float32 vout_imag = v_add(v_mul(vsin, vreal), v_mul(vcos, vimag));

            v_store(out_real + d, vout_real);
            v_store(out_imag + d, vout_imag);
        }
#endif
        // scalar tail
        for (; d < dhalf; ++d)
        {
            float r = real[d];
            float im = imag[d];
            float s = sin_cache[d];
            float c = cos_cache[d];
            out_real[d] = r * c - im * s;
            out_imag[d] = im * c + r * s;
        }
    }

}

static void rotateInterlieved(){}



// https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html#rotaryembedding-23
class RotaryEmbeddingLayerImpl CV_FINAL : public RotaryEmbeddingLayer   {
 public:
    RotaryEmbeddingLayerImpl(const LayerParams &params) {
        setParamsFrom(params);
        num_heads = params.get<int>("num_heads");
        rotary_embedding_dim = params.get<int>("rotary_embedding_dim");
        interlieved = params.get<int>("interlieved", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckTrue(
            inputs.size() == 4 || (inputs[1].size() > 2 && inputs[2].size() > 2),
            "provide position_ids or specify sin_cache and cos_cahe in format BxTxD");
        const int ndims = inputs[0].dims;
        // const int dim_head = ndims == 4 ? inputs[0][3] : inputs[0][2] / num_heads;
        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        Mat& x = inputs[0];
        Mat& cos_cache = inputs[1];
        Mat& sin_cache = inputs[2];



    }

 private:
    int num_heads;
    int interlieved;
    int rotary_embedding_dim;
};

Ptr<RotaryEmbeddingLayer> RotaryEmbeddingLayer::create(const LayerParams &params) {
    return makePtr<RotaryEmbeddingLayerImpl>(params);
}


}} // cv::dnn
