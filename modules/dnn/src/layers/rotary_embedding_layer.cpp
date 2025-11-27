// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/softmax.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {


static void gather(
    const uchar* data,  const size_t* pos_ids,
    const int batch_size, const int seq_len, const int seq_len_max,
    const int dhalf, const int elem_size,
    uchar* out
)
{

    Range r(0, seq_len * batch_size);
    bool outOfRangeIdx = false;
    for (int i = r.start; i < r.end; ++i)
    {
        const int t = i % seq_len;
        const int b = i / seq_len;

        size_t pos_id = pos_ids[b * seq_len + t];
        if (pos_id >= seq_len_max)
        {
            outOfRangeIdx = true;
            break;
        }
        const uchar* data_in_ptr = data + pos_id * dhalf * elem_size;
        uchar* out_ptr = out + ( b * seq_len + t ) * dhalf * elem_size;

        memcpy(out_ptr, data_in_ptr, dhalf * elem_size);
    }

    if (outOfRangeIdx)
    {
        CV_Error(Error::StsOutOfRange, "some of indices are outside of range");
    }
}

static void rotate(
    const float*data_in, float* data_out,
    const float*cos_cache, const float*sin_cache,
    const int batch_size, const int seq_len, const int n_heads, const int dim_head,
    const int rotary_dim, const bool is_data_4d
){
    auto fn = [&](const Range &r) {
        const int dhalf = rotary_dim / 2;
        for (int i = r.start; i < r.end; ++i)
        {
            const int t = i % seq_len;
            const int b = i / (n_heads * seq_len);
            const int n = (i - b * n_heads * seq_len - t) / seq_len;

            const int offset = (is_data_4d ?
                seq_len * (b * n_heads + n ) + t :
                n_heads * (b * seq_len + t ) + n) * dim_head;

            const float* real = data_in + offset;
            const float* imag = data_in + offset + dhalf;

            const float*sin_cache_ptr = sin_cache + (b * seq_len + t) * dhalf;
            const float*cos_cache_ptr = cos_cache + (b * seq_len + t) * dhalf;

            float* out_real = data_out + offset;
            float* out_imag = data_out + offset + dhalf;

            int d = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
            const size_t w = VTraits<v_float32>::vlanes();
            for (; d + w <= dhalf; d += w)
            {
                // Alignment requirement: if CV_STRONG_ALIGNMENT=1 then passed pointer must be aligned (sizeof(lane type) should be enough
                // should be fullfilled by design
                v_float32 vreal = vx_load(real + d);
                v_float32 vimag = vx_load(imag + d);
                v_float32 vsin  = vx_load(sin_cache_ptr + d);
                v_float32 vcos  = vx_load(cos_cache_ptr + d);

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
                float s = sin_cache_ptr[d];
                float c = cos_cache_ptr[d];
                out_real[d] = r * c - im * s;
                out_imag[d] = im * c + r * s;
            }

            // copy not rotated part
            if (dim_head > rotary_dim)
            {
                const float* data_in_ptr = data_in + offset + rotary_dim;
                float* out_ptr = data_out + offset + rotary_dim;
                memcpy(out_ptr, data_in_ptr, (dim_head - rotary_dim) * sizeof(float));
            }
        }
    };

    const size_t loops = n_heads * seq_len * batch_size;
    double nstripes = loops * dim_head * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}

static void rotate_interleaved(
    const float*data_in, float* data_out,
    const float*cos_cache, const float*sin_cache,
    const int batch_size, const int seq_len, const int n_heads, const int dim_head,
    const int rotary_dim, const bool is_data_4d
){
    auto fn = [&](const Range &r) {
        const int dhalf = rotary_dim / 2;

        for (int i = r.start; i < r.end; ++i)
        {
            const int t = i % seq_len;
            const int b = i / (n_heads * seq_len);
            const int n = (i - b * n_heads * seq_len - t) / seq_len;

            const int offset = (is_data_4d ?
                seq_len * (b * n_heads + n ) + t :
                n_heads * (b * seq_len + t ) + n) * dim_head;

            const float* data_in_ptr = data_in + offset;
            float* out_ptr = data_out + offset;

            const float*sin_cache_ptr = sin_cache + (b * seq_len + t) * dhalf;
            const float*cos_cache_ptr = cos_cache + (b * seq_len + t) * dhalf;

            int d = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
            const size_t w = VTraits<v_float32>::vlanes();
            for (; d + w <= dhalf; d += w)
            {
                v_float32 vimag, vreal;
                v_float32 vsin  = vx_load(sin_cache_ptr + d);
                v_float32 vcos  = vx_load(cos_cache_ptr + d);
                v_load_deinterleave(data_in_ptr + 2*d, vreal, vimag);

                v_float32 vout_real = v_sub(v_mul(vcos, vreal), v_mul(vsin, vimag));
                v_float32 vout_imag = v_add(v_mul(vsin, vreal), v_mul(vcos, vimag));

                v_store_interleave(out_ptr + 2*d, vout_real, vout_imag);
            }
    #endif
            // scalar tail
            for (; d < dhalf; ++d)
            {
                float r = data_in_ptr[2*d];
                float im = data_in_ptr[2*d + 1];
                float s = sin_cache_ptr[d];
                float c = cos_cache_ptr[d];
                out_ptr[2*d] = r * c - im * s;
                out_ptr[2*d + 1] = im * c + r * s;
            }
            // copy not rotated part
            if (dim_head > rotary_dim)
            {
                const float* data_in_ptr_tail = data_in_ptr + rotary_dim;
                float* out_ptr_tail = out_ptr + rotary_dim;
                memcpy(out_ptr_tail, data_in_ptr_tail, (dim_head - rotary_dim) * sizeof(float));
            }
        }
    };

    const size_t loops = n_heads * seq_len * batch_size;
    double nstripes = loops * dim_head * (1 / 1024.0);
    parallel_for_(Range(0, loops), fn, nstripes);
}


// https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html#rotaryembedding-23
class RotaryEmbeddingLayerImpl CV_FINAL : public RotaryEmbeddingLayer   {
 public:
    RotaryEmbeddingLayerImpl(const LayerParams &params) {
        setParamsFrom(params);
        num_heads = params.get<int>("num_heads", -1);
        rotary_embedding_dim = params.get<int>("rotary_embedding_dim", 0);
        interleaved = params.get<int>("interleaved", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                              const int requiredOutputs,
                              const int requiredInternals,
                              std::vector<MatType>&outputs,
                              std::vector<MatType>&internals) const CV_OVERRIDE {
        const bool do_gather = inputs.size() > 3;
        outputs.assign(1, inputs[0]);
        if (do_gather)
        {
            internals.push_back(inputs[1]); // cos
            internals.push_back(inputs[2]); // sin
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {

        CV_CheckTrue(inputs.size() >= 3, "RotaryEmbeddingLayer: at least three inputs are required");

        const MatShape& x_input_shape = inputs[0];
        CV_CheckTrue(
            x_input_shape.dims == 4 || num_heads > -1,
            "RotaryEmbeddingLayer: input must have 4 dimensions or num_heads must be specified"
        );
        const MatShape& cos_cache_shape = inputs[1];
        const MatShape& sin_cache_shape = inputs[2];
        CV_CheckTrue(
            cos_cache_shape.dims == sin_cache_shape.dims,
            "RotaryEmbeddingLayer: cos_cache and sin_cache must have the same number of dimensions"
        );
        CV_CheckTrue(
            cos_cache_shape.dims == 3 || inputs.size() > 3,
            "RotaryEmbeddingLayer: provide position_ids or specify sin_cache and cos_cahe in format BxTxD"
        );
        CV_CheckTrue(
            cos_cache_shape.dims == cos_cache_shape.dims,
            "RotaryEmbeddingLayer: cos_cache and sin_cache must have the same number of dimensions"
        );
        for (int i = 0; i < cos_cache_shape.dims; ++i)
        {
            CV_CheckTrue(
                cos_cache_shape[i] == sin_cache_shape[i],
                "RotaryEmbeddingLayer: cos_cache and sin_cache must have the same shape"
            );
        }

        outputs.assign(1, inputs[0]);

        const bool do_gather = inputs.size() > 3;
        if (do_gather)
        {
            CV_CheckTrue(
                inputs[3].dims == 2,
                "RotaryEmbeddingLayer: position_ids must have 2 dimensions (BxT)"
            );
            CV_CheckTrue(
                inputs[1].dims == 2 && inputs[2].dims == 2,
                "RotaryEmbeddingLayer: when using position_ids, sin_cache and cos_cache must have 2 dimensions (TxD)"
            );
            const int batch_size = static_cast<int>(inputs[3][0]);
            const int seq_len = static_cast<int>(inputs[3][1]);
            const int d_half = static_cast<int>(cos_cache_shape[cos_cache_shape.dims - 1]);
            // cos
            internals.push_back(MatShape{batch_size, seq_len, d_half});
            // sin
            internals.push_back(MatShape{batch_size, seq_len, d_half});
        } else {
            CV_CheckTrue(
                cos_cache_shape.dims == 3,
                "RotaryEmbeddingLayer: sin_cache and cos_cache must have 3 dimensions (BxTxD) when position_ids are not provided"
            );
        }

        if (x_input_shape.dims == 4)
        {
            CV_CheckTrue(x_input_shape[3] % 2 == 0,
                "RotaryEmbeddingLayer: head size must be even");
        } else if( x_input_shape.dims == 3 ) {
            CV_CheckTrue(num_heads > 0, "RotaryEmbeddingLayer: num_heads must be provided for 3d input");
            CV_CheckTrue(x_input_shape[2] % num_heads == 0,
                "RotaryEmbeddingLayer: input's last dimension must be divisible by num_heads");
        }
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);
        const bool do_gather = inputs.size() > 3;

        if (num_heads == -1)
            // if input is 3d and num_heads is not given (-1),
            // getMemoryShapes throws error
            // so here input must be 4d
            num_heads = inputs[0].size[1];

        const Mat& input = inputs[0];
        Mat& output = outputs[0];

        const Mat& cos_cache = do_gather ? internals[0] : inputs[1];
        const Mat& sin_cache = do_gather ? internals[1] : inputs[2];

        const bool is_data_4d = input.dims == 4;
        const int dim_head = is_data_4d ? input.size[3] : (input.size[2] / num_heads);
        const int rotary_dim = rotary_embedding_dim > 0 ? rotary_embedding_dim : dim_head;
        CV_CheckTrue(rotary_dim % 2 == 0, "RotaryEmbeddingLayer: rotary_dim must be even");
        const int seq_len = is_data_4d ? input.size[2] : input.size[1];

        if (do_gather)
        {
            const Mat& position_ids = inputs[3];
            const int batch_size = position_ids.size[0];
            const int seq_len = position_ids.size[1];
            const int seq_len_max = inputs[1].size[0];
            const int dhalf = rotary_dim / 2;
            const Mat* caches[2] = { &inputs[1], &inputs[2] };
            for (int i = 0; i < 2; ++i)
            {
                gather(
                    caches[i]->ptr<uchar>(), position_ids.ptr<size_t>(),
                    batch_size, seq_len, seq_len_max,
                    dhalf, sizeof(float),
                    internals[i].ptr<uchar>()
                );
            }
        }

        if (interleaved)
        {
            rotate_interleaved(
                input.ptr<const float>(), output.ptr<float>(),
                cos_cache.ptr<const float>(), sin_cache.ptr<const float>(),
                input.size[0], seq_len, num_heads, dim_head,
                rotary_dim, is_data_4d
            );
        }
        else
        {
            rotate(
                input.ptr<const float>(), output.ptr<float>(),
                cos_cache.ptr<const float>(), sin_cache.ptr<const float>(),
                input.size[0], seq_len, num_heads, dim_head,
                rotary_dim, is_data_4d
            );
        }
    }

 private:
    int num_heads = -1;
    int interleaved;
    int rotary_embedding_dim = -1;
};


Ptr<RotaryEmbeddingLayer> RotaryEmbeddingLayer::create(const LayerParams &params) {
    return makePtr<RotaryEmbeddingLayerImpl>(params);
}


}} // cv::dnn
