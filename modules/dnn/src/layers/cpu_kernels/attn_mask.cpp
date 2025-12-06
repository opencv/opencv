// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "../../precomp.hpp"
#include "softmax.hpp"

namespace cv { namespace dnn {


static inline v_float32 load_int_mask_as_f32(const int32_t* ptr) {
    v_int32 v = vx_load(ptr);
    return v_reinterpret_as_f32(v_ne(v, v_setzero_s32()));
}

static inline v_float32 load_int_mask_as_f32(const uint8_t* ptr) {
    v_uint32 v = vx_load_expand_q(ptr);
    return v_reinterpret_as_f32(v_ne(v, v_setzero_u32()));
}

static inline v_float32 load_int_mask_as_f32(const int8_t* ptr) {
    v_int32 v = vx_load_expand_q(ptr);
    return v_reinterpret_as_f32(v_ne(v, v_setzero_s32()));
}

static inline v_float32 load_int_mask_as_f32(const uint16_t* ptr) {
    v_uint32 v = vx_load_expand(ptr);
    return v_reinterpret_as_f32(v_ne(v, v_setzero_u32()));
}

template<typename T>
inline v_float32 load_int_mask_as_f32(const T* ptr) {
    return v_setzero_f32();
}


template <typename T>
static void apply_mask_int_kernel(
    float* weights, const T* mask,
    int tmax,
    float min_val)
{
    int t = 0;
#if CV_SIMD
    if (sizeof(T) <= 4)  {
        const int w = VTraits<v_float32>::nlanes;
        for (; t <= tmax - w; t += w)
        {
            v_float32 v_weight = vx_load(&weights[t]);
            v_float32 v_mask = load_int_mask_as_f32(&mask[t]);
            v_weight = v_select(v_mask, v_weight, v_setall_f32(min_val));
            vx_store(&weights[t], v_weight);
        }
    }
#endif
    for (; t < tmax; t++)
    {
        if (!(mask[t] != 0)) {
            weights[t] = min_val;
        }
    }
}

void apply_mask_int(Mat &att_weights, const Mat &att_mask,
    const int seq_len_kv, const int seq_len_q, const float min_val,
    const bool has_mask, const bool is_causal)
{

    const int loops = att_weights.size[0] * att_weights.size[1];
    const int stripeSize = seq_len_q * seq_len_kv;

    float* weights_data = att_weights.ptr<float>();
    for (int i = 0; i < loops; i++){
        const size_t offset = i * stripeSize;

        for (int tq = 0; tq < seq_len_q; tq++){
            const int tmax = is_causal ? std::min(tq + 1, seq_len_kv) : seq_len_kv;
            if (has_mask) {
                switch(att_mask.depth())
                {
                    case CV_Bool:
                    case CV_8U:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const uint8_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_8S:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const int8_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_16U:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const uint16_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_16S:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const int16_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_32U:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const uint32_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_32S:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const int32_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_64U:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const uint64_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    case CV_64S:
                        apply_mask_int_kernel(
                            &weights_data[offset + tq * seq_len_kv],
                            att_mask.ptr<const int64_t>() + offset + tq * seq_len_kv,
                            tmax, min_val);
                        break;
                    default:
                        CV_Error(Error::StsUnsupportedFormat, "Unsupported mask data type in apply_mask_int");
                }
            }
            if (is_causal) {
                for (int tk = tmax; tk < seq_len_kv; tk++) {
                    weights_data[offset + tq * seq_len_kv + tk] = min_val;
                }
            }
        }
    }
}


void apply_mask_float(Mat &att_weights, const Mat &att_mask,
        const int seq_len_kv, const int seq_len_q, const float min_val,
        const bool has_mask, const bool is_causal) {

    const int loops = att_weights.size[0] * att_weights.size[1];
    const int stripeSize = seq_len_q * seq_len_kv;
    float* weights_data = att_weights.ptr<float>();
    const float* mask_data = att_mask.ptr<float>();
    for (int i = 0; i < loops; i++){
        const size_t offset = i * stripeSize;

        for (int tq = 0; tq < seq_len_q; tq++){
            const int tmax = is_causal ? std::min(tq + 1, seq_len_kv) : seq_len_kv;
            int tk = 0;

            if (has_mask) {
#if CV_SIMD
                const int w = VTraits<v_float32>::nlanes;
                for (; tk <= tmax - w; tk += w)
                {
                    v_float32 v_weight = vx_load(&weights_data[offset + tq * seq_len_kv + tk]);
                    v_float32 v_mask = vx_load(&mask_data[offset + tq * seq_len_kv + tk]);
                    v_float32 v_result = v_add(v_weight, v_mask);
                    vx_store(&weights_data[offset + tq * seq_len_kv + tk], v_result);
                }
#endif
                for (; tk < tmax; tk++)
                {
                    weights_data[offset + tq * seq_len_kv + tk] +=
                        mask_data[offset + tq * seq_len_kv + tk];
                }
            }

            if (is_causal) {
                for (tk = tmax; tk < seq_len_kv; tk++) {
                    weights_data[offset + tq * seq_len_kv + tk] = min_val;
                }
            }
        }
    }
}


}}
                // else {
                //     v_float32 mask_val = vx_load(&mask_data[offset + tq * seq_len_kv + tk]);
                //     v_float32 weight_val = vx_load(&weights_data[offset + tq * seq_len_kv + tk]);
                //     v_float32 result = v_add(weight_val, mask_val);
                //     vx_store(&weights_data[offset + tq * seq_len_kv + tk], result);
                // }