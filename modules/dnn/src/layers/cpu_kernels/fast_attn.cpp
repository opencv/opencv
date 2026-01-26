// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

static inline v_float32 load_int_mask_as_f32(const int32_t* ptr) {
    v_int32 v = vx_load(ptr);
    return v_reinterpret_as_f32(v_ne(v, vx_setall_s32(0)));
}

static inline v_float32 load_int_mask_as_f32(const uint8_t* ptr) {
    v_uint32 v = vx_load_expand_q(ptr);
    return v_reinterpret_as_f32(v_ne(v, vx_setall_u32(0)));
}

static inline v_float32 load_int_mask_as_f32(const int8_t* ptr) {
    v_int32 v = vx_load_expand_q(ptr);
    return v_reinterpret_as_f32(v_ne(v, vx_setall_s32(0)));
}

static inline v_float32 load_int_mask_as_f32(const uint16_t* ptr) {
    v_uint32 v = vx_load_expand(ptr);
    return v_reinterpret_as_f32(v_ne(v, vx_setall_u32(0)));
}

template<typename T>
inline v_float32 load_int_mask_as_f32(const T* ptr) {
    return vx_setall_f32(0.f);
}


template <typename MaskT>
struct MaskPolicyFloat {
    static inline v_float32 load_mask(const MaskT* ptr) { return vx_load((const float*)ptr); }
    static inline v_float32 load_mask_scalar(const MaskT* ptr) { return vx_setall_f32((float)*ptr); }
    static inline v_float32 apply(v_float32 val, v_float32 mask, v_float32 min_val) { return v_add(val, mask); }
    static inline float apply_scalar(float val, MaskT mask, float min_val) { return val + mask; }
};

template <typename MaskT>
struct MaskPolicyInt {
    static inline v_float32 load_mask(const MaskT* ptr) { return load_int_mask_as_f32(ptr); }
    static inline v_float32 load_mask_scalar(const MaskT* ptr) {
        return (*ptr != 0) ? v_reinterpret_as_f32(vx_setall_u32(0xFFFFFFFF)) : vx_setall_f32(0.f);
    }
    static inline v_float32 apply(v_float32 val, v_float32 mask, v_float32 min_val) { return v_select(mask, val, min_val); }
    static inline float apply_scalar(float val, MaskT mask, float min_val) { return (mask != 0) ? val : min_val; }
};

struct MaskPolicyNone {
    static inline v_float32 load_mask(const void* ptr) { return vx_setzero_f32(); }
    static inline v_float32 load_mask_scalar(const void* ptr) { return vx_setzero_f32(); }
    static inline v_float32 apply(v_float32 val, v_float32 mask, v_float32 min_val) { return val; }
    static inline float apply_scalar(float val, int mask, float min_val) { return val; }
};

template <typename MaskT, typename Policy>
void run_fused_softmax(
    Mat &att_weights, const Mat &att_mask,
    const float softcap, const bool do_softcap,
    const float threshold,
    const float min_val, const bool is_causal)
{
    const int batch_size = att_weights.size[0];
    const int n_heads = att_weights.size[1];
    const int seq_len_q = att_weights.size[2];
    const int seq_len_kv = att_weights.size[3];

    float* data = att_weights.ptr<float>();
    const MaskT* mask_data = att_mask.empty() ? nullptr : att_mask.ptr<MaskT>();

    size_t mask_step_b = 0, mask_step_h = 0, mask_step_q = 0, mask_step_k = 0;
    if (mask_data) {
        int dims_m = att_mask.dims;
        int offset_dim = 4 - dims_m;

        auto get_size = [&](int i) { return (i < offset_dim) ? 1 : att_mask.size[i - offset_dim]; };
        auto get_step = [&](int i) { return (i < offset_dim) ? 0 : att_mask.step[i - offset_dim] / sizeof(MaskT); };

        if (get_size(0) > 1) mask_step_b = get_step(0);
        if (get_size(1) > 1) mask_step_h = get_step(1);
        if (get_size(2) > 1) mask_step_q = get_step(2);
        if (get_size(3) > 1) mask_step_k = 1;
    }

    size_t total_tasks = (size_t)batch_size * n_heads;

    parallel_for_(Range(0, (int)total_tasks), [&](const Range &range) {
        for (int i = range.start; i < range.end; i++) {
            int b = i / n_heads;
            int h = i % n_heads;

            size_t offset = (size_t)i * seq_len_q * seq_len_kv;
            size_t mask_offset_q = b * mask_step_b + h * mask_step_h;

            for (int tq = 0; tq < seq_len_q; tq++){
            const int tmax = is_causal ? std::min(tq + 1, seq_len_kv) : seq_len_kv;
            float maxVal = -FLT_MAX;
            int tk = 0;
#if CV_SIMD
            const int w = VTraits<v_float32>::vlanes();
            v_float32 v_max_val = vx_setall_f32(maxVal);
            v_float32 v_softcap = vx_setall_f32(softcap);
            v_float32 v_inv_softcap = vx_setall_f32(1.f / softcap);
            v_float32 v_threshold = vx_setall_f32(threshold);
            v_float32 v_minus_threshold = vx_setall_f32(-threshold);
            v_float32 v_one = vx_setall_f32(1.f);
            v_float32 v_minus_one = vx_setall_f32(-1.f);
            v_float32 v_minus_two = vx_setall_f32(-2.f);
            v_float32 v_min_val = vx_setall_f32(min_val);

            for (; tk <= tmax - w; tk += w) {
                v_float32 v_val = vx_load(&data[offset + tk]);

                if (mask_data) {
                    v_float32 v_mask_val;
                    if (mask_step_k)
                        v_mask_val = Policy::load_mask(&mask_data[mask_offset_q + tk]);
                    else
                        v_mask_val = Policy::load_mask_scalar(&mask_data[mask_offset_q]);
                    v_val = Policy::apply(v_val, v_mask_val, v_min_val);
                }

                if (do_softcap) {
                    v_float32 v_scaled = v_mul(v_val, v_inv_softcap);

                    v_float32 v_mask_pos = v_gt(v_scaled, v_threshold);
                    v_float32 v_mask_neg = v_lt(v_scaled, v_minus_threshold);

                    v_float32 v_scaled_safe = v_select(v_mask_neg, vx_setall_f32(0.f), v_scaled);

                    v_float32 v_exp_part = v_exp(v_mul(v_minus_two, v_scaled_safe));
                    v_float32 v_tanh = v_div(v_sub(v_one, v_exp_part), v_add(v_one, v_exp_part));

                    v_float32 v_res = v_select(v_mask_pos, v_one,
                                        v_select(v_mask_neg, v_minus_one, v_tanh));

                    v_val = v_mul(v_res, v_softcap);
                }
                vx_store(&data[offset + tk], v_val);
                v_max_val = v_max(v_max_val, v_val);
            }
            maxVal = v_reduce_max(v_max_val);
#endif
            for (; tk < tmax; tk++){
                if (mask_data) {
                    size_t mask_idx = mask_offset_q + (mask_step_k ? tk : 0);
                    data[offset + tk] = Policy::apply_scalar(data[offset + tk], mask_data[mask_idx], min_val);
                }

                if (do_softcap) {
                    float softcap_val = data[offset + tk] / softcap;
                    if (softcap_val > threshold)
                        data[offset + tk ] = 1.f;
                    else if(softcap_val < -threshold)
                        data[offset + tk ] = -1.f;
                    else
                        data[offset + tk] = (1.f - expf(-2 *softcap_val)) / (1.f + expf(-2 *softcap_val));
                    data[offset + tk] *= softcap;
                }
                maxVal = std::max(maxVal, data[offset + tk]);
            }

            for (; tk < seq_len_kv; tk++) {
                data[offset + tk] = min_val;
            }

            float sum = 0.f;
            tk = 0;
#if CV_SIMD
            v_float32 v_sum = vx_setzero_f32();
            v_float32 v_max_val_shift = vx_setall_f32(maxVal);
            for (; tk <= seq_len_kv - w; tk += w) {
                v_float32 v_val = vx_load(&data[offset + tk]);
                v_val = v_sub(v_val, v_max_val_shift);
                v_val = v_exp(v_val);
                v_sum = v_add(v_sum, v_val);
                vx_store(&data[offset + tk], v_val);
            }
            sum = v_reduce_sum(v_sum);
#endif
            for (; tk < seq_len_kv; tk++){
                data[offset + tk] = expf(data[offset + tk] - maxVal);
                sum += data[offset + tk];
            }

            float inv_sum = 1.f / sum;
            tk = 0;
#if CV_SIMD
            v_float32 v_inv_sum = vx_setall_f32(inv_sum);
            for (; tk <= seq_len_kv - w; tk += w) {
                v_float32 v_val = vx_load(&data[offset + tk]);
                v_val = v_mul(v_val, v_inv_sum);
                vx_store(&data[offset + tk], v_val);
            }
#endif
            for (; tk < seq_len_kv; tk++){
                data[offset + tk] *= inv_sum;
            }

            offset += seq_len_kv;
            mask_offset_q += mask_step_q;
        }
    }
    });
}

void fused_softmax_softcap_mask(
        Mat &att_weights,const Mat &att_mask,
        const float softcap, const bool do_softcap,
        const float threshold,
        const float min_val, const bool has_mask, const bool is_causal
){
    if (has_mask) {
        switch(att_mask.depth())
        {
            case CV_32F:
                run_fused_softmax<float, MaskPolicyFloat<float>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_Bool:
            case CV_8U:
                run_fused_softmax<uint8_t, MaskPolicyInt<uint8_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_8S:
                run_fused_softmax<int8_t, MaskPolicyInt<int8_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_16U:
                run_fused_softmax<uint16_t, MaskPolicyInt<uint16_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_16S:
                run_fused_softmax<int16_t, MaskPolicyInt<int16_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_32U:
                run_fused_softmax<uint32_t, MaskPolicyInt<uint32_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_32S:
                run_fused_softmax<int32_t, MaskPolicyInt<int32_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_64U:
                run_fused_softmax<uint64_t, MaskPolicyInt<uint64_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            case CV_64S:
                run_fused_softmax<int64_t, MaskPolicyInt<int64_t>>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
                break;
            default:
                CV_Error(Error::StsUnsupportedFormat, "Unsupported mask data type in fused_softmax_softcap_mask");
        }
    } else {
        run_fused_softmax<float, MaskPolicyNone>(att_weights, att_mask, softcap, do_softcap, threshold, min_val, is_causal);
    }
}
}}
