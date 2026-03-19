"test_basic_conv_with_padding", // (assert failed) !blobs.empty() in initCUDA
"test_basic_conv_without_padding", // (assert failed) !blobs.empty() in initCUDA
"test_cast_DOUBLE_to_FLOAT",
"test_conv_with_autopad_same", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_and_asymmetric_padding", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_no_padding", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_padding", // (assert failed) !blobs.empty() in initCUDA
"test_cumsum_1d",
"test_cumsum_1d_exclusive",
"test_cumsum_1d_reverse",
"test_cumsum_1d_reverse_exclusive",
"test_cumsum_2d_axis_0",
"test_cumsum_2d_axis_1",
"test_cumsum_2d_negative_axis",
"test_dropout_default_ratio",
"test_einsum_batch_diagonal",
"test_einsum_batch_matmul",
"test_einsum_sum",
"test_einsum_transpose",
"test_logsoftmax_large_number", // fp16 accuracy issue
"test_logsoftmax_large_number_expanded", // fp16 accuracy issue
"test_maxpool_with_argmax_2d_precomputed_pads", // assertion failed mat.type() == CV_32F
"test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded", // crash: https://github.com/opencv/opencv/issues/25471
"test_reduce_prod_default_axes_keepdims_example", // fallback to cpu, accuracy
"test_reduce_prod_default_axes_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_default_axes_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_do_not_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_negative_axes_keepdims_random", // fallback to cpu, accuracy
"test_pow", // fp16 accuracy issue
"test_softmax_large_number", // fp16 accuracy issue
"test_softmax_large_number_expanded", // fp16 accuracy issue
"test_tan", // fp16 accuracy issue
"test_dequantizelinear_blocked", // Issue https://github.com/opencv/opencv/issues/25999
"test_quantizelinear", // Issue https://github.com/opencv/opencv/issues/25999
"test_quantizelinear_axis", // Issue https://github.com/opencv/opencv/issues/25999
"test_quantizelinear_blocked", // Issue https://github.com/opencv/opencv/issues/25999
"test_max_float64",
"test_min_float64",
"test_mod_mixed_sign_float64",
"test_attention_3d_attn_mask",
"test_attention_3d_causal",
"test_attention_3d_diff_heads_sizes",
"test_attention_3d_diff_heads_sizes_attn_mask",
"test_attention_3d_diff_heads_sizes_causal",
"test_attention_3d_diff_heads_sizes_softcap",
"test_attention_3d_diff_heads_sizes_scaled",
"test_attention_3d_gqa",
"test_attention_3d_gqa_attn_mask",
"test_attention_3d_gqa_causal",
"test_attention_3d_gqa_scaled",
"test_attention_3d_gqa_softcap",
"test_attention_3d_scaled",
"test_attention_3d_softcap",
"test_attention_3d_transpose_verification",
"test_attention_4d",
"test_attention_4d_attn_mask",
"test_attention_4d_attn_mask_3d",
"test_attention_4d_attn_mask_3d_causal",
"test_attention_4d_attn_mask_4d",
"test_attention_4d_attn_mask_4d_causal",
"test_attention_4d_attn_mask_bool",
"test_attention_4d_attn_mask_bool_4d",
"test_attention_4d_causal",
"test_attention_4d_diff_heads_sizes",
"test_attention_4d_diff_heads_sizes_attn_mask",
"test_attention_4d_diff_heads_sizes_causal",
"test_attention_4d_diff_heads_sizes_scaled",
"test_attention_4d_diff_heads_sizes_softcap",
"test_attention_4d_gqa",
"test_attention_4d_gqa_attn_mask",
"test_attention_4d_gqa_causal",
"test_attention_4d_gqa_scaled",
"test_attention_4d_gqa_softcap",
"test_attention_4d_scaled",
"test_attention_4d_softcap",
"test_attention_4d_attn_mask_bool",
"test_attention_4d_attn_mask_bool_4d",