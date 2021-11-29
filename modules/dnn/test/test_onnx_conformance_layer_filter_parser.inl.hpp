// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// not a standalone file, see test_onnx_conformance.cpp
#if 0
cout << "Filtering is disabled: PARSER" << endl;
#else

#define SKIP_TAGS \
    CV_TEST_TAG_DNN_SKIP_PARSER, \
    CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE
#define SKIP applyTestTag(SKIP_TAGS)
#define SKIP_(...) applyTestTag(__VA_ARGS__, SKIP_TAGS)


ASSERT_FALSE(name.empty());

#define EOF_LABEL exit_filter_parser
#define BEGIN_SWITCH() \
if (name.empty() /*false*/) \
{

#define CASE(t) \
    goto EOF_LABEL; \
} \
if (name == #t) \
{ \
    filterApplied = true;

#define END_SWITCH() \
    goto EOF_LABEL; \
} \
EOF_LABEL:

bool filterApplied = false;

// Update note: execute <opencv_extra>/testdata/dnn/onnx/generate_conformance_list.py
BEGIN_SWITCH()
CASE(test_abs)
    SKIP;
CASE(test_acos)
    SKIP;
CASE(test_acos_example)
    SKIP;
CASE(test_acosh)
    SKIP;
CASE(test_acosh_example)
    SKIP;
CASE(test_adagrad)
    SKIP;
CASE(test_adagrad_multiple)
    SKIP;
CASE(test_adam)
    SKIP;
CASE(test_adam_multiple)
    SKIP;
CASE(test_add)
    // pass
CASE(test_add_bcast)
    // pass
CASE(test_add_uint8)
    SKIP;
CASE(test_and2d)
    SKIP;
CASE(test_and3d)
    SKIP;
CASE(test_and4d)
    SKIP;
CASE(test_and_bcast3v1d)
    SKIP;
CASE(test_and_bcast3v2d)
    SKIP;
CASE(test_and_bcast4v2d)
    SKIP;
CASE(test_and_bcast4v3d)
    SKIP;
CASE(test_and_bcast4v4d)
    SKIP;
CASE(test_argmax_default_axis_example)
    SKIP;
CASE(test_argmax_default_axis_example_select_last_index)
    SKIP;
CASE(test_argmax_default_axis_random)
    SKIP;
CASE(test_argmax_default_axis_random_select_last_index)
    SKIP;
CASE(test_argmax_keepdims_example)
    SKIP;
CASE(test_argmax_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmax_keepdims_random)
    SKIP;
CASE(test_argmax_keepdims_random_select_last_index)
    SKIP;
CASE(test_argmax_negative_axis_keepdims_example)
    SKIP;
CASE(test_argmax_negative_axis_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmax_negative_axis_keepdims_random)
    SKIP;
CASE(test_argmax_negative_axis_keepdims_random_select_last_index)
    SKIP;
CASE(test_argmax_no_keepdims_example)
    SKIP;
CASE(test_argmax_no_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmax_no_keepdims_random)
    SKIP;
CASE(test_argmax_no_keepdims_random_select_last_index)
    SKIP;
CASE(test_argmin_default_axis_example)
    SKIP;
CASE(test_argmin_default_axis_example_select_last_index)
    SKIP;
CASE(test_argmin_default_axis_random)
    SKIP;
CASE(test_argmin_default_axis_random_select_last_index)
    SKIP;
CASE(test_argmin_keepdims_example)
    SKIP;
CASE(test_argmin_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmin_keepdims_random)
    SKIP;
CASE(test_argmin_keepdims_random_select_last_index)
    SKIP;
CASE(test_argmin_negative_axis_keepdims_example)
    SKIP;
CASE(test_argmin_negative_axis_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmin_negative_axis_keepdims_random)
    SKIP;
CASE(test_argmin_negative_axis_keepdims_random_select_last_index)
    SKIP;
CASE(test_argmin_no_keepdims_example)
    SKIP;
CASE(test_argmin_no_keepdims_example_select_last_index)
    SKIP;
CASE(test_argmin_no_keepdims_random)
    SKIP;
CASE(test_argmin_no_keepdims_random_select_last_index)
    SKIP;
CASE(test_asin)
    SKIP;
CASE(test_asin_example)
    SKIP;
CASE(test_asinh)
    SKIP;
CASE(test_asinh_example)
    SKIP;
CASE(test_atan)
    SKIP;
CASE(test_atan_example)
    SKIP;
CASE(test_atanh)
    SKIP;
CASE(test_atanh_example)
    SKIP;
CASE(test_averagepool_1d_default)
    // pass
CASE(test_averagepool_2d_ceil)
    // pass
CASE(test_averagepool_2d_default)
    // pass
CASE(test_averagepool_2d_pads)
    // pass
CASE(test_averagepool_2d_pads_count_include_pad)
    // pass
CASE(test_averagepool_2d_precomputed_pads)
    // pass
CASE(test_averagepool_2d_precomputed_pads_count_include_pad)
    // pass
CASE(test_averagepool_2d_precomputed_same_upper)
    // pass
CASE(test_averagepool_2d_precomputed_strides)
    // pass
CASE(test_averagepool_2d_same_lower)
    // pass
CASE(test_averagepool_2d_same_upper)
    // pass
CASE(test_averagepool_2d_strides)
    // pass
CASE(test_averagepool_3d_default)
    // pass
CASE(test_basic_conv_with_padding)
    // pass
CASE(test_basic_conv_without_padding)
    // pass
CASE(test_basic_convinteger)
    SKIP;
CASE(test_batchnorm_epsilon)
    SKIP;
CASE(test_batchnorm_epsilon_training_mode)
    SKIP;
CASE(test_batchnorm_example)
    SKIP;
CASE(test_batchnorm_example_training_mode)
    SKIP;
CASE(test_bernoulli)
    SKIP;
CASE(test_bernoulli_double)
    SKIP;
CASE(test_bernoulli_double_expanded)
    SKIP;
CASE(test_bernoulli_expanded)
    SKIP;
CASE(test_bernoulli_seed)
    SKIP;
CASE(test_bernoulli_seed_expanded)
    SKIP;
CASE(test_bitshift_left_uint16)
    SKIP;
CASE(test_bitshift_left_uint32)
    SKIP;
CASE(test_bitshift_left_uint64)
    SKIP;
CASE(test_bitshift_left_uint8)
    SKIP;
CASE(test_bitshift_right_uint16)
    SKIP;
CASE(test_bitshift_right_uint32)
    SKIP;
CASE(test_bitshift_right_uint64)
    SKIP;
CASE(test_bitshift_right_uint8)
    SKIP;
CASE(test_cast_BFLOAT16_to_FLOAT)
    SKIP;
CASE(test_cast_DOUBLE_to_FLOAT)
    SKIP;
CASE(test_cast_DOUBLE_to_FLOAT16)
    SKIP;
CASE(test_cast_FLOAT16_to_DOUBLE)
    SKIP;
CASE(test_cast_FLOAT16_to_FLOAT)
    SKIP;
CASE(test_cast_FLOAT_to_BFLOAT16)
    SKIP;
CASE(test_cast_FLOAT_to_DOUBLE)
    SKIP;
CASE(test_cast_FLOAT_to_FLOAT16)
    SKIP;
CASE(test_cast_FLOAT_to_STRING)
    // pass
CASE(test_cast_STRING_to_FLOAT)
    // pass
CASE(test_castlike_BFLOAT16_to_FLOAT)
    SKIP;
CASE(test_castlike_BFLOAT16_to_FLOAT_expanded)
    SKIP;
CASE(test_castlike_DOUBLE_to_FLOAT)
    SKIP;
CASE(test_castlike_DOUBLE_to_FLOAT16)
    SKIP;
CASE(test_castlike_DOUBLE_to_FLOAT16_expanded)
    SKIP;
CASE(test_castlike_DOUBLE_to_FLOAT_expanded)
    SKIP;
CASE(test_castlike_FLOAT16_to_DOUBLE)
    SKIP;
CASE(test_castlike_FLOAT16_to_DOUBLE_expanded)
    SKIP;
CASE(test_castlike_FLOAT16_to_FLOAT)
    SKIP;
CASE(test_castlike_FLOAT16_to_FLOAT_expanded)
    SKIP;
CASE(test_castlike_FLOAT_to_BFLOAT16)
    SKIP;
CASE(test_castlike_FLOAT_to_BFLOAT16_expanded)
    SKIP;
CASE(test_castlike_FLOAT_to_DOUBLE)
    SKIP;
CASE(test_castlike_FLOAT_to_DOUBLE_expanded)
    SKIP;
CASE(test_castlike_FLOAT_to_FLOAT16)
    SKIP;
CASE(test_castlike_FLOAT_to_FLOAT16_expanded)
    SKIP;
CASE(test_castlike_FLOAT_to_STRING)
    SKIP;
CASE(test_castlike_FLOAT_to_STRING_expanded)
    // pass
CASE(test_castlike_STRING_to_FLOAT)
    SKIP;
CASE(test_castlike_STRING_to_FLOAT_expanded)
    // pass
CASE(test_ceil)
    SKIP;
CASE(test_ceil_example)
    SKIP;
CASE(test_celu)
    SKIP;
CASE(test_celu_expanded)
    // pass
CASE(test_clip)
    // pass
CASE(test_clip_default_inbounds)
    // pass
CASE(test_clip_default_int8_inbounds)
    SKIP;
CASE(test_clip_default_int8_max)
    SKIP;
CASE(test_clip_default_int8_min)
    SKIP;
CASE(test_clip_default_max)
    // pass
CASE(test_clip_default_min)
    // pass
CASE(test_clip_example)
    // pass
CASE(test_clip_inbounds)
    // pass
CASE(test_clip_outbounds)
    // pass
CASE(test_clip_splitbounds)
    // pass
CASE(test_compress_0)
    SKIP;
CASE(test_compress_1)
    SKIP;
CASE(test_compress_default_axis)
    SKIP;
CASE(test_compress_negative_axis)
    SKIP;
CASE(test_concat_1d_axis_0)
    // pass
CASE(test_concat_1d_axis_negative_1)
    // pass
CASE(test_concat_2d_axis_0)
    // pass
CASE(test_concat_2d_axis_1)
    // pass
CASE(test_concat_2d_axis_negative_1)
    // pass
CASE(test_concat_2d_axis_negative_2)
    // pass
CASE(test_concat_3d_axis_0)
    // pass
CASE(test_concat_3d_axis_1)
    // pass
CASE(test_concat_3d_axis_2)
    // pass
CASE(test_concat_3d_axis_negative_1)
    // pass
CASE(test_concat_3d_axis_negative_2)
    // pass
CASE(test_concat_3d_axis_negative_3)
    // pass
CASE(test_constant)
    SKIP;
CASE(test_constant_pad)
    SKIP;
CASE(test_constantofshape_float_ones)
    SKIP;
CASE(test_constantofshape_int_shape_zero)
    SKIP;
CASE(test_constantofshape_int_zeros)
    SKIP;
CASE(test_conv_with_autopad_same)
    // pass
CASE(test_conv_with_strides_and_asymmetric_padding)
    // pass
CASE(test_conv_with_strides_no_padding)
    // pass
CASE(test_conv_with_strides_padding)
    // pass
CASE(test_convinteger_with_padding)
    SKIP;
CASE(test_convinteger_without_padding)
    SKIP;
CASE(test_convtranspose)
    SKIP;
CASE(test_convtranspose_1d)
    SKIP;
CASE(test_convtranspose_3d)
    SKIP;
CASE(test_convtranspose_autopad_same)
    SKIP;
CASE(test_convtranspose_dilations)
    SKIP;
CASE(test_convtranspose_kernel_shape)
    SKIP;
CASE(test_convtranspose_output_shape)
    SKIP;
CASE(test_convtranspose_pad)
    SKIP;
CASE(test_convtranspose_pads)
    SKIP;
CASE(test_convtranspose_with_kernel)
    SKIP;
CASE(test_cos)
    SKIP;
CASE(test_cos_example)
    SKIP;
CASE(test_cosh)
    SKIP;
CASE(test_cosh_example)
    SKIP;
CASE(test_cumsum_1d)
    SKIP;
CASE(test_cumsum_1d_exclusive)
    SKIP;
CASE(test_cumsum_1d_reverse)
    SKIP;
CASE(test_cumsum_1d_reverse_exclusive)
    SKIP;
CASE(test_cumsum_2d_axis_0)
    SKIP;
CASE(test_cumsum_2d_axis_1)
    SKIP;
CASE(test_cumsum_2d_negative_axis)
    SKIP;
CASE(test_depthtospace_crd_mode)
    SKIP;
CASE(test_depthtospace_crd_mode_example)
    SKIP;
CASE(test_depthtospace_dcr_mode)
    SKIP;
CASE(test_depthtospace_example)
    SKIP;
CASE(test_dequantizelinear)
    SKIP;
CASE(test_dequantizelinear_axis)
    SKIP;
CASE(test_det_2d)
    SKIP;
CASE(test_det_nd)
    SKIP;
CASE(test_div)
    // pass
CASE(test_div_bcast)
    // pass
CASE(test_div_example)
    SKIP;
CASE(test_div_uint8)
    SKIP;
CASE(test_dropout_default)
    // pass
CASE(test_dropout_default_mask)
    SKIP;
CASE(test_dropout_default_mask_ratio)
    SKIP;
CASE(test_dropout_default_old)
    // pass
CASE(test_dropout_default_ratio)
    // pass
CASE(test_dropout_random_old)
    // pass
CASE(test_dynamicquantizelinear)
    SKIP;
CASE(test_dynamicquantizelinear_expanded)
    SKIP;
CASE(test_dynamicquantizelinear_max_adjusted)
    SKIP;
CASE(test_dynamicquantizelinear_max_adjusted_expanded)
    SKIP;
CASE(test_dynamicquantizelinear_min_adjusted)
    SKIP;
CASE(test_dynamicquantizelinear_min_adjusted_expanded)
    SKIP;
CASE(test_edge_pad)
    SKIP;
CASE(test_einsum_batch_diagonal)
    SKIP;
CASE(test_einsum_batch_matmul)
    SKIP;
CASE(test_einsum_inner_prod)
    SKIP;
CASE(test_einsum_sum)
    SKIP;
CASE(test_einsum_transpose)
    SKIP;
CASE(test_elu)
    // pass
CASE(test_elu_default)
    // pass
CASE(test_elu_example)
    // pass
CASE(test_equal)
    SKIP;
CASE(test_equal_bcast)
    SKIP;
CASE(test_erf)
    SKIP;
CASE(test_exp)
    // pass
CASE(test_exp_example)
    // pass
CASE(test_expand_dim_changed)
    SKIP;
CASE(test_expand_dim_unchanged)
    SKIP;
CASE(test_eyelike_populate_off_main_diagonal)
    SKIP;
CASE(test_eyelike_with_dtype)
    SKIP;
CASE(test_eyelike_without_dtype)
    SKIP;
CASE(test_flatten_axis0)
    // pass
CASE(test_flatten_axis1)
    // pass
CASE(test_flatten_axis2)
    // pass
CASE(test_flatten_axis3)
    // pass
CASE(test_flatten_default_axis)
    // pass
CASE(test_flatten_negative_axis1)
    // pass
CASE(test_flatten_negative_axis2)
    // pass
CASE(test_flatten_negative_axis3)
    // pass
CASE(test_flatten_negative_axis4)
    // pass
CASE(test_floor)
    SKIP;
CASE(test_floor_example)
    SKIP;
CASE(test_gather_0)
    SKIP;
CASE(test_gather_1)
    SKIP;
CASE(test_gather_2d_indices)
    SKIP;
CASE(test_gather_elements_0)
    SKIP;
CASE(test_gather_elements_1)
    SKIP;
CASE(test_gather_elements_negative_indices)
    SKIP;
CASE(test_gather_negative_indices)
    SKIP;
CASE(test_gathernd_example_float32)
    SKIP;
CASE(test_gathernd_example_int32)
    SKIP;
CASE(test_gathernd_example_int32_batch_dim1)
    SKIP;
CASE(test_gemm_all_attributes)
    SKIP;
CASE(test_gemm_alpha)
    SKIP;
CASE(test_gemm_beta)
    SKIP;
CASE(test_gemm_default_matrix_bias)
    SKIP;
CASE(test_gemm_default_no_bias)
    SKIP;
CASE(test_gemm_default_scalar_bias)
    SKIP;
CASE(test_gemm_default_single_elem_vector_bias)
    SKIP;
CASE(test_gemm_default_vector_bias)
    SKIP;
CASE(test_gemm_default_zero_bias)
    SKIP;
CASE(test_gemm_transposeA)
    SKIP;
CASE(test_gemm_transposeB)
    SKIP;
CASE(test_globalaveragepool)
    // pass
CASE(test_globalaveragepool_precomputed)
    // pass
CASE(test_globalmaxpool)
    // pass
CASE(test_globalmaxpool_precomputed)
    // pass
CASE(test_greater)
    SKIP;
CASE(test_greater_bcast)
    SKIP;
CASE(test_greater_equal)
    SKIP;
CASE(test_greater_equal_bcast)
    SKIP;
CASE(test_greater_equal_bcast_expanded)
    SKIP;
CASE(test_greater_equal_expanded)
    SKIP;
CASE(test_gridsample)
    SKIP;
CASE(test_gridsample_aligncorners_true)
    SKIP;
CASE(test_gridsample_bicubic)
    SKIP;
CASE(test_gridsample_bilinear)
    SKIP;
CASE(test_gridsample_border_padding)
    SKIP;
CASE(test_gridsample_nearest)
    SKIP;
CASE(test_gridsample_reflection_padding)
    SKIP;
CASE(test_gridsample_zeros_padding)
    SKIP;
CASE(test_gru_batchwise)
    SKIP;
CASE(test_gru_defaults)
    SKIP;
CASE(test_gru_seq_length)
    SKIP;
CASE(test_gru_with_initial_bias)
    SKIP;
CASE(test_hardmax_axis_0)
    SKIP;
CASE(test_hardmax_axis_1)
    SKIP;
CASE(test_hardmax_axis_2)
    SKIP;
CASE(test_hardmax_default_axis)
    SKIP;
CASE(test_hardmax_example)
    SKIP;
CASE(test_hardmax_negative_axis)
    SKIP;
CASE(test_hardmax_one_hot)
    SKIP;
CASE(test_hardsigmoid)
    SKIP;
CASE(test_hardsigmoid_default)
    SKIP;
CASE(test_hardsigmoid_example)
    SKIP;
CASE(test_hardswish)
    SKIP;
CASE(test_hardswish_expanded)
    SKIP;
CASE(test_identity)
    // pass
CASE(test_identity_opt)
    SKIP;
CASE(test_identity_sequence)
    SKIP;
CASE(test_if)
    SKIP;
CASE(test_if_opt)
    SKIP;
CASE(test_if_seq)
    SKIP;
CASE(test_instancenorm_epsilon)
    SKIP;
CASE(test_instancenorm_example)
    SKIP;
CASE(test_isinf)
    SKIP;
CASE(test_isinf_negative)
    SKIP;
CASE(test_isinf_positive)
    SKIP;
CASE(test_isnan)
    SKIP;
CASE(test_leakyrelu)
    // pass
CASE(test_leakyrelu_default)
    // pass
CASE(test_leakyrelu_example)
    // pass
CASE(test_less)
    SKIP;
CASE(test_less_bcast)
    SKIP;
CASE(test_less_equal)
    SKIP;
CASE(test_less_equal_bcast)
    SKIP;
CASE(test_less_equal_bcast_expanded)
    SKIP;
CASE(test_less_equal_expanded)
    SKIP;
CASE(test_log)
    SKIP;
CASE(test_log_example)
    SKIP;
CASE(test_logsoftmax_axis_0)
    // pass
CASE(test_logsoftmax_axis_0_expanded)
    // pass
CASE(test_logsoftmax_axis_1)
    // pass
CASE(test_logsoftmax_axis_1_expanded)
    // pass
CASE(test_logsoftmax_axis_2)
    // pass
CASE(test_logsoftmax_axis_2_expanded)
    // pass
CASE(test_logsoftmax_default_axis)
    // pass
CASE(test_logsoftmax_default_axis_expanded)
    // pass
CASE(test_logsoftmax_example_1)
    // pass
CASE(test_logsoftmax_example_1_expanded)
    // pass
CASE(test_logsoftmax_large_number)
    // pass
CASE(test_logsoftmax_large_number_expanded)
    // pass
CASE(test_logsoftmax_negative_axis)
    // pass
CASE(test_logsoftmax_negative_axis_expanded)
    // pass
CASE(test_loop11)
    SKIP;
CASE(test_loop13_seq)
    SKIP;
CASE(test_loop16_seq_none)
    SKIP;
CASE(test_lrn)
    // pass
CASE(test_lrn_default)
    // pass
CASE(test_lstm_batchwise)
    SKIP;
CASE(test_lstm_defaults)
    SKIP;
CASE(test_lstm_with_initial_bias)
    SKIP;
CASE(test_lstm_with_peepholes)
    SKIP;
CASE(test_matmul_2d)
    // pass
CASE(test_matmul_3d)
    // pass
CASE(test_matmul_4d)
    // pass
CASE(test_matmulinteger)
    SKIP;
CASE(test_max_example)
    SKIP;
CASE(test_max_float16)
    SKIP;
CASE(test_max_float32)
    SKIP;
CASE(test_max_float64)
    SKIP;
CASE(test_max_int16)
    SKIP;
CASE(test_max_int32)
    SKIP;
CASE(test_max_int64)
    SKIP;
CASE(test_max_int8)
    SKIP;
CASE(test_max_one_input)
    SKIP;
CASE(test_max_two_inputs)
    SKIP;
CASE(test_max_uint16)
    SKIP;
CASE(test_max_uint32)
    SKIP;
CASE(test_max_uint64)
    SKIP;
CASE(test_max_uint8)
    SKIP;
CASE(test_maxpool_1d_default)
    // pass
CASE(test_maxpool_2d_ceil)
    // pass
CASE(test_maxpool_2d_default)
    // pass
CASE(test_maxpool_2d_dilations)
    // pass
CASE(test_maxpool_2d_pads)
    // pass
CASE(test_maxpool_2d_precomputed_pads)
    // pass
CASE(test_maxpool_2d_precomputed_same_upper)
    // pass
CASE(test_maxpool_2d_precomputed_strides)
    // pass
CASE(test_maxpool_2d_same_lower)
    // pass
CASE(test_maxpool_2d_same_upper)
    // pass
CASE(test_maxpool_2d_strides)
    // pass
CASE(test_maxpool_2d_uint8)
    SKIP;
CASE(test_maxpool_3d_default)
    // pass
CASE(test_maxpool_with_argmax_2d_precomputed_pads)
    // pass
CASE(test_maxpool_with_argmax_2d_precomputed_strides)
    // pass
CASE(test_maxunpool_export_with_output_shape)
    SKIP;
CASE(test_maxunpool_export_without_output_shape)
    SKIP;
CASE(test_mean_example)
    SKIP;
CASE(test_mean_one_input)
    SKIP;
CASE(test_mean_two_inputs)
    SKIP;
CASE(test_min_example)
    SKIP;
CASE(test_min_float16)
    SKIP;
CASE(test_min_float32)
    SKIP;
CASE(test_min_float64)
    SKIP;
CASE(test_min_int16)
    SKIP;
CASE(test_min_int32)
    SKIP;
CASE(test_min_int64)
    SKIP;
CASE(test_min_int8)
    SKIP;
CASE(test_min_one_input)
    SKIP;
CASE(test_min_two_inputs)
    SKIP;
CASE(test_min_uint16)
    SKIP;
CASE(test_min_uint32)
    SKIP;
CASE(test_min_uint64)
    SKIP;
CASE(test_min_uint8)
    SKIP;
CASE(test_mod_broadcast)
    SKIP;
CASE(test_mod_int64_fmod)
    SKIP;
CASE(test_mod_mixed_sign_float16)
    SKIP;
CASE(test_mod_mixed_sign_float32)
    SKIP;
CASE(test_mod_mixed_sign_float64)
    SKIP;
CASE(test_mod_mixed_sign_int16)
    SKIP;
CASE(test_mod_mixed_sign_int32)
    SKIP;
CASE(test_mod_mixed_sign_int64)
    SKIP;
CASE(test_mod_mixed_sign_int8)
    SKIP;
CASE(test_mod_uint16)
    SKIP;
CASE(test_mod_uint32)
    SKIP;
CASE(test_mod_uint64)
    SKIP;
CASE(test_mod_uint8)
    SKIP;
CASE(test_momentum)
    SKIP;
CASE(test_momentum_multiple)
    SKIP;
CASE(test_mul)
    // pass
CASE(test_mul_bcast)
    // pass
CASE(test_mul_example)
    SKIP;
CASE(test_mul_uint8)
    SKIP;
CASE(test_mvn)
    SKIP;
CASE(test_mvn_expanded)
    SKIP;
CASE(test_neg)
    // pass
CASE(test_neg_example)
    // pass
CASE(test_nesterov_momentum)
    SKIP;
CASE(test_nllloss_NC)
    SKIP;
CASE(test_nllloss_NC_expanded)
    SKIP;
CASE(test_nllloss_NCd1)
    SKIP;
CASE(test_nllloss_NCd1_expanded)
    SKIP;
CASE(test_nllloss_NCd1_ii)
    SKIP;
CASE(test_nllloss_NCd1_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1_mean_weight_negative_ii)
    SKIP;
CASE(test_nllloss_NCd1_mean_weight_negative_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1_weight)
    SKIP;
CASE(test_nllloss_NCd1_weight_expanded)
    SKIP;
CASE(test_nllloss_NCd1_weight_ii)
    SKIP;
CASE(test_nllloss_NCd1_weight_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2)
    SKIP;
CASE(test_nllloss_NCd1d2_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_no_weight_reduction_mean_ii)
    SKIP;
CASE(test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_reduction_mean)
    SKIP;
CASE(test_nllloss_NCd1d2_reduction_mean_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_reduction_sum)
    SKIP;
CASE(test_nllloss_NCd1d2_reduction_sum_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_mean)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_mean_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_ii)
    SKIP;
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2d3_none_no_weight_negative_ii)
    SKIP;
CASE(test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2d3_sum_weight_high_ii)
    SKIP;
CASE(test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2d3d4d5_mean_weight)
    SKIP;
CASE(test_nllloss_NCd1d2d3d4d5_mean_weight_expanded)
    SKIP;
CASE(test_nllloss_NCd1d2d3d4d5_none_no_weight)
    SKIP;
CASE(test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded)
    SKIP;
CASE(test_nonmaxsuppression_center_point_box_format)
    SKIP;
CASE(test_nonmaxsuppression_flipped_coordinates)
    SKIP;
CASE(test_nonmaxsuppression_identical_boxes)
    SKIP;
CASE(test_nonmaxsuppression_limit_output_size)
    SKIP;
CASE(test_nonmaxsuppression_single_box)
    SKIP;
CASE(test_nonmaxsuppression_suppress_by_IOU)
    SKIP;
CASE(test_nonmaxsuppression_suppress_by_IOU_and_scores)
    SKIP;
CASE(test_nonmaxsuppression_two_batches)
    SKIP;
CASE(test_nonmaxsuppression_two_classes)
    SKIP;
CASE(test_nonzero_example)
    SKIP;
CASE(test_not_2d)
    SKIP;
CASE(test_not_3d)
    SKIP;
CASE(test_not_4d)
    SKIP;
CASE(test_onehot_negative_indices)
    SKIP;
CASE(test_onehot_with_axis)
    SKIP;
CASE(test_onehot_with_negative_axis)
    SKIP;
CASE(test_onehot_without_axis)
    SKIP;
CASE(test_optional_get_element)
    SKIP;
CASE(test_optional_get_element_sequence)
    SKIP;
CASE(test_optional_has_element)
    SKIP;
CASE(test_optional_has_element_empty)
    SKIP;
CASE(test_or2d)
    SKIP;
CASE(test_or3d)
    SKIP;
CASE(test_or4d)
    SKIP;
CASE(test_or_bcast3v1d)
    SKIP;
CASE(test_or_bcast3v2d)
    SKIP;
CASE(test_or_bcast4v2d)
    SKIP;
CASE(test_or_bcast4v3d)
    SKIP;
CASE(test_or_bcast4v4d)
    SKIP;
CASE(test_pow)
    SKIP;
CASE(test_pow_bcast_array)
    SKIP;
CASE(test_pow_bcast_scalar)
    SKIP;
CASE(test_pow_example)
    SKIP;
CASE(test_pow_types_float)
    SKIP;
CASE(test_pow_types_float32_int32)
    SKIP;
CASE(test_pow_types_float32_int64)
    SKIP;
CASE(test_pow_types_float32_uint32)
    SKIP;
CASE(test_pow_types_float32_uint64)
    SKIP;
CASE(test_pow_types_int)
    SKIP;
CASE(test_pow_types_int32_float32)
    SKIP;
CASE(test_pow_types_int32_int32)
    SKIP;
CASE(test_pow_types_int64_float32)
    SKIP;
CASE(test_pow_types_int64_int64)
    SKIP;
CASE(test_prelu_broadcast)
    SKIP;
CASE(test_prelu_example)
    SKIP;
CASE(test_qlinearconv)
    SKIP;
CASE(test_qlinearmatmul_2D)
    SKIP;
CASE(test_qlinearmatmul_3D)
    SKIP;
CASE(test_quantizelinear)
    SKIP;
CASE(test_quantizelinear_axis)
    SKIP;
CASE(test_range_float_type_positive_delta)
    SKIP;
CASE(test_range_float_type_positive_delta_expanded)
    SKIP;
CASE(test_range_int32_type_negative_delta)
    SKIP;
CASE(test_range_int32_type_negative_delta_expanded)
    SKIP;
CASE(test_reciprocal)
    SKIP;
CASE(test_reciprocal_example)
    SKIP;
CASE(test_reduce_l1_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_l1_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_l1_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_l1_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_l1_keep_dims_example)
    SKIP;
CASE(test_reduce_l1_keep_dims_random)
    SKIP;
CASE(test_reduce_l1_negative_axes_keep_dims_example)
    SKIP;
CASE(test_reduce_l1_negative_axes_keep_dims_random)
    SKIP;
CASE(test_reduce_l2_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_l2_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_l2_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_l2_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_l2_keep_dims_example)
    SKIP;
CASE(test_reduce_l2_keep_dims_random)
    SKIP;
CASE(test_reduce_l2_negative_axes_keep_dims_example)
    SKIP;
CASE(test_reduce_l2_negative_axes_keep_dims_random)
    SKIP;
CASE(test_reduce_log_sum)
    SKIP;
CASE(test_reduce_log_sum_asc_axes)
    SKIP;
CASE(test_reduce_log_sum_default)
    SKIP;
CASE(test_reduce_log_sum_desc_axes)
    SKIP;
CASE(test_reduce_log_sum_exp_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_log_sum_exp_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_log_sum_exp_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_log_sum_exp_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_log_sum_exp_keepdims_example)
    SKIP;
CASE(test_reduce_log_sum_exp_keepdims_random)
    SKIP;
CASE(test_reduce_log_sum_exp_negative_axes_keepdims_example)
    SKIP;
CASE(test_reduce_log_sum_exp_negative_axes_keepdims_random)
    SKIP;
CASE(test_reduce_log_sum_negative_axes)
    SKIP;
CASE(test_reduce_max_default_axes_keepdim_example)
    SKIP;
CASE(test_reduce_max_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_max_do_not_keepdims_example)
    // pass
CASE(test_reduce_max_do_not_keepdims_random)
    // pass
CASE(test_reduce_max_keepdims_example)
    // pass
CASE(test_reduce_max_keepdims_random)
    // pass
CASE(test_reduce_max_negative_axes_keepdims_example)
    // pass
CASE(test_reduce_max_negative_axes_keepdims_random)
    // pass
CASE(test_reduce_mean_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_mean_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_mean_do_not_keepdims_example)
    // pass
CASE(test_reduce_mean_do_not_keepdims_random)
    // pass
CASE(test_reduce_mean_keepdims_example)
    // pass
CASE(test_reduce_mean_keepdims_random)
    // pass
CASE(test_reduce_mean_negative_axes_keepdims_example)
    // pass
CASE(test_reduce_mean_negative_axes_keepdims_random)
    // pass
CASE(test_reduce_min_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_min_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_min_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_min_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_min_keepdims_example)
    SKIP;
CASE(test_reduce_min_keepdims_random)
    SKIP;
CASE(test_reduce_min_negative_axes_keepdims_example)
    SKIP;
CASE(test_reduce_min_negative_axes_keepdims_random)
    SKIP;
CASE(test_reduce_prod_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_prod_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_prod_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_prod_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_prod_keepdims_example)
    SKIP;
CASE(test_reduce_prod_keepdims_random)
    SKIP;
CASE(test_reduce_prod_negative_axes_keepdims_example)
    SKIP;
CASE(test_reduce_prod_negative_axes_keepdims_random)
    SKIP;
CASE(test_reduce_sum_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_sum_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_sum_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_sum_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_sum_empty_axes_input_noop_example)
    SKIP;
CASE(test_reduce_sum_empty_axes_input_noop_random)
    SKIP;
CASE(test_reduce_sum_keepdims_example)
    SKIP;
CASE(test_reduce_sum_keepdims_random)
    SKIP;
CASE(test_reduce_sum_negative_axes_keepdims_example)
    SKIP;
CASE(test_reduce_sum_negative_axes_keepdims_random)
    SKIP;
CASE(test_reduce_sum_square_default_axes_keepdims_example)
    SKIP;
CASE(test_reduce_sum_square_default_axes_keepdims_random)
    SKIP;
CASE(test_reduce_sum_square_do_not_keepdims_example)
    SKIP;
CASE(test_reduce_sum_square_do_not_keepdims_random)
    SKIP;
CASE(test_reduce_sum_square_keepdims_example)
    SKIP;
CASE(test_reduce_sum_square_keepdims_random)
    SKIP;
CASE(test_reduce_sum_square_negative_axes_keepdims_example)
    SKIP;
CASE(test_reduce_sum_square_negative_axes_keepdims_random)
    SKIP;
CASE(test_reflect_pad)
    SKIP;
CASE(test_relu)
    // pass
CASE(test_reshape_allowzero_reordered)
    SKIP;
CASE(test_reshape_extended_dims)
    SKIP;
CASE(test_reshape_negative_dim)
    SKIP;
CASE(test_reshape_negative_extended_dims)
    SKIP;
CASE(test_reshape_one_dim)
    SKIP;
CASE(test_reshape_reduced_dims)
    SKIP;
CASE(test_reshape_reordered_all_dims)
    SKIP;
CASE(test_reshape_reordered_last_dims)
    SKIP;
CASE(test_reshape_zero_and_negative_dim)
    SKIP;
CASE(test_reshape_zero_dim)
    SKIP;
CASE(test_resize_downsample_scales_cubic)
    SKIP;
CASE(test_resize_downsample_scales_cubic_A_n0p5_exclude_outside)
    SKIP;
CASE(test_resize_downsample_scales_cubic_align_corners)
    SKIP;
CASE(test_resize_downsample_scales_linear)
    SKIP;
CASE(test_resize_downsample_scales_linear_align_corners)
    SKIP;
CASE(test_resize_downsample_scales_nearest)
    SKIP;
CASE(test_resize_downsample_sizes_cubic)
    SKIP;
CASE(test_resize_downsample_sizes_linear_pytorch_half_pixel)
    SKIP;
CASE(test_resize_downsample_sizes_nearest)
    SKIP;
CASE(test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn)
    SKIP;
CASE(test_resize_tf_crop_and_resize)
    SKIP;
CASE(test_resize_upsample_scales_cubic)
    SKIP;
CASE(test_resize_upsample_scales_cubic_A_n0p5_exclude_outside)
    SKIP;
CASE(test_resize_upsample_scales_cubic_align_corners)
    SKIP;
CASE(test_resize_upsample_scales_cubic_asymmetric)
    SKIP;
CASE(test_resize_upsample_scales_linear)
    SKIP;
CASE(test_resize_upsample_scales_linear_align_corners)
    SKIP;
CASE(test_resize_upsample_scales_nearest)
    SKIP;
CASE(test_resize_upsample_sizes_cubic)
    SKIP;
CASE(test_resize_upsample_sizes_nearest)
    SKIP;
CASE(test_resize_upsample_sizes_nearest_ceil_half_pixel)
    SKIP;
CASE(test_resize_upsample_sizes_nearest_floor_align_corners)
    SKIP;
CASE(test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric)
    SKIP;
CASE(test_reversesequence_batch)
    SKIP;
CASE(test_reversesequence_time)
    SKIP;
CASE(test_rnn_seq_length)
    SKIP;
CASE(test_roialign_aligned_false)
    SKIP;
CASE(test_roialign_aligned_true)
    SKIP;
CASE(test_round)
    SKIP;
CASE(test_scan9_sum)
    SKIP;
CASE(test_scan_sum)
    SKIP;
CASE(test_scatter_elements_with_axis)
    SKIP;
CASE(test_scatter_elements_with_duplicate_indices)
    SKIP;
CASE(test_scatter_elements_with_negative_indices)
    SKIP;
CASE(test_scatter_elements_without_axis)
    SKIP;
CASE(test_scatter_with_axis)
    SKIP;
CASE(test_scatter_without_axis)
    SKIP;
CASE(test_scatternd)
    SKIP;
CASE(test_scatternd_add)
    SKIP;
CASE(test_scatternd_multiply)
    SKIP;
CASE(test_sce_NCd1_mean_weight_negative_ii)
    SKIP;
CASE(test_sce_NCd1_mean_weight_negative_ii_expanded)
    SKIP;
CASE(test_sce_NCd1_mean_weight_negative_ii_log_prob)
    SKIP;
CASE(test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii)
    SKIP;
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob)
    SKIP;
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3_sum_weight_high_ii)
    SKIP;
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_log_prob)
    SKIP;
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_mean_weight)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_mean_weight_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_mean_weight_log_prob)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_none_no_weight)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_expanded)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_log_prob)
    SKIP;
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded)
    SKIP;
CASE(test_sce_mean)
    SKIP;
CASE(test_sce_mean_3d)
    SKIP;
CASE(test_sce_mean_3d_expanded)
    SKIP;
CASE(test_sce_mean_3d_log_prob)
    SKIP;
CASE(test_sce_mean_3d_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_expanded)
    SKIP;
CASE(test_sce_mean_log_prob)
    SKIP;
CASE(test_sce_mean_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii)
    SKIP;
CASE(test_sce_mean_no_weight_ii_3d)
    SKIP;
CASE(test_sce_mean_no_weight_ii_3d_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii_3d_log_prob)
    SKIP;
CASE(test_sce_mean_no_weight_ii_3d_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii_4d)
    SKIP;
CASE(test_sce_mean_no_weight_ii_4d_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii_4d_log_prob)
    SKIP;
CASE(test_sce_mean_no_weight_ii_4d_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii_expanded)
    SKIP;
CASE(test_sce_mean_no_weight_ii_log_prob)
    SKIP;
CASE(test_sce_mean_no_weight_ii_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_weight)
    SKIP;
CASE(test_sce_mean_weight_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii)
    SKIP;
CASE(test_sce_mean_weight_ii_3d)
    SKIP;
CASE(test_sce_mean_weight_ii_3d_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii_3d_log_prob)
    SKIP;
CASE(test_sce_mean_weight_ii_3d_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii_4d)
    SKIP;
CASE(test_sce_mean_weight_ii_4d_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii_4d_log_prob)
    SKIP;
CASE(test_sce_mean_weight_ii_4d_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii_expanded)
    SKIP;
CASE(test_sce_mean_weight_ii_log_prob)
    SKIP;
CASE(test_sce_mean_weight_ii_log_prob_expanded)
    SKIP;
CASE(test_sce_mean_weight_log_prob)
    SKIP;
CASE(test_sce_mean_weight_log_prob_expanded)
    SKIP;
CASE(test_sce_none)
    SKIP;
CASE(test_sce_none_expanded)
    SKIP;
CASE(test_sce_none_log_prob)
    SKIP;
CASE(test_sce_none_log_prob_expanded)
    SKIP;
CASE(test_sce_none_weights)
    SKIP;
CASE(test_sce_none_weights_expanded)
    SKIP;
CASE(test_sce_none_weights_log_prob)
    SKIP;
CASE(test_sce_none_weights_log_prob_expanded)
    SKIP;
CASE(test_sce_sum)
    SKIP;
CASE(test_sce_sum_expanded)
    SKIP;
CASE(test_sce_sum_log_prob)
    SKIP;
CASE(test_sce_sum_log_prob_expanded)
    SKIP;
CASE(test_selu)
    SKIP;
CASE(test_selu_default)
    SKIP;
CASE(test_selu_example)
    SKIP;
CASE(test_sequence_insert_at_back)
    SKIP;
CASE(test_sequence_insert_at_front)
    SKIP;
CASE(test_shape)
    SKIP;
CASE(test_shape_clip_end)
    SKIP;
CASE(test_shape_clip_start)
    SKIP;
CASE(test_shape_end_1)
    SKIP;
CASE(test_shape_end_negative_1)
    SKIP;
CASE(test_shape_example)
    SKIP;
CASE(test_shape_start_1)
    SKIP;
CASE(test_shape_start_1_end_2)
    SKIP;
CASE(test_shape_start_1_end_negative_1)
    SKIP;
CASE(test_shape_start_negative_1)
    SKIP;
CASE(test_shrink_hard)
    SKIP;
CASE(test_shrink_soft)
    SKIP;
CASE(test_sigmoid)
    // pass
CASE(test_sigmoid_example)
    // pass
CASE(test_sign)
    SKIP;
CASE(test_simple_rnn_batchwise)
    SKIP;
CASE(test_simple_rnn_defaults)
    SKIP;
CASE(test_simple_rnn_with_initial_bias)
    SKIP;
CASE(test_sin)
    SKIP;
CASE(test_sin_example)
    SKIP;
CASE(test_sinh)
    SKIP;
CASE(test_sinh_example)
    SKIP;
CASE(test_size)
    SKIP;
CASE(test_size_example)
    SKIP;
CASE(test_slice)
    SKIP;
CASE(test_slice_default_axes)
    SKIP;
CASE(test_slice_default_steps)
    SKIP;
CASE(test_slice_end_out_of_bounds)
    SKIP;
CASE(test_slice_neg)
    SKIP;
CASE(test_slice_neg_steps)
    SKIP;
CASE(test_slice_negative_axes)
    SKIP;
CASE(test_slice_start_out_of_bounds)
    SKIP;
CASE(test_softmax_axis_0)
    // pass
CASE(test_softmax_axis_0_expanded)
    // pass
CASE(test_softmax_axis_1)
    // pass
CASE(test_softmax_axis_1_expanded)
    // pass
CASE(test_softmax_axis_2)
    // pass
CASE(test_softmax_axis_2_expanded)
    // pass
CASE(test_softmax_default_axis)
    // pass
CASE(test_softmax_default_axis_expanded)
    // pass
CASE(test_softmax_example)
    // pass
CASE(test_softmax_example_expanded)
    // pass
CASE(test_softmax_large_number)
    // pass
CASE(test_softmax_large_number_expanded)
    // pass
CASE(test_softmax_negative_axis)
    // pass
CASE(test_softmax_negative_axis_expanded)
    // pass
CASE(test_softplus)
    SKIP;
CASE(test_softplus_example)
    SKIP;
CASE(test_softsign)
    SKIP;
CASE(test_softsign_example)
    SKIP;
CASE(test_spacetodepth)
    SKIP;
CASE(test_spacetodepth_example)
    SKIP;
CASE(test_split_equal_parts_1d)
    // pass
CASE(test_split_equal_parts_2d)
    // pass
CASE(test_split_equal_parts_default_axis)
    // pass
CASE(test_split_variable_parts_1d)
    SKIP;
CASE(test_split_variable_parts_2d)
    SKIP;
CASE(test_split_variable_parts_default_axis)
    SKIP;
CASE(test_split_zero_size_splits)
    SKIP;
CASE(test_sqrt)
    SKIP;
CASE(test_sqrt_example)
    SKIP;
CASE(test_squeeze)
    SKIP;
CASE(test_squeeze_negative_axes)
    SKIP;
CASE(test_strnormalizer_export_monday_casesensintive_lower)
    SKIP;
CASE(test_strnormalizer_export_monday_casesensintive_nochangecase)
    SKIP;
CASE(test_strnormalizer_export_monday_casesensintive_upper)
    SKIP;
CASE(test_strnormalizer_export_monday_empty_output)
    SKIP;
CASE(test_strnormalizer_export_monday_insensintive_upper_twodim)
    SKIP;
CASE(test_strnormalizer_nostopwords_nochangecase)
    SKIP;
CASE(test_sub)
    // pass
CASE(test_sub_bcast)
    // pass
CASE(test_sub_example)
    SKIP;
CASE(test_sub_uint8)
    SKIP;
CASE(test_sum_example)
    SKIP;
CASE(test_sum_one_input)
    // pass
CASE(test_sum_two_inputs)
    SKIP;
CASE(test_tan)
    SKIP;
CASE(test_tan_example)
    SKIP;
CASE(test_tanh)
    // pass
CASE(test_tanh_example)
    // pass
CASE(test_tfidfvectorizer_tf_batch_onlybigrams_skip0)
    SKIP;
CASE(test_tfidfvectorizer_tf_batch_onlybigrams_skip5)
    SKIP;
CASE(test_tfidfvectorizer_tf_batch_uniandbigrams_skip5)
    SKIP;
CASE(test_tfidfvectorizer_tf_only_bigrams_skip0)
    SKIP;
CASE(test_tfidfvectorizer_tf_onlybigrams_levelempty)
    SKIP;
CASE(test_tfidfvectorizer_tf_onlybigrams_skip5)
    SKIP;
CASE(test_tfidfvectorizer_tf_uniandbigrams_skip5)
    SKIP;
CASE(test_thresholdedrelu)
    SKIP;
CASE(test_thresholdedrelu_default)
    SKIP;
CASE(test_thresholdedrelu_example)
    SKIP;
CASE(test_tile)
    SKIP;
CASE(test_tile_precomputed)
    SKIP;
CASE(test_top_k)
    SKIP;
CASE(test_top_k_negative_axis)
    SKIP;
CASE(test_top_k_smallest)
    SKIP;
CASE(test_training_dropout)
    SKIP;
CASE(test_training_dropout_default)
    SKIP;
CASE(test_training_dropout_default_mask)
    SKIP;
CASE(test_training_dropout_mask)
    SKIP;
CASE(test_training_dropout_zero_ratio)
    SKIP;
CASE(test_training_dropout_zero_ratio_mask)
    SKIP;
CASE(test_transpose_all_permutations_0)
    // pass
CASE(test_transpose_all_permutations_1)
    // pass
CASE(test_transpose_all_permutations_2)
    // pass
CASE(test_transpose_all_permutations_3)
    // pass
CASE(test_transpose_all_permutations_4)
    // pass
CASE(test_transpose_all_permutations_5)
    // pass
CASE(test_transpose_default)
    // pass
CASE(test_tril)
    SKIP;
CASE(test_tril_neg)
    SKIP;
CASE(test_tril_one_row_neg)
    SKIP;
CASE(test_tril_out_neg)
    SKIP;
CASE(test_tril_out_pos)
    SKIP;
CASE(test_tril_pos)
    SKIP;
CASE(test_tril_square)
    SKIP;
CASE(test_tril_square_neg)
    SKIP;
CASE(test_tril_zero)
    SKIP;
CASE(test_triu)
    SKIP;
CASE(test_triu_neg)
    SKIP;
CASE(test_triu_one_row)
    SKIP;
CASE(test_triu_out_neg_out)
    SKIP;
CASE(test_triu_out_pos)
    SKIP;
CASE(test_triu_pos)
    SKIP;
CASE(test_triu_square)
    SKIP;
CASE(test_triu_square_neg)
    SKIP;
CASE(test_triu_zero)
    SKIP;
CASE(test_unique_not_sorted_without_axis)
    SKIP;
CASE(test_unique_sorted_with_axis)
    SKIP;
CASE(test_unique_sorted_with_axis_3d)
    SKIP;
CASE(test_unique_sorted_with_negative_axis)
    SKIP;
CASE(test_unique_sorted_without_axis)
    SKIP;
CASE(test_unsqueeze_axis_0)
    SKIP;
CASE(test_unsqueeze_axis_1)
    SKIP;
CASE(test_unsqueeze_axis_2)
    SKIP;
CASE(test_unsqueeze_axis_3)
    // pass
CASE(test_unsqueeze_negative_axes)
    SKIP;
CASE(test_unsqueeze_three_axes)
    SKIP;
CASE(test_unsqueeze_two_axes)
    SKIP;
CASE(test_unsqueeze_unsorted_axes)
    SKIP;
CASE(test_upsample_nearest)
    // pass
CASE(test_where_example)
    SKIP;
CASE(test_where_long_example)
    SKIP;
CASE(test_xor2d)
    SKIP;
CASE(test_xor3d)
    SKIP;
CASE(test_xor4d)
    SKIP;
CASE(test_xor_bcast3v1d)
    SKIP;
CASE(test_xor_bcast3v2d)
    SKIP;
CASE(test_xor_bcast4v2d)
    SKIP;
CASE(test_xor_bcast4v3d)
    SKIP;
CASE(test_xor_bcast4v4d)
    SKIP;
END_SWITCH()
#undef EOF_LABEL
#undef BEGIN_SWITCH
#undef CASE
#undef END_SWITCH
if (!filterApplied)
{
    ADD_FAILURE() << "Parser: unknown test='" << name << "'. Update filter configuration";
}

#undef SKIP_TAGS
#undef SKIP
#undef SKIP_

#endif
