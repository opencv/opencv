// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// not a standalone file, see test_onnx_conformance.cpp
#if 0
cout << "Filtering is disabled: OpenVINO" << endl;
#else


#if 0
// Stats for --gtest_filter=*ONNX_conformance*NGRAPH*
[ SKIPSTAT ] TAG='dnn_skip_ie_myriadx' skip 48 tests
[ SKIPSTAT ] TAG='dnn_skip_ie' skip 0 tests (149 times in extra skip list)
[ SKIPSTAT ] TAG='dnn_skip_ie_ngraph' skip 0 tests (149 times in extra skip list)
[ SKIPSTAT ] TAG='dnn_skip_ie_cpu' skip 29 tests
[ SKIPSTAT ] TAG='dnn_skip_ie_ocl' skip 34 tests
[ SKIPSTAT ] TAG='dnn_skip_ie_ocl_fp16' skip 38 tests
#endif


#define SKIP_TAGS \
    CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, \
    CV_TEST_TAG_DNN_SKIP_IE, \
    CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE
#define SKIP_(...) applyTestTag(__VA_ARGS__, SKIP_TAGS)
#define SKIP applyTestTag(tag_target_skip, SKIP_TAGS)
#define SKIP_CPU if (target == DNN_TARGET_CPU) applyTestTag(tag_target_skip, SKIP_TAGS)
#define SKIP_NON_CPU if (target != DNN_TARGET_CPU) applyTestTag(tag_target_skip, SKIP_TAGS)
#define SKIP_OPENCL if (target == DNN_TARGET_OPENCL) applyTestTag(tag_target_skip, SKIP_TAGS)
#define SKIP_OPENCL_FP16 if (target == DNN_TARGET_OPENCL_FP16) applyTestTag(tag_target_skip, SKIP_TAGS)
#define SKIP_MYRIAD if (target == DNN_TARGET_MYRIAD) applyTestTag(tag_target_skip, SKIP_TAGS)

std::string tag_target_skip =
    (target == DNN_TARGET_CPU) ? CV_TEST_TAG_DNN_SKIP_IE_CPU :
    (target == DNN_TARGET_OPENCL) ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL :
    (target == DNN_TARGET_OPENCL_FP16) ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16 :
    (target == DNN_TARGET_MYRIAD) ? CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X :
    "";

ASSERT_FALSE(name.empty());

#define EOF_LABEL exit_filter_opencv
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

#if INF_ENGINE_VER_MAJOR_EQ(2021040000) || INF_ENGINE_VER_MAJOR_EQ(2022010000)
#define SKIP_SET_1 1
#else
#define SKIP_SET_1 0
#endif

// Update note: execute <opencv_extra>/testdata/dnn/onnx/generate_conformance_list.py
BEGIN_SWITCH()
CASE(test_abs)
    // no filter
CASE(test_acos)
    // no filter
CASE(test_acos_example)
    // no filter
CASE(test_acosh)
    // no filter
CASE(test_acosh_example)
    // no filter
CASE(test_adagrad)
    // no filter
CASE(test_adagrad_multiple)
    // no filter
CASE(test_adam)
    // no filter
CASE(test_adam_multiple)
    // no filter
CASE(test_add)
    // no filter
CASE(test_add_bcast)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_add_uint8)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_and2d)
    // no filter
CASE(test_and3d)
    // no filter
CASE(test_and4d)
    // no filter
CASE(test_and_bcast3v1d)
    // no filter
CASE(test_and_bcast3v2d)
    // no filter
CASE(test_and_bcast4v2d)
    // no filter
CASE(test_and_bcast4v3d)
    // no filter
CASE(test_and_bcast4v4d)
    // no filter
CASE(test_argmax_default_axis_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_default_axis_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_default_axis_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_default_axis_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_negative_axis_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_negative_axis_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_negative_axis_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_negative_axis_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_no_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_no_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_no_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmax_no_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_default_axis_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_default_axis_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_default_axis_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_default_axis_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_negative_axis_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_negative_axis_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_negative_axis_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_negative_axis_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_no_keepdims_example)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_no_keepdims_example_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_no_keepdims_random)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_argmin_no_keepdims_random_select_last_index)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_asin)
    // no filter
CASE(test_asin_example)
    // no filter
CASE(test_asinh)
    // no filter
CASE(test_asinh_example)
    // no filter
CASE(test_atan)
    // no filter
CASE(test_atan_example)
    // no filter
CASE(test_atanh)
    // no filter
CASE(test_atanh_example)
    // no filter
CASE(test_averagepool_1d_default)
    // no filter
CASE(test_averagepool_2d_ceil)
    // no filter
CASE(test_averagepool_2d_default)
    // no filter
CASE(test_averagepool_2d_pads)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_averagepool_2d_pads_count_include_pad)
#if SKIP_SET_1
    SKIP_CPU;
    // MYRIAD is ok
    SKIP_OPENCL;
    SKIP_OPENCL_FP16;
#endif
CASE(test_averagepool_2d_precomputed_pads)
    // no filter
CASE(test_averagepool_2d_precomputed_pads_count_include_pad)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_averagepool_2d_precomputed_same_upper)
    // no filter
CASE(test_averagepool_2d_precomputed_strides)
    // no filter
CASE(test_averagepool_2d_same_lower)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_averagepool_2d_same_upper)
    // no filter
CASE(test_averagepool_2d_strides)
    // no filter
CASE(test_averagepool_3d_default)
    // no filter
CASE(test_basic_conv_with_padding)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_basic_conv_without_padding)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_basic_convinteger)
    // no filter
CASE(test_batchnorm_epsilon)
    // no filter
CASE(test_batchnorm_epsilon_training_mode)
    // no filter
CASE(test_batchnorm_example)
    // no filter
CASE(test_batchnorm_example_training_mode)
    // no filter
CASE(test_bernoulli)
    // no filter
CASE(test_bernoulli_double)
    // no filter
CASE(test_bernoulli_double_expanded)
    // no filter
CASE(test_bernoulli_expanded)
    // no filter
CASE(test_bernoulli_seed)
    // no filter
CASE(test_bernoulli_seed_expanded)
    // no filter
CASE(test_bitshift_left_uint16)
    // no filter
CASE(test_bitshift_left_uint32)
    // no filter
CASE(test_bitshift_left_uint64)
    // no filter
CASE(test_bitshift_left_uint8)
    // no filter
CASE(test_bitshift_right_uint16)
    // no filter
CASE(test_bitshift_right_uint32)
    // no filter
CASE(test_bitshift_right_uint64)
    // no filter
CASE(test_bitshift_right_uint8)
    // no filter
CASE(test_cast_BFLOAT16_to_FLOAT)
    // no filter
CASE(test_cast_DOUBLE_to_FLOAT)
    // no filter
CASE(test_cast_DOUBLE_to_FLOAT16)
    // no filter
CASE(test_cast_FLOAT16_to_DOUBLE)
    // no filter
CASE(test_cast_FLOAT16_to_FLOAT)
    // no filter
CASE(test_cast_FLOAT_to_BFLOAT16)
    // no filter
CASE(test_cast_FLOAT_to_DOUBLE)
    // no filter
CASE(test_cast_FLOAT_to_FLOAT16)
    // no filter
CASE(test_cast_FLOAT_to_STRING)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_cast_STRING_to_FLOAT)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_castlike_BFLOAT16_to_FLOAT)
    // no filter
CASE(test_castlike_BFLOAT16_to_FLOAT_expanded)
    // no filter
CASE(test_castlike_DOUBLE_to_FLOAT)
    // no filter
CASE(test_castlike_DOUBLE_to_FLOAT16)
    // no filter
CASE(test_castlike_DOUBLE_to_FLOAT16_expanded)
    // no filter
CASE(test_castlike_DOUBLE_to_FLOAT_expanded)
    // no filter
CASE(test_castlike_FLOAT16_to_DOUBLE)
    // no filter
CASE(test_castlike_FLOAT16_to_DOUBLE_expanded)
    // no filter
CASE(test_castlike_FLOAT16_to_FLOAT)
    // no filter
CASE(test_castlike_FLOAT16_to_FLOAT_expanded)
    // no filter
CASE(test_castlike_FLOAT_to_BFLOAT16)
    // no filter
CASE(test_castlike_FLOAT_to_BFLOAT16_expanded)
    // no filter
CASE(test_castlike_FLOAT_to_DOUBLE)
    // no filter
CASE(test_castlike_FLOAT_to_DOUBLE_expanded)
    // no filter
CASE(test_castlike_FLOAT_to_FLOAT16)
    // no filter
CASE(test_castlike_FLOAT_to_FLOAT16_expanded)
    // no filter
CASE(test_castlike_FLOAT_to_STRING)
    // no filter
CASE(test_castlike_FLOAT_to_STRING_expanded)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_castlike_STRING_to_FLOAT)
    // no filter
CASE(test_castlike_STRING_to_FLOAT_expanded)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_ceil)
    // no filter
CASE(test_ceil_example)
    // no filter
CASE(test_celu)
    // no filter
CASE(test_celu_expanded)
    // no filter
CASE(test_clip)
    // no filter
CASE(test_clip_default_inbounds)
    // no filter
CASE(test_clip_default_int8_inbounds)
    // no filter
CASE(test_clip_default_int8_max)
    // no filter
CASE(test_clip_default_int8_min)
    // no filter
CASE(test_clip_default_max)
    // no filter
CASE(test_clip_default_min)
    // no filter
CASE(test_clip_example)
    // no filter
CASE(test_clip_inbounds)
    // no filter
CASE(test_clip_outbounds)
    // no filter
CASE(test_clip_splitbounds)
    // no filter
CASE(test_compress_0)
    // no filter
CASE(test_compress_1)
    // no filter
CASE(test_compress_default_axis)
    // no filter
CASE(test_compress_negative_axis)
    // no filter
CASE(test_concat_1d_axis_0)
    // no filter
CASE(test_concat_1d_axis_negative_1)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_concat_2d_axis_0)
    // no filter
CASE(test_concat_2d_axis_1)
    // no filter
CASE(test_concat_2d_axis_negative_1)
    // no filter
CASE(test_concat_2d_axis_negative_2)
    // no filter
CASE(test_concat_3d_axis_0)
    // no filter
CASE(test_concat_3d_axis_1)
    // no filter
CASE(test_concat_3d_axis_2)
    // no filter
CASE(test_concat_3d_axis_negative_1)
    // no filter
CASE(test_concat_3d_axis_negative_2)
    // no filter
CASE(test_concat_3d_axis_negative_3)
    // no filter
CASE(test_constant)
    // no filter
CASE(test_constant_pad)
    // no filter
CASE(test_constantofshape_float_ones)
    // no filter
CASE(test_constantofshape_int_shape_zero)
    // no filter
CASE(test_constantofshape_int_zeros)
    // no filter
CASE(test_conv_with_autopad_same)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_conv_with_strides_and_asymmetric_padding)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_conv_with_strides_no_padding)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_conv_with_strides_padding)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_convinteger_with_padding)
    // no filter
CASE(test_convinteger_without_padding)
    // no filter
CASE(test_convtranspose)
    // no filter
CASE(test_convtranspose_1d)
    // no filter
CASE(test_convtranspose_3d)
    // no filter
CASE(test_convtranspose_autopad_same)
    // no filter
CASE(test_convtranspose_dilations)
    // no filter
CASE(test_convtranspose_kernel_shape)
    // no filter
CASE(test_convtranspose_output_shape)
    // no filter
CASE(test_convtranspose_pad)
    // no filter
CASE(test_convtranspose_pads)
    // no filter
CASE(test_convtranspose_with_kernel)
    // no filter
CASE(test_cos)
    // no filter
CASE(test_cos_example)
    // no filter
CASE(test_cosh)
    // no filter
CASE(test_cosh_example)
    // no filter
CASE(test_cumsum_1d)
    // no filter
CASE(test_cumsum_1d_exclusive)
    // no filter
CASE(test_cumsum_1d_reverse)
    // no filter
CASE(test_cumsum_1d_reverse_exclusive)
    // no filter
CASE(test_cumsum_2d_axis_0)
    // no filter
CASE(test_cumsum_2d_axis_1)
    // no filter
CASE(test_cumsum_2d_negative_axis)
    // no filter
CASE(test_depthtospace_crd_mode)
    // no filter
CASE(test_depthtospace_crd_mode_example)
    // no filter
CASE(test_depthtospace_dcr_mode)
    // no filter
CASE(test_depthtospace_example)
    // no filter
CASE(test_dequantizelinear)
    SKIP;
CASE(test_dequantizelinear_axis)
    SKIP;
CASE(test_dequantizelinear_blocked)
    SKIP;
CASE(test_det_2d)
    // no filter
CASE(test_det_nd)
    // no filter
CASE(test_div)
    // no filter
CASE(test_div_bcast)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_div_example)
    // no filter
CASE(test_div_uint8)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_dropout_default)
    // no filter
CASE(test_dropout_default_mask)
    // no filter
CASE(test_dropout_default_mask_ratio)
    // no filter
CASE(test_dropout_default_old)
    // no filter
CASE(test_dropout_default_ratio)
    // no filter
CASE(test_dropout_random_old)
    // no filter
CASE(test_dynamicquantizelinear)
    // no filter
CASE(test_dynamicquantizelinear_expanded)
    // no filter
CASE(test_dynamicquantizelinear_max_adjusted)
    // no filter
CASE(test_dynamicquantizelinear_max_adjusted_expanded)
    // no filter
CASE(test_dynamicquantizelinear_min_adjusted)
    // no filter
CASE(test_dynamicquantizelinear_min_adjusted_expanded)
    // no filter
CASE(test_edge_pad)
    // no filter
CASE(test_einsum_batch_diagonal)
    SKIP;
CASE(test_einsum_batch_matmul)
    // no filter
CASE(test_einsum_inner_prod)
    // no filter
CASE(test_einsum_sum)
    // no filter
CASE(test_einsum_transpose)
    // no filter
CASE(test_elu)
    // no filter
CASE(test_elu_default)
    // no filter
CASE(test_elu_example)
    // no filter
CASE(test_equal)
    // no filter
CASE(test_equal_bcast)
    // no filter
CASE(test_erf)
    // no filter
CASE(test_exp)
    // no filter
CASE(test_exp_example)
    // no filter
CASE(test_expand_dim_changed)
    // no filter
CASE(test_expand_dim_unchanged)
    // no filter
CASE(test_eyelike_populate_off_main_diagonal)
    // no filter
CASE(test_eyelike_with_dtype)
    // no filter
CASE(test_eyelike_without_dtype)
    // no filter
CASE(test_flatten_axis0)
    // no filter
CASE(test_flatten_axis1)
    // no filter
CASE(test_flatten_axis2)
    // no filter
CASE(test_flatten_axis3)
    // no filter
CASE(test_flatten_default_axis)
    // no filter
CASE(test_flatten_negative_axis1)
    // no filter
CASE(test_flatten_negative_axis2)
    // no filter
CASE(test_flatten_negative_axis3)
    // no filter
CASE(test_flatten_negative_axis4)
    // no filter
CASE(test_floor)
    // no filter
CASE(test_floor_example)
    // no filter
CASE(test_gather_0)
    // no filter
CASE(test_gather_1)
    // no filter
CASE(test_gather_2d_indices)
    // no filter
CASE(test_gather_elements_0)
    // no filter
CASE(test_gather_elements_1)
    // no filter
CASE(test_gather_elements_negative_indices)
    // no filter
CASE(test_gather_negative_indices)
    // no filter
CASE(test_gathernd_example_float32)
    // no filter
CASE(test_gathernd_example_int32)
    // no filter
CASE(test_gathernd_example_int32_batch_dim1)
    // no filter
CASE(test_gemm_all_attributes)
    // no filter
CASE(test_gemm_alpha)
    // no filter
CASE(test_gemm_beta)
    // no filter
CASE(test_gemm_default_matrix_bias)
    // no filter
CASE(test_gemm_default_no_bias)
    // no filter
CASE(test_gemm_default_scalar_bias)
    // no filter
CASE(test_gemm_default_single_elem_vector_bias)
    // no filter
CASE(test_gemm_default_vector_bias)
    // no filter
CASE(test_gemm_default_zero_bias)
    // no filter
CASE(test_gemm_transposeA)
    // no filter
CASE(test_gemm_transposeB)
    // no filter
CASE(test_globalaveragepool)
    // no filter
CASE(test_globalaveragepool_precomputed)
    // no filter
CASE(test_globalmaxpool)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_globalmaxpool_precomputed)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_greater)
    // no filter
CASE(test_greater_bcast)
    // no filter
CASE(test_greater_equal)
    // no filter
CASE(test_greater_equal_bcast)
    // no filter
CASE(test_greater_equal_bcast_expanded)
    // no filter
CASE(test_greater_equal_expanded)
    // no filter
CASE(test_gridsample)
    // no filter
CASE(test_gridsample_aligncorners_true)
    // no filter
CASE(test_gridsample_bicubic)
    // no filter
CASE(test_gridsample_bilinear)
    // no filter
CASE(test_gridsample_border_padding)
    // no filter
CASE(test_gridsample_nearest)
    // no filter
CASE(test_gridsample_reflection_padding)
    // no filter
CASE(test_gridsample_zeros_padding)
    // no filter
CASE(test_group_normalization_epsilon)
    // no filter
CASE(test_group_normalization_example)
    // no filter
CASE(test_gru_batchwise)
    // no filter
CASE(test_gru_defaults)
    // no filter
CASE(test_gru_seq_length)
    // no filter
CASE(test_gru_with_initial_bias)
    // no filter
CASE(test_hardmax_axis_0)
    // no filter
CASE(test_hardmax_axis_1)
    // no filter
CASE(test_hardmax_axis_2)
    // no filter
CASE(test_hardmax_default_axis)
    // no filter
CASE(test_hardmax_example)
    // no filter
CASE(test_hardmax_negative_axis)
    // no filter
CASE(test_hardmax_one_hot)
    // no filter
CASE(test_hardsigmoid)
    // no filter
CASE(test_hardsigmoid_default)
    // no filter
CASE(test_hardsigmoid_example)
    // no filter
CASE(test_hardswish)
    // no filter
CASE(test_hardswish_expanded)
    // no filter
CASE(test_identity)
    // no filter
CASE(test_identity_opt)
    // no filter
CASE(test_identity_sequence)
    // no filter
CASE(test_if)
    // no filter
CASE(test_if_opt)
    // no filter
CASE(test_if_seq)
    // no filter
CASE(test_instancenorm_epsilon)
    // no filter
CASE(test_instancenorm_example)
    // no filter
CASE(test_isinf)
    // no filter
CASE(test_isinf_negative)
    // no filter
CASE(test_isinf_positive)
    // no filter
CASE(test_isnan)
    // no filter
CASE(test_layer_normalization_2d_axis0)
    // no filter
CASE(test_layer_normalization_2d_axis1)
    // no filter
CASE(test_layer_normalization_2d_axis_negative_1)
    // no filter
CASE(test_layer_normalization_2d_axis_negative_2)
    // no filter
CASE(test_layer_normalization_3d_axis0_epsilon)
    // no filter
CASE(test_layer_normalization_3d_axis1_epsilon)
    // no filter
CASE(test_layer_normalization_3d_axis2_epsilon)
    // no filter
CASE(test_layer_normalization_3d_axis_negative_1_epsilon)
    // no filter
CASE(test_layer_normalization_3d_axis_negative_2_epsilon)
    // no filter
CASE(test_layer_normalization_3d_axis_negative_3_epsilon)
    // no filter
CASE(test_layer_normalization_4d_axis0)
    // no filter
CASE(test_layer_normalization_4d_axis1)
    // no filter
CASE(test_layer_normalization_4d_axis2)
    // no filter
CASE(test_layer_normalization_4d_axis3)
    // no filter
CASE(test_layer_normalization_4d_axis_negative_1)
    // no filter
CASE(test_layer_normalization_4d_axis_negative_2)
    // no filter
CASE(test_layer_normalization_4d_axis_negative_3)
    // no filter
CASE(test_layer_normalization_4d_axis_negative_4)
    // no filter
CASE(test_layer_normalization_default_axis)
    // no filter
CASE(test_leakyrelu)
    // no filter
CASE(test_leakyrelu_default)
    // no filter
CASE(test_leakyrelu_example)
    // no filter
CASE(test_less)
    // no filter
CASE(test_less_bcast)
    // no filter
CASE(test_less_equal)
    // no filter
CASE(test_less_equal_bcast)
    // no filter
CASE(test_less_equal_bcast_expanded)
    // no filter
CASE(test_less_equal_expanded)
    // no filter
CASE(test_log)
    // no filter
CASE(test_log_example)
    // no filter
CASE(test_logsoftmax_axis_0)
#if SKIP_SET_1
    SKIP_OPENCL;
    SKIP_OPENCL_FP16;
#endif
CASE(test_logsoftmax_axis_0_expanded)
#if SKIP_SET_1
    SKIP_OPENCL;
    SKIP_OPENCL_FP16;
#endif
CASE(test_logsoftmax_axis_1)
    // no filter
CASE(test_logsoftmax_axis_1_expanded)
    // no filter
CASE(test_logsoftmax_axis_2)
    // no filter
CASE(test_logsoftmax_axis_2_expanded)
    // no filter
CASE(test_logsoftmax_default_axis)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_logsoftmax_default_axis_expanded)
    // no filter
CASE(test_logsoftmax_example_1)
    // no filter
CASE(test_logsoftmax_example_1_expanded)
    // no filter
CASE(test_logsoftmax_large_number)
#if SKIP_SET_1
    SKIP_OPENCL_FP16;
    SKIP_MYRIAD;
#endif
CASE(test_logsoftmax_large_number_expanded)
#if SKIP_SET_1
    SKIP_OPENCL_FP16;
    SKIP_MYRIAD;
#endif
CASE(test_logsoftmax_negative_axis)
    // no filter
CASE(test_logsoftmax_negative_axis_expanded)
    // no filter
CASE(test_loop11)
    // no filter
CASE(test_loop13_seq)
    // no filter
CASE(test_loop16_seq_none)
    // no filter
CASE(test_lrn)
    // no filter
CASE(test_lrn_default)
    // no filter
CASE(test_lstm_batchwise)
    // no filter
CASE(test_lstm_defaults)
    // no filter
CASE(test_lstm_with_initial_bias)
    // no filter
CASE(test_lstm_with_peepholes)
    // no filter
CASE(test_matmul_2d)
    // no filter
CASE(test_matmul_3d)
    // no filter
CASE(test_matmul_4d)
    // no filter
CASE(test_matmulinteger)
    // no filter
CASE(test_max_example)
    // no filter
CASE(test_max_float16)
    // no filter
CASE(test_max_float32)
    // no filter
CASE(test_max_float64)
    // no filter
CASE(test_max_int16)
    // no filter
CASE(test_max_int32)
    // no filter
CASE(test_max_int64)
    // no filter
CASE(test_max_int8)
    // no filter
CASE(test_max_one_input)
    // no filter
CASE(test_max_two_inputs)
    // no filter
CASE(test_max_uint16)
    // no filter
CASE(test_max_uint32)
    // no filter
CASE(test_max_uint64)
    // no filter
CASE(test_max_uint8)
    // no filter
CASE(test_maxpool_1d_default)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_ceil)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_default)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_dilations)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxpool_2d_pads)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_precomputed_pads)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_precomputed_same_upper)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_precomputed_strides)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_same_lower)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxpool_2d_same_upper)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_strides)
#if SKIP_SET_1
    SKIP_MYRIAD;
#endif
CASE(test_maxpool_2d_uint8)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxpool_3d_default)
#if SKIP_SET_1
    SKIP_NON_CPU;
#endif
CASE(test_maxpool_with_argmax_2d_precomputed_pads)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxpool_with_argmax_2d_precomputed_strides)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxunpool_export_with_output_shape)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_maxunpool_export_without_output_shape)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_mean_example)
    // no filter
CASE(test_mean_one_input)
    // no filter
CASE(test_mean_two_inputs)
    // no filter
CASE(test_min_example)
    // no filter
CASE(test_min_float16)
    // no filter
CASE(test_min_float32)
    // no filter
CASE(test_min_float64)
    // no filter
CASE(test_min_int16)
    // no filter
CASE(test_min_int32)
    // no filter
CASE(test_min_int64)
    // no filter
CASE(test_min_int8)
    // no filter
CASE(test_min_one_input)
    // no filter
CASE(test_min_two_inputs)
    // no filter
CASE(test_min_uint16)
    // no filter
CASE(test_min_uint32)
    // no filter
CASE(test_min_uint64)
    // no filter
CASE(test_min_uint8)
    // no filter
CASE(test_mod_broadcast)
    // no filter
CASE(test_mod_int64_fmod)
    // no filter
CASE(test_mod_mixed_sign_float16)
    // no filter
    if (target == DNN_TARGET_OPENCL)
    {
        default_l1 = 0.0011;  // Expected: (normL1) <= (l1), actual: 0.00104141 vs 1e-05
        default_lInf = 0.0016;  // Expected: (normInf) <= (lInf), actual: 0.00156212 vs 0.0001
    }
CASE(test_mod_mixed_sign_float32)
    // no filter
    if (target == DNN_TARGET_OPENCL)
    {
        default_l1 = 0.0011;  // Expected: (normL1) <= (l1), actual: 0.00104141 vs 1e-05
        default_lInf = 0.0016;  // Expected: (normInf) <= (lInf), actual: 0.00156212 vs 0.0001
    }
CASE(test_mod_mixed_sign_float64)
    // no filter
    if (target == DNN_TARGET_OPENCL)
    {
        default_l1 = 0.0011;  // Expected: (normL1) <= (l1), actual: 0.00104167 vs 1e-05
        default_lInf = 0.0016;  // Expected: (normInf) <= (lInf), actual: 0.00156251 vs 0.0001
    }
CASE(test_mod_mixed_sign_int16)
    // no filter
CASE(test_mod_mixed_sign_int32)
    // no filter
CASE(test_mod_mixed_sign_int64)
    // no filter
CASE(test_mod_mixed_sign_int8)
    // no filter
CASE(test_mod_uint16)
    // no filter
CASE(test_mod_uint32)
    // no filter
CASE(test_mod_uint64)
    // no filter
CASE(test_mod_uint8)
    // no filter
CASE(test_momentum)
    // no filter
CASE(test_momentum_multiple)
    // no filter
CASE(test_mul)
    // no filter
CASE(test_mul_bcast)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_mul_example)
    // no filter
CASE(test_mul_uint8)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_mvn)
    // no filter
CASE(test_mvn_expanded)
    // no filter
CASE(test_neg)
    // no filter
CASE(test_neg_example)
    // no filter
CASE(test_nesterov_momentum)
    // no filter
CASE(test_nllloss_NC)
    // no filter
CASE(test_nllloss_NC_expanded)
    // no filter
CASE(test_nllloss_NCd1)
    // no filter
CASE(test_nllloss_NCd1_expanded)
    // no filter
CASE(test_nllloss_NCd1_ii)
    // no filter
CASE(test_nllloss_NCd1_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1_mean_weight_negative_ii)
    // no filter
CASE(test_nllloss_NCd1_mean_weight_negative_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1_weight)
    // no filter
CASE(test_nllloss_NCd1_weight_expanded)
    // no filter
CASE(test_nllloss_NCd1_weight_ii)
    // no filter
CASE(test_nllloss_NCd1_weight_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1d2)
    // no filter
CASE(test_nllloss_NCd1d2_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_no_weight_reduction_mean_ii)
    // no filter
CASE(test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_reduction_mean)
    // no filter
CASE(test_nllloss_NCd1d2_reduction_mean_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_reduction_sum)
    // no filter
CASE(test_nllloss_NCd1d2_reduction_sum_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_mean)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_mean_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_expanded)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_ii)
    // no filter
CASE(test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1d2d3_none_no_weight_negative_ii)
    // no filter
CASE(test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1d2d3_sum_weight_high_ii)
    // no filter
CASE(test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded)
    // no filter
CASE(test_nllloss_NCd1d2d3d4d5_mean_weight)
    // no filter
CASE(test_nllloss_NCd1d2d3d4d5_mean_weight_expanded)
    // no filter
CASE(test_nllloss_NCd1d2d3d4d5_none_no_weight)
    // no filter
CASE(test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded)
    // no filter
CASE(test_nonmaxsuppression_center_point_box_format)
    // no filter
CASE(test_nonmaxsuppression_flipped_coordinates)
    // no filter
CASE(test_nonmaxsuppression_identical_boxes)
    // no filter
CASE(test_nonmaxsuppression_limit_output_size)
    // no filter
CASE(test_nonmaxsuppression_single_box)
    // no filter
CASE(test_nonmaxsuppression_suppress_by_IOU)
    // no filter
CASE(test_nonmaxsuppression_suppress_by_IOU_and_scores)
    // no filter
CASE(test_nonmaxsuppression_two_batches)
    // no filter
CASE(test_nonmaxsuppression_two_classes)
    // no filter
CASE(test_nonzero_example)
    // no filter
CASE(test_not_2d)
    // no filter
CASE(test_not_3d)
    // no filter
CASE(test_not_4d)
    // no filter
CASE(test_onehot_negative_indices)
    // no filter
CASE(test_onehot_with_axis)
    // no filter
CASE(test_onehot_with_negative_axis)
    // no filter
CASE(test_onehot_without_axis)
    // no filter
CASE(test_optional_get_element)
    // no filter
CASE(test_optional_get_element_sequence)
    // no filter
CASE(test_optional_has_element)
    // no filter
CASE(test_optional_has_element_empty)
    // no filter
CASE(test_or2d)
    // no filter
CASE(test_or3d)
    // no filter
CASE(test_or4d)
    // no filter
CASE(test_or_bcast3v1d)
    // no filter
CASE(test_or_bcast3v2d)
    // no filter
CASE(test_or_bcast4v2d)
    // no filter
CASE(test_or_bcast4v3d)
    // no filter
CASE(test_or_bcast4v4d)
    // no filter
CASE(test_pow)
    // no filter
CASE(test_pow_bcast_array)
    // no filter
CASE(test_pow_bcast_scalar)
    // no filter
CASE(test_pow_example)
    // no filter
CASE(test_pow_types_float)
    // no filter
CASE(test_pow_types_float32_int32)
    // no filter
CASE(test_pow_types_float32_int64)
    // no filter
CASE(test_pow_types_float32_uint32)
    // no filter
CASE(test_pow_types_float32_uint64)
    // no filter
CASE(test_pow_types_int)
    // no filter
CASE(test_pow_types_int32_float32)
    // no filter
CASE(test_pow_types_int32_int32)
    // no filter
CASE(test_pow_types_int64_float32)
    // no filter
CASE(test_pow_types_int64_int64)
    // no filter
CASE(test_prelu_broadcast)
    // no filter
CASE(test_prelu_example)
    // no filter
CASE(test_qlinearconv)
    // no filter
CASE(test_qlinearmatmul_2D)
    // no filter
CASE(test_qlinearmatmul_3D)
    // no filter
CASE(test_quantizelinear)
    SKIP;
CASE(test_quantizelinear_axis)
    SKIP;
CASE(test_quantizelinear_blocked)
    SKIP;
CASE(test_range_float_type_positive_delta)
    // no filter
CASE(test_range_float_type_positive_delta_expanded)
    // no filter
CASE(test_range_int32_type_negative_delta)
    // no filter
CASE(test_range_int32_type_negative_delta_expanded)
    // no filter
CASE(test_reciprocal)
    // no filter
CASE(test_reciprocal_example)
    // no filter
CASE(test_reduce_l1_default_axes_keepdims_example)
    // no filter
CASE(test_reduce_l1_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_l1_do_not_keepdims_example)
    // no filter
CASE(test_reduce_l1_do_not_keepdims_random)
    // no filter
CASE(test_reduce_l1_keep_dims_example)
    // no filter
CASE(test_reduce_l1_keep_dims_random)
    // no filter
CASE(test_reduce_l1_negative_axes_keep_dims_example)
    // no filter
CASE(test_reduce_l1_negative_axes_keep_dims_random)
    // no filter
CASE(test_reduce_l2_default_axes_keepdims_example)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00490189 vs 0.004)
#endif
CASE(test_reduce_l2_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_l2_do_not_keepdims_example)
    // no filter
CASE(test_reduce_l2_do_not_keepdims_random)
    // no filter
CASE(test_reduce_l2_keep_dims_example)
    // no filter
CASE(test_reduce_l2_keep_dims_random)
    // no filter
CASE(test_reduce_l2_negative_axes_keep_dims_example)
    // no filter
CASE(test_reduce_l2_negative_axes_keep_dims_random)
    // no filter
CASE(test_reduce_log_sum)
    // no filter
CASE(test_reduce_log_sum_asc_axes)
    // no filter
CASE(test_reduce_log_sum_default)
    // no filter
CASE(test_reduce_log_sum_desc_axes)
    // no filter
CASE(test_reduce_log_sum_exp_default_axes_keepdims_example)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00671387 vs 0.004
#endif
CASE(test_reduce_log_sum_exp_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_log_sum_exp_do_not_keepdims_example)
    // no filter
CASE(test_reduce_log_sum_exp_do_not_keepdims_random)
    // no filter
CASE(test_reduce_log_sum_exp_keepdims_example)
    // no filter
CASE(test_reduce_log_sum_exp_keepdims_random)
    // no filter
CASE(test_reduce_log_sum_exp_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_log_sum_exp_negative_axes_keepdims_random)
    // no filter
CASE(test_reduce_log_sum_negative_axes)
    // no filter
CASE(test_reduce_max_default_axes_keepdim_example)
    // no filter
CASE(test_reduce_max_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_max_do_not_keepdims_example)
    // no filter
CASE(test_reduce_max_do_not_keepdims_random)
    // no filter
CASE(test_reduce_max_keepdims_example)
    // no filter
CASE(test_reduce_max_keepdims_random)
    // no filter
CASE(test_reduce_max_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_max_negative_axes_keepdims_random)
    // no filter
CASE(test_reduce_mean_default_axes_keepdims_example)
    // no filter
CASE(test_reduce_mean_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_mean_do_not_keepdims_example)
    // no filter
CASE(test_reduce_mean_do_not_keepdims_random)
    // no filter
CASE(test_reduce_mean_keepdims_example)
    // no filter
CASE(test_reduce_mean_keepdims_random)
    // no filter
CASE(test_reduce_mean_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_mean_negative_axes_keepdims_random)
    // no filter
CASE(test_reduce_min_default_axes_keepdims_example)
    // no filter
CASE(test_reduce_min_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_min_do_not_keepdims_example)
    // no filter
CASE(test_reduce_min_do_not_keepdims_random)
    // no filter
CASE(test_reduce_min_keepdims_example)
    // no filter
CASE(test_reduce_min_keepdims_random)
    // no filter
CASE(test_reduce_min_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_min_negative_axes_keepdims_random)
    // no filter
CASE(test_reduce_prod_default_axes_keepdims_example)
#if SKIP_SET_1
    SKIP_MYRIAD;  // accuracy (Expected: (normL1) <= (l1), actual: inf vs 0.004)
#endif
CASE(test_reduce_prod_default_axes_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 5;  // Expected: (normL1) <= (l1), actual: 2.66211 vs 0.004  |ref| = 24621.337890625
        default_lInf = 5;  // Expected: (normInf) <= (lInf), actual: 2.66211 vs 0.02  |ref| = 24621.337890625
    }
#endif
CASE(test_reduce_prod_do_not_keepdims_example)
    // no filter
CASE(test_reduce_prod_do_not_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
    }
#endif
CASE(test_reduce_prod_keepdims_example)
    // no filter
CASE(test_reduce_prod_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
    }
#endif
#if INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
    }
#endif
CASE(test_reduce_prod_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_prod_negative_axes_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
    }
#endif
#if INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
    }
#endif
CASE(test_reduce_sum_default_axes_keepdims_example)
    // no filter
CASE(test_reduce_sum_default_axes_keepdims_random)
    // no filter
CASE(test_reduce_sum_do_not_keepdims_example)
    // no filter
CASE(test_reduce_sum_do_not_keepdims_random)
    // no filter
CASE(test_reduce_sum_empty_axes_input_noop_example)
    // no filter
CASE(test_reduce_sum_empty_axes_input_noop_random)
    // no filter
CASE(test_reduce_sum_keepdims_example)
    // no filter
CASE(test_reduce_sum_keepdims_random)
    // no filter
CASE(test_reduce_sum_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_sum_negative_axes_keepdims_random)
    // no filter
CASE(test_reduce_sum_square_default_axes_keepdims_example)
    // no filter
CASE(test_reduce_sum_square_default_axes_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.0183411 vs 0.004
#endif
CASE(test_reduce_sum_square_do_not_keepdims_example)
    // no filter
CASE(test_reduce_sum_square_do_not_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
    }
#endif
#if INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        default_l1 = 0.01f;  // Expected: (normL1) <= (l1), actual: 0.00723048 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0201416 vs 0.02
    }
#endif
CASE(test_reduce_sum_square_keepdims_example)
    // no filter
CASE(test_reduce_sum_square_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
    }
#endif
#if INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
    }
#endif
CASE(test_reduce_sum_square_negative_axes_keepdims_example)
    // no filter
CASE(test_reduce_sum_square_negative_axes_keepdims_random)
#if SKIP_SET_1
    if (target == DNN_TARGET_MYRIAD)
    {
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
    }
#endif
#if INF_ENGINE_VER_MAJOR_EQ(2022010000)
    if (target == DNN_TARGET_OPENCL_FP16)
    {
        default_l1 = 0.05f;  // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004
        default_lInf = 0.05f;  // Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
    }
#endif
CASE(test_reflect_pad)
    // no filter
CASE(test_relu)
    // no filter
CASE(test_reshape_allowzero_reordered)
    // no filter
CASE(test_reshape_extended_dims)
    // no filter
CASE(test_reshape_negative_dim)
    // no filter
CASE(test_reshape_negative_extended_dims)
    // no filter
CASE(test_reshape_one_dim)
    // no filter
CASE(test_reshape_reduced_dims)
    // no filter
CASE(test_reshape_reordered_all_dims)
    // no filter
CASE(test_reshape_reordered_last_dims)
    // no filter
CASE(test_reshape_zero_and_negative_dim)
    // no filter
CASE(test_reshape_zero_dim)
    // no filter
CASE(test_resize_downsample_scales_cubic)
    // no filter
CASE(test_resize_downsample_scales_cubic_A_n0p5_exclude_outside)
    // no filter
CASE(test_resize_downsample_scales_cubic_align_corners)
    // no filter
CASE(test_resize_downsample_scales_linear)
    // no filter
CASE(test_resize_downsample_scales_linear_align_corners)
    // no filter
CASE(test_resize_downsample_scales_nearest)
    // no filter
CASE(test_resize_downsample_sizes_cubic)
    // no filter
CASE(test_resize_downsample_sizes_linear_pytorch_half_pixel)
    // no filter
CASE(test_resize_downsample_sizes_nearest)
    // no filter
CASE(test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn)
    // no filter
CASE(test_resize_tf_crop_and_resize)
    // no filter
CASE(test_resize_upsample_scales_cubic)
    // no filter
CASE(test_resize_upsample_scales_cubic_A_n0p5_exclude_outside)
    // no filter
CASE(test_resize_upsample_scales_cubic_align_corners)
    // no filter
CASE(test_resize_upsample_scales_cubic_asymmetric)
    // no filter
CASE(test_resize_upsample_scales_linear)
    // no filter
CASE(test_resize_upsample_scales_linear_align_corners)
    // no filter
CASE(test_resize_upsample_scales_nearest)
    // no filter
CASE(test_resize_upsample_sizes_cubic)
    // no filter
CASE(test_resize_upsample_sizes_nearest)
    // no filter
CASE(test_resize_upsample_sizes_nearest_ceil_half_pixel)
    // no filter
CASE(test_resize_upsample_sizes_nearest_floor_align_corners)
    // no filter
CASE(test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric)
    // no filter
CASE(test_reversesequence_batch)
    // no filter
CASE(test_reversesequence_time)
    // no filter
CASE(test_rnn_seq_length)
    // no filter
CASE(test_roialign_aligned_false)
    // no filter
CASE(test_roialign_aligned_true)
    // no filter
CASE(test_round)
    // no filter
CASE(test_scan9_sum)
    // no filter
CASE(test_scan_sum)
    // no filter
CASE(test_scatter_elements_with_axis)
    // no filter
CASE(test_scatter_elements_with_duplicate_indices)
    // no filter
CASE(test_scatter_elements_with_negative_indices)
    // no filter
CASE(test_scatter_elements_with_reduction_max)
    // no filter
CASE(test_scatter_elements_with_reduction_min)
    // no filter
CASE(test_scatter_elements_without_axis)
    // no filter
CASE(test_scatter_with_axis)
    // no filter
CASE(test_scatter_without_axis)
    // no filter
CASE(test_scatternd)
    // no filter
CASE(test_scatternd_add)
    // no filter
CASE(test_scatternd_max)
    // no filter
CASE(test_scatternd_min)
    // no filter
CASE(test_scatternd_multiply)
    // no filter
CASE(test_sce_NCd1_mean_weight_negative_ii)
    // no filter
CASE(test_sce_NCd1_mean_weight_negative_ii_expanded)
    // no filter
CASE(test_sce_NCd1_mean_weight_negative_ii_log_prob)
    // no filter
CASE(test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded)
    // no filter
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii)
    // no filter
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded)
    // no filter
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob)
    // no filter
CASE(test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded)
    // no filter
CASE(test_sce_NCd1d2d3_sum_weight_high_ii)
    // no filter
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_expanded)
    // no filter
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_log_prob)
    // no filter
CASE(test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_mean_weight)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_mean_weight_expanded)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_mean_weight_log_prob)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_none_no_weight)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_expanded)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_log_prob)
    // no filter
CASE(test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded)
    // no filter
CASE(test_sce_mean)
    // no filter
CASE(test_sce_mean_3d)
    // no filter
CASE(test_sce_mean_3d_expanded)
    // no filter
CASE(test_sce_mean_3d_log_prob)
    // no filter
CASE(test_sce_mean_3d_log_prob_expanded)
    // no filter
CASE(test_sce_mean_expanded)
    // no filter
CASE(test_sce_mean_log_prob)
    // no filter
CASE(test_sce_mean_log_prob_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii)
    // no filter
CASE(test_sce_mean_no_weight_ii_3d)
    // no filter
CASE(test_sce_mean_no_weight_ii_3d_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii_3d_log_prob)
    // no filter
CASE(test_sce_mean_no_weight_ii_3d_log_prob_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii_4d)
    // no filter
CASE(test_sce_mean_no_weight_ii_4d_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii_4d_log_prob)
    // no filter
CASE(test_sce_mean_no_weight_ii_4d_log_prob_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii_expanded)
    // no filter
CASE(test_sce_mean_no_weight_ii_log_prob)
    // no filter
CASE(test_sce_mean_no_weight_ii_log_prob_expanded)
    // no filter
CASE(test_sce_mean_weight)
    // no filter
CASE(test_sce_mean_weight_expanded)
    // no filter
CASE(test_sce_mean_weight_ii)
    // no filter
CASE(test_sce_mean_weight_ii_3d)
    // no filter
CASE(test_sce_mean_weight_ii_3d_expanded)
    // no filter
CASE(test_sce_mean_weight_ii_3d_log_prob)
    // no filter
CASE(test_sce_mean_weight_ii_3d_log_prob_expanded)
    // no filter
CASE(test_sce_mean_weight_ii_4d)
    // no filter
CASE(test_sce_mean_weight_ii_4d_expanded)
    // no filter
CASE(test_sce_mean_weight_ii_4d_log_prob)
    // no filter
CASE(test_sce_mean_weight_ii_4d_log_prob_expanded)
    // no filter
CASE(test_sce_mean_weight_ii_expanded)
    // no filter
CASE(test_sce_mean_weight_ii_log_prob)
    // no filter
CASE(test_sce_mean_weight_ii_log_prob_expanded)
    // no filter
CASE(test_sce_mean_weight_log_prob)
    // no filter
CASE(test_sce_mean_weight_log_prob_expanded)
    // no filter
CASE(test_sce_none)
    // no filter
CASE(test_sce_none_expanded)
    // no filter
CASE(test_sce_none_log_prob)
    // no filter
CASE(test_sce_none_log_prob_expanded)
    // no filter
CASE(test_sce_none_weights)
    // no filter
CASE(test_sce_none_weights_expanded)
    // no filter
CASE(test_sce_none_weights_log_prob)
    // no filter
CASE(test_sce_none_weights_log_prob_expanded)
    // no filter
CASE(test_sce_sum)
    // no filter
CASE(test_sce_sum_expanded)
    // no filter
CASE(test_sce_sum_log_prob)
    // no filter
CASE(test_sce_sum_log_prob_expanded)
    // no filter
CASE(test_selu)
    // no filter
CASE(test_selu_default)
    // no filter
CASE(test_selu_example)
    // no filter
CASE(test_sequence_insert_at_back)
    // no filter
CASE(test_sequence_insert_at_front)
    // no filter
CASE(test_shape)
    // no filter
CASE(test_shape_clip_end)
    // no filter
CASE(test_shape_clip_start)
    // no filter
CASE(test_shape_end_1)
    // no filter
CASE(test_shape_end_negative_1)
    // no filter
CASE(test_shape_example)
    // no filter
CASE(test_shape_start_1)
    // no filter
CASE(test_shape_start_1_end_2)
    // no filter
CASE(test_shape_start_1_end_negative_1)
    // no filter
CASE(test_shape_start_negative_1)
    // no filter
CASE(test_shrink_hard)
    // no filter
CASE(test_shrink_soft)
    // no filter
CASE(test_sigmoid)
    // no filter
CASE(test_sigmoid_example)
    // no filter
CASE(test_sign)
    // no filter
CASE(test_simple_rnn_batchwise)
    // no filter
CASE(test_simple_rnn_defaults)
    // no filter
CASE(test_simple_rnn_with_initial_bias)
    // no filter
CASE(test_sin)
    // no filter
CASE(test_sin_example)
    // no filter
CASE(test_sinh)
    // no filter
CASE(test_sinh_example)
    // no filter
CASE(test_size)
    // no filter
CASE(test_size_example)
    // no filter
CASE(test_slice)
    // no filter
CASE(test_slice_default_axes)
    // no filter
CASE(test_slice_default_steps)
    // no filter
CASE(test_slice_end_out_of_bounds)
    // no filter
CASE(test_slice_neg)
    // no filter
CASE(test_slice_neg_steps)
    // no filter
CASE(test_slice_negative_axes)
    // no filter
CASE(test_slice_start_out_of_bounds)
    // no filter
CASE(test_softmax_axis_0)
#if SKIP_SET_1
    SKIP_OPENCL;
    SKIP_OPENCL_FP16;
#endif
CASE(test_softmax_axis_0_expanded)
#if SKIP_SET_1
    SKIP_OPENCL;
    SKIP_OPENCL_FP16;
#endif
CASE(test_softmax_axis_1)
    // no filter
CASE(test_softmax_axis_1_expanded)
    // no filter
CASE(test_softmax_axis_2)
    // no filter
CASE(test_softmax_axis_2_expanded)
    // no filter
CASE(test_softmax_default_axis)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_softmax_default_axis_expanded)
    // no filter
CASE(test_softmax_example)
    // no filter
CASE(test_softmax_example_expanded)
    // no filter
CASE(test_softmax_large_number)
#if SKIP_SET_1
    SKIP_OPENCL_FP16;
    SKIP_MYRIAD;
#endif
CASE(test_softmax_large_number_expanded)
#if SKIP_SET_1
    SKIP_OPENCL_FP16;
    SKIP_MYRIAD;
#endif
CASE(test_softmax_negative_axis)
    // no filter
CASE(test_softmax_negative_axis_expanded)
    // no filter
CASE(test_softplus)
    // no filter
CASE(test_softplus_example)
    // no filter
CASE(test_softsign)
    // no filter
CASE(test_softsign_example)
    // no filter
CASE(test_spacetodepth)
    // no filter
CASE(test_spacetodepth_example)
    // no filter
CASE(test_split_equal_parts_1d)
    // no filter
CASE(test_split_equal_parts_2d)
    // no filter
CASE(test_split_equal_parts_default_axis)
    // no filter
CASE(test_split_variable_parts_1d)
    // no filter
CASE(test_split_variable_parts_2d)
    // no filter
CASE(test_split_variable_parts_default_axis)
    // no filter
CASE(test_split_zero_size_splits)
    // no filter
CASE(test_sqrt)
    // no filter
CASE(test_sqrt_example)
    // no filter
CASE(test_squeeze)
    // no filter
CASE(test_squeeze_negative_axes)
    // no filter
CASE(test_strnormalizer_export_monday_casesensintive_lower)
    // no filter
CASE(test_strnormalizer_export_monday_casesensintive_nochangecase)
    // no filter
CASE(test_strnormalizer_export_monday_casesensintive_upper)
    // no filter
CASE(test_strnormalizer_export_monday_empty_output)
    // no filter
CASE(test_strnormalizer_export_monday_insensintive_upper_twodim)
    // no filter
CASE(test_strnormalizer_nostopwords_nochangecase)
    // no filter
CASE(test_sub)
    // no filter
CASE(test_sub_bcast)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_sub_example)
    // no filter
CASE(test_sub_uint8)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_sum_example)
    // no filter
CASE(test_sum_one_input)
    // no filter
CASE(test_sum_two_inputs)
    // no filter
CASE(test_tan)
    // no filter
CASE(test_tan_example)
    // no filter
CASE(test_tanh)
    // no filter
CASE(test_tanh_example)
    // no filter
CASE(test_tfidfvectorizer_tf_batch_onlybigrams_skip0)
    // no filter
CASE(test_tfidfvectorizer_tf_batch_onlybigrams_skip5)
    // no filter
CASE(test_tfidfvectorizer_tf_batch_uniandbigrams_skip5)
    // no filter
CASE(test_tfidfvectorizer_tf_only_bigrams_skip0)
    // no filter
CASE(test_tfidfvectorizer_tf_onlybigrams_levelempty)
    // no filter
CASE(test_tfidfvectorizer_tf_onlybigrams_skip5)
    // no filter
CASE(test_tfidfvectorizer_tf_uniandbigrams_skip5)
    // no filter
CASE(test_thresholdedrelu)
    // no filter
CASE(test_thresholdedrelu_default)
    // no filter
CASE(test_thresholdedrelu_example)
    // no filter
CASE(test_tile)
    // no filter
CASE(test_tile_precomputed)
    // no filter
CASE(test_top_k)
    // no filter
CASE(test_top_k_negative_axis)
    // no filter
CASE(test_top_k_smallest)
    // no filter
CASE(test_training_dropout)
    // no filter
CASE(test_training_dropout_default)
    // no filter
CASE(test_training_dropout_default_mask)
    // no filter
CASE(test_training_dropout_mask)
    // no filter
CASE(test_training_dropout_zero_ratio)
    // no filter
CASE(test_training_dropout_zero_ratio_mask)
    // no filter
CASE(test_transpose_all_permutations_0)
    // no filter
CASE(test_transpose_all_permutations_1)
    // no filter
CASE(test_transpose_all_permutations_2)
    // no filter
CASE(test_transpose_all_permutations_3)
    // no filter
CASE(test_transpose_all_permutations_4)
    // no filter
CASE(test_transpose_all_permutations_5)
    // no filter
CASE(test_transpose_default)
    // no filter
CASE(test_tril)
    // no filter
CASE(test_tril_neg)
    // no filter
CASE(test_tril_one_row_neg)
    // no filter
CASE(test_tril_out_neg)
    // no filter
CASE(test_tril_out_pos)
    // no filter
CASE(test_tril_pos)
    // no filter
CASE(test_tril_square)
    // no filter
CASE(test_tril_square_neg)
    // no filter
CASE(test_tril_zero)
    // no filter
CASE(test_triu)
    // no filter
CASE(test_triu_neg)
    // no filter
CASE(test_triu_one_row)
    // no filter
CASE(test_triu_out_neg_out)
    // no filter
CASE(test_triu_out_pos)
    // no filter
CASE(test_triu_pos)
    // no filter
CASE(test_triu_square)
    // no filter
CASE(test_triu_square_neg)
    // no filter
CASE(test_triu_zero)
    // no filter
CASE(test_unique_not_sorted_without_axis)
    // no filter
CASE(test_unique_sorted_with_axis)
    // no filter
CASE(test_unique_sorted_with_axis_3d)
    // no filter
CASE(test_unique_sorted_with_negative_axis)
    // no filter
CASE(test_unique_sorted_without_axis)
    // no filter
CASE(test_unsqueeze_axis_0)
    // no filter
CASE(test_unsqueeze_axis_1)
    // no filter
CASE(test_unsqueeze_axis_2)
    // no filter
CASE(test_unsqueeze_axis_3)
    // no filter
CASE(test_unsqueeze_negative_axes)
    // no filter
CASE(test_unsqueeze_three_axes)
    // no filter
CASE(test_unsqueeze_two_axes)
    // no filter
CASE(test_unsqueeze_unsorted_axes)
    // no filter
CASE(test_upsample_nearest)
#if SKIP_SET_1
    SKIP;
#endif
CASE(test_where_example)
    // no filter
CASE(test_where_long_example)
    // no filter
CASE(test_xor2d)
    // no filter
CASE(test_xor3d)
    // no filter
CASE(test_xor4d)
    // no filter
CASE(test_xor_bcast3v1d)
    // no filter
CASE(test_xor_bcast3v2d)
    // no filter
CASE(test_xor_bcast4v2d)
    // no filter
CASE(test_xor_bcast4v3d)
    // no filter
CASE(test_xor_bcast4v4d)
    // no filter
END_SWITCH()
#undef EOF_LABEL
#undef BEGIN_SWITCH
#undef CASE
#undef END_SWITCH
if (!filterApplied)
{
    ADD_FAILURE() << "OpenVINO backend: unknown test='" << name << "'. Update filter configuration";
}

#undef SKIP_TAGS
#undef SKIP_
#undef SKIP
#undef SKIP_CPU
#undef SKIP_NON_CPU
#undef SKIP_OPENCL
#undef SKIP_OPENCL_FP16
#undef SKIP_MYRIAD

#endif
