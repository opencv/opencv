// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"
#include <set>
#include <string>
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#if defined(_MSC_VER)  // workaround for 32-bit MSVC compiler
#pragma optimize("", off)
#endif


#define CV_TEST_TAG_DNN_ERROR_PARSER "dnn_error_parser"
#define CV_TEST_TAG_DNN_ERROR_NET_SETUP "dnn_error_net_setup"
#define CV_TEST_TAG_DNN_ERROR_FORWARD "dnn_error_forward"
#define CV_TEST_TAG_DNN_LAYER_FALLBACK "dnn_layer_fallback"
#define CV_TEST_TAG_DNN_NO_ACCURACY_CHECK "dnn_no_accuracy_check"


namespace opencv_test {

struct TestCase
{
    const char* name;
    uint32_t inputs;
    uint32_t outputs;
};

static const TestCase testConformanceConfig[] = {
    {"test_abs", 1, 1},
    {"test_acos", 1, 1},
    {"test_acos_example", 1, 1},
    {"test_acosh", 1, 1},
    {"test_acosh_example", 1, 1},
    {"test_adagrad", 5, 2},
    {"test_adagrad_multiple", 8, 4},
    {"test_adam", 6, 3},
    {"test_adam_multiple", 10, 6},
    {"test_add", 2, 1},
    {"test_add_bcast", 2, 1},
    {"test_add_uint8", 2, 1},
    {"test_and2d", 2, 1},
    {"test_and3d", 2, 1},
    {"test_and4d", 2, 1},
    {"test_and_bcast3v1d", 2, 1},
    {"test_and_bcast3v2d", 2, 1},
    {"test_and_bcast4v2d", 2, 1},
    {"test_and_bcast4v3d", 2, 1},
    {"test_and_bcast4v4d", 2, 1},
    {"test_argmax_default_axis_example", 1, 1},
    {"test_argmax_default_axis_example_select_last_index", 1, 1},
    {"test_argmax_default_axis_random", 1, 1},
    {"test_argmax_default_axis_random_select_last_index", 1, 1},
    {"test_argmax_keepdims_example", 1, 1},
    {"test_argmax_keepdims_example_select_last_index", 1, 1},
    {"test_argmax_keepdims_random", 1, 1},
    {"test_argmax_keepdims_random_select_last_index", 1, 1},
    {"test_argmax_negative_axis_keepdims_example", 1, 1},
    {"test_argmax_negative_axis_keepdims_example_select_last_index", 1, 1},
    {"test_argmax_negative_axis_keepdims_random", 1, 1},
    {"test_argmax_negative_axis_keepdims_random_select_last_index", 1, 1},
    {"test_argmax_no_keepdims_example", 1, 1},
    {"test_argmax_no_keepdims_example_select_last_index", 1, 1},
    {"test_argmax_no_keepdims_random", 1, 1},
    {"test_argmax_no_keepdims_random_select_last_index", 1, 1},
    {"test_argmin_default_axis_example", 1, 1},
    {"test_argmin_default_axis_example_select_last_index", 1, 1},
    {"test_argmin_default_axis_random", 1, 1},
    {"test_argmin_default_axis_random_select_last_index", 1, 1},
    {"test_argmin_keepdims_example", 1, 1},
    {"test_argmin_keepdims_example_select_last_index", 1, 1},
    {"test_argmin_keepdims_random", 1, 1},
    {"test_argmin_keepdims_random_select_last_index", 1, 1},
    {"test_argmin_negative_axis_keepdims_example", 1, 1},
    {"test_argmin_negative_axis_keepdims_example_select_last_index", 1, 1},
    {"test_argmin_negative_axis_keepdims_random", 1, 1},
    {"test_argmin_negative_axis_keepdims_random_select_last_index", 1, 1},
    {"test_argmin_no_keepdims_example", 1, 1},
    {"test_argmin_no_keepdims_example_select_last_index", 1, 1},
    {"test_argmin_no_keepdims_random", 1, 1},
    {"test_argmin_no_keepdims_random_select_last_index", 1, 1},
    {"test_asin", 1, 1},
    {"test_asin_example", 1, 1},
    {"test_asinh", 1, 1},
    {"test_asinh_example", 1, 1},
    {"test_atan", 1, 1},
    {"test_atan_example", 1, 1},
    {"test_atanh", 1, 1},
    {"test_atanh_example", 1, 1},
    {"test_averagepool_1d_default", 1, 1},
    {"test_averagepool_2d_ceil", 1, 1},
    {"test_averagepool_2d_default", 1, 1},
    {"test_averagepool_2d_pads", 1, 1},
    {"test_averagepool_2d_pads_count_include_pad", 1, 1},
    {"test_averagepool_2d_precomputed_pads", 1, 1},
    {"test_averagepool_2d_precomputed_pads_count_include_pad", 1, 1},
    {"test_averagepool_2d_precomputed_same_upper", 1, 1},
    {"test_averagepool_2d_precomputed_strides", 1, 1},
    {"test_averagepool_2d_same_lower", 1, 1},
    {"test_averagepool_2d_same_upper", 1, 1},
    {"test_averagepool_2d_strides", 1, 1},
    {"test_averagepool_3d_default", 1, 1},
    {"test_basic_conv_with_padding", 2, 1},
    {"test_basic_conv_without_padding", 2, 1},
    {"test_basic_convinteger", 3, 1},
    {"test_batchnorm_epsilon", 5, 1},
    {"test_batchnorm_epsilon_training_mode", 5, 3},
    {"test_batchnorm_example", 5, 1},
    {"test_batchnorm_example_training_mode", 5, 3},
    {"test_bernoulli", 1, 1},
    {"test_bernoulli_double", 1, 1},
    {"test_bernoulli_double_expanded", 1, 1},
    {"test_bernoulli_expanded", 1, 1},
    {"test_bernoulli_seed", 1, 1},
    {"test_bernoulli_seed_expanded", 1, 1},
    {"test_bitshift_left_uint16", 2, 1},
    {"test_bitshift_left_uint32", 2, 1},
    {"test_bitshift_left_uint64", 2, 1},
    {"test_bitshift_left_uint8", 2, 1},
    {"test_bitshift_right_uint16", 2, 1},
    {"test_bitshift_right_uint32", 2, 1},
    {"test_bitshift_right_uint64", 2, 1},
    {"test_bitshift_right_uint8", 2, 1},
    {"test_cast_BFLOAT16_to_FLOAT", 1, 1},
    {"test_cast_DOUBLE_to_FLOAT", 1, 1},
    {"test_cast_DOUBLE_to_FLOAT16", 1, 1},
    {"test_cast_FLOAT16_to_DOUBLE", 1, 1},
    {"test_cast_FLOAT16_to_FLOAT", 1, 1},
    {"test_cast_FLOAT_to_BFLOAT16", 1, 1},
    {"test_cast_FLOAT_to_DOUBLE", 1, 1},
    {"test_cast_FLOAT_to_FLOAT16", 1, 1},
    {"test_cast_FLOAT_to_STRING", 1, 1},
    {"test_cast_STRING_to_FLOAT", 1, 1},
    {"test_castlike_BFLOAT16_to_FLOAT", 2, 1},
    {"test_castlike_BFLOAT16_to_FLOAT_expanded", 2, 1},
    {"test_castlike_DOUBLE_to_FLOAT", 2, 1},
    {"test_castlike_DOUBLE_to_FLOAT16", 2, 1},
    {"test_castlike_DOUBLE_to_FLOAT16_expanded", 2, 1},
    {"test_castlike_DOUBLE_to_FLOAT_expanded", 2, 1},
    {"test_castlike_FLOAT16_to_DOUBLE", 2, 1},
    {"test_castlike_FLOAT16_to_DOUBLE_expanded", 2, 1},
    {"test_castlike_FLOAT16_to_FLOAT", 2, 1},
    {"test_castlike_FLOAT16_to_FLOAT_expanded", 2, 1},
    {"test_castlike_FLOAT_to_BFLOAT16", 2, 1},
    {"test_castlike_FLOAT_to_BFLOAT16_expanded", 2, 1},
    {"test_castlike_FLOAT_to_DOUBLE", 2, 1},
    {"test_castlike_FLOAT_to_DOUBLE_expanded", 2, 1},
    {"test_castlike_FLOAT_to_FLOAT16", 2, 1},
    {"test_castlike_FLOAT_to_FLOAT16_expanded", 2, 1},
    {"test_castlike_FLOAT_to_STRING", 2, 1},
    {"test_castlike_FLOAT_to_STRING_expanded", 2, 1},
    {"test_castlike_STRING_to_FLOAT", 2, 1},
    {"test_castlike_STRING_to_FLOAT_expanded", 2, 1},
    {"test_ceil", 1, 1},
    {"test_ceil_example", 1, 1},
    {"test_celu", 1, 1},
    {"test_celu_expanded", 1, 1},
    {"test_clip", 3, 1},
    {"test_clip_default_inbounds", 1, 1},
    {"test_clip_default_int8_inbounds", 1, 1},
    {"test_clip_default_int8_max", 2, 1},
    {"test_clip_default_int8_min", 2, 1},
    {"test_clip_default_max", 2, 1},
    {"test_clip_default_min", 2, 1},
    {"test_clip_example", 3, 1},
    {"test_clip_inbounds", 3, 1},
    {"test_clip_outbounds", 3, 1},
    {"test_clip_splitbounds", 3, 1},
    {"test_compress_0", 2, 1},
    {"test_compress_1", 2, 1},
    {"test_compress_default_axis", 2, 1},
    {"test_compress_negative_axis", 2, 1},
    {"test_concat_1d_axis_0", 2, 1},
    {"test_concat_1d_axis_negative_1", 2, 1},
    {"test_concat_2d_axis_0", 2, 1},
    {"test_concat_2d_axis_1", 2, 1},
    {"test_concat_2d_axis_negative_1", 2, 1},
    {"test_concat_2d_axis_negative_2", 2, 1},
    {"test_concat_3d_axis_0", 2, 1},
    {"test_concat_3d_axis_1", 2, 1},
    {"test_concat_3d_axis_2", 2, 1},
    {"test_concat_3d_axis_negative_1", 2, 1},
    {"test_concat_3d_axis_negative_2", 2, 1},
    {"test_concat_3d_axis_negative_3", 2, 1},
    {"test_constant", 0, 1},
    {"test_constant_pad", 3, 1},
    {"test_constantofshape_float_ones", 1, 1},
    {"test_constantofshape_int_shape_zero", 1, 1},
    {"test_constantofshape_int_zeros", 1, 1},
    {"test_conv_with_autopad_same", 2, 1},
    {"test_conv_with_strides_and_asymmetric_padding", 2, 1},
    {"test_conv_with_strides_no_padding", 2, 1},
    {"test_conv_with_strides_padding", 2, 1},
    {"test_convinteger_with_padding", 3, 1},
    {"test_convinteger_without_padding", 3, 1},
    {"test_convtranspose", 2, 1},
    {"test_convtranspose_1d", 2, 1},
    {"test_convtranspose_3d", 2, 1},
    {"test_convtranspose_autopad_same", 2, 1},
    {"test_convtranspose_dilations", 2, 1},
    {"test_convtranspose_kernel_shape", 2, 1},
    {"test_convtranspose_output_shape", 2, 1},
    {"test_convtranspose_pad", 2, 1},
    {"test_convtranspose_pads", 2, 1},
    {"test_convtranspose_with_kernel", 2, 1},
    {"test_cos", 1, 1},
    {"test_cos_example", 1, 1},
    {"test_cosh", 1, 1},
    {"test_cosh_example", 1, 1},
    {"test_cumsum_1d", 2, 1},
    {"test_cumsum_1d_exclusive", 2, 1},
    {"test_cumsum_1d_reverse", 2, 1},
    {"test_cumsum_1d_reverse_exclusive", 2, 1},
    {"test_cumsum_2d_axis_0", 2, 1},
    {"test_cumsum_2d_axis_1", 2, 1},
    {"test_cumsum_2d_negative_axis", 2, 1},
    {"test_depthtospace_crd_mode", 1, 1},
    {"test_depthtospace_crd_mode_example", 1, 1},
    {"test_depthtospace_dcr_mode", 1, 1},
    {"test_depthtospace_example", 1, 1},
    {"test_dequantizelinear", 3, 1},
    {"test_dequantizelinear_axis", 3, 1},
    {"test_dequantizelinear_blocked", 3, 1},
    {"test_det_2d", 1, 1},
    {"test_det_nd", 1, 1},
    {"test_div", 2, 1},
    {"test_div_bcast", 2, 1},
    {"test_div_example", 2, 1},
    {"test_div_uint8", 2, 1},
    {"test_dropout_default", 1, 1},
    {"test_dropout_default_mask", 1, 2},
    {"test_dropout_default_mask_ratio", 2, 2},
    {"test_dropout_default_old", 1, 1},
    {"test_dropout_default_ratio", 2, 1},
    {"test_dropout_random_old", 1, 1},
    {"test_dynamicquantizelinear", 1, 3},
    {"test_dynamicquantizelinear_expanded", 1, 3},
    {"test_dynamicquantizelinear_max_adjusted", 1, 3},
    {"test_dynamicquantizelinear_max_adjusted_expanded", 1, 3},
    {"test_dynamicquantizelinear_min_adjusted", 1, 3},
    {"test_dynamicquantizelinear_min_adjusted_expanded", 1, 3},
    {"test_edge_pad", 2, 1},
    {"test_einsum_batch_diagonal", 1, 1},
    {"test_einsum_batch_matmul", 2, 1},
    {"test_einsum_inner_prod", 2, 1},
    {"test_einsum_sum", 1, 1},
    {"test_einsum_transpose", 1, 1},
    {"test_elu", 1, 1},
    {"test_elu_default", 1, 1},
    {"test_elu_default_expanded_ver18", 1, 1},
    {"test_elu_example", 1, 1},
    {"test_elu_example_expanded_ver18", 1, 1},
    {"test_elu_expanded_ver18", 1, 1},
    {"test_equal", 2, 1},
    {"test_equal_bcast", 2, 1},
    {"test_erf", 1, 1},
    {"test_exp", 1, 1},
    {"test_exp_example", 1, 1},
    {"test_expand_dim_changed", 2, 1},
    {"test_expand_dim_unchanged", 2, 1},
    {"test_eyelike_populate_off_main_diagonal", 1, 1},
    {"test_eyelike_with_dtype", 1, 1},
    {"test_eyelike_without_dtype", 1, 1},
    {"test_flatten_axis0", 1, 1},
    {"test_flatten_axis1", 1, 1},
    {"test_flatten_axis2", 1, 1},
    {"test_flatten_axis3", 1, 1},
    {"test_flatten_default_axis", 1, 1},
    {"test_flatten_negative_axis1", 1, 1},
    {"test_flatten_negative_axis2", 1, 1},
    {"test_flatten_negative_axis3", 1, 1},
    {"test_flatten_negative_axis4", 1, 1},
    {"test_floor", 1, 1},
    {"test_floor_example", 1, 1},
    {"test_gather_0", 2, 1},
    {"test_gather_1", 2, 1},
    {"test_gather_2d_indices", 2, 1},
    {"test_gather_elements_0", 2, 1},
    {"test_gather_elements_1", 2, 1},
    {"test_gather_elements_negative_indices", 2, 1},
    {"test_gather_negative_indices", 2, 1},
    {"test_gathernd_example_float32", 2, 1},
    {"test_gathernd_example_int32", 2, 1},
    {"test_gathernd_example_int32_batch_dim1", 2, 1},
    {"test_gelu_default_1", 1, 1},
    {"test_gelu_default_1_expanded", 1, 1},
    {"test_gelu_default_2", 1, 1},
    {"test_gelu_default_2_expanded", 1, 1},
    {"test_gelu_tanh_1", 1, 1},
    {"test_gelu_tanh_1_expanded", 1, 1},
    {"test_gelu_tanh_2", 1, 1},
    {"test_gelu_tanh_2_expanded", 1, 1},
    {"test_gemm_all_attributes", 3, 1},
    {"test_gemm_alpha", 3, 1},
    {"test_gemm_beta", 3, 1},
    {"test_gemm_default_matrix_bias", 3, 1},
    {"test_gemm_default_no_bias", 2, 1},
    {"test_gemm_default_scalar_bias", 3, 1},
    {"test_gemm_default_single_elem_vector_bias", 3, 1},
    {"test_gemm_default_vector_bias", 3, 1},
    {"test_gemm_default_zero_bias", 3, 1},
    {"test_gemm_transposeA", 3, 1},
    {"test_gemm_transposeB", 3, 1},
    {"test_globalaveragepool", 1, 1},
    {"test_globalaveragepool_precomputed", 1, 1},
    {"test_globalmaxpool", 1, 1},
    {"test_globalmaxpool_precomputed", 1, 1},
    {"test_greater", 2, 1},
    {"test_greater_bcast", 2, 1},
    {"test_greater_equal", 2, 1},
    {"test_greater_equal_bcast", 2, 1},
    {"test_greater_equal_bcast_expanded", 2, 1},
    {"test_greater_equal_expanded", 2, 1},
    {"test_gridsample", 2, 1},
    {"test_gridsample_aligncorners_true", 2, 1},
    {"test_gridsample_bicubic", 2, 1},
    {"test_gridsample_bilinear", 2, 1},
    {"test_gridsample_border_padding", 2, 1},
    {"test_gridsample_nearest", 2, 1},
    {"test_gridsample_reflection_padding", 2, 1},
    {"test_gridsample_zeros_padding", 2, 1},
    {"test_group_normalization_epsilon", 3, 1},
    {"test_group_normalization_example", 3, 1},
    {"test_gru_batchwise", 3, 2},
    {"test_gru_defaults", 3, 1},
    {"test_gru_seq_length", 4, 1},
    {"test_gru_with_initial_bias", 4, 1},
    {"test_hardmax_axis_0", 1, 1},
    {"test_hardmax_axis_1", 1, 1},
    {"test_hardmax_axis_2", 1, 1},
    {"test_hardmax_default_axis", 1, 1},
    {"test_hardmax_example", 1, 1},
    {"test_hardmax_negative_axis", 1, 1},
    {"test_hardmax_one_hot", 1, 1},
    {"test_hardsigmoid", 1, 1},
    {"test_hardsigmoid_default", 1, 1},
    {"test_hardsigmoid_example", 1, 1},
    {"test_hardswish", 1, 1},
    {"test_hardswish_expanded", 1, 1},
    {"test_identity", 1, 1},
    {"test_identity_opt", 1, 1},
    {"test_identity_sequence", 1, 1},
    {"test_if", 1, 1},
    {"test_if_opt", 1, 1},
    {"test_if_seq", 1, 1},
    {"test_instancenorm_epsilon", 3, 1},
    {"test_instancenorm_example", 3, 1},
    {"test_isinf", 1, 1},
    {"test_isinf_negative", 1, 1},
    {"test_isinf_positive", 1, 1},
    {"test_isnan", 1, 1},
    {"test_layer_normalization_2d_axis0", 3, 1},
    {"test_layer_normalization_2d_axis1", 3, 1},
    {"test_layer_normalization_2d_axis_negative_1", 3, 1},
    {"test_layer_normalization_2d_axis_negative_2", 3, 1},
    {"test_layer_normalization_3d_axis0_epsilon", 3, 1},
    {"test_layer_normalization_3d_axis1_epsilon", 3, 1},
    {"test_layer_normalization_3d_axis2_epsilon", 3, 1},
    {"test_layer_normalization_3d_axis_negative_1_epsilon", 3, 1},
    {"test_layer_normalization_3d_axis_negative_2_epsilon", 3, 1},
    {"test_layer_normalization_3d_axis_negative_3_epsilon", 3, 1},
    {"test_layer_normalization_4d_axis0", 3, 1},
    {"test_layer_normalization_4d_axis1", 3, 1},
    {"test_layer_normalization_4d_axis2", 3, 1},
    {"test_layer_normalization_4d_axis3", 3, 1},
    {"test_layer_normalization_4d_axis_negative_1", 3, 1},
    {"test_layer_normalization_4d_axis_negative_2", 3, 1},
    {"test_layer_normalization_4d_axis_negative_3", 3, 1},
    {"test_layer_normalization_4d_axis_negative_4", 3, 1},
    {"test_layer_normalization_default_axis", 3, 1},
    {"test_leakyrelu", 1, 1},
    {"test_leakyrelu_default", 1, 1},
    {"test_leakyrelu_example", 1, 1},
    {"test_less", 2, 1},
    {"test_less_bcast", 2, 1},
    {"test_less_equal", 2, 1},
    {"test_less_equal_bcast", 2, 1},
    {"test_less_equal_bcast_expanded", 2, 1},
    {"test_less_equal_expanded", 2, 1},
    {"test_log", 1, 1},
    {"test_log_example", 1, 1},
    {"test_logsoftmax_axis_0", 1, 1},
    {"test_logsoftmax_axis_0_expanded", 1, 1},
    {"test_logsoftmax_axis_1", 1, 1},
    {"test_logsoftmax_axis_1_expanded", 1, 1},
    {"test_logsoftmax_axis_2", 1, 1},
    {"test_logsoftmax_axis_2_expanded", 1, 1},
    {"test_logsoftmax_default_axis", 1, 1},
    {"test_logsoftmax_default_axis_expanded", 1, 1},
    {"test_logsoftmax_example_1", 1, 1},
    {"test_logsoftmax_example_1_expanded", 1, 1},
    {"test_logsoftmax_large_number", 1, 1},
    {"test_logsoftmax_large_number_expanded", 1, 1},
    {"test_logsoftmax_negative_axis", 1, 1},
    {"test_logsoftmax_negative_axis_expanded", 1, 1},
    {"test_loop11", 3, 2},
    {"test_loop13_seq", 3, 1},
    {"test_loop16_seq_none", 3, 1},
    {"test_lrn", 1, 1},
    {"test_lrn_default", 1, 1},
    {"test_lstm_batchwise", 3, 2},
    {"test_lstm_defaults", 3, 1},
    {"test_lstm_with_initial_bias", 4, 1},
    {"test_lstm_with_peepholes", 8, 1},
    {"test_matmul_2d", 2, 1},
    {"test_matmul_3d", 2, 1},
    {"test_matmul_4d", 2, 1},
    {"test_matmulinteger", 4, 1},
    {"test_max_example", 3, 1},
    {"test_max_float16", 2, 1},
    {"test_max_float32", 2, 1},
    {"test_max_float64", 2, 1},
    {"test_max_int16", 2, 1},
    {"test_max_int32", 2, 1},
    {"test_max_int64", 2, 1},
    {"test_max_int8", 2, 1},
    {"test_max_one_input", 1, 1},
    {"test_max_two_inputs", 2, 1},
    {"test_max_uint16", 2, 1},
    {"test_max_uint32", 2, 1},
    {"test_max_uint64", 2, 1},
    {"test_max_uint8", 2, 1},
    {"test_maxpool_1d_default", 1, 1},
    {"test_maxpool_2d_ceil", 1, 1},
    {"test_maxpool_2d_default", 1, 1},
    {"test_maxpool_2d_dilations", 1, 1},
    {"test_maxpool_2d_pads", 1, 1},
    {"test_maxpool_2d_precomputed_pads", 1, 1},
    {"test_maxpool_2d_precomputed_same_upper", 1, 1},
    {"test_maxpool_2d_precomputed_strides", 1, 1},
    {"test_maxpool_2d_same_lower", 1, 1},
    {"test_maxpool_2d_same_upper", 1, 1},
    {"test_maxpool_2d_strides", 1, 1},
    {"test_maxpool_2d_uint8", 1, 1},
    {"test_maxpool_3d_default", 1, 1},
    {"test_maxpool_with_argmax_2d_precomputed_pads", 1, 2},
    {"test_maxpool_with_argmax_2d_precomputed_strides", 1, 2},
    {"test_maxunpool_export_with_output_shape", 3, 1},
    {"test_maxunpool_export_without_output_shape", 2, 1},
    {"test_mean_example", 3, 1},
    {"test_mean_one_input", 1, 1},
    {"test_mean_two_inputs", 2, 1},
    {"test_min_example", 3, 1},
    {"test_min_float16", 2, 1},
    {"test_min_float32", 2, 1},
    {"test_min_float64", 2, 1},
    {"test_min_int16", 2, 1},
    {"test_min_int32", 2, 1},
    {"test_min_int64", 2, 1},
    {"test_min_int8", 2, 1},
    {"test_min_one_input", 1, 1},
    {"test_min_two_inputs", 2, 1},
    {"test_min_uint16", 2, 1},
    {"test_min_uint32", 2, 1},
    {"test_min_uint64", 2, 1},
    {"test_min_uint8", 2, 1},
    {"test_mish", 1, 1},
    {"test_mish_expanded", 1, 1},
    {"test_mod_broadcast", 2, 1},
    {"test_mod_int64_fmod", 2, 1},
    {"test_mod_mixed_sign_float16", 2, 1},
    {"test_mod_mixed_sign_float32", 2, 1},
    {"test_mod_mixed_sign_float64", 2, 1},
    {"test_mod_mixed_sign_int16", 2, 1},
    {"test_mod_mixed_sign_int32", 2, 1},
    {"test_mod_mixed_sign_int64", 2, 1},
    {"test_mod_mixed_sign_int8", 2, 1},
    {"test_mod_uint16", 2, 1},
    {"test_mod_uint32", 2, 1},
    {"test_mod_uint64", 2, 1},
    {"test_mod_uint8", 2, 1},
    {"test_momentum", 5, 2},
    {"test_momentum_multiple", 8, 4},
    {"test_mul", 2, 1},
    {"test_mul_bcast", 2, 1},
    {"test_mul_example", 2, 1},
    {"test_mul_uint8", 2, 1},
    {"test_mvn", 1, 1},
    {"test_mvn_expanded", 1, 1},
    {"test_neg", 1, 1},
    {"test_neg_example", 1, 1},
    {"test_nesterov_momentum", 5, 2},
    {"test_nllloss_NC", 2, 1},
    {"test_nllloss_NC_expanded", 2, 1},
    {"test_nllloss_NCd1", 2, 1},
    {"test_nllloss_NCd1_expanded", 2, 1},
    {"test_nllloss_NCd1_ii", 2, 1},
    {"test_nllloss_NCd1_ii_expanded", 2, 1},
    {"test_nllloss_NCd1_mean_weight_negative_ii", 3, 1},
    {"test_nllloss_NCd1_mean_weight_negative_ii_expanded", 3, 1},
    {"test_nllloss_NCd1_weight", 3, 1},
    {"test_nllloss_NCd1_weight_expanded", 3, 1},
    {"test_nllloss_NCd1_weight_ii", 3, 1},
    {"test_nllloss_NCd1_weight_ii_expanded", 3, 1},
    {"test_nllloss_NCd1d2", 2, 1},
    {"test_nllloss_NCd1d2_expanded", 2, 1},
    {"test_nllloss_NCd1d2_no_weight_reduction_mean_ii", 2, 1},
    {"test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded", 2, 1},
    {"test_nllloss_NCd1d2_reduction_mean", 2, 1},
    {"test_nllloss_NCd1d2_reduction_mean_expanded", 2, 1},
    {"test_nllloss_NCd1d2_reduction_sum", 2, 1},
    {"test_nllloss_NCd1d2_reduction_sum_expanded", 2, 1},
    {"test_nllloss_NCd1d2_with_weight", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_expanded", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_mean", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_mean_expanded", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_sum", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_sum_expanded", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_sum_ii", 3, 1},
    {"test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded", 3, 1},
    {"test_nllloss_NCd1d2d3_none_no_weight_negative_ii", 2, 1},
    {"test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded", 2, 1},
    {"test_nllloss_NCd1d2d3_sum_weight_high_ii", 3, 1},
    {"test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded", 3, 1},
    {"test_nllloss_NCd1d2d3d4d5_mean_weight", 3, 1},
    {"test_nllloss_NCd1d2d3d4d5_mean_weight_expanded", 3, 1},
    {"test_nllloss_NCd1d2d3d4d5_none_no_weight", 2, 1},
    {"test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded", 2, 1},
    {"test_nonmaxsuppression_center_point_box_format", 5, 1},
    {"test_nonmaxsuppression_flipped_coordinates", 5, 1},
    {"test_nonmaxsuppression_identical_boxes", 5, 1},
    {"test_nonmaxsuppression_limit_output_size", 5, 1},
    {"test_nonmaxsuppression_single_box", 5, 1},
    {"test_nonmaxsuppression_suppress_by_IOU", 5, 1},
    {"test_nonmaxsuppression_suppress_by_IOU_and_scores", 5, 1},
    {"test_nonmaxsuppression_two_batches", 5, 1},
    {"test_nonmaxsuppression_two_classes", 5, 1},
    {"test_nonzero_example", 1, 1},
    {"test_not_2d", 1, 1},
    {"test_not_3d", 1, 1},
    {"test_not_4d", 1, 1},
    {"test_onehot_negative_indices", 3, 1},
    {"test_onehot_with_axis", 3, 1},
    {"test_onehot_with_negative_axis", 3, 1},
    {"test_onehot_without_axis", 3, 1},
    {"test_optional_get_element", 1, 1},
    {"test_optional_get_element_sequence", 1, 1},
    {"test_optional_has_element", 1, 1},
    {"test_optional_has_element_empty", 1, 1},
    {"test_or2d", 2, 1},
    {"test_or3d", 2, 1},
    {"test_or4d", 2, 1},
    {"test_or_bcast3v1d", 2, 1},
    {"test_or_bcast3v2d", 2, 1},
    {"test_or_bcast4v2d", 2, 1},
    {"test_or_bcast4v3d", 2, 1},
    {"test_or_bcast4v4d", 2, 1},
    {"test_pow", 2, 1},
    {"test_pow_bcast_array", 2, 1},
    {"test_pow_bcast_scalar", 2, 1},
    {"test_pow_example", 2, 1},
    {"test_pow_types_float", 2, 1},
    {"test_pow_types_float32_int32", 2, 1},
    {"test_pow_types_float32_int64", 2, 1},
    {"test_pow_types_float32_uint32", 2, 1},
    {"test_pow_types_float32_uint64", 2, 1},
    {"test_pow_types_int", 2, 1},
    {"test_pow_types_int32_float32", 2, 1},
    {"test_pow_types_int32_int32", 2, 1},
    {"test_pow_types_int64_float32", 2, 1},
    {"test_pow_types_int64_int64", 2, 1},
    {"test_prelu_broadcast", 2, 1},
    {"test_prelu_example", 2, 1},
    {"test_qlinearconv", 8, 1},
    {"test_qlinearmatmul_2D", 8, 1},
    {"test_qlinearmatmul_3D", 8, 1},
    {"test_quantizelinear", 3, 1},
    {"test_quantizelinear_axis", 3, 1},
    {"test_quantizelinear_blocked", 3, 1},
    {"test_range_float_type_positive_delta", 3, 1},
    {"test_range_float_type_positive_delta_expanded", 3, 1},
    {"test_range_int32_type_negative_delta", 3, 1},
    {"test_range_int32_type_negative_delta_expanded", 3, 1},
    {"test_reciprocal", 1, 1},
    {"test_reciprocal_example", 1, 1},
    {"test_reduce_l1_default_axes_keepdims_example", 1, 1},
    {"test_reduce_l1_default_axes_keepdims_random", 1, 1},
    {"test_reduce_l1_do_not_keepdims_example", 1, 1},
    {"test_reduce_l1_do_not_keepdims_random", 1, 1},
    {"test_reduce_l1_keep_dims_example", 1, 1},
    {"test_reduce_l1_keep_dims_random", 1, 1},
    {"test_reduce_l1_negative_axes_keep_dims_example", 1, 1},
    {"test_reduce_l1_negative_axes_keep_dims_random", 1, 1},
    {"test_reduce_l2_default_axes_keepdims_example", 1, 1},
    {"test_reduce_l2_default_axes_keepdims_random", 1, 1},
    {"test_reduce_l2_do_not_keepdims_example", 1, 1},
    {"test_reduce_l2_do_not_keepdims_random", 1, 1},
    {"test_reduce_l2_keep_dims_example", 1, 1},
    {"test_reduce_l2_keep_dims_random", 1, 1},
    {"test_reduce_l2_negative_axes_keep_dims_example", 1, 1},
    {"test_reduce_l2_negative_axes_keep_dims_random", 1, 1},
    {"test_reduce_log_sum", 1, 1},
    {"test_reduce_log_sum_asc_axes", 1, 1},
    {"test_reduce_log_sum_default", 1, 1},
    {"test_reduce_log_sum_desc_axes", 1, 1},
    {"test_reduce_log_sum_exp_default_axes_keepdims_example", 1, 1},
    {"test_reduce_log_sum_exp_default_axes_keepdims_random", 1, 1},
    {"test_reduce_log_sum_exp_do_not_keepdims_example", 1, 1},
    {"test_reduce_log_sum_exp_do_not_keepdims_random", 1, 1},
    {"test_reduce_log_sum_exp_keepdims_example", 1, 1},
    {"test_reduce_log_sum_exp_keepdims_random", 1, 1},
    {"test_reduce_log_sum_exp_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_log_sum_exp_negative_axes_keepdims_random", 1, 1},
    {"test_reduce_log_sum_negative_axes", 1, 1},
    {"test_reduce_max_default_axes_keepdim_example", 1, 1},
    {"test_reduce_max_default_axes_keepdims_random", 1, 1},
    {"test_reduce_max_do_not_keepdims_example", 1, 1},
    {"test_reduce_max_do_not_keepdims_random", 1, 1},
    {"test_reduce_max_keepdims_example", 1, 1},
    {"test_reduce_max_keepdims_random", 1, 1},
    {"test_reduce_max_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_max_negative_axes_keepdims_random", 1, 1},
    {"test_reduce_mean_default_axes_keepdims_example", 1, 1},
    {"test_reduce_mean_default_axes_keepdims_random", 1, 1},
    {"test_reduce_mean_do_not_keepdims_example", 1, 1},
    {"test_reduce_mean_do_not_keepdims_random", 1, 1},
    {"test_reduce_mean_keepdims_example", 1, 1},
    {"test_reduce_mean_keepdims_random", 1, 1},
    {"test_reduce_mean_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_mean_negative_axes_keepdims_random", 1, 1},
    {"test_reduce_min_default_axes_keepdims_example", 1, 1},
    {"test_reduce_min_default_axes_keepdims_random", 1, 1},
    {"test_reduce_min_do_not_keepdims_example", 1, 1},
    {"test_reduce_min_do_not_keepdims_random", 1, 1},
    {"test_reduce_min_keepdims_example", 1, 1},
    {"test_reduce_min_keepdims_random", 1, 1},
    {"test_reduce_min_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_min_negative_axes_keepdims_random", 1, 1},
    {"test_reduce_prod_default_axes_keepdims_example", 1, 1},
    {"test_reduce_prod_default_axes_keepdims_random", 1, 1},
    {"test_reduce_prod_do_not_keepdims_example", 1, 1},
    {"test_reduce_prod_do_not_keepdims_random", 1, 1},
    {"test_reduce_prod_keepdims_example", 1, 1},
    {"test_reduce_prod_keepdims_random", 1, 1},
    {"test_reduce_prod_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_prod_negative_axes_keepdims_random", 1, 1},
    {"test_reduce_sum_default_axes_keepdims_example", 2, 1},
    {"test_reduce_sum_default_axes_keepdims_random", 2, 1},
    {"test_reduce_sum_do_not_keepdims_example", 2, 1},
    {"test_reduce_sum_do_not_keepdims_random", 2, 1},
    {"test_reduce_sum_empty_axes_input_noop_example", 2, 1},
    {"test_reduce_sum_empty_axes_input_noop_random", 2, 1},
    {"test_reduce_sum_keepdims_example", 2, 1},
    {"test_reduce_sum_keepdims_random", 2, 1},
    {"test_reduce_sum_negative_axes_keepdims_example", 2, 1},
    {"test_reduce_sum_negative_axes_keepdims_random", 2, 1},
    {"test_reduce_sum_square_default_axes_keepdims_example", 1, 1},
    {"test_reduce_sum_square_default_axes_keepdims_random", 1, 1},
    {"test_reduce_sum_square_do_not_keepdims_example", 1, 1},
    {"test_reduce_sum_square_do_not_keepdims_random", 1, 1},
    {"test_reduce_sum_square_keepdims_example", 1, 1},
    {"test_reduce_sum_square_keepdims_random", 1, 1},
    {"test_reduce_sum_square_negative_axes_keepdims_example", 1, 1},
    {"test_reduce_sum_square_negative_axes_keepdims_random", 1, 1},
    {"test_reflect_pad", 2, 1},
    {"test_relu", 1, 1},
    {"test_reshape_allowzero_reordered", 2, 1},
    {"test_reshape_extended_dims", 2, 1},
    {"test_reshape_negative_dim", 2, 1},
    {"test_reshape_negative_extended_dims", 2, 1},
    {"test_reshape_one_dim", 2, 1},
    {"test_reshape_reduced_dims", 2, 1},
    {"test_reshape_reordered_all_dims", 2, 1},
    {"test_reshape_reordered_last_dims", 2, 1},
    {"test_reshape_zero_and_negative_dim", 2, 1},
    {"test_reshape_zero_dim", 2, 1},
    {"test_resize_downsample_scales_cubic", 2, 1},
    {"test_resize_downsample_scales_cubic_A_n0p5_exclude_outside", 2, 1},
    {"test_resize_downsample_scales_cubic_align_corners", 2, 1},
    {"test_resize_downsample_scales_linear", 2, 1},
    {"test_resize_downsample_scales_linear_align_corners", 2, 1},
    {"test_resize_downsample_scales_nearest", 2, 1},
    {"test_resize_downsample_sizes_cubic", 2, 1},
    {"test_resize_downsample_sizes_linear_pytorch_half_pixel", 2, 1},
    {"test_resize_downsample_sizes_nearest", 2, 1},
    {"test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn", 2, 1},
    {"test_resize_tf_crop_and_resize", 3, 1},
    {"test_resize_upsample_scales_cubic", 2, 1},
    {"test_resize_upsample_scales_cubic_A_n0p5_exclude_outside", 2, 1},
    {"test_resize_upsample_scales_cubic_align_corners", 2, 1},
    {"test_resize_upsample_scales_cubic_asymmetric", 2, 1},
    {"test_resize_upsample_scales_linear", 2, 1},
    {"test_resize_upsample_scales_linear_align_corners", 2, 1},
    {"test_resize_upsample_scales_nearest", 2, 1},
    {"test_resize_upsample_sizes_cubic", 2, 1},
    {"test_resize_upsample_sizes_nearest", 2, 1},
    {"test_resize_upsample_sizes_nearest_ceil_half_pixel", 2, 1},
    {"test_resize_upsample_sizes_nearest_floor_align_corners", 2, 1},
    {"test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", 2, 1},
    {"test_reversesequence_batch", 2, 1},
    {"test_reversesequence_time", 2, 1},
    {"test_rnn_seq_length", 4, 1},
    {"test_roialign_aligned_false", 3, 1},
    {"test_roialign_aligned_true", 3, 1},
    {"test_round", 1, 1},
    {"test_scan9_sum", 2, 2},
    {"test_scan_sum", 2, 2},
    {"test_scatter_elements_with_axis", 3, 1},
    {"test_scatter_elements_with_duplicate_indices", 3, 1},
    {"test_scatter_elements_with_negative_indices", 3, 1},
    {"test_scatter_elements_with_reduction_max", 3, 1},
    {"test_scatter_elements_with_reduction_min", 3, 1},
    {"test_scatter_elements_without_axis", 3, 1},
    {"test_scatter_with_axis", 3, 1},
    {"test_scatter_without_axis", 3, 1},
    {"test_scatternd", 3, 1},
    {"test_scatternd_add", 3, 1},
    {"test_scatternd_max", 3, 1},
    {"test_scatternd_min", 3, 1},
    {"test_scatternd_multiply", 3, 1},
    {"test_sce_NCd1_mean_weight_negative_ii", 3, 1},
    {"test_sce_NCd1_mean_weight_negative_ii_expanded", 3, 1},
    {"test_sce_NCd1_mean_weight_negative_ii_log_prob", 3, 2},
    {"test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded", 3, 2},
    {"test_sce_NCd1d2d3_none_no_weight_negative_ii", 2, 1},
    {"test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded", 2, 1},
    {"test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob", 2, 2},
    {"test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded", 2, 2},
    {"test_sce_NCd1d2d3_sum_weight_high_ii", 3, 1},
    {"test_sce_NCd1d2d3_sum_weight_high_ii_expanded", 3, 1},
    {"test_sce_NCd1d2d3_sum_weight_high_ii_log_prob", 3, 2},
    {"test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded", 3, 2},
    {"test_sce_NCd1d2d3d4d5_mean_weight", 3, 1},
    {"test_sce_NCd1d2d3d4d5_mean_weight_expanded", 3, 1},
    {"test_sce_NCd1d2d3d4d5_mean_weight_log_prob", 3, 2},
    {"test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded", 3, 2},
    {"test_sce_NCd1d2d3d4d5_none_no_weight", 2, 1},
    {"test_sce_NCd1d2d3d4d5_none_no_weight_expanded", 2, 1},
    {"test_sce_NCd1d2d3d4d5_none_no_weight_log_prob", 2, 2},
    {"test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", 2, 2},
    {"test_sce_mean", 2, 1},
    {"test_sce_mean_3d", 2, 1},
    {"test_sce_mean_3d_expanded", 2, 1},
    {"test_sce_mean_3d_log_prob", 2, 2},
    {"test_sce_mean_3d_log_prob_expanded", 2, 2},
    {"test_sce_mean_expanded", 2, 1},
    {"test_sce_mean_log_prob", 2, 2},
    {"test_sce_mean_log_prob_expanded", 2, 2},
    {"test_sce_mean_no_weight_ii", 2, 1},
    {"test_sce_mean_no_weight_ii_3d", 2, 1},
    {"test_sce_mean_no_weight_ii_3d_expanded", 2, 1},
    {"test_sce_mean_no_weight_ii_3d_log_prob", 2, 2},
    {"test_sce_mean_no_weight_ii_3d_log_prob_expanded", 2, 2},
    {"test_sce_mean_no_weight_ii_4d", 2, 1},
    {"test_sce_mean_no_weight_ii_4d_expanded", 2, 1},
    {"test_sce_mean_no_weight_ii_4d_log_prob", 2, 2},
    {"test_sce_mean_no_weight_ii_4d_log_prob_expanded", 2, 2},
    {"test_sce_mean_no_weight_ii_expanded", 2, 1},
    {"test_sce_mean_no_weight_ii_log_prob", 2, 2},
    {"test_sce_mean_no_weight_ii_log_prob_expanded", 2, 2},
    {"test_sce_mean_weight", 3, 1},
    {"test_sce_mean_weight_expanded", 3, 1},
    {"test_sce_mean_weight_ii", 3, 1},
    {"test_sce_mean_weight_ii_3d", 3, 1},
    {"test_sce_mean_weight_ii_3d_expanded", 3, 1},
    {"test_sce_mean_weight_ii_3d_log_prob", 3, 2},
    {"test_sce_mean_weight_ii_3d_log_prob_expanded", 3, 2},
    {"test_sce_mean_weight_ii_4d", 3, 1},
    {"test_sce_mean_weight_ii_4d_expanded", 3, 1},
    {"test_sce_mean_weight_ii_4d_log_prob", 3, 2},
    {"test_sce_mean_weight_ii_4d_log_prob_expanded", 3, 2},
    {"test_sce_mean_weight_ii_expanded", 3, 1},
    {"test_sce_mean_weight_ii_log_prob", 3, 2},
    {"test_sce_mean_weight_ii_log_prob_expanded", 3, 2},
    {"test_sce_mean_weight_log_prob", 3, 2},
    {"test_sce_mean_weight_log_prob_expanded", 3, 2},
    {"test_sce_none", 2, 1},
    {"test_sce_none_expanded", 2, 1},
    {"test_sce_none_log_prob", 2, 2},
    {"test_sce_none_log_prob_expanded", 2, 2},
    {"test_sce_none_weights", 3, 1},
    {"test_sce_none_weights_expanded", 3, 1},
    {"test_sce_none_weights_log_prob", 3, 2},
    {"test_sce_none_weights_log_prob_expanded", 3, 2},
    {"test_sce_sum", 2, 1},
    {"test_sce_sum_expanded", 2, 1},
    {"test_sce_sum_log_prob", 2, 2},
    {"test_sce_sum_log_prob_expanded", 2, 2},
    {"test_selu", 1, 1},
    {"test_selu_default", 1, 1},
    {"test_selu_default_expanded_ver18", 1, 1},
    {"test_selu_example", 1, 1},
    {"test_selu_example_expanded_ver18", 1, 1},
    {"test_selu_expanded_ver18", 1, 1},
    {"test_sequence_insert_at_back", 2, 1},
    {"test_sequence_insert_at_front", 3, 1},
    {"test_shape", 1, 1},
    {"test_shape_clip_end", 1, 1},
    {"test_shape_clip_start", 1, 1},
    {"test_shape_end_1", 1, 1},
    {"test_shape_end_negative_1", 1, 1},
    {"test_shape_example", 1, 1},
    {"test_shape_start_1", 1, 1},
    {"test_shape_start_1_end_2", 1, 1},
    {"test_shape_start_1_end_negative_1", 1, 1},
    {"test_shape_start_negative_1", 1, 1},
    {"test_shrink_hard", 1, 1},
    {"test_shrink_soft", 1, 1},
    {"test_sigmoid", 1, 1},
    {"test_sigmoid_example", 1, 1},
    {"test_sign", 1, 1},
    {"test_simple_rnn_batchwise", 3, 2},
    {"test_simple_rnn_defaults", 3, 1},
    {"test_simple_rnn_with_initial_bias", 4, 1},
    {"test_sin", 1, 1},
    {"test_sin_example", 1, 1},
    {"test_sinh", 1, 1},
    {"test_sinh_example", 1, 1},
    {"test_size", 1, 1},
    {"test_size_example", 1, 1},
    {"test_slice", 5, 1},
    {"test_slice_default_axes", 3, 1},
    {"test_slice_default_steps", 4, 1},
    {"test_slice_end_out_of_bounds", 5, 1},
    {"test_slice_neg", 5, 1},
    {"test_slice_neg_steps", 5, 1},
    {"test_slice_negative_axes", 4, 1},
    {"test_slice_start_out_of_bounds", 5, 1},
    {"test_softmax_axis_0", 1, 1},
    {"test_softmax_axis_0_expanded", 1, 1},
    {"test_softmax_axis_1", 1, 1},
    {"test_softmax_axis_1_expanded", 1, 1},
    {"test_softmax_axis_2", 1, 1},
    {"test_softmax_axis_2_expanded", 1, 1},
    {"test_softmax_default_axis", 1, 1},
    {"test_softmax_default_axis_expanded", 1, 1},
    {"test_softmax_example", 1, 1},
    {"test_softmax_example_expanded", 1, 1},
    {"test_softmax_large_number", 1, 1},
    {"test_softmax_large_number_expanded", 1, 1},
    {"test_softmax_negative_axis", 1, 1},
    {"test_softmax_negative_axis_expanded", 1, 1},
    {"test_softplus", 1, 1},
    {"test_softplus_example", 1, 1},
    {"test_softsign", 1, 1},
    {"test_softsign_example", 1, 1},
    {"test_spacetodepth", 1, 1},
    {"test_spacetodepth_example", 1, 1},
    {"test_split_equal_parts_1d", 1, 3},
    {"test_split_equal_parts_2d", 1, 2},
    {"test_split_equal_parts_default_axis", 1, 3},
    {"test_split_variable_parts_1d", 2, 2},
    {"test_split_variable_parts_2d", 2, 2},
    {"test_split_variable_parts_default_axis", 2, 2},
    {"test_split_zero_size_splits", 2, 3},
    {"test_sqrt", 1, 1},
    {"test_sqrt_example", 1, 1},
    {"test_squeeze", 2, 1},
    {"test_squeeze_negative_axes", 2, 1},
    {"test_strnormalizer_export_monday_casesensintive_lower", 1, 1},
    {"test_strnormalizer_export_monday_casesensintive_nochangecase", 1, 1},
    {"test_strnormalizer_export_monday_casesensintive_upper", 1, 1},
    {"test_strnormalizer_export_monday_empty_output", 1, 1},
    {"test_strnormalizer_export_monday_insensintive_upper_twodim", 1, 1},
    {"test_strnormalizer_nostopwords_nochangecase", 1, 1},
    {"test_sub", 2, 1},
    {"test_sub_bcast", 2, 1},
    {"test_sub_example", 2, 1},
    {"test_sub_uint8", 2, 1},
    {"test_sum_example", 3, 1},
    {"test_sum_one_input", 1, 1},
    {"test_sum_two_inputs", 2, 1},
    {"test_tan", 1, 1},
    {"test_tan_example", 1, 1},
    {"test_tanh", 1, 1},
    {"test_tanh_example", 1, 1},
    {"test_tfidfvectorizer_tf_batch_onlybigrams_skip0", 1, 1},
    {"test_tfidfvectorizer_tf_batch_onlybigrams_skip5", 1, 1},
    {"test_tfidfvectorizer_tf_batch_uniandbigrams_skip5", 1, 1},
    {"test_tfidfvectorizer_tf_only_bigrams_skip0", 1, 1},
    {"test_tfidfvectorizer_tf_onlybigrams_levelempty", 1, 1},
    {"test_tfidfvectorizer_tf_onlybigrams_skip5", 1, 1},
    {"test_tfidfvectorizer_tf_uniandbigrams_skip5", 1, 1},
    {"test_thresholdedrelu", 1, 1},
    {"test_thresholdedrelu_default", 1, 1},
    {"test_thresholdedrelu_example", 1, 1},
    {"test_tile", 2, 1},
    {"test_tile_precomputed", 2, 1},
    {"test_top_k", 2, 2},
    {"test_top_k_negative_axis", 2, 2},
    {"test_top_k_smallest", 2, 2},
    {"test_training_dropout", 3, 1},
    {"test_training_dropout_default", 3, 1},
    {"test_training_dropout_default_mask", 3, 2},
    {"test_training_dropout_mask", 3, 2},
    {"test_training_dropout_zero_ratio", 3, 1},
    {"test_training_dropout_zero_ratio_mask", 3, 2},
    {"test_transpose_all_permutations_0", 1, 1},
    {"test_transpose_all_permutations_1", 1, 1},
    {"test_transpose_all_permutations_2", 1, 1},
    {"test_transpose_all_permutations_3", 1, 1},
    {"test_transpose_all_permutations_4", 1, 1},
    {"test_transpose_all_permutations_5", 1, 1},
    {"test_transpose_default", 1, 1},
    {"test_tril", 1, 1},
    {"test_tril_neg", 2, 1},
    {"test_tril_one_row_neg", 1, 1},
    {"test_tril_out_neg", 2, 1},
    {"test_tril_out_pos", 2, 1},
    {"test_tril_pos", 2, 1},
    {"test_tril_square", 1, 1},
    {"test_tril_square_neg", 2, 1},
    {"test_tril_zero", 2, 1},
    {"test_triu", 1, 1},
    {"test_triu_neg", 2, 1},
    {"test_triu_one_row", 2, 1},
    {"test_triu_out_neg_out", 2, 1},
    {"test_triu_out_pos", 2, 1},
    {"test_triu_pos", 2, 1},
    {"test_triu_square", 1, 1},
    {"test_triu_square_neg", 2, 1},
    {"test_triu_zero", 2, 1},
    {"test_unique_not_sorted_without_axis", 1, 4},
    {"test_unique_sorted_with_axis", 1, 4},
    {"test_unique_sorted_with_axis_3d", 1, 4},
    {"test_unique_sorted_with_negative_axis", 1, 4},
    {"test_unique_sorted_without_axis", 1, 4},
    {"test_unsqueeze_axis_0", 2, 1},
    {"test_unsqueeze_axis_1", 2, 1},
    {"test_unsqueeze_axis_2", 2, 1},
    {"test_unsqueeze_axis_3", 1, 1},
    {"test_unsqueeze_negative_axes", 2, 1},
    {"test_unsqueeze_three_axes", 2, 1},
    {"test_unsqueeze_two_axes", 2, 1},
    {"test_unsqueeze_unsorted_axes", 2, 1},
    {"test_upsample_nearest", 2, 1},
    {"test_where_example", 3, 1},
    {"test_where_long_example", 3, 1},
    {"test_xor2d", 2, 1},
    {"test_xor3d", 2, 1},
    {"test_xor4d", 2, 1},
    {"test_xor_bcast3v1d", 2, 1},
    {"test_xor_bcast3v2d", 2, 1},
    {"test_xor_bcast4v2d", 2, 1},
    {"test_xor_bcast4v3d", 2, 1},
    {"test_xor_bcast4v4d", 2, 1},
};


std::ostream& operator<<(std::ostream& os, const TestCase& test_case)
{
    return os << test_case.name;
}

typedef tuple<TestCase, tuple<Backend, Target> > ONNXConfParams;

std::string printOnnxConfParams(const testing::TestParamInfo<ONNXConfParams>& params)
{
    TestCase test_case = get<0>(params.param);
    Backend backend = get<0>(get<1>(params.param));
    Target target = get<1>(get<1>(params.param));

    std::stringstream ss;
    ss << test_case.name << "_";
    PrintTo(backend, &ss);
    ss << "_";
    PrintTo(target, &ss);

    return ss.str();
}

class Test_ONNX_conformance : public TestWithParam<ONNXConfParams>
{
public:

    TestCase test_case;
    Backend backend;
    Target target;

    double default_l1;
    double default_lInf;

    static std::set<std::string> parser_deny_list;
    static std::set<std::string> global_deny_list;
    static std::set<std::string> opencv_deny_list;
    static std::set<std::string> opencl_fp16_deny_list;
    static std::set<std::string> opencl_deny_list;
    static std::set<std::string> cpu_deny_list;
#ifdef HAVE_HALIDE
    static std::set<std::string> halide_deny_list;
#endif
#ifdef HAVE_VULKAN
    static std::set<std::string> vulkan_deny_list;
#endif
#ifdef HAVE_CUDA
    static std::set<std::string> cuda_deny_list;
    static std::set<std::string> cuda_fp16_deny_list;
#endif

    Test_ONNX_conformance()
    {
        test_case = get<0>(GetParam());
        backend = get<0>(get<1>(GetParam()));
        target = get<1>(get<1>(GetParam()));

        if (target == DNN_TARGET_CUDA_FP16 || target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD || target == DNN_TARGET_CPU_FP16)
        {
            default_l1 = 7e-3;
            default_lInf = 2e-2;
        }
        else
        {
            default_l1 = 1e-5;
            default_lInf = 1e-4;
        }
    }

    bool checkFallbacks(Net& net) const
    {
        // Check if all the layers are supported with current backend and target.
        // Some layers might be fused so their timings equal to zero.
        std::vector<double> timings;
        net.getPerfProfile(timings);
        std::vector<std::string> names = net.getLayerNames();
        CV_CheckEQ(names.size(), timings.size(), "DNN critical error");

        bool hasFallbacks = false;
        for (int i = 0; i < names.size(); ++i)
        {
            Ptr<dnn::Layer> l = net.getLayer(net.getLayerId(names[i]));
            bool fused = timings[i] == 0.;
            if ((!l->supportBackend(backend) || l->preferableTarget != target) && !fused)
            {
                hasFallbacks = true;
                std::cout << "FALLBACK: Layer [" << l->type << "]:[" << l->name << "] is expected to have backend implementation" << endl;
            }
        }
        return hasFallbacks;
    }

    static void SetUpTestCase()
    {
        parser_deny_list = {
            #include "test_onnx_conformance_layer_parser_denylist.inl.hpp"
        };

        global_deny_list = {
            #include "test_onnx_conformance_layer_filter_opencv_all_denylist.inl.hpp"
        };

        opencv_deny_list = {
            #include "test_onnx_conformance_layer_filter_opencv_denylist.inl.hpp"
        };

        opencl_fp16_deny_list = {
            #include "test_onnx_conformance_layer_filter_opencv_ocl_fp16_denylist.inl.hpp"
        };

        opencl_deny_list = {
            #include "test_onnx_conformance_layer_filter_opencv_ocl_fp32_denylist.inl.hpp"
        };

        cpu_deny_list = {
            #include "test_onnx_conformance_layer_filter_opencv_cpu_denylist.inl.hpp"
        };

#ifdef HAVE_HALIDE
        halide_deny_list = {
            #include "test_onnx_conformance_layer_filter__halide_denylist.inl.hpp"
        };
#endif

#ifdef HAVE_VULKAN
        vulkan_deny_list = {
            #include "test_onnx_conformance_layer_filter__vulkan_denylist.inl.hpp"
        };
#endif

#ifdef HAVE_CUDA
        cuda_deny_list = {
            #include "test_onnx_conformance_layer_filter__cuda_denylist.inl.hpp"
        };
        cuda_fp16_deny_list = {
            #include "test_onnx_conformance_layer_filter__cuda_fp16_denylist.inl.hpp"
        };
#endif
    }

};

std::set<std::string> Test_ONNX_conformance::parser_deny_list;
std::set<std::string> Test_ONNX_conformance::global_deny_list;
std::set<std::string> Test_ONNX_conformance::opencv_deny_list;
std::set<std::string> Test_ONNX_conformance::opencl_fp16_deny_list;
std::set<std::string> Test_ONNX_conformance::opencl_deny_list;
std::set<std::string> Test_ONNX_conformance::cpu_deny_list;
#ifdef HAVE_HALIDE
std::set<std::string> Test_ONNX_conformance::halide_deny_list;
#endif
#ifdef HAVE_VULKAN
std::set<std::string> Test_ONNX_conformance::vulkan_deny_list;
#endif
#ifdef HAVE_CUDA
std::set<std::string> Test_ONNX_conformance::cuda_deny_list;
std::set<std::string> Test_ONNX_conformance::cuda_fp16_deny_list;
#endif

TEST_P(Test_ONNX_conformance, Layer_Test)
{
    const std::string& name = test_case.name;
    ASSERT_FALSE(name.empty());

    bool checkLayersFallbacks = true;
    bool checkAccuracy = true;

    // SKIP when the test case is in the parser deny list.
    if (parser_deny_list.find(name) != parser_deny_list.end())
    {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_PARSER, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
    }

    // SKIP when the test case is in the global deny list.
    if (global_deny_list.find(name) != global_deny_list.end())
    {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_GLOBAL, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
    }

    if (backend == DNN_BACKEND_OPENCV)
    {
        if (opencv_deny_list.find(name) != opencv_deny_list.end())
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }
        if ((target == DNN_TARGET_OPENCL_FP16) && (opencl_fp16_deny_list.find(name) != opencl_fp16_deny_list.end()))
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16, CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }
        if ((target == DNN_TARGET_OPENCL) && (opencl_deny_list.find(name) != opencl_deny_list.end()))
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL, CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }
        if ((target == DNN_TARGET_CPU) && (cpu_deny_list.find(name) != cpu_deny_list.end()))
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU, CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }

        if (name == "test_pow") {
            default_lInf = 0.00013; // Expected: (normInf) <= (lInf), actual: 0.00012207 vs 0.0001
        }
        if (name == "test_gelu_tanh_1") {
            default_l1 = 0.00011; // Expected: (normL1) <= (l1), actual: 0.000101805 vs 1e-05
            default_lInf = 0.00016; // Expected: (normInf) <= (lInf), actual: 0.000152707 vs 0.0001
        }
        if (name == "test_gelu_tanh_2") {
            if (target == DNN_TARGET_OPENCL_FP16) {
                default_l1 = 0.00016; // Expected: (normL1) <= (l1), actual: 0.000157223 vs 9e-05
                default_lInf = 0.0016; // Expected: (normInf) <= (lInf), actual: 0.00153041 vs 0.0005
            } else {
                default_l1 = 9e-5; // Expected: (normL1) <= (l1), actual: 8.80073e-05 vs 1e-05
                default_lInf = 0.0005; // Expected: (normInf) <= (lInf), actual: 0.000455521 vs 0.0001
            }
        }
    }
#ifdef HAVE_HALIDE
    else if (backend == DNN_BACKEND_HALIDE)
    {
        if (halide_deny_list.find(name) != halide_deny_list.end())
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_HALIDE, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }
    }
#endif
#ifdef HAVE_INF_ENGINE
    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        #include "test_onnx_conformance_layer_filter__openvino.inl.hpp"
    }
#endif
#ifdef HAVE_VULKAN
    else if (backend == DNN_BACKEND_VKCOM)
    {
        if (vulkan_deny_list.find(name) != vulkan_deny_list.end())
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_VULKAN, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }

        if (name == "test_gelu_tanh_1") {
            default_l1 = 0.00011; // Expected: (normL1) <= (l1), actual: 0.000101805 vs 1e-05
            default_lInf = 0.00016; // Expected: (normInf) <= (lInf), actual: 0.000152707 vs 0.0001
        }
        if (name == "test_gelu_tanh_2") {
            default_l1 = 9e-5; // Expected: (normL1) <= (l1), actual: 8.80073e-05 vs 1e-05
            default_lInf = 0.0005; // Expected: (normInf) <= (lInf), actual: 0.000455521 vs 0.0001
        }
    }
#endif
#ifdef HAVE_CUDA
    else if (backend == DNN_BACKEND_CUDA)
    {
        if (target == DNN_TARGET_CUDA && cuda_deny_list.find(name) != cuda_deny_list.end())
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }
        if (target == DNN_TARGET_CUDA_FP16 && cuda_fp16_deny_list.find(name) != cuda_fp16_deny_list.end())
        {
            applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16, CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE);
        }

        if (name == "test_gelu_tanh_1") {
            default_l1 = 0.00011; // Expected: (normL1) <= (l1), actual: 0.000101815 vs 1e-05
            default_lInf = 0.00016; // Expected: (normInf) <= (lInf), actual: 0.000152737 vs 0.0001
        }
        if (name == "test_gelu_tanh_2") {
            if (target == DNN_TARGET_CUDA_FP16) {
                default_l1 = 0.00023; // Expected: (normL1) <= (l1), actual: 0.000220591 vs 9e-05
                default_lInf = 0.0023; // Expected: (normInf) <= (lInf), actual: 0.00220466 vs 0.0005
            } else {
                default_l1 = 9e-5; // Expected: (normL1) <= (l1), actual: 8.80127e-05 vs 1e-05
                default_lInf = 0.0005; // Expected: (normInf) <= (lInf), actual: 0.000455445 vs 0.0001
            }
        }
    }
#endif
    else
    {
        std::ostringstream ss;
        ss << "No test filter available for backend ";
        PrintTo(backend, &ss);
        ss << ". Run test by default";
        std::cout << ss.str() << std::endl;
    }

    std::vector<Mat> inputs;
    std::vector<Mat> ref_outputs;

    std::string prefix = cv::format("dnn/onnx/conformance/node/%s", test_case.name);

    Net net;
    try
    {
        std::string model_path = findDataFile(prefix + "/model.onnx");

        //cout << "Read ONNX inputs..." << endl;
        for (int i = 0; i < test_case.inputs; ++i)
        {
            Mat input = readTensorFromONNX(findDataFile(prefix + cv::format("/test_data_set_0/input_%d.pb", i)));
            inputs.push_back(input);
        }

        //cout << "Read ONNX reference outputs..." << endl;
        for (int i = 0; i < test_case.outputs; ++i)
        {
            Mat output = readTensorFromONNX(findDataFile(prefix + cv::format("/test_data_set_0/output_%d.pb", i)));
            ref_outputs.push_back(output);
        }

        //cout << "Parse model..." << endl;
        net = readNetFromONNX(model_path);
        if (net.empty())
        {
            applyTestTag(CV_TEST_TAG_DNN_ERROR_PARSER);
        }
    }
    catch (...)
    {
        cout << "Exception during ONNX model parse / loading input / loading reference data!" << endl;
        applyTestTag(CV_TEST_TAG_DNN_ERROR_PARSER);
        throw;
    }
    ASSERT_FALSE(net.empty());

    std::vector<std::string> inputNames;
    for (int i = 0; i < inputs.size(); ++i)
        inputNames.push_back(cv::format("%d", i));
    net.setInputsNames(inputNames);

    try
    {
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        for (int i = 0; i < inputs.size(); ++i)
        {
            net.setInput(inputs[i], inputNames[i]);
        }
    }
    catch (...)
    {
        cout << "Exception during network configuration!" << endl;
        applyTestTag(CV_TEST_TAG_DNN_ERROR_NET_SETUP);
        throw;
    }

    std::vector<std::string> layerNames = net.getUnconnectedOutLayersNames();
    std::vector<Mat> outputs;
    try
    {
        net.forward(outputs, layerNames);
    }
    catch (...)
    {
        cout << "Exception during net.forward() call!" << endl;
        applyTestTag(CV_TEST_TAG_DNN_ERROR_FORWARD);
        throw;
    }
    ASSERT_GE(outputs.size(), 1);

    if (checkLayersFallbacks && checkFallbacks(net))
    {
        applyTestTag(CV_TEST_TAG_DNN_LAYER_FALLBACK);
    }

    if (checkAccuracy)
    {
        try
        {
            if (ref_outputs.size() == 1)
            {
                // probably we found random unconnected layers.
                normAssert(ref_outputs[0], outputs[0], "", default_l1, default_lInf);
            }
            else
            {
                ASSERT_EQ(outputs.size(), ref_outputs.size());
                for (size_t i = 0; i < ref_outputs.size(); ++i)
                {
                    normAssert(ref_outputs[i], outputs[i], "", default_l1, default_lInf);
                }
            }
        }
        catch (...)
        {
            cout << "Exception during accuracy check!" << endl;
            throw;
        }
    }
    else
    {
        applyTestTag(CV_TEST_TAG_DNN_NO_ACCURACY_CHECK);
    }

    if (!HasFailure())
        cout << "Test passed!" << endl;
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_conformance,
    testing::Combine(
        testing::ValuesIn(testConformanceConfig),
        dnnBackendsAndTargets(/*withInferenceEngine=*/true, /*withHalide=*/true)
    ),
    printOnnxConfParams
);

}
