#!/usr/bin/env python
from __future__ import print_function
import os
import re
import glob

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests, unittest

# Mirrors modules/dnn/test/test_onnx_conformance.cpp through cv2.dnn bindings, since that C++ suite never calls into Python and so can't catch binding-layer regressions.
# CONFORMANCE_TESTS matches C++'s testConformanceConfig[] exactly; OpenCV/CPU only, since bindings are backend-agnostic.
# Denylisted cases are read from OpenCV's own C++ .inl.hpp files at test time (reused, not duplicated) -- see _cpp_denylist_reasons().
# KNOWN_SKIPS is different: cases C++ passes but that fail only through the Python bindings.
CONFORMANCE_TESTS = (
    "test_abs", "test_acos", "test_acos_example",
    "test_acosh", "test_acosh_example", "test_adagrad",
    "test_adagrad_multiple", "test_adam", "test_adam_multiple",
    "test_add", "test_add_bcast", "test_add_int16",
    "test_add_int8", "test_add_uint16", "test_add_uint32",
    "test_add_uint64", "test_add_uint8", "test_affine_grid_2d",
    "test_affine_grid_2d_align_corners", "test_affine_grid_2d_align_corners_expanded", "test_affine_grid_2d_expanded",
    "test_affine_grid_3d", "test_affine_grid_3d_align_corners", "test_affine_grid_3d_align_corners_expanded",
    "test_affine_grid_3d_expanded", "test_ai_onnx_ml_array_feature_extractor", "test_ai_onnx_ml_binarizer",
    "test_ai_onnx_ml_label_encoder_string_int", "test_ai_onnx_ml_label_encoder_string_int_no_default", "test_ai_onnx_ml_label_encoder_tensor_mapping",
    "test_ai_onnx_ml_label_encoder_tensor_value_only_mapping", "test_ai_onnx_ml_tree_ensemble_set_membership", "test_ai_onnx_ml_tree_ensemble_single_tree",
    "test_and2d", "test_and3d", "test_and4d",
    "test_and_bcast3v1d", "test_and_bcast3v2d", "test_and_bcast4v2d",
    "test_and_bcast4v3d", "test_and_bcast4v4d", "test_argmax_default_axis_example",
    "test_argmax_default_axis_example_select_last_index", "test_argmax_default_axis_random", "test_argmax_default_axis_random_select_last_index",
    "test_argmax_keepdims_example", "test_argmax_keepdims_example_select_last_index", "test_argmax_keepdims_random",
    "test_argmax_keepdims_random_select_last_index", "test_argmax_negative_axis_keepdims_example", "test_argmax_negative_axis_keepdims_example_select_last_index",
    "test_argmax_negative_axis_keepdims_random", "test_argmax_negative_axis_keepdims_random_select_last_index", "test_argmax_no_keepdims_example",
    "test_argmax_no_keepdims_example_select_last_index", "test_argmax_no_keepdims_random", "test_argmax_no_keepdims_random_select_last_index",
    "test_argmin_default_axis_example", "test_argmin_default_axis_example_select_last_index", "test_argmin_default_axis_random",
    "test_argmin_default_axis_random_select_last_index", "test_argmin_keepdims_example", "test_argmin_keepdims_example_select_last_index",
    "test_argmin_keepdims_random", "test_argmin_keepdims_random_select_last_index", "test_argmin_negative_axis_keepdims_example",
    "test_argmin_negative_axis_keepdims_example_select_last_index", "test_argmin_negative_axis_keepdims_random", "test_argmin_negative_axis_keepdims_random_select_last_index",
    "test_argmin_no_keepdims_example", "test_argmin_no_keepdims_example_select_last_index", "test_argmin_no_keepdims_random",
    "test_argmin_no_keepdims_random_select_last_index", "test_asin", "test_asin_example",
    "test_asinh", "test_asinh_example", "test_atan",
    "test_atan_example", "test_atanh", "test_atanh_example",
    "test_attention_3d", "test_attention_3d_attn_mask", "test_attention_3d_attn_mask_expanded",
    "test_attention_3d_causal", "test_attention_3d_causal_expanded", "test_attention_3d_diff_heads_sizes",
    "test_attention_3d_diff_heads_sizes_attn_mask", "test_attention_3d_diff_heads_sizes_attn_mask_expanded", "test_attention_3d_diff_heads_sizes_causal",
    "test_attention_3d_diff_heads_sizes_causal_expanded", "test_attention_3d_diff_heads_sizes_expanded", "test_attention_3d_diff_heads_sizes_scaled",
    "test_attention_3d_diff_heads_sizes_scaled_expanded", "test_attention_3d_diff_heads_sizes_softcap", "test_attention_3d_diff_heads_sizes_softcap_expanded",
    "test_attention_3d_diff_heads_with_past_and_present", "test_attention_3d_diff_heads_with_past_and_present_expanded", "test_attention_3d_expanded",
    "test_attention_3d_gqa", "test_attention_3d_gqa_attn_mask", "test_attention_3d_gqa_attn_mask_expanded",
    "test_attention_3d_gqa_causal", "test_attention_3d_gqa_causal_expanded", "test_attention_3d_gqa_expanded",
    "test_attention_3d_gqa_scaled", "test_attention_3d_gqa_scaled_expanded", "test_attention_3d_gqa_softcap",
    "test_attention_3d_gqa_softcap_expanded", "test_attention_3d_gqa_with_past_and_present", "test_attention_3d_gqa_with_past_and_present_expanded",
    "test_attention_3d_scaled", "test_attention_3d_scaled_expanded", "test_attention_3d_softcap",
    "test_attention_3d_softcap_expanded", "test_attention_3d_transpose_verification", "test_attention_3d_transpose_verification_expanded",
    "test_attention_3d_with_past_and_present", "test_attention_3d_with_past_and_present_expanded", "test_attention_3d_with_past_and_present_qk_matmul",
    "test_attention_3d_with_past_and_present_qk_matmul_bias", "test_attention_3d_with_past_and_present_qk_matmul_bias_expanded", "test_attention_3d_with_past_and_present_qk_matmul_expanded",
    "test_attention_3d_with_past_and_present_qk_matmul_softcap", "test_attention_3d_with_past_and_present_qk_matmul_softcap_expanded", "test_attention_3d_with_past_and_present_qk_matmul_softmax",
    "test_attention_3d_with_past_and_present_qk_matmul_softmax_expanded", "test_attention_4d", "test_attention_4d_attn_mask",
    "test_attention_4d_attn_mask_3d", "test_attention_4d_attn_mask_3d_causal", "test_attention_4d_attn_mask_3d_causal_expanded",
    "test_attention_4d_attn_mask_3d_expanded", "test_attention_4d_attn_mask_4d", "test_attention_4d_attn_mask_4d_causal",
    "test_attention_4d_attn_mask_4d_causal_expanded", "test_attention_4d_attn_mask_4d_expanded", "test_attention_4d_attn_mask_bool",
    "test_attention_4d_attn_mask_bool_4d", "test_attention_4d_attn_mask_bool_4d_expanded", "test_attention_4d_attn_mask_bool_expanded",
    "test_attention_4d_attn_mask_expanded", "test_attention_4d_causal", "test_attention_4d_causal_expanded",
    "test_attention_4d_diff_heads_mask4d_padded_kv", "test_attention_4d_diff_heads_mask4d_padded_kv_expanded", "test_attention_4d_diff_heads_sizes",
    "test_attention_4d_diff_heads_sizes_attn_mask", "test_attention_4d_diff_heads_sizes_attn_mask_expanded", "test_attention_4d_diff_heads_sizes_causal",
    "test_attention_4d_diff_heads_sizes_causal_expanded", "test_attention_4d_diff_heads_sizes_expanded", "test_attention_4d_diff_heads_sizes_scaled",
    "test_attention_4d_diff_heads_sizes_scaled_expanded", "test_attention_4d_diff_heads_sizes_softcap", "test_attention_4d_diff_heads_sizes_softcap_expanded",
    "test_attention_4d_diff_heads_with_past_and_present", "test_attention_4d_diff_heads_with_past_and_present_expanded", "test_attention_4d_diff_heads_with_past_and_present_mask3d",
    "test_attention_4d_diff_heads_with_past_and_present_mask3d_expanded", "test_attention_4d_diff_heads_with_past_and_present_mask4d", "test_attention_4d_diff_heads_with_past_and_present_mask4d_expanded",
    "test_attention_4d_expanded", "test_attention_4d_fp16", "test_attention_4d_fp16_expanded",
    "test_attention_4d_gqa", "test_attention_4d_gqa_attn_mask", "test_attention_4d_gqa_attn_mask_expanded",
    "test_attention_4d_gqa_causal", "test_attention_4d_gqa_causal_expanded", "test_attention_4d_gqa_expanded",
    "test_attention_4d_gqa_scaled", "test_attention_4d_gqa_scaled_expanded", "test_attention_4d_gqa_softcap",
    "test_attention_4d_gqa_softcap_expanded", "test_attention_4d_gqa_with_past_and_present", "test_attention_4d_gqa_with_past_and_present_expanded",
    "test_attention_4d_gqa_with_past_and_present_fp16", "test_attention_4d_gqa_with_past_and_present_fp16_expanded", "test_attention_4d_scaled",
    "test_attention_4d_scaled_expanded", "test_attention_4d_softcap", "test_attention_4d_softcap_expanded",
    "test_attention_4d_softcap_neginf_mask", "test_attention_4d_softcap_neginf_mask_expanded", "test_attention_4d_softcap_neginf_mask_poison",
    "test_attention_4d_softcap_neginf_mask_poison_expanded", "test_attention_4d_with_past_and_present", "test_attention_4d_with_past_and_present_expanded",
    "test_attention_4d_with_past_and_present_qk_matmul", "test_attention_4d_with_past_and_present_qk_matmul_bias", "test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask",
    "test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal", "test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal_expanded", "test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_expanded",
    "test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask", "test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal", "test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal_expanded",
    "test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_expanded", "test_attention_4d_with_past_and_present_qk_matmul_bias_expanded", "test_attention_4d_with_past_and_present_qk_matmul_expanded",
    "test_attention_4d_with_qk_matmul", "test_attention_4d_with_qk_matmul_bias", "test_attention_4d_with_qk_matmul_bias_expanded",
    "test_attention_4d_with_qk_matmul_expanded", "test_attention_4d_with_qk_matmul_softcap", "test_attention_4d_with_qk_matmul_softcap_expanded",
    "test_attention_4d_with_qk_matmul_softmax", "test_attention_4d_with_qk_matmul_softmax_expanded", "test_averagepool_1d_default",
    "test_averagepool_2d_ceil", "test_averagepool_2d_ceil_last_window_starts_on_pad", "test_averagepool_2d_default",
    "test_averagepool_2d_dilations", "test_averagepool_2d_pads", "test_averagepool_2d_pads_count_include_pad",
    "test_averagepool_2d_precomputed_pads", "test_averagepool_2d_precomputed_pads_count_include_pad", "test_averagepool_2d_precomputed_same_upper",
    "test_averagepool_2d_precomputed_strides", "test_averagepool_2d_same_lower", "test_averagepool_2d_same_upper",
    "test_averagepool_2d_strides", "test_averagepool_3d_default", "test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_False",
    "test_averagepool_3d_dilations_large_count_include_pad_is_0_ceil_mode_is_True", "test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False", "test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True",
    "test_averagepool_3d_dilations_small", "test_basic_conv_with_padding", "test_basic_conv_without_padding",
    "test_basic_convinteger", "test_basic_deform_conv_with_padding", "test_basic_deform_conv_without_padding",
    "test_batchnorm_epsilon", "test_batchnorm_epsilon_training_mode", "test_batchnorm_example",
    "test_batchnorm_example_training_mode", "test_bernoulli", "test_bernoulli_double",
    "test_bernoulli_double_expanded", "test_bernoulli_expanded", "test_bernoulli_seed",
    "test_bernoulli_seed_expanded", "test_bitcast_2d_float32_to_int32", "test_bitcast_bool_to_uint8",
    "test_bitcast_float32_to_int32", "test_bitcast_float64_to_int64", "test_bitcast_int32_to_float32",
    "test_bitcast_int64_to_float64", "test_bitcast_int8_to_uint8", "test_bitcast_scalar_float32_to_int32",
    "test_bitcast_uint16_to_int16", "test_bitcast_uint32_to_int32", "test_bitshift_left_uint16",
    "test_bitshift_left_uint32", "test_bitshift_left_uint64", "test_bitshift_left_uint8",
    "test_bitshift_right_uint16", "test_bitshift_right_uint32", "test_bitshift_right_uint64",
    "test_bitshift_right_uint8", "test_bitwise_and_i16_3d", "test_bitwise_and_i32_2d",
    "test_bitwise_and_ui64_bcast_3v1d", "test_bitwise_and_ui8_bcast_4v3d", "test_bitwise_not_2d",
    "test_bitwise_not_3d", "test_bitwise_not_4d", "test_bitwise_or_i16_4d",
    "test_bitwise_or_i32_2d", "test_bitwise_or_ui64_bcast_3v1d", "test_bitwise_or_ui8_bcast_4v3d",
    "test_bitwise_xor_i16_3d", "test_bitwise_xor_i32_2d", "test_bitwise_xor_ui64_bcast_3v1d",
    "test_bitwise_xor_ui8_bcast_4v3d", "test_blackmanwindow", "test_blackmanwindow_expanded",
    "test_blackmanwindow_symmetric", "test_blackmanwindow_symmetric_expanded", "test_cast_BFLOAT16_to_FLOAT",
    "test_cast_DOUBLE_to_FLOAT", "test_cast_DOUBLE_to_FLOAT16", "test_cast_FLOAT16_to_DOUBLE",
    "test_cast_FLOAT16_to_FLOAT", "test_cast_FLOAT16_to_FLOAT4E2M1", "test_cast_FLOAT16_to_FLOAT8E4M3FN",
    "test_cast_FLOAT16_to_FLOAT8E4M3FNUZ", "test_cast_FLOAT16_to_FLOAT8E5M2", "test_cast_FLOAT16_to_FLOAT8E5M2FNUZ",
    "test_cast_FLOAT16_to_INT2", "test_cast_FLOAT16_to_INT4", "test_cast_FLOAT16_to_UINT2",
    "test_cast_FLOAT16_to_UINT4", "test_cast_FLOAT4E2M1_to_FLOAT", "test_cast_FLOAT4E2M1_to_FLOAT16",
    "test_cast_FLOAT8E4M3FNUZ_to_FLOAT", "test_cast_FLOAT8E4M3FNUZ_to_FLOAT16", "test_cast_FLOAT8E4M3FN_to_FLOAT",
    "test_cast_FLOAT8E4M3FN_to_FLOAT16", "test_cast_FLOAT8E5M2FNUZ_to_FLOAT", "test_cast_FLOAT8E5M2FNUZ_to_FLOAT16",
    "test_cast_FLOAT8E5M2_to_FLOAT", "test_cast_FLOAT8E5M2_to_FLOAT16", "test_cast_FLOAT_to_BFLOAT16",
    "test_cast_FLOAT_to_DOUBLE", "test_cast_FLOAT_to_FLOAT16", "test_cast_FLOAT_to_FLOAT4E2M1",
    "test_cast_FLOAT_to_FLOAT8E4M3FN", "test_cast_FLOAT_to_FLOAT8E4M3FNUZ", "test_cast_FLOAT_to_FLOAT8E5M2",
    "test_cast_FLOAT_to_FLOAT8E5M2FNUZ", "test_cast_FLOAT_to_INT2", "test_cast_FLOAT_to_INT4",
    "test_cast_FLOAT_to_STRING", "test_cast_FLOAT_to_UINT2", "test_cast_FLOAT_to_UINT4",
    "test_cast_INT2_to_FLOAT", "test_cast_INT2_to_FLOAT16", "test_cast_INT2_to_INT8",
    "test_cast_INT4_to_FLOAT", "test_cast_INT4_to_FLOAT16", "test_cast_INT4_to_INT8",
    "test_cast_STRING_to_FLOAT", "test_cast_UINT2_to_FLOAT", "test_cast_UINT2_to_FLOAT16",
    "test_cast_UINT2_to_UINT8", "test_cast_UINT4_to_FLOAT", "test_cast_UINT4_to_FLOAT16",
    "test_cast_UINT4_to_UINT8", "test_cast_e8m0_FLOAT16_to_FLOAT8E8M0", "test_cast_e8m0_FLOAT8E8M0_to_FLOAT",
    "test_cast_e8m0_FLOAT8E8M0_to_FLOAT16", "test_cast_e8m0_FLOAT_to_FLOAT8E8M0", "test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN",
    "test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ", "test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2", "test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN", "test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ", "test_cast_no_saturate_FLOAT_to_FLOAT8E5M2",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ", "test_castlike_BFLOAT16_to_FLOAT", "test_castlike_BFLOAT16_to_FLOAT_expanded",
    "test_castlike_DOUBLE_to_FLOAT", "test_castlike_DOUBLE_to_FLOAT16", "test_castlike_DOUBLE_to_FLOAT16_expanded",
    "test_castlike_DOUBLE_to_FLOAT_expanded", "test_castlike_FLOAT16_to_DOUBLE", "test_castlike_FLOAT16_to_DOUBLE_expanded",
    "test_castlike_FLOAT16_to_FLOAT", "test_castlike_FLOAT16_to_FLOAT4E2M1", "test_castlike_FLOAT16_to_FLOAT4E2M1_expanded",
    "test_castlike_FLOAT16_to_FLOAT8E4M3FN", "test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ", "test_castlike_FLOAT16_to_FLOAT8E4M3FNUZ_expanded",
    "test_castlike_FLOAT16_to_FLOAT8E4M3FN_expanded", "test_castlike_FLOAT16_to_FLOAT8E5M2", "test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ",
    "test_castlike_FLOAT16_to_FLOAT8E5M2FNUZ_expanded", "test_castlike_FLOAT16_to_FLOAT8E5M2_expanded", "test_castlike_FLOAT16_to_FLOAT_expanded",
    "test_castlike_FLOAT16_to_INT2", "test_castlike_FLOAT16_to_INT2_expanded", "test_castlike_FLOAT16_to_INT4",
    "test_castlike_FLOAT16_to_INT4_expanded", "test_castlike_FLOAT16_to_UINT2", "test_castlike_FLOAT16_to_UINT2_expanded",
    "test_castlike_FLOAT16_to_UINT4", "test_castlike_FLOAT16_to_UINT4_expanded", "test_castlike_FLOAT4E2M1_to_FLOAT",
    "test_castlike_FLOAT4E2M1_to_FLOAT16", "test_castlike_FLOAT4E2M1_to_FLOAT16_expanded", "test_castlike_FLOAT4E2M1_to_FLOAT_expanded",
    "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT", "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16", "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT16_expanded",
    "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded", "test_castlike_FLOAT8E4M3FN_to_FLOAT", "test_castlike_FLOAT8E4M3FN_to_FLOAT16",
    "test_castlike_FLOAT8E4M3FN_to_FLOAT16_expanded", "test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded", "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT",
    "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16", "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT16_expanded", "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded",
    "test_castlike_FLOAT8E5M2_to_FLOAT", "test_castlike_FLOAT8E5M2_to_FLOAT16", "test_castlike_FLOAT8E5M2_to_FLOAT16_expanded",
    "test_castlike_FLOAT8E5M2_to_FLOAT_expanded", "test_castlike_FLOAT_to_BFLOAT16", "test_castlike_FLOAT_to_BFLOAT16_expanded",
    "test_castlike_FLOAT_to_DOUBLE", "test_castlike_FLOAT_to_DOUBLE_expanded", "test_castlike_FLOAT_to_FLOAT16",
    "test_castlike_FLOAT_to_FLOAT16_expanded", "test_castlike_FLOAT_to_FLOAT4E2M1", "test_castlike_FLOAT_to_FLOAT4E2M1_expanded",
    "test_castlike_FLOAT_to_FLOAT8E4M3FN", "test_castlike_FLOAT_to_FLOAT8E4M3FNUZ", "test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded",
    "test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded", "test_castlike_FLOAT_to_FLOAT8E5M2", "test_castlike_FLOAT_to_FLOAT8E5M2FNUZ",
    "test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded", "test_castlike_FLOAT_to_FLOAT8E5M2_expanded", "test_castlike_FLOAT_to_INT2",
    "test_castlike_FLOAT_to_INT2_expanded", "test_castlike_FLOAT_to_INT4", "test_castlike_FLOAT_to_INT4_expanded",
    "test_castlike_FLOAT_to_STRING", "test_castlike_FLOAT_to_STRING_expanded", "test_castlike_FLOAT_to_UINT2",
    "test_castlike_FLOAT_to_UINT2_expanded", "test_castlike_FLOAT_to_UINT4", "test_castlike_FLOAT_to_UINT4_expanded",
    "test_castlike_INT2_to_FLOAT", "test_castlike_INT2_to_FLOAT16", "test_castlike_INT2_to_FLOAT16_expanded",
    "test_castlike_INT2_to_FLOAT_expanded", "test_castlike_INT2_to_INT8", "test_castlike_INT2_to_INT8_expanded",
    "test_castlike_INT4_to_FLOAT", "test_castlike_INT4_to_FLOAT16", "test_castlike_INT4_to_FLOAT16_expanded",
    "test_castlike_INT4_to_FLOAT_expanded", "test_castlike_INT4_to_INT8", "test_castlike_INT4_to_INT8_expanded",
    "test_castlike_STRING_to_FLOAT", "test_castlike_STRING_to_FLOAT_expanded", "test_castlike_UINT2_to_FLOAT",
    "test_castlike_UINT2_to_FLOAT16", "test_castlike_UINT2_to_FLOAT16_expanded", "test_castlike_UINT2_to_FLOAT_expanded",
    "test_castlike_UINT2_to_UINT8", "test_castlike_UINT2_to_UINT8_expanded", "test_castlike_UINT4_to_FLOAT",
    "test_castlike_UINT4_to_FLOAT16", "test_castlike_UINT4_to_FLOAT16_expanded", "test_castlike_UINT4_to_FLOAT_expanded",
    "test_castlike_UINT4_to_UINT8", "test_castlike_UINT4_to_UINT8_expanded", "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN",
    "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ", "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ_expanded", "test_castlike_no_saturate_FLOAT16_to_FLOAT8E4M3FN_expanded",
    "test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2", "test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ", "test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ_expanded",
    "test_castlike_no_saturate_FLOAT16_to_FLOAT8E5M2_expanded", "test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN", "test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ",
    "test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ_expanded", "test_castlike_no_saturate_FLOAT_to_FLOAT8E4M3FN_expanded", "test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2",
    "test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ", "test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ_expanded", "test_castlike_no_saturate_FLOAT_to_FLOAT8E5M2_expanded",
    "test_causal_conv_with_state_b1_c1_degenerate", "test_causal_conv_with_state_b1_c1_degenerate_expanded", "test_causal_conv_with_state_basic",
    "test_causal_conv_with_state_basic_expanded", "test_causal_conv_with_state_decode_step", "test_causal_conv_with_state_decode_step_expanded",
    "test_causal_conv_with_state_fp16", "test_causal_conv_with_state_fp16_expanded", "test_causal_conv_with_state_kernel_size_one",
    "test_causal_conv_with_state_kernel_size_one_expanded", "test_causal_conv_with_state_short_input_no_past_state", "test_causal_conv_with_state_short_input_no_past_state_expanded",
    "test_causal_conv_with_state_silu", "test_causal_conv_with_state_silu_expanded", "test_causal_conv_with_state_silu_fp16",
    "test_causal_conv_with_state_silu_fp16_expanded", "test_causal_conv_with_state_silu_with_past_state", "test_causal_conv_with_state_silu_with_past_state_expanded",
    "test_causal_conv_with_state_swish_alias", "test_causal_conv_with_state_swish_alias_expanded", "test_causal_conv_with_state_with_bias",
    "test_causal_conv_with_state_with_bias_and_past_state", "test_causal_conv_with_state_with_bias_and_past_state_expanded", "test_causal_conv_with_state_with_bias_expanded",
    "test_causal_conv_with_state_with_past_state", "test_causal_conv_with_state_with_past_state_expanded", "test_ceil",
    "test_ceil_example", "test_celu", "test_celu_expanded",
    "test_center_crop_pad_crop", "test_center_crop_pad_crop_and_pad", "test_center_crop_pad_crop_and_pad_expanded",
    "test_center_crop_pad_crop_axes_chw", "test_center_crop_pad_crop_axes_chw_expanded", "test_center_crop_pad_crop_axes_hwc",
    "test_center_crop_pad_crop_axes_hwc_expanded", "test_center_crop_pad_crop_expanded", "test_center_crop_pad_crop_negative_axes_hwc",
    "test_center_crop_pad_crop_negative_axes_hwc_expanded", "test_center_crop_pad_pad", "test_center_crop_pad_pad_expanded",
    "test_clip", "test_clip_default_inbounds", "test_clip_default_inbounds_expanded",
    "test_clip_default_int8_inbounds", "test_clip_default_int8_inbounds_expanded", "test_clip_default_int8_max",
    "test_clip_default_int8_max_expanded", "test_clip_default_int8_min", "test_clip_default_int8_min_expanded",
    "test_clip_default_max", "test_clip_default_max_expanded", "test_clip_default_min",
    "test_clip_default_min_expanded", "test_clip_example", "test_clip_example_expanded",
    "test_clip_expanded", "test_clip_inbounds", "test_clip_inbounds_expanded",
    "test_clip_min_greater_than_max", "test_clip_min_greater_than_max_expanded", "test_clip_outbounds",
    "test_clip_outbounds_expanded", "test_clip_splitbounds", "test_clip_splitbounds_expanded",
    "test_col2im", "test_col2im_5d", "test_col2im_dilations",
    "test_col2im_pads", "test_col2im_strides", "test_compress_0",
    "test_compress_1", "test_compress_default_axis", "test_compress_negative_axis",
    "test_concat_1d_axis_0", "test_concat_1d_axis_negative_1", "test_concat_2d_axis_0",
    "test_concat_2d_axis_1", "test_concat_2d_axis_negative_1", "test_concat_2d_axis_negative_2",
    "test_concat_3d_axis_0", "test_concat_3d_axis_1", "test_concat_3d_axis_2",
    "test_concat_3d_axis_negative_1", "test_concat_3d_axis_negative_2", "test_concat_3d_axis_negative_3",
    "test_constant", "test_constant_pad", "test_constant_pad_axes",
    "test_constant_pad_negative_axes", "test_constantofshape_float_ones", "test_constantofshape_int_shape_zero",
    "test_constantofshape_int_zeros", "test_conv_with_autopad_same", "test_conv_with_strides_and_asymmetric_padding",
    "test_conv_with_strides_no_padding", "test_conv_with_strides_padding", "test_convinteger_with_padding",
    "test_convinteger_without_padding", "test_convtranspose", "test_convtranspose_1d",
    "test_convtranspose_3d", "test_convtranspose_autopad_same", "test_convtranspose_dilations",
    "test_convtranspose_group_2", "test_convtranspose_group_2_image_3", "test_convtranspose_kernel_shape",
    "test_convtranspose_output_shape", "test_convtranspose_pad", "test_convtranspose_pads",
    "test_convtranspose_with_kernel", "test_cos", "test_cos_example",
    "test_cosh", "test_cosh_example", "test_cumprod_1d",
    "test_cumprod_1d_exclusive", "test_cumprod_1d_int32_exclusive", "test_cumprod_1d_reverse",
    "test_cumprod_1d_reverse_exclusive", "test_cumprod_2d_axis_0", "test_cumprod_2d_axis_1",
    "test_cumprod_2d_int32", "test_cumprod_2d_negative_axis", "test_cumsum_1d",
    "test_cumsum_1d_exclusive", "test_cumsum_1d_int32_exclusive", "test_cumsum_1d_reverse",
    "test_cumsum_1d_reverse_exclusive", "test_cumsum_2d_axis_0", "test_cumsum_2d_axis_1",
    "test_cumsum_2d_int32", "test_cumsum_2d_negative_axis", "test_deform_conv_with_mask_bias",
    "test_deform_conv_with_multiple_offset_groups", "test_depthtospace_crd_mode", "test_depthtospace_crd_mode_example",
    "test_depthtospace_dcr_mode", "test_depthtospace_example", "test_dequantizelinear",
    "test_dequantizelinear_axis", "test_dequantizelinear_blocked", "test_dequantizelinear_e4m3fn",
    "test_dequantizelinear_e4m3fn_float16", "test_dequantizelinear_e4m3fn_zero_point", "test_dequantizelinear_e5m2",
    "test_dequantizelinear_float4e2m1", "test_dequantizelinear_int16", "test_dequantizelinear_int2",
    "test_dequantizelinear_int4", "test_dequantizelinear_uint16", "test_dequantizelinear_uint2",
    "test_dequantizelinear_uint4", "test_det_2d", "test_det_nd",
    "test_dft", "test_dft_axis", "test_dft_axis_opset19",
    "test_dft_inverse", "test_dft_inverse_opset19", "test_dft_irfft",
    "test_dft_irfft_opset19", "test_dft_opset19", "test_dft_rfft",
    "test_dft_rfft_opset19", "test_div", "test_div_bcast",
    "test_div_example", "test_div_int16", "test_div_int32_trunc",
    "test_div_int8", "test_div_uint16", "test_div_uint32",
    "test_div_uint64", "test_div_uint8", "test_dropout_default",
    "test_dropout_default_mask", "test_dropout_default_mask_ratio", "test_dropout_default_old",
    "test_dropout_default_ratio", "test_dropout_random_old", "test_dynamicquantizelinear",
    "test_dynamicquantizelinear_expanded", "test_dynamicquantizelinear_max_adjusted", "test_dynamicquantizelinear_max_adjusted_expanded",
    "test_dynamicquantizelinear_min_adjusted", "test_dynamicquantizelinear_min_adjusted_expanded", "test_edge_pad",
    "test_einsum_batch_diagonal", "test_einsum_batch_matmul", "test_einsum_inner_prod",
    "test_einsum_scalar", "test_einsum_sum", "test_einsum_transpose",
    "test_elu", "test_elu_default", "test_elu_default_expanded_ver18",
    "test_elu_example", "test_elu_example_expanded_ver18", "test_elu_expanded_ver18",
    "test_equal", "test_equal_bcast", "test_equal_int16",
    "test_equal_int8", "test_equal_string", "test_equal_string_broadcast",
    "test_equal_uint16", "test_equal_uint32", "test_equal_uint64",
    "test_equal_uint8", "test_erf", "test_exp",
    "test_exp_example", "test_expand_dim_changed", "test_expand_dim_unchanged",
    "test_eyelike_populate_off_main_diagonal", "test_eyelike_with_dtype", "test_eyelike_without_dtype",
    "test_flatten_axis0", "test_flatten_axis1", "test_flatten_axis2",
    "test_flatten_axis3", "test_flatten_default_axis", "test_flatten_negative_axis1",
    "test_flatten_negative_axis2", "test_flatten_negative_axis3", "test_flatten_negative_axis4",
    "test_flexattention", "test_flexattention_causal_mask", "test_flexattention_causal_mask_expanded_ver26",
    "test_flexattention_diff_head_sizes", "test_flexattention_diff_head_sizes_expanded_ver26", "test_flexattention_double",
    "test_flexattention_double_expanded_ver26", "test_flexattention_expanded_ver26", "test_flexattention_fp16",
    "test_flexattention_fp16_expanded_ver26", "test_flexattention_gqa", "test_flexattention_gqa_expanded_ver26",
    "test_flexattention_prob_mod", "test_flexattention_prob_mod_expanded_ver26", "test_flexattention_relative_positional",
    "test_flexattention_relative_positional_expanded_ver26", "test_flexattention_scaled", "test_flexattention_scaled_expanded_ver26",
    "test_flexattention_score_mod", "test_flexattention_score_mod_expanded_ver26", "test_flexattention_soft_cap",
    "test_flexattention_soft_cap_expanded_ver26", "test_floor", "test_floor_example",
    "test_gather_0", "test_gather_1", "test_gather_2d_indices",
    "test_gather_elements_0", "test_gather_elements_1", "test_gather_elements_negative_indices",
    "test_gather_negative_indices", "test_gathernd_example_float32", "test_gathernd_example_int32",
    "test_gathernd_example_int32_batch_dim1", "test_gelu_default_1", "test_gelu_default_1_expanded",
    "test_gelu_default_2", "test_gelu_default_2_expanded", "test_gelu_tanh_1",
    "test_gelu_tanh_1_expanded", "test_gelu_tanh_2", "test_gelu_tanh_2_expanded",
    "test_gemm_all_attributes", "test_gemm_alpha", "test_gemm_beta",
    "test_gemm_default_matrix_bias", "test_gemm_default_no_bias", "test_gemm_default_scalar_bias",
    "test_gemm_default_single_elem_vector_bias", "test_gemm_default_vector_bias", "test_gemm_default_zero_bias",
    "test_gemm_transposeA", "test_gemm_transposeB", "test_globalaveragepool",
    "test_globalaveragepool_precomputed", "test_globalmaxpool", "test_globalmaxpool_precomputed",
    "test_greater", "test_greater_bcast", "test_greater_equal",
    "test_greater_equal_bcast", "test_greater_equal_bcast_expanded", "test_greater_equal_expanded",
    "test_greater_equal_int16", "test_greater_equal_int16_expanded", "test_greater_equal_int8",
    "test_greater_equal_int8_expanded", "test_greater_equal_uint16", "test_greater_equal_uint16_expanded",
    "test_greater_equal_uint32", "test_greater_equal_uint32_expanded", "test_greater_equal_uint64",
    "test_greater_equal_uint64_expanded", "test_greater_equal_uint8", "test_greater_equal_uint8_expanded",
    "test_greater_int16", "test_greater_int8", "test_greater_uint16",
    "test_greater_uint32", "test_greater_uint64", "test_greater_uint8",
    "test_gridsample", "test_gridsample_aligncorners_true", "test_gridsample_bicubic",
    "test_gridsample_bicubic_align_corners_0_additional_1", "test_gridsample_bicubic_align_corners_1_additional_1", "test_gridsample_bilinear",
    "test_gridsample_bilinear_align_corners_0_additional_1", "test_gridsample_bilinear_align_corners_1_additional_1", "test_gridsample_border_padding",
    "test_gridsample_nearest", "test_gridsample_nearest_align_corners_0_additional_1", "test_gridsample_nearest_align_corners_1_additional_1",
    "test_gridsample_reflection_padding", "test_gridsample_volumetric_bilinear_align_corners_0", "test_gridsample_volumetric_bilinear_align_corners_1",
    "test_gridsample_volumetric_nearest_align_corners_0", "test_gridsample_volumetric_nearest_align_corners_1", "test_gridsample_zeros_padding",
    "test_group_normalization_epsilon", "test_group_normalization_epsilon_expanded", "test_group_normalization_example",
    "test_group_normalization_example_expanded", "test_gru_batchwise", "test_gru_defaults",
    "test_gru_seq_length", "test_gru_with_initial_bias", "test_hammingwindow",
    "test_hammingwindow_expanded", "test_hammingwindow_symmetric", "test_hammingwindow_symmetric_expanded",
    "test_hannwindow", "test_hannwindow_expanded", "test_hannwindow_symmetric",
    "test_hannwindow_symmetric_expanded", "test_hardmax_axis_0", "test_hardmax_axis_1",
    "test_hardmax_axis_2", "test_hardmax_default_axis", "test_hardmax_example",
    "test_hardmax_negative_axis", "test_hardmax_one_hot", "test_hardsigmoid",
    "test_hardsigmoid_default", "test_hardsigmoid_default_expanded_ver18", "test_hardsigmoid_example",
    "test_hardsigmoid_example_expanded_ver18", "test_hardsigmoid_expanded_ver18", "test_hardswish",
    "test_hardswish_expanded", "test_identity", "test_identity_opt",
    "test_identity_sequence", "test_if", "test_if_opt",
    "test_if_seq", "test_image_decoder_decode_bmp_rgb", "test_image_decoder_decode_jpeg2k_rgb",
    "test_image_decoder_decode_jpeg_bgr", "test_image_decoder_decode_jpeg_grayscale", "test_image_decoder_decode_jpeg_rgb",
    "test_image_decoder_decode_png_rgb", "test_image_decoder_decode_pnm_rgb", "test_image_decoder_decode_tiff_rgb",
    "test_image_decoder_decode_webp_rgb", "test_instancenorm_epsilon", "test_instancenorm_example",
    "test_isinf", "test_isinf_float16", "test_isinf_negative",
    "test_isinf_positive", "test_isnan", "test_isnan_float16",
    "test_l1normalization_axis_0", "test_l1normalization_axis_1", "test_l1normalization_axis_last",
    "test_l2normalization_axis_0", "test_l2normalization_axis_1", "test_layer_normalization_2d_axis0",
    "test_layer_normalization_2d_axis0_expanded", "test_layer_normalization_2d_axis0_expanded_ver18", "test_layer_normalization_2d_axis1",
    "test_layer_normalization_2d_axis1_expanded", "test_layer_normalization_2d_axis1_expanded_ver18", "test_layer_normalization_2d_axis_negative_1",
    "test_layer_normalization_2d_axis_negative_1_expanded", "test_layer_normalization_2d_axis_negative_1_expanded_ver18", "test_layer_normalization_2d_axis_negative_2",
    "test_layer_normalization_2d_axis_negative_2_expanded", "test_layer_normalization_2d_axis_negative_2_expanded_ver18", "test_layer_normalization_3d_axis0_epsilon",
    "test_layer_normalization_3d_axis0_epsilon_expanded", "test_layer_normalization_3d_axis0_epsilon_expanded_ver18", "test_layer_normalization_3d_axis1_epsilon",
    "test_layer_normalization_3d_axis1_epsilon_expanded", "test_layer_normalization_3d_axis1_epsilon_expanded_ver18", "test_layer_normalization_3d_axis2_epsilon",
    "test_layer_normalization_3d_axis2_epsilon_expanded", "test_layer_normalization_3d_axis2_epsilon_expanded_ver18", "test_layer_normalization_3d_axis_negative_1_epsilon",
    "test_layer_normalization_3d_axis_negative_1_epsilon_expanded", "test_layer_normalization_3d_axis_negative_1_epsilon_expanded_ver18", "test_layer_normalization_3d_axis_negative_2_epsilon",
    "test_layer_normalization_3d_axis_negative_2_epsilon_expanded", "test_layer_normalization_3d_axis_negative_2_epsilon_expanded_ver18", "test_layer_normalization_3d_axis_negative_3_epsilon",
    "test_layer_normalization_3d_axis_negative_3_epsilon_expanded", "test_layer_normalization_3d_axis_negative_3_epsilon_expanded_ver18", "test_layer_normalization_4d_axis0",
    "test_layer_normalization_4d_axis0_expanded", "test_layer_normalization_4d_axis0_expanded_ver18", "test_layer_normalization_4d_axis1",
    "test_layer_normalization_4d_axis1_expanded", "test_layer_normalization_4d_axis1_expanded_ver18", "test_layer_normalization_4d_axis2",
    "test_layer_normalization_4d_axis2_expanded", "test_layer_normalization_4d_axis2_expanded_ver18", "test_layer_normalization_4d_axis3",
    "test_layer_normalization_4d_axis3_expanded", "test_layer_normalization_4d_axis3_expanded_ver18", "test_layer_normalization_4d_axis_negative_1",
    "test_layer_normalization_4d_axis_negative_1_expanded", "test_layer_normalization_4d_axis_negative_1_expanded_ver18", "test_layer_normalization_4d_axis_negative_2",
    "test_layer_normalization_4d_axis_negative_2_expanded", "test_layer_normalization_4d_axis_negative_2_expanded_ver18", "test_layer_normalization_4d_axis_negative_3",
    "test_layer_normalization_4d_axis_negative_3_expanded", "test_layer_normalization_4d_axis_negative_3_expanded_ver18", "test_layer_normalization_4d_axis_negative_4",
    "test_layer_normalization_4d_axis_negative_4_expanded", "test_layer_normalization_4d_axis_negative_4_expanded_ver18", "test_layer_normalization_default_axis",
    "test_layer_normalization_default_axis_expanded", "test_layer_normalization_default_axis_expanded_ver18", "test_leakyrelu",
    "test_leakyrelu_default", "test_leakyrelu_default_expanded", "test_leakyrelu_example",
    "test_leakyrelu_example_expanded", "test_leakyrelu_expanded", "test_less",
    "test_less_bcast", "test_less_equal", "test_less_equal_bcast",
    "test_less_equal_bcast_expanded", "test_less_equal_expanded", "test_less_equal_int16",
    "test_less_equal_int16_expanded", "test_less_equal_int8", "test_less_equal_int8_expanded",
    "test_less_equal_uint16", "test_less_equal_uint16_expanded", "test_less_equal_uint32",
    "test_less_equal_uint32_expanded", "test_less_equal_uint64", "test_less_equal_uint64_expanded",
    "test_less_equal_uint8", "test_less_equal_uint8_expanded", "test_less_int16",
    "test_less_int8", "test_less_uint16", "test_less_uint32",
    "test_less_uint64", "test_less_uint8", "test_linear_attention_decode_step",
    "test_linear_attention_decode_step_expanded", "test_linear_attention_delta", "test_linear_attention_delta_expanded",
    "test_linear_attention_explicit_scale", "test_linear_attention_explicit_scale_expanded", "test_linear_attention_fp16",
    "test_linear_attention_fp16_expanded", "test_linear_attention_gated", "test_linear_attention_gated_delta",
    "test_linear_attention_gated_delta_beta_scalar", "test_linear_attention_gated_delta_beta_scalar_expanded", "test_linear_attention_gated_delta_expanded",
    "test_linear_attention_gated_delta_gqa", "test_linear_attention_gated_delta_gqa_expanded", "test_linear_attention_gated_delta_mqa",
    "test_linear_attention_gated_delta_mqa_expanded", "test_linear_attention_gated_expanded", "test_linear_attention_gated_per_head_decay",
    "test_linear_attention_gated_per_head_decay_expanded", "test_linear_attention_linear", "test_linear_attention_linear_expanded",
    "test_linear_attention_linear_t1_no_past", "test_linear_attention_linear_t1_no_past_expanded", "test_linear_attention_no_past_explicit_zeros",
    "test_linear_attention_no_past_explicit_zeros_expanded", "test_linear_attention_prefill_with_past", "test_linear_attention_prefill_with_past_expanded",
    "test_log", "test_log_example", "test_logsoftmax_axis_0",
    "test_logsoftmax_axis_0_expanded", "test_logsoftmax_axis_0_expanded_ver18", "test_logsoftmax_axis_1",
    "test_logsoftmax_axis_1_expanded", "test_logsoftmax_axis_1_expanded_ver18", "test_logsoftmax_axis_2",
    "test_logsoftmax_axis_2_expanded", "test_logsoftmax_axis_2_expanded_ver18", "test_logsoftmax_default_axis",
    "test_logsoftmax_default_axis_expanded", "test_logsoftmax_default_axis_expanded_ver18", "test_logsoftmax_example_1",
    "test_logsoftmax_example_1_expanded", "test_logsoftmax_example_1_expanded_ver18", "test_logsoftmax_large_number",
    "test_logsoftmax_large_number_expanded", "test_logsoftmax_large_number_expanded_ver18", "test_logsoftmax_negative_axis",
    "test_logsoftmax_negative_axis_expanded", "test_logsoftmax_negative_axis_expanded_ver18", "test_loop11",
    "test_loop13_seq", "test_loop16_seq_none", "test_lpnormalization_default",
    "test_lppool_1d_default", "test_lppool_2d_default", "test_lppool_2d_dilations",
    "test_lppool_2d_pads", "test_lppool_2d_same_lower", "test_lppool_2d_same_upper",
    "test_lppool_2d_strides", "test_lppool_3d_default", "test_lrn",
    "test_lrn_default", "test_lstm_batchwise", "test_lstm_defaults",
    "test_lstm_with_initial_bias", "test_lstm_with_peepholes", "test_matmul_1d_1d",
    "test_matmul_1d_3d", "test_matmul_2d", "test_matmul_3d",
    "test_matmul_4d", "test_matmul_4d_1d", "test_matmul_bcast",
    "test_matmulinteger", "test_max_example", "test_max_float16",
    "test_max_float32", "test_max_float64", "test_max_int16",
    "test_max_int32", "test_max_int64", "test_max_int8",
    "test_max_one_input", "test_max_two_inputs", "test_max_uint16",
    "test_max_uint32", "test_max_uint64", "test_max_uint8",
    "test_maxpool_1d_default", "test_maxpool_2d_ceil", "test_maxpool_2d_ceil_output_size_reduce_by_one",
    "test_maxpool_2d_default", "test_maxpool_2d_dilations", "test_maxpool_2d_pads",
    "test_maxpool_2d_precomputed_pads", "test_maxpool_2d_precomputed_same_upper", "test_maxpool_2d_precomputed_strides",
    "test_maxpool_2d_same_lower", "test_maxpool_2d_same_upper", "test_maxpool_2d_strides",
    "test_maxpool_2d_uint8", "test_maxpool_3d_default", "test_maxpool_3d_dilations",
    "test_maxpool_3d_dilations_use_ref_impl", "test_maxpool_3d_dilations_use_ref_impl_large", "test_maxpool_with_argmax_2d_precomputed_pads",
    "test_maxpool_with_argmax_2d_precomputed_strides", "test_maxunpool_export_with_output_shape", "test_maxunpool_export_without_output_shape",
    "test_mean_example", "test_mean_one_input", "test_mean_two_inputs",
    "test_melweightmatrix", "test_min_example", "test_min_float16",
    "test_min_float32", "test_min_float64", "test_min_int16",
    "test_min_int32", "test_min_int64", "test_min_int8",
    "test_min_one_input", "test_min_two_inputs", "test_min_uint16",
    "test_min_uint32", "test_min_uint64", "test_min_uint8",
    "test_mish", "test_mish_expanded", "test_mod_broadcast",
    "test_mod_int64_fmod", "test_mod_mixed_sign_float16", "test_mod_mixed_sign_float32",
    "test_mod_mixed_sign_float64", "test_mod_mixed_sign_int16", "test_mod_mixed_sign_int32",
    "test_mod_mixed_sign_int64", "test_mod_mixed_sign_int8", "test_mod_uint16",
    "test_mod_uint32", "test_mod_uint64", "test_mod_uint8",
    "test_momentum", "test_momentum_multiple", "test_mul",
    "test_mul_bcast", "test_mul_example", "test_mul_int16",
    "test_mul_int8", "test_mul_uint16", "test_mul_uint32",
    "test_mul_uint64", "test_mul_uint8", "test_mvn",
    "test_mvn_expanded", "test_mvn_expanded_ver18", "test_neg",
    "test_neg_example", "test_nesterov_momentum", "test_nllloss_NC",
    "test_nllloss_NC_expanded", "test_nllloss_NCd1", "test_nllloss_NCd1_expanded",
    "test_nllloss_NCd1_ii", "test_nllloss_NCd1_ii_expanded", "test_nllloss_NCd1_mean_weight_negative_ii",
    "test_nllloss_NCd1_mean_weight_negative_ii_expanded", "test_nllloss_NCd1_weight", "test_nllloss_NCd1_weight_expanded",
    "test_nllloss_NCd1_weight_ii", "test_nllloss_NCd1_weight_ii_expanded", "test_nllloss_NCd1d2",
    "test_nllloss_NCd1d2_expanded", "test_nllloss_NCd1d2_no_weight_reduction_mean_ii", "test_nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded",
    "test_nllloss_NCd1d2_reduction_mean", "test_nllloss_NCd1d2_reduction_mean_expanded", "test_nllloss_NCd1d2_reduction_sum",
    "test_nllloss_NCd1d2_reduction_sum_expanded", "test_nllloss_NCd1d2_with_weight", "test_nllloss_NCd1d2_with_weight_expanded",
    "test_nllloss_NCd1d2_with_weight_reduction_mean", "test_nllloss_NCd1d2_with_weight_reduction_mean_expanded", "test_nllloss_NCd1d2_with_weight_reduction_sum",
    "test_nllloss_NCd1d2_with_weight_reduction_sum_expanded", "test_nllloss_NCd1d2_with_weight_reduction_sum_ii", "test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded",
    "test_nllloss_NCd1d2d3_none_no_weight_negative_ii", "test_nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded", "test_nllloss_NCd1d2d3_sum_weight_high_ii",
    "test_nllloss_NCd1d2d3_sum_weight_high_ii_expanded", "test_nllloss_NCd1d2d3d4d5_mean_weight", "test_nllloss_NCd1d2d3d4d5_mean_weight_expanded",
    "test_nllloss_NCd1d2d3d4d5_none_no_weight", "test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded", "test_nonmaxsuppression_center_point_box_format",
    "test_nonmaxsuppression_flipped_coordinates", "test_nonmaxsuppression_identical_boxes", "test_nonmaxsuppression_iou_threshold_boundary",
    "test_nonmaxsuppression_limit_output_size", "test_nonmaxsuppression_single_box", "test_nonmaxsuppression_suppress_by_IOU",
    "test_nonmaxsuppression_suppress_by_IOU_and_scores", "test_nonmaxsuppression_two_batches", "test_nonmaxsuppression_two_classes",
    "test_nonzero_example", "test_not_2d", "test_not_3d",
    "test_not_4d", "test_onehot_negative_indices", "test_onehot_with_axis",
    "test_onehot_with_negative_axis", "test_onehot_without_axis", "test_optional_get_element",
    "test_optional_get_element_optional_sequence", "test_optional_get_element_optional_tensor", "test_optional_get_element_sequence",
    "test_optional_get_element_tensor", "test_optional_has_element", "test_optional_has_element_empty",
    "test_optional_has_element_empty_no_input_name_optional_input", "test_optional_has_element_empty_no_input_name_tensor_input", "test_optional_has_element_empty_no_input_optional_input",
    "test_optional_has_element_empty_no_input_tensor_input", "test_optional_has_element_empty_optional_input", "test_optional_has_element_optional_input",
    "test_optional_has_element_tensor_input", "test_or2d", "test_or3d",
    "test_or4d", "test_or_bcast3v1d", "test_or_bcast3v2d",
    "test_or_bcast4v2d", "test_or_bcast4v3d", "test_or_bcast4v4d",
    "test_pow", "test_pow_bcast_array", "test_pow_bcast_scalar",
    "test_pow_example", "test_pow_types_float32_int32", "test_pow_types_float32_int64",
    "test_pow_types_float32_uint32", "test_pow_types_float32_uint64", "test_pow_types_int",
    "test_pow_types_int32_float32", "test_pow_types_int32_int32", "test_pow_types_int64_float32",
    "test_pow_types_int64_int64", "test_prelu_broadcast", "test_prelu_broadcast_expanded",
    "test_prelu_example", "test_prelu_example_expanded", "test_qlinearconv",
    "test_qlinearmatmul_2D", "test_qlinearmatmul_2D_int8_float16", "test_qlinearmatmul_2D_int8_float32",
    "test_qlinearmatmul_2D_uint8_float16", "test_qlinearmatmul_2D_uint8_float32", "test_qlinearmatmul_3D",
    "test_qlinearmatmul_3D_int8_float16", "test_qlinearmatmul_3D_int8_float32", "test_qlinearmatmul_3D_uint8_float16",
    "test_qlinearmatmul_3D_uint8_float32", "test_quantizelinear", "test_quantizelinear_axis",
    "test_quantizelinear_blocked", "test_quantizelinear_blocked_asymmetric", "test_quantizelinear_blocked_symmetric",
    "test_quantizelinear_e4m3fn", "test_quantizelinear_e5m2", "test_quantizelinear_float4e2m1",
    "test_quantizelinear_int16", "test_quantizelinear_int2", "test_quantizelinear_int4",
    "test_quantizelinear_uint16", "test_quantizelinear_uint2", "test_quantizelinear_uint4",
    "test_range_bfloat16_type_positive_delta", "test_range_bfloat16_type_positive_delta_expanded", "test_range_float16_type_positive_delta",
    "test_range_float16_type_positive_delta_expanded", "test_range_float_type_positive_delta", "test_range_float_type_positive_delta_expanded",
    "test_range_int32_type_negative_delta", "test_range_int32_type_negative_delta_expanded", "test_reciprocal",
    "test_reciprocal_example", "test_reduce_l1_default_axes_keepdims_example", "test_reduce_l1_default_axes_keepdims_example_expanded",
    "test_reduce_l1_default_axes_keepdims_random", "test_reduce_l1_default_axes_keepdims_random_expanded", "test_reduce_l1_do_not_keepdims_example",
    "test_reduce_l1_do_not_keepdims_example_expanded", "test_reduce_l1_do_not_keepdims_random", "test_reduce_l1_do_not_keepdims_random_expanded",
    "test_reduce_l1_empty_set", "test_reduce_l1_empty_set_expanded", "test_reduce_l1_keep_dims_example",
    "test_reduce_l1_keep_dims_example_expanded", "test_reduce_l1_keep_dims_random", "test_reduce_l1_keep_dims_random_expanded",
    "test_reduce_l1_negative_axes_keep_dims_example", "test_reduce_l1_negative_axes_keep_dims_example_expanded", "test_reduce_l1_negative_axes_keep_dims_random",
    "test_reduce_l1_negative_axes_keep_dims_random_expanded", "test_reduce_l2_default_axes_keepdims_example", "test_reduce_l2_default_axes_keepdims_example_expanded",
    "test_reduce_l2_default_axes_keepdims_random", "test_reduce_l2_default_axes_keepdims_random_expanded", "test_reduce_l2_do_not_keepdims_example",
    "test_reduce_l2_do_not_keepdims_example_expanded", "test_reduce_l2_do_not_keepdims_random", "test_reduce_l2_do_not_keepdims_random_expanded",
    "test_reduce_l2_empty_set", "test_reduce_l2_empty_set_expanded", "test_reduce_l2_keep_dims_example",
    "test_reduce_l2_keep_dims_example_expanded", "test_reduce_l2_keep_dims_random", "test_reduce_l2_keep_dims_random_expanded",
    "test_reduce_l2_negative_axes_keep_dims_example", "test_reduce_l2_negative_axes_keep_dims_example_expanded", "test_reduce_l2_negative_axes_keep_dims_random",
    "test_reduce_l2_negative_axes_keep_dims_random_expanded", "test_reduce_log_sum", "test_reduce_log_sum_asc_axes",
    "test_reduce_log_sum_asc_axes_expanded", "test_reduce_log_sum_default", "test_reduce_log_sum_default_expanded",
    "test_reduce_log_sum_desc_axes", "test_reduce_log_sum_desc_axes_expanded", "test_reduce_log_sum_empty_set",
    "test_reduce_log_sum_empty_set_expanded", "test_reduce_log_sum_exp_default_axes_keepdims_example", "test_reduce_log_sum_exp_default_axes_keepdims_example_expanded",
    "test_reduce_log_sum_exp_default_axes_keepdims_random", "test_reduce_log_sum_exp_default_axes_keepdims_random_expanded", "test_reduce_log_sum_exp_do_not_keepdims_example",
    "test_reduce_log_sum_exp_do_not_keepdims_example_expanded", "test_reduce_log_sum_exp_do_not_keepdims_random", "test_reduce_log_sum_exp_do_not_keepdims_random_expanded",
    "test_reduce_log_sum_exp_empty_set", "test_reduce_log_sum_exp_empty_set_expanded", "test_reduce_log_sum_exp_keepdims_example",
    "test_reduce_log_sum_exp_keepdims_example_expanded", "test_reduce_log_sum_exp_keepdims_random", "test_reduce_log_sum_exp_keepdims_random_expanded",
    "test_reduce_log_sum_exp_negative_axes_keepdims_example", "test_reduce_log_sum_exp_negative_axes_keepdims_example_expanded", "test_reduce_log_sum_exp_negative_axes_keepdims_random",
    "test_reduce_log_sum_exp_negative_axes_keepdims_random_expanded", "test_reduce_log_sum_negative_axes", "test_reduce_log_sum_negative_axes_expanded",
    "test_reduce_max_bool_inputs", "test_reduce_max_default_axes_keepdim_example", "test_reduce_max_default_axes_keepdims_random",
    "test_reduce_max_do_not_keepdims_example", "test_reduce_max_do_not_keepdims_random", "test_reduce_max_empty_set",
    "test_reduce_max_keepdims_example", "test_reduce_max_keepdims_random", "test_reduce_max_negative_axes_keepdims_example",
    "test_reduce_max_negative_axes_keepdims_random", "test_reduce_mean_default_axes_keepdims_example", "test_reduce_mean_default_axes_keepdims_random",
    "test_reduce_mean_do_not_keepdims_example", "test_reduce_mean_do_not_keepdims_random", "test_reduce_mean_keepdims_example",
    "test_reduce_mean_keepdims_random", "test_reduce_mean_negative_axes_keepdims_example", "test_reduce_mean_negative_axes_keepdims_random",
    "test_reduce_min_bool_inputs", "test_reduce_min_default_axes_keepdims_example", "test_reduce_min_default_axes_keepdims_random",
    "test_reduce_min_do_not_keepdims_example", "test_reduce_min_do_not_keepdims_random", "test_reduce_min_empty_set",
    "test_reduce_min_keepdims_example", "test_reduce_min_keepdims_random", "test_reduce_min_negative_axes_keepdims_example",
    "test_reduce_min_negative_axes_keepdims_random", "test_reduce_prod_default_axes_keepdims_example", "test_reduce_prod_default_axes_keepdims_random",
    "test_reduce_prod_do_not_keepdims_example", "test_reduce_prod_do_not_keepdims_random", "test_reduce_prod_empty_set",
    "test_reduce_prod_keepdims_example", "test_reduce_prod_keepdims_random", "test_reduce_prod_negative_axes_keepdims_example",
    "test_reduce_prod_negative_axes_keepdims_random", "test_reduce_sum_default_axes_keepdims_example", "test_reduce_sum_default_axes_keepdims_random",
    "test_reduce_sum_do_not_keepdims_example", "test_reduce_sum_do_not_keepdims_random", "test_reduce_sum_empty_axes_input_noop",
    "test_reduce_sum_empty_axes_input_noop_example", "test_reduce_sum_empty_axes_input_noop_random", "test_reduce_sum_empty_set",
    "test_reduce_sum_empty_set_non_reduced_axis_zero", "test_reduce_sum_keepdims_example", "test_reduce_sum_keepdims_random",
    "test_reduce_sum_negative_axes_keepdims_example", "test_reduce_sum_negative_axes_keepdims_random", "test_reduce_sum_square_default_axes_keepdims_example",
    "test_reduce_sum_square_default_axes_keepdims_example_expanded", "test_reduce_sum_square_default_axes_keepdims_random", "test_reduce_sum_square_default_axes_keepdims_random_expanded",
    "test_reduce_sum_square_do_not_keepdims_example", "test_reduce_sum_square_do_not_keepdims_example_expanded", "test_reduce_sum_square_do_not_keepdims_random",
    "test_reduce_sum_square_do_not_keepdims_random_expanded", "test_reduce_sum_square_empty_set", "test_reduce_sum_square_empty_set_expanded",
    "test_reduce_sum_square_keepdims_example", "test_reduce_sum_square_keepdims_example_expanded", "test_reduce_sum_square_keepdims_random",
    "test_reduce_sum_square_keepdims_random_expanded", "test_reduce_sum_square_negative_axes_keepdims_example", "test_reduce_sum_square_negative_axes_keepdims_example_expanded",
    "test_reduce_sum_square_negative_axes_keepdims_random", "test_reduce_sum_square_negative_axes_keepdims_random_expanded", "test_reflect_pad",
    "test_regex_full_match_basic", "test_regex_full_match_email_domain", "test_regex_full_match_empty",
    "test_relu", "test_relu_expanded_ver18", "test_reshape_allowzero_reordered",
    "test_reshape_extended_dims", "test_reshape_negative_dim", "test_reshape_negative_extended_dims",
    "test_reshape_one_dim", "test_reshape_reduced_dims", "test_reshape_reordered_all_dims",
    "test_reshape_reordered_last_dims", "test_reshape_zero_and_negative_dim", "test_reshape_zero_dim",
    "test_resize_downsample_scales_cubic", "test_resize_downsample_scales_cubic_A_n0p5_exclude_outside", "test_resize_downsample_scales_cubic_align_corners",
    "test_resize_downsample_scales_cubic_antialias", "test_resize_downsample_scales_linear", "test_resize_downsample_scales_linear_align_corners",
    "test_resize_downsample_scales_linear_antialias", "test_resize_downsample_scales_linear_half_pixel_symmetric", "test_resize_downsample_scales_nearest",
    "test_resize_downsample_sizes_cubic", "test_resize_downsample_sizes_cubic_antialias", "test_resize_downsample_sizes_linear_antialias",
    "test_resize_downsample_sizes_linear_pytorch_half_pixel", "test_resize_downsample_sizes_nearest", "test_resize_downsample_sizes_nearest_not_larger",
    "test_resize_downsample_sizes_nearest_not_smaller", "test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn", "test_resize_tf_crop_and_resize",
    "test_resize_tf_crop_and_resize_axes_2_3", "test_resize_tf_crop_and_resize_axes_3_2", "test_resize_tf_crop_and_resize_extrapolation_value",
    "test_resize_upsample_scales_cubic", "test_resize_upsample_scales_cubic_A_n0p5_exclude_outside", "test_resize_upsample_scales_cubic_align_corners",
    "test_resize_upsample_scales_cubic_asymmetric", "test_resize_upsample_scales_linear", "test_resize_upsample_scales_linear_align_corners",
    "test_resize_upsample_scales_linear_half_pixel_symmetric", "test_resize_upsample_scales_nearest", "test_resize_upsample_scales_nearest_axes_2_3",
    "test_resize_upsample_scales_nearest_axes_3_2", "test_resize_upsample_sizes_cubic", "test_resize_upsample_sizes_nearest",
    "test_resize_upsample_sizes_nearest_axes_2_3", "test_resize_upsample_sizes_nearest_axes_3_2", "test_resize_upsample_sizes_nearest_ceil_half_pixel",
    "test_resize_upsample_sizes_nearest_floor_align_corners", "test_resize_upsample_sizes_nearest_not_larger", "test_resize_upsample_sizes_nearest_not_smaller",
    "test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "test_reversesequence_batch", "test_reversesequence_time",
    "test_rms_normalization_2d_axis0", "test_rms_normalization_2d_axis0_expanded", "test_rms_normalization_2d_axis1",
    "test_rms_normalization_2d_axis1_expanded", "test_rms_normalization_2d_axis_negative_1", "test_rms_normalization_2d_axis_negative_1_expanded",
    "test_rms_normalization_2d_axis_negative_2", "test_rms_normalization_2d_axis_negative_2_expanded", "test_rms_normalization_3d_axis0_epsilon",
    "test_rms_normalization_3d_axis0_epsilon_expanded", "test_rms_normalization_3d_axis1_epsilon", "test_rms_normalization_3d_axis1_epsilon_expanded",
    "test_rms_normalization_3d_axis2_epsilon", "test_rms_normalization_3d_axis2_epsilon_expanded", "test_rms_normalization_3d_axis_negative_1_epsilon",
    "test_rms_normalization_3d_axis_negative_1_epsilon_expanded", "test_rms_normalization_3d_axis_negative_2_epsilon", "test_rms_normalization_3d_axis_negative_2_epsilon_expanded",
    "test_rms_normalization_3d_axis_negative_3_epsilon", "test_rms_normalization_3d_axis_negative_3_epsilon_expanded", "test_rms_normalization_4d_axis0",
    "test_rms_normalization_4d_axis0_expanded", "test_rms_normalization_4d_axis1", "test_rms_normalization_4d_axis1_expanded",
    "test_rms_normalization_4d_axis2", "test_rms_normalization_4d_axis2_expanded", "test_rms_normalization_4d_axis3",
    "test_rms_normalization_4d_axis3_expanded", "test_rms_normalization_4d_axis_negative_1", "test_rms_normalization_4d_axis_negative_1_expanded",
    "test_rms_normalization_4d_axis_negative_2", "test_rms_normalization_4d_axis_negative_2_expanded", "test_rms_normalization_4d_axis_negative_3",
    "test_rms_normalization_4d_axis_negative_3_expanded", "test_rms_normalization_4d_axis_negative_4", "test_rms_normalization_4d_axis_negative_4_expanded",
    "test_rms_normalization_default_axis", "test_rms_normalization_default_axis_expanded", "test_rnn_seq_length",
    "test_roialign_aligned_false", "test_roialign_aligned_true", "test_roialign_mode_max",
    "test_rotary_embedding", "test_rotary_embedding_3d_input", "test_rotary_embedding_3d_input_expanded",
    "test_rotary_embedding_expanded", "test_rotary_embedding_interleaved", "test_rotary_embedding_interleaved_expanded",
    "test_rotary_embedding_no_position_ids", "test_rotary_embedding_no_position_ids_expanded", "test_rotary_embedding_no_position_ids_interleaved",
    "test_rotary_embedding_no_position_ids_interleaved_expanded", "test_rotary_embedding_no_position_ids_rotary_dim", "test_rotary_embedding_no_position_ids_rotary_dim_expanded",
    "test_rotary_embedding_with_interleaved_rotary_dim", "test_rotary_embedding_with_interleaved_rotary_dim_expanded", "test_rotary_embedding_with_rotary_dim",
    "test_rotary_embedding_with_rotary_dim_expanded", "test_round", "test_scan9_multi_state",
    "test_scan9_scalar", "test_scan9_sum", "test_scan_sum",
    "test_scatter_elements_with_axis", "test_scatter_elements_with_duplicate_indices", "test_scatter_elements_with_negative_indices",
    "test_scatter_elements_with_reduction_max", "test_scatter_elements_with_reduction_min", "test_scatter_elements_with_reduction_mul",
    "test_scatter_elements_without_axis", "test_scatter_with_axis", "test_scatter_without_axis",
    "test_scatternd", "test_scatternd_add", "test_scatternd_max",
    "test_scatternd_min", "test_scatternd_multiply", "test_sce_NCd1_mean_weight_negative_ii",
    "test_sce_NCd1_mean_weight_negative_ii_expanded", "test_sce_NCd1_mean_weight_negative_ii_log_prob", "test_sce_NCd1_mean_weight_negative_ii_log_prob_expanded",
    "test_sce_NCd1d2d3_none_no_weight_negative_ii", "test_sce_NCd1d2d3_none_no_weight_negative_ii_expanded", "test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob",
    "test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded", "test_sce_NCd1d2d3_sum_weight_high_ii", "test_sce_NCd1d2d3_sum_weight_high_ii_expanded",
    "test_sce_NCd1d2d3_sum_weight_high_ii_log_prob", "test_sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded", "test_sce_NCd1d2d3d4d5_mean_weight",
    "test_sce_NCd1d2d3d4d5_mean_weight_expanded", "test_sce_NCd1d2d3d4d5_mean_weight_log_prob", "test_sce_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
    "test_sce_NCd1d2d3d4d5_none_no_weight", "test_sce_NCd1d2d3d4d5_none_no_weight_expanded", "test_sce_NCd1d2d3d4d5_none_no_weight_log_prob",
    "test_sce_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "test_sce_mean", "test_sce_mean_3d",
    "test_sce_mean_3d_expanded", "test_sce_mean_3d_log_prob", "test_sce_mean_3d_log_prob_expanded",
    "test_sce_mean_expanded", "test_sce_mean_log_prob", "test_sce_mean_log_prob_expanded",
    "test_sce_mean_no_weight_ii", "test_sce_mean_no_weight_ii_3d", "test_sce_mean_no_weight_ii_3d_expanded",
    "test_sce_mean_no_weight_ii_3d_log_prob", "test_sce_mean_no_weight_ii_3d_log_prob_expanded", "test_sce_mean_no_weight_ii_4d",
    "test_sce_mean_no_weight_ii_4d_expanded", "test_sce_mean_no_weight_ii_4d_log_prob", "test_sce_mean_no_weight_ii_4d_log_prob_expanded",
    "test_sce_mean_no_weight_ii_expanded", "test_sce_mean_no_weight_ii_log_prob", "test_sce_mean_no_weight_ii_log_prob_expanded",
    "test_sce_mean_weight", "test_sce_mean_weight_expanded", "test_sce_mean_weight_ii",
    "test_sce_mean_weight_ii_3d", "test_sce_mean_weight_ii_3d_expanded", "test_sce_mean_weight_ii_3d_log_prob",
    "test_sce_mean_weight_ii_3d_log_prob_expanded", "test_sce_mean_weight_ii_4d", "test_sce_mean_weight_ii_4d_expanded",
    "test_sce_mean_weight_ii_4d_log_prob", "test_sce_mean_weight_ii_4d_log_prob_expanded", "test_sce_mean_weight_ii_expanded",
    "test_sce_mean_weight_ii_log_prob", "test_sce_mean_weight_ii_log_prob_expanded", "test_sce_mean_weight_log_prob",
    "test_sce_mean_weight_log_prob_expanded", "test_sce_none", "test_sce_none_expanded",
    "test_sce_none_log_prob", "test_sce_none_log_prob_expanded", "test_sce_none_weights",
    "test_sce_none_weights_expanded", "test_sce_none_weights_log_prob", "test_sce_none_weights_log_prob_expanded",
    "test_sce_sum", "test_sce_sum_expanded", "test_sce_sum_log_prob",
    "test_sce_sum_log_prob_expanded", "test_selu", "test_selu_default",
    "test_selu_default_expanded_ver18", "test_selu_example", "test_selu_example_expanded_ver18",
    "test_selu_expanded_ver18", "test_sequence_insert_at_back", "test_sequence_insert_at_front",
    "test_sequence_map_add_1_sequence_1_tensor", "test_sequence_map_add_1_sequence_1_tensor_expanded", "test_sequence_map_add_2_sequences",
    "test_sequence_map_add_2_sequences_expanded", "test_sequence_map_extract_shapes", "test_sequence_map_extract_shapes_expanded",
    "test_sequence_map_identity_1_sequence", "test_sequence_map_identity_1_sequence_1_tensor", "test_sequence_map_identity_1_sequence_1_tensor_expanded",
    "test_sequence_map_identity_1_sequence_expanded", "test_sequence_map_identity_2_sequences", "test_sequence_map_identity_2_sequences_expanded",
    "test_shape", "test_shape_clip_end", "test_shape_clip_start",
    "test_shape_end_1", "test_shape_end_negative_1", "test_shape_example",
    "test_shape_start_1", "test_shape_start_1_end_2", "test_shape_start_1_end_negative_1",
    "test_shape_start_greater_than_end", "test_shape_start_negative_1", "test_shrink_hard",
    "test_shrink_hard_expanded_ver18", "test_shrink_soft", "test_shrink_soft_expanded_ver18",
    "test_sigmoid", "test_sigmoid_example", "test_sign",
    "test_simple_rnn_batchwise", "test_simple_rnn_defaults", "test_simple_rnn_with_initial_bias",
    "test_sin", "test_sin_example", "test_sinh",
    "test_sinh_example", "test_size", "test_size_example",
    "test_slice", "test_slice_default_axes", "test_slice_default_steps",
    "test_slice_end_out_of_bounds", "test_slice_neg", "test_slice_neg_steps",
    "test_slice_negative_axes", "test_slice_start_out_of_bounds", "test_softmax_axis_0",
    "test_softmax_axis_0_expanded", "test_softmax_axis_0_expanded_ver18", "test_softmax_axis_1",
    "test_softmax_axis_1_expanded", "test_softmax_axis_1_expanded_ver18", "test_softmax_axis_2",
    "test_softmax_axis_2_expanded", "test_softmax_axis_2_expanded_ver18", "test_softmax_default_axis",
    "test_softmax_default_axis_expanded", "test_softmax_default_axis_expanded_ver18", "test_softmax_example",
    "test_softmax_example_expanded", "test_softmax_example_expanded_ver18", "test_softmax_large_number",
    "test_softmax_large_number_expanded", "test_softmax_large_number_expanded_ver18", "test_softmax_negative_axis",
    "test_softmax_negative_axis_expanded", "test_softmax_negative_axis_expanded_ver18", "test_softplus",
    "test_softplus_example", "test_softplus_example_expanded_ver18", "test_softplus_expanded_ver18",
    "test_softsign", "test_softsign_example", "test_softsign_example_expanded_ver18",
    "test_softsign_expanded_ver18", "test_spacetodepth", "test_spacetodepth_example",
    "test_split_1d_uneven_split_opset18", "test_split_2d_uneven_split_opset18", "test_split_equal_parts_1d",
    "test_split_equal_parts_1d_opset13", "test_split_equal_parts_1d_opset18", "test_split_equal_parts_2d",
    "test_split_equal_parts_2d_opset13", "test_split_equal_parts_default_axis", "test_split_equal_parts_default_axis_opset13",
    "test_split_equal_parts_default_axis_opset18", "test_split_to_sequence_1", "test_split_to_sequence_2",
    "test_split_to_sequence_nokeepdims", "test_split_variable_parts_1d", "test_split_variable_parts_1d_opset13",
    "test_split_variable_parts_1d_opset18", "test_split_variable_parts_2d", "test_split_variable_parts_2d_opset13",
    "test_split_variable_parts_2d_opset18", "test_split_variable_parts_default_axis", "test_split_variable_parts_default_axis_opset13",
    "test_split_variable_parts_default_axis_opset18", "test_split_zero_size_splits", "test_split_zero_size_splits_opset13",
    "test_split_zero_size_splits_opset18", "test_sqrt", "test_sqrt_example",
    "test_squeeze", "test_squeeze_negative_axes", "test_stft",
    "test_stft_with_window", "test_string_concat", "test_string_concat_broadcasting",
    "test_string_concat_empty_string", "test_string_concat_utf8", "test_string_concat_zero_dimensional",
    "test_string_split_basic", "test_string_split_consecutive_delimiters", "test_string_split_empty_string_delimiter",
    "test_string_split_empty_tensor", "test_string_split_maxsplit", "test_string_split_no_delimiter",
    "test_strnormalizer_export_monday_casesensintive_lower", "test_strnormalizer_export_monday_casesensintive_nochangecase", "test_strnormalizer_export_monday_casesensintive_upper",
    "test_strnormalizer_export_monday_empty_output", "test_strnormalizer_export_monday_insensintive_upper_twodim", "test_strnormalizer_nostopwords_nochangecase",
    "test_sub", "test_sub_bcast", "test_sub_example",
    "test_sub_int16", "test_sub_int8", "test_sub_uint16",
    "test_sub_uint32", "test_sub_uint64", "test_sub_uint8",
    "test_sum_example", "test_sum_one_input", "test_sum_two_inputs",
    "test_swish", "test_swish_expanded", "test_tan",
    "test_tan_example", "test_tanh", "test_tanh_example",
    "test_tensorscatter", "test_tensorscatter_3d", "test_tensorscatter_circular",
    "test_tfidfvectorizer_tf_batch_onlybigrams_skip0", "test_tfidfvectorizer_tf_batch_onlybigrams_skip5", "test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
    "test_tfidfvectorizer_tf_only_bigrams_skip0", "test_tfidfvectorizer_tf_onlybigrams_levelempty", "test_tfidfvectorizer_tf_onlybigrams_skip5",
    "test_tfidfvectorizer_tf_uniandbigrams_skip5", "test_thresholdedrelu", "test_thresholdedrelu_default",
    "test_thresholdedrelu_default_expanded_ver18", "test_thresholdedrelu_example", "test_thresholdedrelu_example_expanded_ver18",
    "test_thresholdedrelu_expanded_ver18", "test_tile", "test_tile_precomputed",
    "test_top_k", "test_top_k_negative_axis", "test_top_k_same_values",
    "test_top_k_same_values_2d", "test_top_k_same_values_largest", "test_top_k_smallest",
    "test_top_k_uint64", "test_training_dropout", "test_training_dropout_default",
    "test_training_dropout_default_mask", "test_training_dropout_mask", "test_training_dropout_zero_ratio",
    "test_training_dropout_zero_ratio_mask", "test_transpose_all_permutations_0", "test_transpose_all_permutations_1",
    "test_transpose_all_permutations_2", "test_transpose_all_permutations_3", "test_transpose_all_permutations_4",
    "test_transpose_all_permutations_5", "test_transpose_default", "test_tril",
    "test_tril_neg", "test_tril_one_row_neg", "test_tril_out_neg",
    "test_tril_out_pos", "test_tril_pos", "test_tril_square",
    "test_tril_square_neg", "test_tril_zero", "test_triu",
    "test_triu_neg", "test_triu_one_row", "test_triu_out_neg_out",
    "test_triu_out_pos", "test_triu_pos", "test_triu_square",
    "test_triu_square_neg", "test_triu_zero", "test_unique_length_1",
    "test_unique_not_sorted_without_axis", "test_unique_sorted_with_axis", "test_unique_sorted_with_axis_3d",
    "test_unique_sorted_with_negative_axis", "test_unique_sorted_without_axis", "test_unsqueeze_axis_0",
    "test_unsqueeze_axis_1", "test_unsqueeze_axis_2", "test_unsqueeze_axis_3",
    "test_unsqueeze_negative_axes", "test_unsqueeze_three_axes", "test_unsqueeze_two_axes",
    "test_unsqueeze_unsorted_axes", "test_upsample_nearest", "test_where_example",
    "test_where_long_example", "test_wrap_pad", "test_xor2d",
    "test_xor3d", "test_xor4d", "test_xor_bcast3v1d",
    "test_xor_bcast3v2d", "test_xor_bcast4v2d", "test_xor_bcast4v3d",
    "test_xor_bcast4v4d",
)

TOLERANCE_OVERRIDES = {
    "test_attention_4d_fp16": (0.0002, 0.001),
    "test_attention_4d_fp16_expanded": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16": (0.0002, 0.001),
    "test_attention_4d_gqa_with_past_and_present_fp16_expanded": (0.0002, 0.001),
    "test_causal_conv_with_state_fp16": (0.0002, 0.002),
    "test_causal_conv_with_state_silu_fp16": (0.0002, 0.002),
    "test_gelu_tanh_1": (0.00011, 0.00016),
    "test_gelu_tanh_2": (9e-05, 0.0005),
    "test_nllloss_NCd1d2_reduction_sum_expanded": (2e-05, 0.0001),
    "test_nllloss_NCd1d2d3d4d5_mean_weight_expanded": (2e-05, 0.0001),
    "test_reduce_prod_default_axes_keepdims_random": (0.002, 0.002),
    "test_reduce_sum_square_default_axes_keepdims_random": (2e-05, 0.0001),
    "test_reduce_sum_square_default_axes_keepdims_random_expanded": (2e-05, 0.0001),
    "test_roialign_aligned_false": (3e-05, 0.0001),
    "test_roialign_aligned_true": (3e-05, 0.0001),
}

# Cases the C++ suite passes but that fail only through the Python bindings.
# Not a port of the C++ denylist; empty is the expected state.
KNOWN_SKIPS = {
    # "test_name": "reason",
}

L1_DEFAULT = 1e-5
LINF_DEFAULT = 1e-4

# Only parser + opencv_all apply to a CPU-only test; classic/CUDA/Vulkan/OCL denylists don't.
_CPP_DENYLIST_FILES = (
    'test_onnx_conformance_layer_parser_denylist.inl.hpp',
    'test_onnx_conformance_layer_filter_opencv_all_denylist.inl.hpp',
)
_DENYLIST_ENTRY_RE = re.compile(r'^\s*"(test_[A-Za-z0-9_]+)"\s*,(?:\s*//\s*(.*))?', re.M)


def _cpp_denylist_reasons(repoPath):
    """Parse OpenCV's C++ ONNX denylists into {name: reason}; {} if the source tree isn't available."""
    reasons = {}
    if not repoPath:
        return reasons
    test_dir = os.path.join(repoPath, 'modules', 'dnn', 'test')
    for fname in _CPP_DENYLIST_FILES:
        fpath = os.path.join(test_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as fh:
            content = fh.read()
        for name, reason in _DENYLIST_ENTRY_RE.findall(content):
            if name not in reasons:
                reasons[name] = reason.strip() or ('excluded in %s' % fname)
    return reasons


def _pbIndex(path):
    """Sort key for input_N.pb / output_N.pb, matching the C++ test's extractIndex."""
    m = re.search(r'_(\d+)\.pb$', os.path.basename(path))
    return int(m.group(1)) if m else 0


def normAssert(test, ref, actual, msg="", l1=L1_DEFAULT, lInf=LINF_DEFAULT):
    """Python port of cv::dnn normAssert (test_common.impl.hpp): FP16->FP32 upcast, scalar/1-element equivalence, exact NaN/Inf matching, then L1/Linf tolerance."""
    ref = np.asarray(ref)
    actual = np.asarray(actual)

    if ref.dtype == np.float16:
        ref = ref.astype(np.float32)
    if actual.dtype == np.float16:
        actual = actual.astype(np.float32)

    if ref.ndim == 0 and actual.shape == (1,):
        ref = ref.reshape(1)
    elif actual.ndim == 0 and ref.shape == (1,):
        actual = actual.reshape(1)

    if ref.size == 0 or actual.size == 0:
        test.assertEqual(ref.size, actual.size, msg)
        test.assertEqual(ref.shape, actual.shape, msg)
        return

    test.assertEqual(ref.shape, actual.shape, msg)

    ref64 = ref.astype(np.float64)
    act64 = actual.astype(np.float64)

    # inf - inf = nan in IEEE arithmetic, so non-finite values need an exact-equality check, not a subtraction.
    nonfinite = ~np.isfinite(ref64) | ~np.isfinite(act64)
    if np.any(nonfinite):
        test.assertTrue(
            np.array_equal(ref64[nonfinite], act64[nonfinite], equal_nan=True),
            '%s: non-finite value mismatch (NaN/Inf) at %d position(s)' % (msg, int(nonfinite.sum())))

    diff = np.where(nonfinite, 0.0, np.abs(ref64 - act64))
    normL1 = float(diff.sum()) / diff.size
    normInf = float(diff.max())
    test.assertLessEqual(normL1, l1, '%s: normL1=%r (l1=%r)' % (msg, normL1, l1))
    test.assertLessEqual(normInf, lInf, '%s: normInf=%r (lInf=%r)' % (msg, normInf, lInf))


class onnx_conformance_test(NewOpenCVTests):

    _cppDenylistReasons = None  # lazily built once per process, class-level cache

    def setUp(self):
        super(onnx_conformance_test, self).setUp()
        if onnx_conformance_test._cppDenylistReasons is None:
            onnx_conformance_test._cppDenylistReasons = _cpp_denylist_reasons(self.repoPath)

    def find_conformance_model(self, name, required=True):
        relpath = '/'.join(['dnn', 'onnx', 'conformance', 'node', name, 'model.onnx'])
        return self.find_file(relpath, [self.extraTestDataPath], required=required)

    def runConformanceCase(self, name):
        if name in KNOWN_SKIPS:
            self.skipTest(KNOWN_SKIPS[name])

        cpp_reason = self._cppDenylistReasons.get(name)
        if cpp_reason is not None:
            self.skipTest('excluded by OpenCV\'s C++ ONNX conformance denylist: %s' % cpp_reason)

        model_path = self.find_conformance_model(name, required=False)
        node_dir = os.path.dirname(model_path)
        dataset_dir = os.path.join(node_dir, 'test_data_set_0')

        inputFiles = sorted(glob.glob(os.path.join(dataset_dir, 'input_*.pb')), key=_pbIndex)
        outputFiles = sorted(glob.glob(os.path.join(dataset_dir, 'output_*.pb')), key=_pbIndex)
        if not outputFiles:
            self.skipTest('No reference data in %s (opencv_extra out of date?)' % dataset_dir)

        inputs = [cv.dnn.readTensorFromONNX(f) for f in inputFiles]
        refs = [cv.dnn.readTensorFromONNX(f) for f in outputFiles]

        net = cv.dnn.readNetFromONNX(model_path)
        self.assertFalse(net.empty(), 'failed to parse %s' % model_path)

        inputNames = [str(i) for i in range(len(inputs))]
        net.setInputsNames(inputNames)
        for inputName, inp in zip(inputNames, inputs):
            net.setInput(inp, inputName)

        outputs = net.forward(net.getUnconnectedOutLayersNames())
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        self.assertGreaterEqual(len(outputs), len(refs),
                                 '%s: expected >= %d outputs, got %d' % (name, len(refs), len(outputs)))

        l1, lInf = TOLERANCE_OVERRIDES.get(name, (L1_DEFAULT, LINF_DEFAULT))
        for i in range(len(refs)):
            normAssert(self, refs[i], outputs[i], msg=name, l1=l1, lInf=lInf)


def _makeConformanceTest(caseName):
    def test(self):
        self.runConformanceCase(caseName)
    test.__name__ = str(caseName)
    return test


for _case_name in CONFORMANCE_TESTS:
    setattr(onnx_conformance_test, _case_name, _makeConformanceTest(_case_name))
del _case_name


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
