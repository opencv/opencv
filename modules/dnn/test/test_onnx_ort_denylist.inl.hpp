// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2026, BigVision LLC, all rights reserved.

// OpenCV ORT (ONNX Runtime) Engine Deny List
// Tests are skipped when running with OPENCV_FORCE_DNN_ENGINE=ENGINE_ORT (4)

// --- ONNX conformance suite tests (Test_ONNX_conformance) ---
"test_gelu_subgraph",
"test_gelu_approximation_subgraph",
"test_layer_norm_subgraph",
"test_layer_norm_no_fusion_subgraph",
"test_softmax_subgraph",
"test_hardswish_subgraph",
"test_celu_subgraph",
"test_normalize_subgraph",
"test_batch_normalization_subgraph",
"test_expand_subgraph",
"test_mish_subgraph",
"test_attention_subgraph",
"test_biased_matmul_subgraph",
"test_dropout_default",
"test_dropout_default_mask",
"test_dropout_default_mask_ratio",
"test_dropout_default_old",
"test_dropout_default_ratio",
"test_dropout_random_old",
"test_linear",
"test_average_pool",
"test_batch_normalization",
"test_gemm_default_matrix_bias",
"test_gemm_default_no_bias",
"test_quantized_matmul",
"test_reduce_mean_axis1",
"test_reduce_sum",
"test_equal_same_dims",
"test_resize_nearest",
"test_if_layer_resize",
"test_yunet_input_size_mismatch",
"test_keypoints_pose_input_size_mismatch",
"test_resize_unfused",
"test_multi_inputs",
"test_dynamic_resize",
"test_trilu_tril_one_row1d",
"test_random_normal_like",
"test_random_normal_like_basic",
"test_random_normal_like_complex",

// --- Test_ONNX_layers / Test_ONNX_nets tests (test_onnx_importer.cpp) ---
"Dropout",
"Linear",
"ReduceMean",
"ReduceSum",
"CompareSameDims_EQ",
"AveragePooling",
"BatchNormalization",
"Multiplication",
"Constant",
"Resize",
"ResizeUnfused",
"MultyInputs",
"DynamicResize",
"trilu_tril_one_row1D",
// alexnet.onnx declares 224x224 input but reference was generated with 227x227 (Caffe-era); ORT enforces shape strictly while CLASSIC/NEW silently accept any size.
"Alexnet",
// Invalid ONNX model - missing required 'consumed_inputs' attribute in BatchNormalization (opset 1, ancient WinMLTools-exported model).
"TinyYolov2",
// Legacy ONNX spatial=0 attribute on BatchNorm causes ORT dimension mismatch; reconvert with modern exporter to fix 1D vs 3D scale error with spatial=1.
"LResNet100E_IR",
"RandomNormalLike_basic",
"RandomNormalLike_complex",
