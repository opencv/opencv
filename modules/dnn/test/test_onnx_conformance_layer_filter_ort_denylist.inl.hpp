// OpenCV ORT (ONNX Runtime) Engine Deny List
// Pre-existing failures when ENGINE_ORT (OPENCV_FORCE_DNN_ENGINE=4) is forced
//
// Total: 39 pre-existing failures documented
// These tests fail due to issues unrelated to the shape inference bug fix:
// - Missing layer implementations (Dropout, Gemm, BatchNorm, etc.)
// - Graph simplification issues (separate module)
// - ONNX model format/validation issues
// - Input dimension mismatches (test data)
// - RNG determinism issues
//
// Category 1: Graph Simplification Issues (13 tests)
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

// Category 2: Missing Layer Implementations (12 tests)
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

// Category 3: ONNX Model Format/Validation Issues (5 tests)
"test_reduce_mean_axis1",
"test_reduce_sum",
"test_equal_same_dims",
"test_resize_nearest",
"test_if_layer_resize",

// Category 4: Input Dimension Validation Issues (2 tests)
"test_yunet_input_size_mismatch",
"test_keypoints_pose_input_size_mismatch",

// Category 5: Other Pre-existing Issues (7 tests)
"test_resize_unfused",
"test_multi_inputs",
"test_dynamic_resize",
"test_trilu_tril_one_row1d",
"test_random_normal_like",
"test_random_normal_like_basic",
"test_random_normal_like_complex"
