"test_averagepool_3d_default",
"test_dequantizelinear",
"test_dequantizelinear_axis",
"test_dequantizelinear_blocked",
"test_dropout_default_ratio",
"test_globalmaxpool",
"test_globalmaxpool_precomputed",
"test_logsoftmax_large_number",
"test_logsoftmax_large_number_expanded",
"test_maxpool_1d_default",
"test_maxpool_2d_ceil",
"test_maxpool_2d_default",
"test_maxpool_2d_pads",
"test_maxpool_2d_precomputed_pads",
"test_maxpool_2d_precomputed_same_upper",
"test_maxpool_2d_precomputed_strides",
"test_maxpool_2d_same_upper",
"test_maxpool_2d_strides",
"test_maxpool_3d_default",
"test_quantizelinear",
"test_quantizelinear_axis",
"test_quantizelinear_blocked",
"test_softmax_large_number",
"test_softmax_large_number_expanded",
"test_split_equal_parts_1d",
"test_split_equal_parts_2d",
"test_split_equal_parts_default_axis",
"test_tan",
"test_reduce_l2_default_axes_keepdims_example", // Expected: (normL1) <= (l1), actual: 0.00490189 vs 0.004
"test_reduce_log_sum_exp_default_axes_keepdims_example", // Expected: (normL1) <= (l1), actual: 0.00671387 vs 0.004
"test_reduce_prod_default_axes_keepdims_example", // Expected: (normL1) <= (l1), actual: inf vs 0.004
"test_reduce_prod_default_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 18.6621 vs 0.004, Expected: (normInf) <= (lInf), actual: 18.6621 vs 0.02
"test_reduce_prod_do_not_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_prod_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_prod_negative_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_sum_square_default_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.0183411 vs 0.004
"test_reduce_sum_square_do_not_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
"test_reduce_sum_square_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
"test_reduce_sum_square_negative_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
"test_scatter_elements_with_axis",
"test_scatter_elements_with_duplicate_indices",
"test_scatter_elements_with_negative_indices",
"test_scatter_elements_with_reduction_max",
"test_scatter_elements_with_reduction_min",
"test_scatter_elements_without_axis",
"test_scatter_with_axis",
"test_scatter_without_axis",
"test_scatternd",
"test_scatternd_add",
"test_scatternd_max",
"test_scatternd_min",
"test_scatternd_multiply",
