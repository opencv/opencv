"test_averagepool_3d_default",
"test_dequantizelinear",
"test_dequantizelinear_axis",
"test_dequantizelinear_blocked",
"test_logsoftmax_large_number",
"test_logsoftmax_large_number_expanded",
"test_maxpool_3d_default",
"test_pow",
"test_quantizelinear",
"test_quantizelinear_axis",
"test_quantizelinear_blocked",
"test_softmax_large_number",
"test_softmax_large_number_expanded",
"test_tan",
"test_reduce_prod_default_axes_keepdims_example", // Expected: (normL1) <= (l1), actual: inf vs 0.004
"test_reduce_prod_default_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 18.6621 vs 0.004, Expected: (normInf) <= (lInf), actual: 18.6621 vs 0.02
"test_reduce_prod_do_not_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_prod_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_prod_negative_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.00436729 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0201836 vs 0.02
"test_reduce_sum_square_default_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.0183411 vs 0.004
"test_reduce_sum_square_do_not_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
"test_reduce_sum_square_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
"test_reduce_sum_square_negative_axes_keepdims_random", // Expected: (normL1) <= (l1), actual: 0.010789 vs 0.004, Expected: (normInf) <= (lInf), actual: 0.0290298 vs 0.02
