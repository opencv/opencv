"test_add_bcast",
"test_add_uint8",  // output type mismatch
#if 1  // output type mismatch CV_32F vs expected 32S
"test_argmax_default_axis_example",
"test_argmax_default_axis_example_select_last_index",
"test_argmax_default_axis_random",
"test_argmax_default_axis_random_select_last_index",
"test_argmax_keepdims_example",
"test_argmax_keepdims_example_select_last_index",
"test_argmax_keepdims_random",
"test_argmax_keepdims_random_select_last_index",
"test_argmax_negative_axis_keepdims_example",
"test_argmax_negative_axis_keepdims_example_select_last_index",
"test_argmax_negative_axis_keepdims_random",
"test_argmax_negative_axis_keepdims_random_select_last_index",
"test_argmax_no_keepdims_example",
"test_argmax_no_keepdims_example_select_last_index",
"test_argmax_no_keepdims_random",
"test_argmax_no_keepdims_random_select_last_index",
"test_argmin_default_axis_example",
"test_argmin_default_axis_example_select_last_index",
"test_argmin_default_axis_random",
"test_argmin_default_axis_random_select_last_index",
"test_argmin_keepdims_example",
"test_argmin_keepdims_example_select_last_index",
"test_argmin_keepdims_random",
"test_argmin_keepdims_random_select_last_index",
"test_argmin_negative_axis_keepdims_example",
"test_argmin_negative_axis_keepdims_example_select_last_index",
"test_argmin_negative_axis_keepdims_random",
"test_argmin_negative_axis_keepdims_random_select_last_index",
"test_argmin_no_keepdims_example",
"test_argmin_no_keepdims_example_select_last_index",
"test_argmin_no_keepdims_random",
"test_argmin_no_keepdims_random_select_last_index",
#endif
"test_averagepool_2d_pads_count_include_pad",
"test_averagepool_2d_precomputed_pads_count_include_pad",
"test_averagepool_2d_same_lower",
"test_cast_FLOAT_to_STRING",
"test_cast_STRING_to_FLOAT",
"test_castlike_FLOAT_to_STRING_expanded",
"test_castlike_STRING_to_FLOAT_expanded",
"test_concat_1d_axis_negative_1", // 1d support is required
"test_div_uint8",  // output type mismatch
"test_maxpool_2d_dilations",
"test_maxpool_2d_same_lower",
"test_maxpool_2d_uint8",  // output type mismatch
"test_maxpool_with_argmax_2d_precomputed_pads",
"test_maxpool_with_argmax_2d_precomputed_strides",
"test_maxunpool_export_with_output_shape",  // exception during net.forward() call
"test_mul_uint8",  // output type mismatch
"test_sub_bcast", // 1d support is required
"test_sub_uint8",  // output type mismatch
"test_upsample_nearest",
"test_div_bcast", // remove when 1D Mat is supported
"test_mul_bcast", // remove when 1D Mat is supported
