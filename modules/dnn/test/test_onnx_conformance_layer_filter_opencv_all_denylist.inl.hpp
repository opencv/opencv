// "test_add_bcast",
"test_add_uint8",  // output type mismatch //FAILS
// #if 1  // output type mismatch CV_32F vs expected 32S
// "test_argmax_default_axis_example",
// "test_argmax_default_axis_example_select_last_index",
// "test_argmax_default_axis_random",
// "test_argmax_default_axis_random_select_last_index",
// "test_argmax_keepdims_example",
// "test_argmax_keepdims_example_select_last_index",
// "test_argmax_keepdims_random",
// "test_argmax_keepdims_random_select_last_index",
// "test_argmax_negative_axis_keepdims_example",
// "test_argmax_negative_axis_keepdims_example_select_last_index",
// "test_argmax_negative_axis_keepdims_random",
// "test_argmax_negative_axis_keepdims_random_select_last_index",
// "test_argmax_no_keepdims_example",
// "test_argmax_no_keepdims_example_select_last_index",
// "test_argmax_no_keepdims_random",
// "test_argmax_no_keepdims_random_select_last_index",
// "test_argmin_default_axis_example",
// "test_argmin_default_axis_example_select_last_index",
// "test_argmin_default_axis_random",
// "test_argmin_default_axis_random_select_last_index",
// "test_argmin_keepdims_example",
// "test_argmin_keepdims_example_select_last_index",
// "test_argmin_keepdims_random",
// "test_argmin_keepdims_random_select_last_index",
// "test_argmin_negative_axis_keepdims_example",
// "test_argmin_negative_axis_keepdims_example_select_last_index",
// "test_argmin_negative_axis_keepdims_random",
// "test_argmin_negative_axis_keepdims_random_select_last_index",
// "test_argmin_no_keepdims_example",
// "test_argmin_no_keepdims_example_select_last_index",
// "test_argmin_no_keepdims_random",
// "test_argmin_no_keepdims_random_select_last_index",
// #endif
"test_averagepool_2d_pads_count_include_pad", //FAILS
"test_averagepool_2d_precomputed_pads_count_include_pad", //FAILS
"test_averagepool_2d_same_lower", //FAILS
"test_cast_FLOAT_to_STRING", //FAILS
"test_cast_STRING_to_FLOAT", //FAILS
"test_castlike_FLOAT_to_STRING_expanded", //FAILS
"test_castlike_STRING_to_FLOAT_expanded", //FAILS
"test_concat_1d_axis_negative_1", // 1d support is required
"test_div_uint8",  // output type mismatch //FAILS
"test_maxpool_2d_dilations", //FAILS
"test_maxpool_2d_same_lower", //FAILS
"test_maxpool_2d_uint8",  // output type mismatch //FAILS
"test_maxpool_with_argmax_2d_precomputed_pads",
"test_maxpool_with_argmax_2d_precomputed_strides", //FAILS
"test_maxunpool_export_with_output_shape",  // exception during net.forward() call //FAILS
"test_mul_uint8",  // output type mismatch //FAILS
// "test_sub_bcast",
"test_sub_uint8",  // output type mismatch //FAILS
"test_upsample_nearest", //FAILS
// "test_div_bcast", // remove when 1D Mat is supported
// "test_mul_bcast", // remove when 1D Mat is supported
