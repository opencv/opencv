"test_averagepool_2d_pads_count_include_pad",  // wrong output
"test_averagepool_2d_precomputed_pads_count_include_pad", // wrong output
"test_averagepool_2d_same_lower", // wrong output
"test_lppool_2d_same_lower", // wrong output (same SAME_LOWER padding issue)
"test_cast_FLOAT_to_STRING", // Unsupported type in function 'parseCast'
"test_cast_STRING_to_FLOAT", // unexception during net.forward() call
"test_castlike_FLOAT_to_STRING_expanded", // Unsupported type in function 'parseCast'
"test_castlike_STRING_to_FLOAT_expanded", // unexception during net.forward() call
"test_maxpool_2d_dilations", // output size mismatch in NORMASSERT
"test_maxpool_2d_same_lower", // wrong output
"test_maxpool_with_argmax_2d_precomputed_strides", // wrong output
"test_maxunpool_export_with_output_shape",  // unexception during net.forward() call
"test_upsample_nearest", // Dimension mismatch of input
"test_flexattention_double_expanded_ver26", // Softmax kernel is fp32-only; fp64 decomposition unsupported (fused test_flexattention_double passes)
