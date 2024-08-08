"test_basic_conv_with_padding", // (assert failed) !blobs.empty() in initCUDA
"test_basic_conv_without_padding", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_autopad_same", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_and_asymmetric_padding", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_no_padding", // (assert failed) !blobs.empty() in initCUDA
"test_conv_with_strides_padding", // (assert failed) !blobs.empty() in initCUDA
"test_dropout_default_ratio",
"test_logsoftmax_large_number", // fp16 accuracy issue
"test_logsoftmax_large_number_expanded", // fp16 accuracy issue
"test_maxpool_with_argmax_2d_precomputed_pads", // assertion failed mat.type() == CV_32F
"test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded", // crash: https://github.com/opencv/opencv/issues/25471
"test_reduce_prod_default_axes_keepdims_example", // fallback to cpu, accuracy
"test_reduce_prod_default_axes_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_default_axes_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_do_not_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_keepdims_random", // fallback to cpu, accuracy
"test_reduce_sum_square_negative_axes_keepdims_random", // fallback to cpu, accuracy
"test_pow", // fp16 accuracy issue
"test_softmax_large_number", // fp16 accuracy issue
"test_softmax_large_number_expanded", // fp16 accuracy issue
"test_tan", // fp16 accuracy issue
