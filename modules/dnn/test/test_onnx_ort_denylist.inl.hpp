// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.


// OpenCV ORT (ONNX Runtime) Engine Deny List
// Tests are skipped when running with OPENCV_FORCE_DNN_ENGINE=ENGINE_ORT (4)

// --- Test_ONNX_layers / Test_ONNX_nets tests (test_onnx_importer.cpp) ---
// Trilu 1D is OpenCV extension; ONNX spec requires rank>=2, ORT enforces strictly.
"trilu_tril_one_row1D",
// alexnet.onnx input 224x224 but reference generated with 227x227; ORT enforces shape strictly.
"Alexnet",
// Invalid model: missing 'consumed_inputs' in BatchNormalization opset 1 (WinMLTools export).
"TinyYolov2",
// BatchNorm spatial=0 causes ORT dimension mismatch; reconvert with modern exporter.
"LResNet100E_IR",
"RandomNormalLike_basic",
"RandomNormalLike_complex",

// --- Test_ONNX_conformance tests ---
// No ORT CPU kernel for Pow with float32+uint32/uint64 WIP : https://github.com/microsoft/onnxruntime/pull/25728
"test_pow_types_float32_uint32",
"test_pow_types_float32_uint64",
// RoiAlign opset 22 not in ORT CPU EP; mode=max incorrect WIP : https://github.com/microsoft/onnxruntime/issues/6146
"test_roialign_mode_max",
// Clip int8 expanded uses Where(int8); int8 commented out in ORT WIP : https://github.com/microsoft/onnxruntime/issues/1080 https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/where_op.cc
"test_clip_default_int8_max_expanded",
"test_clip_default_int8_min_expanded",
// BitShift uint16 commented out in ORT WIP : https://github.com/microsoft/onnxruntime/blob/v1.25.1/onnxruntime/core/providers/cpu/math/element_wise_ops.cc
"test_bitshift_left_uint16",
"test_bitshift_right_uint16",
// Min/Max int16/uint16 not in ORT EnabledMin/Max12Types WIP : https://github.com/microsoft/onnxruntime/blob/v1.25.1/onnxruntime/core/providers/cpu/math/element_wise_ops.cc
"test_min_int16",
"test_min_uint16",
"test_max_int16",
"test_max_uint16",
// TopK uint64 not registered in ORT CPU EP WIP : https://github.com/microsoft/onnxruntime/blob/v1.25.1/onnxruntime/core/providers/cpu/math/top_k.cc
"test_top_k_uint64",
// ORT CPU EP does not support batchwise (layout=1) GRU/LSTM WIP : https://github.com/opencv/opencv/issues/26456
"test_gru_batchwise",
"test_lstm_batchwise",
// DFT: ORT float32 precision vs numpy higher precision; normL1 ~8e-5 exceeds 1e-5 threshold
"test_dft",
"test_dft_axis_opset19",
"test_dft_opset19",

// PriorBox is a custom OpenCV domain op; not registered in ORT opset 22
"PriorBox_ONNX",
