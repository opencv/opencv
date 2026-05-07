// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2026, BigVision LLC, all rights reserved.

// OpenCV ORT (ONNX Runtime) Engine Deny List
// Tests are skipped when running with OPENCV_FORCE_DNN_ENGINE=ENGINE_ORT (4)

// --- Test_ONNX_layers / Test_ONNX_nets tests (test_onnx_importer.cpp) ---
// Trilu with 1D input is an OpenCV extension; ONNX spec requires rank>=2, ORT enforces strictly.
"trilu_tril_one_row1D",
// alexnet.onnx declares 224x224 input but reference was generated with 227x227 (Caffe-era); ORT enforces shape strictly while CLASSIC/NEW silently accept any size.
"Alexnet",
// Invalid ONNX model - missing required 'consumed_inputs' attribute in BatchNormalization (opset 1, ancient WinMLTools-exported model).
"TinyYolov2",
// Legacy ONNX spatial=0 attribute on BatchNorm causes ORT dimension mismatch; reconvert with modern exporter to fix 1D vs 3D scale error with spatial=1.
"LResNet100E_IR",
"RandomNormalLike_basic",
"RandomNormalLike_complex",
