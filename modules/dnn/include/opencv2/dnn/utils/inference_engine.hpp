// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_UTILS_INF_ENGINE_HPP
#define OPENCV_DNN_UTILS_INF_ENGINE_HPP

#include "../dnn.hpp"

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN


/** @brief Release a Myriad device (binded by OpenCV).
 *
 * Single Myriad device cannot be shared across multiple processes which uses
 * Inference Engine's Myriad plugin.
 */
CV_EXPORTS_W void resetMyriadDevice();


/* Values for 'OPENCV_DNN_IE_VPU_TYPE' parameter */
#define CV_DNN_INFERENCE_ENGINE_VPU_TYPE_UNSPECIFIED ""
/// Intel(R) Movidius(TM) Neural Compute Stick, NCS (USB 03e7:2150), Myriad2 (https://software.intel.com/en-us/movidius-ncs)
#define CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2 "Myriad2"
/// Intel(R) Neural Compute Stick 2, NCS2 (USB 03e7:2485), MyriadX (https://software.intel.com/ru-ru/neural-compute-stick)
#define CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X "MyriadX"


/** @brief Returns Inference Engine VPU type.
 *
 * See values of `CV_DNN_INFERENCE_ENGINE_VPU_TYPE_*` macros.
 */
CV_EXPORTS_W cv::String getInferenceEngineVPUType();


CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

#endif // OPENCV_DNN_UTILS_INF_ENGINE_HPP
