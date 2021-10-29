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
CV__DNN_INLINE_NS_BEGIN


/* Values for 'OPENCV_DNN_BACKEND_INFERENCE_ENGINE_TYPE' parameter */
#define CV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_API     "NN_BUILDER"
#define CV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH             "NGRAPH"

/** @brief Returns Inference Engine internal backend API.
 *
 * See values of `CV_DNN_BACKEND_INFERENCE_ENGINE_*` macros.
 *
 * Default value is controlled through `OPENCV_DNN_BACKEND_INFERENCE_ENGINE_TYPE` runtime parameter (environment variable).
 */
CV_EXPORTS_W cv::String getInferenceEngineBackendType();

/** @brief Specify Inference Engine internal backend API.
 *
 * See values of `CV_DNN_BACKEND_INFERENCE_ENGINE_*` macros.
 *
 * @returns previous value of internal backend API
 */
CV_EXPORTS_W cv::String setInferenceEngineBackendType(const cv::String& newBackendType);


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
#define CV_DNN_INFERENCE_ENGINE_CPU_TYPE_ARM_COMPUTE "ARM_COMPUTE"
#define CV_DNN_INFERENCE_ENGINE_CPU_TYPE_X86         "X86"


/** @brief Returns Inference Engine VPU type.
 *
 * See values of `CV_DNN_INFERENCE_ENGINE_VPU_TYPE_*` macros.
 */
CV_EXPORTS_W cv::String getInferenceEngineVPUType();

/** @brief Returns Inference Engine CPU type.
 *
 * Specify OpenVINO plugin: CPU or ARM.
 */
CV_EXPORTS_W cv::String getInferenceEngineCPUType();

/** @brief Release a HDDL plugin.
 */
CV_EXPORTS_W void releaseHDDLPlugin();


CV__DNN_INLINE_NS_END
}} // namespace

#endif // OPENCV_DNN_UTILS_INF_ENGINE_HPP
