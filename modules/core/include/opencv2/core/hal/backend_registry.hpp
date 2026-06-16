// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP
#define OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP

#include "opencv2/core/hal/backend.hpp"

namespace cv { namespace hal {

//
// registerBackend()
// Add a backend to the global registry.
// Backends are checked in registration order —
// first registered = first checked by findBackend().
// Call once at program startup from the plugin loader.
// Does not take ownership — caller manages lifetime.
//
CV_EXPORTS void registerBackend(Backend* backend);

//
// findBackend()
// Find the first registered backend that supports op_id.
// op_id — one of the GPU_OP_* constants in backend.hpp
// Returns Backend* if found.
// Returns nullptr if no backend supports this operation.
// Called every time CV_GPU_RUN fires inside a cv:: function.
//
CV_EXPORTS Backend* findBackend(int op_id);

//
// clearBackends()
// Remove all registered backends from the registry.
// Used in unit tests to reset state between test cases.
// Not called in production code.
//
CV_EXPORTS void clearBackends();

//
// loadBackendPlugins()
// Loads GPU backend plugins via dlopen at startup.
// Thread-safe — runs exactly once.
// Called automatically by a static initializer.
//
CV_EXPORTS void loadBackendPlugins();

}} // cv::hal

#endif // OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP
