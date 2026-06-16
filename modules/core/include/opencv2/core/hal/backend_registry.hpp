// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP
#define OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP

#include "opencv2/core/hal/backend.hpp"

namespace cv { namespace hal {

// Add a backend to the global registry (caller keeps ownership).
CV_EXPORTS void registerBackend(Backend* backend);

// First registered backend, or nullptr; used to grab a backend for setup.
CV_EXPORTS Backend* findBackend();

// Remove all registered backends (test reset only).
CV_EXPORTS void clearBackends();

// Load GPU backend plugins via dlopen at startup (thread-safe, runs once).
CV_EXPORTS void loadBackendPlugins();

}} // cv::hal

#endif // OPENCV_CORE_HAL_BACKEND_REGISTRY_HPP
