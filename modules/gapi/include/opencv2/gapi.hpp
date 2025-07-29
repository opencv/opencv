// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_HPP
#define OPENCV_GAPI_HPP

#include <memory>

/** \defgroup gapi_ref G-API framework
@{
    @defgroup gapi_main_classes G-API Main Classes
    @defgroup gapi_data_objects G-API Data Types
    @{
      @defgroup gapi_meta_args G-API Metadata Descriptors
    @}
    @defgroup gapi_std_backends G-API Standard Backends
    @defgroup gapi_compile_args G-API Graph Compilation Arguments
    @defgroup gapi_serialization G-API Serialization functionality
@}
 */

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gtyped.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/operators.hpp>

// Include these files here to avoid cyclic dependency between
// Desync & GKernel & GComputation & GStreamingCompiled.
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#endif // OPENCV_GAPI_HPP
