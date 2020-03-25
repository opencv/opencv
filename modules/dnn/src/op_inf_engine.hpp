// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_INF_ENGINE_HPP__
#define __OPENCV_DNN_OP_INF_ENGINE_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#include "opencv2/core/async.hpp"
#include "opencv2/core/detail/async_promise.hpp"

#include "opencv2/dnn/utils/inference_engine.hpp"

#ifdef HAVE_INF_ENGINE

#define INF_ENGINE_RELEASE_2020_1 2020010000

#ifndef INF_ENGINE_RELEASE
#warning("IE version have not been provided via command-line. Using 2019.1 by default")
#define INF_ENGINE_RELEASE INF_ENGINE_RELEASE_2020_1
#endif

#define INF_ENGINE_VER_MAJOR_GT(ver) (((INF_ENGINE_RELEASE) / 10000) > ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_GE(ver) (((INF_ENGINE_RELEASE) / 10000) >= ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_LT(ver) (((INF_ENGINE_RELEASE) / 10000) < ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_LE(ver) (((INF_ENGINE_RELEASE) / 10000) <= ((ver) / 10000))
#define INF_ENGINE_VER_MAJOR_EQ(ver) (((INF_ENGINE_RELEASE) / 10000) == ((ver) / 10000))

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
//#define INFERENCE_ENGINE_DEPRECATED  // turn off deprecation warnings from IE
//there is no way to suppress warnings from IE only at this moment, so we are forced to suppress warnings globally
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(disable: 4996)  // was declared deprecated
#endif
#endif  // HAVE_DNN_IE_NN_BUILDER_2019

#if defined(__GNUC__) && INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2020_1)
#pragma GCC visibility push(default)
#endif

#include <inference_engine.hpp>


#if defined(__GNUC__) && INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2020_1)
#pragma GCC visibility pop
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic pop
#endif

#endif  // HAVE_INF_ENGINE

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

Backend& getInferenceEngineBackendTypeParam();


Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats);


CV__DNN_EXPERIMENTAL_NS_BEGIN

bool isMyriadX();

CV__DNN_EXPERIMENTAL_NS_END

InferenceEngine::Core& getCore(const std::string& id);

template<typename T = size_t>
static inline std::vector<T> getShape(const Mat& mat)
{
    std::vector<T> result(mat.dims);
    for (int i = 0; i < mat.dims; i++)
        result[i] = (T)mat.size[i];
    return result;
}


#endif  // HAVE_INF_ENGINE

}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_OP_INF_ENGINE_HPP__
