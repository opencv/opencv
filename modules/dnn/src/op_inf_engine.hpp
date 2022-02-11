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

#define INF_ENGINE_RELEASE_2018R5 2018050000
#define INF_ENGINE_RELEASE_2019R1 2019010000
#define INF_ENGINE_RELEASE_2019R2 2019020000
#define INF_ENGINE_RELEASE_2019R3 2019030000
#define INF_ENGINE_RELEASE_2020_1 2020010000
#define INF_ENGINE_RELEASE_2020_2 2020020000
#define INF_ENGINE_RELEASE_2020_3 2020030000
#define INF_ENGINE_RELEASE_2020_4 2020040000
#define INF_ENGINE_RELEASE_2021_1 2021010000
#define INF_ENGINE_RELEASE_2021_2 2021020000
#define INF_ENGINE_RELEASE_2021_3 2021030000
#define INF_ENGINE_RELEASE_2021_4 2021040000

#ifndef INF_ENGINE_RELEASE
#warning("IE version have not been provided via command-line. Using 2021.4 by default")
#define INF_ENGINE_RELEASE INF_ENGINE_RELEASE_2021_4
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

#include <inference_engine.hpp>

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic pop
#endif

#endif  // HAVE_INF_ENGINE

#define CV_ERROR_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 do { CV_Error(Error::StsNotImplemented, "This OpenCV version is built without Inference Engine NN Builder API support (legacy API is not supported anymore)"); } while (0)

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

Backend& getInferenceEngineBackendTypeParam();

Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats);



CV__DNN_INLINE_NS_BEGIN

bool isMyriadX();

bool isArmComputePlugin();

CV__DNN_INLINE_NS_END


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
