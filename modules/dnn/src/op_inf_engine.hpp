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

#define INF_ENGINE_RELEASE_2020_2 2020020000
#define INF_ENGINE_RELEASE_2020_3 2020030000
#define INF_ENGINE_RELEASE_2020_4 2020040000
#define INF_ENGINE_RELEASE_2021_1 2021010000
#define INF_ENGINE_RELEASE_2021_2 2021020000
#define INF_ENGINE_RELEASE_2021_3 2021030000
#define INF_ENGINE_RELEASE_2021_4 2021040000
#define INF_ENGINE_RELEASE_2022_1 2022010000

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

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
#include <openvino/openvino.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/convert_fp32_to_fp16.hpp>
#else
#include <inference_engine.hpp>
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
//#pragma GCC diagnostic pop
#endif

#endif  // HAVE_INF_ENGINE

#define CV_ERROR_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 do { CV_Error(Error::StsNotImplemented, "This OpenCV version is built without Inference Engine NN Builder API support (legacy API is not supported anymore)"); } while (0)

namespace cv { namespace dnn {

CV__DNN_INLINE_NS_BEGIN
namespace openvino {

// TODO: use std::string as parameter
bool checkTarget(Target target);

}  // namespace openvino
CV__DNN_INLINE_NS_END

#ifdef HAVE_INF_ENGINE

Backend& getInferenceEngineBackendTypeParam();

#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
Mat infEngineBlobToMat(const ov::Tensor& blob);

void infEngineBlobsToMats(const ov::TensorVector& blobs,
                          std::vector<Mat>& mats);
#else
Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

void infEngineBlobsToMats(const std::vector<InferenceEngine::Blob::Ptr>& blobs,
                          std::vector<Mat>& mats);
#endif  // OpenVINO >= 2022.1


CV__DNN_INLINE_NS_BEGIN

void switchToOpenVINOBackend(Net& net);

bool isMyriadX();

bool isArmComputePlugin();

CV__DNN_INLINE_NS_END

// A series of wrappers for classes from OpenVINO API 2.0.
// Need just for less conditional compilation inserts.
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
namespace InferenceEngine {

class CNNNetwork {
public:
    CNNNetwork();

    CNNNetwork(std::shared_ptr<ov::Model> model);

    std::shared_ptr<ov::Model> getFunction() const;

    void serialize(const std::string& xmlPath, const std::string& binPath);

    void reshape(const std::map<std::string, std::vector<size_t> >& shapes);

private:
    std::shared_ptr<ov::Model> model = nullptr;
};

typedef ov::InferRequest InferRequest;

class ExecutableNetwork : public ov::CompiledModel {
public:
    ExecutableNetwork();

    ExecutableNetwork(const ov::CompiledModel& copy);

    ov::InferRequest CreateInferRequest();
};

class Core : public ov::Core {
public:
    std::vector<std::string> GetAvailableDevices();

    void UnregisterPlugin(const std::string& id);

    CNNNetwork ReadNetwork(const std::string& xmlPath, const std::string& binPath);

    ExecutableNetwork LoadNetwork(CNNNetwork net, const std::string& device,
                                  const std::map<std::string, std::string>& config);
};

}
#endif // OpenVINO >= 2022.1

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
