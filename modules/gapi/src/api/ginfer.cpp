// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#include "precomp.hpp"

#include <functional> // hash
#include <numeric> // accumulate
#include <unordered_set>
#include <iterator>

#include <ade/util/algorithm.hpp>

#include <opencv2/gapi/infer.hpp>

cv::gapi::GNetPackage::GNetPackage(std::initializer_list<GNetParam> ii)
    : networks(ii) {
}

std::vector<cv::gapi::GBackend> cv::gapi::GNetPackage::backends() const {
    std::unordered_set<cv::gapi::GBackend> unique_set;
    for (const auto &nn : networks) unique_set.insert(nn.backend);
    return std::vector<cv::gapi::GBackend>(unique_set.begin(), unique_set.end());
}

// FIXME: Inference API is currently only available in full mode
#if !defined(GAPI_STANDALONE)

cv::GInferInputs::GInferInputs()
    : in_blobs(std::make_shared<Map>())
{
}

cv::GMat& cv::GInferInputs::operator[](const std::string& name) {
    return (*in_blobs)[name];
}

const cv::GInferInputs::Map& cv::GInferInputs::getBlobs() const {
    return *in_blobs;
}

void cv::GInferInputs::setInput(const std::string& name, const cv::GMat& value) {
    in_blobs->emplace(name, value);
}

struct cv::GInferOutputs::Priv
{
    Priv(std::shared_ptr<cv::GCall>);

    std::shared_ptr<cv::GCall> call;
    InOutInfo* info = nullptr;
    std::unordered_map<std::string, cv::GMat> out_blobs;
};

cv::GInferOutputs::Priv::Priv(std::shared_ptr<cv::GCall> c)
    : call(std::move(c)), info(cv::util::any_cast<InOutInfo>(&call->params()))
{
}

cv::GInferOutputs::GInferOutputs(std::shared_ptr<cv::GCall> call)
    : m_priv(std::make_shared<cv::GInferOutputs::Priv>(std::move(call)))
{
}

cv::GMat cv::GInferOutputs::at(const std::string& name)
{
    auto it = m_priv->out_blobs.find(name);
    if (it == m_priv->out_blobs.end()) {
        // FIXME: Avoid modifying GKernel
        // Expect output to be always GMat
        m_priv->call->kernel().outShapes.push_back(cv::GShape::GMAT);
        // ...so _empty_ constructor is passed here.
        m_priv->call->kernel().outCtors.emplace_back(cv::util::monostate{});
        int out_idx = static_cast<int>(m_priv->out_blobs.size());
        it = m_priv->out_blobs.emplace(name, m_priv->call->yield(out_idx)).first;
        m_priv->info->out_names.push_back(name);
    }
    return it->second;
}
#endif // GAPI_STANDALONE
