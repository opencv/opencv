// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#include "precomp.hpp"
#include <iostream> // cerr
#include <functional> // hash
#include <numeric> // accumulate

#include <ade/util/algorithm.hpp>

#include "logger.hpp"
#include <opencv2/gapi/gkernel.hpp>

#include "api/gbackend_priv.hpp"

// GKernelPackage public implementation ////////////////////////////////////////
void cv::gapi::GKernelPackage::remove(const cv::gapi::GBackend& backend)
{
    std::vector<std::string> id_deleted_kernels;
    for (const auto& p : m_id_kernels)
    {
        if (p.second.first == backend)
        {
            id_deleted_kernels.push_back(p.first);
        }
    }

    for (const auto& kernel_id : id_deleted_kernels)
    {
        m_id_kernels.erase(kernel_id);
    }
}

bool cv::gapi::GKernelPackage::includesAPI(const std::string &id) const
{
    return ade::util::contains(m_id_kernels, id);
}

void cv::gapi::GKernelPackage::removeAPI(const std::string &id)
{
    m_id_kernels.erase(id);
}

std::size_t cv::gapi::GKernelPackage::size() const
{
    return m_id_kernels.size();
}

const std::vector<cv::GTransform> &cv::gapi::GKernelPackage::get_transformations() const
{
    return m_transformations;
}

std::vector<std::string> cv::gapi::GKernelPackage::get_kernel_ids() const
{
    std::vector<std::string> ids;
    for (auto &&id : m_id_kernels)
    {
        ids.emplace_back(id.first);
    }
    return ids;
}

cv::gapi::GKernelPackage cv::gapi::combine(const GKernelPackage  &lhs,
                                           const GKernelPackage  &rhs)
{

        // If there is a collision, prefer RHS to LHS
        // since RHS package has a precedense, start with its copy
        GKernelPackage result(rhs);
        // now iterate over LHS package and put kernel if and only
        // if there's no such one
        for (const auto& kernel : lhs.m_id_kernels)
        {
            if (!result.includesAPI(kernel.first))
            {
                result.m_id_kernels.emplace(kernel.first, kernel.second);
            }
        }
        for (const auto &transforms : lhs.m_transformations){
            result.m_transformations.push_back(transforms);
        }
        return result;
}

std::pair<cv::gapi::GBackend, cv::GKernelImpl>
cv::gapi::GKernelPackage::lookup(const std::string &id) const
{
    auto kernel_it = m_id_kernels.find(id);
    if (kernel_it != m_id_kernels.end())
    {
        return kernel_it->second;
    }
    // If reached here, kernel was not found.
    util::throw_error(std::logic_error("Kernel " + id + " was not found"));
}

std::vector<cv::gapi::GBackend> cv::gapi::GKernelPackage::backends() const
{
    using kernel_type = std::pair<std::string, std::pair<cv::gapi::GBackend, cv::GKernelImpl>>;
    std::unordered_set<cv::gapi::GBackend> unique_set;
    ade::util::transform(m_id_kernels, std::inserter(unique_set, unique_set.end()),
                                       [](const kernel_type& k) { return k.second.first; });

    return std::vector<cv::gapi::GBackend>(unique_set.begin(), unique_set.end());
}
