// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <iostream> // cerr
#include <functional> // hash
#include <numeric> // accumulate

#include <ade/util/algorithm.hpp>

#include "logger.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gbackend_priv.hpp"

// GKernelPackage public implementation ////////////////////////////////////////
void cv::gapi::GKernelPackage::remove(const cv::gapi::GBackend& backend)
{
    m_backend_kernels.erase(backend);
}

bool cv::gapi::GKernelPackage::includesAPI(const std::string &id) const
{
    // In current form not very efficient (n * log n)
    auto it = std::find_if(m_backend_kernels.begin(),
                           m_backend_kernels.end(),
                           [&id](const M::value_type &p) {
                               return ade::util::contains(p.second, id);
                           });
    return (it != m_backend_kernels.end());
}

void cv::gapi::GKernelPackage::removeAPI(const std::string &id)
{
    for (auto &bk : m_backend_kernels)
    {
        if (ade::util::contains(bk.second, id))
        {
            bk.second.erase(id);
            break;
        }
    }
}

std::size_t cv::gapi::GKernelPackage::size() const
{
    return std::accumulate(m_backend_kernels.begin(),
                           m_backend_kernels.end(),
                           static_cast<std::size_t>(0u),
                           [](std::size_t acc, const M::value_type& v) {
                               return acc + v.second.size();
                           });
}

cv::gapi::GKernelPackage cv::gapi::combine(const GKernelPackage  &lhs,
                                           const GKernelPackage  &rhs)
{

        // If there is a collision, prefer RHS to LHS
        // since RHS package has a precedense, start with its copy
        GKernelPackage result(rhs);
        // now iterate over LHS package and put kernel if and only
        // if there's no such one
        for (const auto &backend : lhs.m_backend_kernels)
        {
            for (const auto &kimpl : backend.second)
            {
                if (!result.includesAPI(kimpl.first))
                    result.m_backend_kernels[backend.first].insert(kimpl);
            }
        }
        return result;
}

std::pair<cv::gapi::GBackend, cv::GKernelImpl>
cv::gapi::GKernelPackage::lookup(const std::string &id) const
{
    // If order is empty, return what comes first
    auto it = std::find_if(m_backend_kernels.begin(),
            m_backend_kernels.end(),
            [&id](const M::value_type &p) {
            return ade::util::contains(p.second, id);
            });
    if (it != m_backend_kernels.end())
    {
        // FIXME: Two lookups!
        return std::make_pair(it->first, it->second.find(id)->second);
    }

    // If reached here, kernel was not found among selected backends.
    util::throw_error(std::logic_error("Kernel " + id + " was not found"));
}

std::vector<cv::gapi::GBackend> cv::gapi::GKernelPackage::backends() const
{
    std::vector<cv::gapi::GBackend> result;
    for (const auto &p : m_backend_kernels) result.emplace_back(p.first);
    return result;
}
