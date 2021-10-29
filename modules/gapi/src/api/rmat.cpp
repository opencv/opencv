// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/rmat.hpp>

using View = cv::RMat::View;

namespace {
cv::GMatDesc checkDesc(const cv::GMatDesc& desc) {
    if (!desc.dims.empty() && desc.chan != -1) {
        cv::util::throw_error(
            std::logic_error("Multidimesional RMat::Views with chan different from -1 are not supported!"));
    }
    return desc;
}

int typeFromDesc(const cv::GMatDesc& desc) {
    // In multidimensional case GMatDesc::chan is -1,
    // change it to 1 when calling CV_MAKE_TYPE
    return CV_MAKE_TYPE(desc.depth, desc.chan == -1 ? 1 : desc.chan);
}

static View::stepsT defaultSteps(const cv::GMatDesc& desc) {
    const auto& dims = desc.dims.empty()
                       ? std::vector<int>{desc.size.height, desc.size.width}
                       : desc.dims;
    View::stepsT steps(dims.size(), 0u);
    auto type = typeFromDesc(desc);
    steps.back() = CV_ELEM_SIZE(type);
    for (int i = static_cast<int>(dims.size())-2; i >= 0; i--) {
        steps[i] = steps[i+1]*dims[i];
    }
    return steps;
}
} // anonymous namespace

View::View(const cv::GMatDesc& desc, uchar* data, size_t step, DestroyCallback&& cb)
    : m_desc(checkDesc(desc))
    , m_data(data)
    , m_steps([this, step](){
        GAPI_Assert(m_desc.dims.empty());
        auto steps = defaultSteps(m_desc);
        if (step != 0u) {
            steps[0] = step;
        }
        return steps;
    }())
    , m_cb(std::move(cb)) {
}

View::View(const cv::GMatDesc& desc, uchar* data, const stepsT &steps, DestroyCallback&& cb)
    : m_desc(checkDesc(desc))
    , m_data(data)
    , m_steps(steps == stepsT{} ? defaultSteps(m_desc): steps)
    , m_cb(std::move(cb)) {
}

int View::type() const { return typeFromDesc(m_desc); }

// There is an issue with default generated operator=(View&&) on Mac:
// it doesn't nullify m_cb of the moved object
View& View::operator=(View&& v) {
    m_desc  = v.m_desc;
    m_data  = v.m_data;
    m_steps = v.m_steps;
    m_cb    = v.m_cb;
    v.m_desc  = {};
    v.m_data  = nullptr;
    v.m_steps = {0u};
    v.m_cb    = nullptr;
    return *this;
}
