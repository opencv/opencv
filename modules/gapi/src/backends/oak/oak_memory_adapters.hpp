// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
#define OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP

#include <memory>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/rmat.hpp>

namespace cv {
namespace gapi {
namespace oak {

// Used for OAK backends outputs only.
// Filled from DepthAI's ImgFrame type and owns the memory.
// Used mainly for CV operations.
class GAPI_EXPORTS OAKMediaAdapter final : public cv::MediaFrame::IAdapter {
public:
    OAKMediaAdapter() = default;
    OAKMediaAdapter(cv::Size sz, cv::MediaFormat fmt, std::vector<uint8_t>&& buffer);
    cv::GFrameDesc meta() const override;
    cv::MediaFrame::View access(cv::MediaFrame::Access) override;
    ~OAKMediaAdapter() = default;
private:
    cv::Size m_sz;
    cv::MediaFormat m_fmt;
    std::vector<uint8_t> m_buffer;
};

// Used for OAK backends outputs only.
// Filled from DepthAI's NNData type and owns the memory.
// Used only for infer operations.
class GAPI_EXPORTS OAKRMatAdapter final : public cv::RMat::Adapter {
public:
    OAKRMatAdapter() = default;
    OAKRMatAdapter(const cv::Size& size, int precision, std::vector<float>&& buffer);
    cv::GMatDesc desc() const override;
    cv::RMat::View access(cv::RMat::Access) override;
    ~OAKRMatAdapter() = default;
private:
    cv::Size m_size;
    int m_precision;
    std::vector<float> m_buffer;
    cv::GMatDesc m_desc;
    cv::Mat m_mat;
};

} // namespace oak
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
