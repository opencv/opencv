// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_RMAT_TESTS_COMMON_HPP
#define OPENCV_GAPI_RMAT_TESTS_COMMON_HPP

#include "../test_precomp.hpp"
#include <opencv2/gapi/rmat.hpp>

namespace opencv_test {
class RMatAdapterRef : public RMat::Adapter {
    cv::Mat& m_mat;
    bool& m_callbackCalled;
public:
    RMatAdapterRef(cv::Mat& m, bool& callbackCalled)
        : m_mat(m), m_callbackCalled(callbackCalled)
    {}
    virtual RMat::View access(RMat::Access access) override {
        if (access == RMat::Access::W) {
            return RMat::View(cv::descr_of(m_mat), m_mat.data, m_mat.step,
                              [this](){
                                  EXPECT_FALSE(m_callbackCalled);
                                  m_callbackCalled = true;
                              });
        } else {
            return RMat::View(cv::descr_of(m_mat), m_mat.data, m_mat.step);
        }
    }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
};

class RMatAdapterCopy : public RMat::Adapter {
    cv::Mat& m_deviceMat;
    cv::Mat  m_hostMat;
    bool& m_callbackCalled;

public:
    RMatAdapterCopy(cv::Mat& m, bool& callbackCalled)
        : m_deviceMat(m), m_hostMat(m.clone()), m_callbackCalled(callbackCalled)
    {}
    virtual RMat::View access(RMat::Access access) override {
        if (access == RMat::Access::W) {
            return RMat::View(cv::descr_of(m_hostMat), m_hostMat.data, m_hostMat.step,
                              [this](){
                                  EXPECT_FALSE(m_callbackCalled);
                                  m_callbackCalled = true;
                                  m_hostMat.copyTo(m_deviceMat);
                              });
        } else {
            m_deviceMat.copyTo(m_hostMat);
            return RMat::View(cv::descr_of(m_hostMat), m_hostMat.data, m_hostMat.step);
        }
    }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_hostMat); }
};
} // namespace opencv_test

#endif // OPENCV_GAPI_RMAT_TESTS_COMMON_HPP
