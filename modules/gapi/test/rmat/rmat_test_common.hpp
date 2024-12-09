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
class RMatAdapterRef : public RMat::IAdapter {
    cv::Mat& m_mat;
    bool& m_callbackCalled;
public:
    RMatAdapterRef(cv::Mat& m, bool& callbackCalled)
        : m_mat(m), m_callbackCalled(callbackCalled)
    {}
    virtual RMat::View access(RMat::Access access) override {
        RMat::View::stepsT steps(m_mat.dims);
        for (int i = 0; i < m_mat.dims; i++) {
            steps[i] = m_mat.step[i];
        }
        if (access == RMat::Access::W) {
            return RMat::View(cv::descr_of(m_mat), m_mat.data, steps,
                              [this](){
                                  EXPECT_FALSE(m_callbackCalled);
                                  m_callbackCalled = true;
                              });
        } else {
            return RMat::View(cv::descr_of(m_mat), m_mat.data, steps);
        }
    }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_mat); }
};

class RMatAdapterCopy : public RMat::IAdapter {
    cv::Mat& m_deviceMat;
    cv::Mat  m_hostMat;
    bool& m_callbackCalled;

public:
    RMatAdapterCopy(cv::Mat& m, bool& callbackCalled)
        : m_deviceMat(m), m_hostMat(m.clone()), m_callbackCalled(callbackCalled)
    {}
    virtual RMat::View access(RMat::Access access) override {
        RMat::View::stepsT steps(m_hostMat.dims);
        for (int i = 0; i < m_hostMat.dims; i++) {
            steps[i] = m_hostMat.step[i];
        }
        if (access == RMat::Access::W) {
            return RMat::View(cv::descr_of(m_hostMat), m_hostMat.data, steps,
                              [this](){
                                  EXPECT_FALSE(m_callbackCalled);
                                  m_callbackCalled = true;
                                  m_hostMat.copyTo(m_deviceMat);
                              });
        } else {
            m_deviceMat.copyTo(m_hostMat);
            return RMat::View(cv::descr_of(m_hostMat), m_hostMat.data, steps);
        }
    }
    virtual cv::GMatDesc desc() const override { return cv::descr_of(m_hostMat); }
};
} // namespace opencv_test

#endif // OPENCV_GAPI_RMAT_TESTS_COMMON_HPP
