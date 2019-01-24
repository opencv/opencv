// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/mat.hpp> //gapi::own::Mat

#include "opencv2/gapi/gmat.hpp"
#include "api/gapi_priv.hpp" // GOrigin

// cv::GMat public implementation //////////////////////////////////////////////
cv::GMat::GMat()
    : m_priv(new GOrigin(GShape::GMAT, GNode::Param()))
{
}

cv::GMat::GMat(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GMAT, n, out))
{
}

cv::GOrigin& cv::GMat::priv()
{
    return *m_priv;
}

const cv::GOrigin& cv::GMat::priv() const
{
    return *m_priv;
}

namespace{
    template <typename T> cv::GMetaArgs vec_descr_of(const std::vector<T> &vec)
        {
        cv::GMetaArgs vec_descr;
        vec_descr.reserve(vec.size());
        for(auto& mat : vec){
            vec_descr.emplace_back(descr_of(mat));
        }
        return vec_descr;
    }
}


#if !defined(GAPI_STANDALONE)
cv::GMatDesc cv::descr_of(const cv::Mat &mat)
{
    return GMatDesc{mat.depth(), mat.channels(), {mat.cols, mat.rows}};
}

cv::GMatDesc cv::descr_of(const cv::UMat &mat)
{
    return GMatDesc{ mat.depth(), mat.channels(),{ mat.cols, mat.rows } };
}

cv::GMetaArgs cv::descr_of(const std::vector<cv::Mat> &vec)
{
    return vec_descr_of(vec);
}

cv::GMetaArgs cv::descr_of(const std::vector<cv::UMat> &vec)
{
    return vec_descr_of(vec);
}
#endif

cv::GMatDesc cv::gapi::own::descr_of(const cv::gapi::own::Mat &mat)
{
    return GMatDesc{mat.depth(), mat.channels(), {mat.cols, mat.rows}};
}

cv::GMetaArgs cv::gapi::own::descr_of(const std::vector<cv::gapi::own::Mat> &vec)
{
    return vec_descr_of(vec);
}

namespace cv {
std::ostream& operator<<(std::ostream& os, const cv::GMatDesc &desc)
{
    switch (desc.depth)
    {
#define TT(X) case CV_##X: os << #X; break;
        TT(8U);
        TT(8S);
        TT(16U);
        TT(16S);
        TT(32S);
        TT(32F);
        TT(64F);
#undef TT
    default:
        os << "(user type "
           << std::hex << desc.depth << std::dec
           << ")";
        break;
    }

    os << "C" << desc.chan << " ";
    os << desc.size.width << "x" << desc.size.height;

    return os;
}
}
