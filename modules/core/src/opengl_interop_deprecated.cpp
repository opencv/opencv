/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#include "opencv2/core/opengl_interop_deprecated.hpp"
#include "opencv2/core/gpumat.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

CvOpenGlFuncTab::~CvOpenGlFuncTab()
{
}

void icvSetOpenGlFuncTab(const CvOpenGlFuncTab*)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

////////////////////////////////////////////////////////////////////////
// GlBuffer

class cv::GlBuffer::Impl
{
};

cv::GlBuffer::GlBuffer(Usage _usage) : rows_(0), cols_(0), type_(0), usage_(_usage)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlBuffer::GlBuffer(int, int, int, Usage _usage) : rows_(0), cols_(0), type_(0), usage_(_usage)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlBuffer::GlBuffer(Size, int, Usage _usage) : rows_(0), cols_(0), type_(0), usage_(_usage)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlBuffer::GlBuffer(InputArray, Usage _usage) : rows_(0), cols_(0), type_(0), usage_(_usage)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlBuffer::create(int, int, int, Usage)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlBuffer::release()
{
}

void cv::GlBuffer::copyFrom(InputArray)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlBuffer::bind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlBuffer::unbind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

Mat cv::GlBuffer::mapHost()
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return Mat();
}

void cv::GlBuffer::unmapHost()
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

GpuMat cv::GlBuffer::mapDevice()
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return GpuMat();
}

void cv::GlBuffer::unmapDevice()
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

template <> void cv::Ptr<cv::GlBuffer::Impl>::delete_obj()
{
    if (obj) delete obj;
}

//////////////////////////////////////////////////////////////////////////////////////////
// GlTexture

class cv::GlTexture::Impl
{
};

cv::GlTexture::GlTexture() : rows_(0), cols_(0), type_(0), buf_(GlBuffer::TEXTURE_BUFFER)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlTexture::GlTexture(int, int, int) : rows_(0), cols_(0), type_(0), buf_(GlBuffer::TEXTURE_BUFFER)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlTexture::GlTexture(Size, int) : rows_(0), cols_(0), type_(0), buf_(GlBuffer::TEXTURE_BUFFER)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

cv::GlTexture::GlTexture(InputArray, bool) : rows_(0), cols_(0), type_(0), buf_(GlBuffer::TEXTURE_BUFFER)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlTexture::create(int, int, int)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlTexture::release()
{
}

void cv::GlTexture::copyFrom(InputArray, bool)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlTexture::bind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlTexture::unbind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

template <> void cv::Ptr<cv::GlTexture::Impl>::delete_obj()
{
    if (obj) delete obj;
}

////////////////////////////////////////////////////////////////////////
// GlArrays

void cv::GlArrays::setVertexArray(InputArray)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlArrays::setColorArray(InputArray, bool)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlArrays::setNormalArray(InputArray)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlArrays::setTexCoordArray(InputArray)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlArrays::bind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlArrays::unbind() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

////////////////////////////////////////////////////////////////////////
// GlFont

cv::GlFont::GlFont(const string& _family, int _height, Weight _weight, Style _style)
    : family_(_family), height_(_height), weight_(_weight), style_(_style), base_(0)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlFont::draw(const char*, int) const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

Ptr<GlFont> cv::GlFont::get(const std::string&, int, Weight, Style)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return Ptr<GlFont>();
}

////////////////////////////////////////////////////////////////////////
// Rendering

void cv::render(const GlTexture&, Rect_<double>, Rect_<double>)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::render(const GlArrays&, int, Scalar)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::render(const string&, const Ptr<GlFont>&, Scalar, Point2d)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

////////////////////////////////////////////////////////////////////////
// GlCamera

cv::GlCamera::GlCamera() :
    eye_(0.0, 0.0, -5.0), center_(0.0, 0.0, 0.0), up_(0.0, 1.0, 0.0),
    pos_(0.0, 0.0, -5.0), yaw_(0.0), pitch_(0.0), roll_(0.0),
    useLookAtParams_(false),

    scale_(1.0, 1.0, 1.0),

    projectionMatrix_(),
    fov_(45.0), aspect_(0.0),
    left_(0.0), right_(1.0), bottom_(1.0), top_(0.0),
    zNear_(-1.0), zFar_(1.0),
    perspectiveProjection_(false)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::lookAt(Point3d, Point3d, Point3d)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setCameraPos(Point3d, double, double, double)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setScale(Point3d)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setProjectionMatrix(const Mat&, bool)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setPerspectiveProjection(double, double, double, double)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setOrthoProjection(double, double, double, double, double, double)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setupProjectionMatrix() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

void cv::GlCamera::setupModelViewMatrix() const
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
}

////////////////////////////////////////////////////////////////////////
// Error handling

bool icvCheckGlError(const char*, const int, const char*)
{
    CV_Error(CV_StsNotImplemented, "This function in deprecated, do not use it");
    return false;
}
