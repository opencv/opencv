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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Ozan Tonkal, ozantonkal@gmail.com
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//  OpenCV Viz module is complete rewrite of
//  PCL visualization module (www.pointclouds.org)
//
//M*/

#include "precomp.hpp"

cv::viz::Viz3d::Viz3d(const String& window_name) : impl_(0) { create(window_name); }

cv::viz::Viz3d::Viz3d(const Viz3d& other) : impl_(other.impl_)
{
    if (impl_)
        CV_XADD(&impl_->ref_counter, 1);
}

cv::viz::Viz3d& cv::viz::Viz3d::operator=(const Viz3d& other)
{
    if (this != &other)
    {
        release();
        impl_ = other.impl_;
        if (impl_)
            CV_XADD(&impl_->ref_counter, 1);
    }
    return *this;
}

cv::viz::Viz3d::~Viz3d() { release(); }

void cv::viz::Viz3d::create(const String &window_name)
{
    if (impl_)
        release();

    if (VizStorage::windowExists(window_name))
        *this = VizStorage::get(window_name);
    else
    {
        impl_ = new VizImpl(window_name);
        impl_->ref_counter = 1;

        // Register the window
        VizStorage::add(*this);
    }
}

void cv::viz::Viz3d::release()
{
    if (impl_ && CV_XADD(&impl_->ref_counter, -1) == 1)
    {
        delete impl_;
        impl_ = 0;
    }

    if (impl_ && impl_->ref_counter == 1)
        VizStorage::removeUnreferenced();

    impl_ = 0;
}

void cv::viz::Viz3d::spin() { impl_->spin(); }
void cv::viz::Viz3d::spinOnce(int time, bool force_redraw) { impl_->spinOnce(time, force_redraw); }
bool cv::viz::Viz3d::wasStopped() const { return impl_->wasStopped(); }

void cv::viz::Viz3d::registerKeyboardCallback(KeyboardCallback callback, void* cookie)
{ impl_->registerKeyboardCallback(callback, cookie); }

void cv::viz::Viz3d::registerMouseCallback(MouseCallback callback, void* cookie)
{ impl_->registerMouseCallback(callback, cookie); }

void cv::viz::Viz3d::showWidget(const String &id, const Widget &widget, const Affine3f &pose) { impl_->showWidget(id, widget, pose); }
void cv::viz::Viz3d::removeWidget(const String &id) { impl_->removeWidget(id); }
cv::viz::Widget cv::viz::Viz3d::getWidget(const String &id) const { return impl_->getWidget(id); }
void cv::viz::Viz3d::removeAllWidgets() { impl_->removeAllWidgets(); }
void cv::viz::Viz3d::setWidgetPose(const String &id, const Affine3f &pose) { impl_->setWidgetPose(id, pose); }
void cv::viz::Viz3d::updateWidgetPose(const String &id, const Affine3f &pose) { impl_->updateWidgetPose(id, pose); }
cv::Affine3f cv::viz::Viz3d::getWidgetPose(const String &id) const { return impl_->getWidgetPose(id); }

void cv::viz::Viz3d::setCamera(const Camera &camera) { impl_->setCamera(camera); }
cv::viz::Camera cv::viz::Viz3d::getCamera() const { return impl_->getCamera(); }
void cv::viz::Viz3d::setViewerPose(const Affine3f &pose) { impl_->setViewerPose(pose); }
cv::Affine3f cv::viz::Viz3d::getViewerPose() { return impl_->getViewerPose(); }

void cv::viz::Viz3d::resetCameraViewpoint(const String &id) { impl_->resetCameraViewpoint(id); }
void cv::viz::Viz3d::resetCamera() { impl_->resetCamera(); }

void cv::viz::Viz3d::convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord) { impl_->convertToWindowCoordinates(pt, window_coord); }
void cv::viz::Viz3d::converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction) { impl_->converTo3DRay(window_coord, origin, direction); }

cv::Size cv::viz::Viz3d::getWindowSize() const { return impl_->getWindowSize(); }
void cv::viz::Viz3d::setWindowSize(const Size &window_size) { impl_->setWindowSize(window_size.width, window_size.height); }
cv::String cv::viz::Viz3d::getWindowName() const { return impl_->getWindowName(); }
void cv::viz::Viz3d::saveScreenshot(const String &file) { impl_->saveScreenshot(file); }
void cv::viz::Viz3d::setWindowPosition(int x, int y) { impl_->setWindowPosition(x,y); }
void cv::viz::Viz3d::setFullScreen(bool mode) { impl_->setFullScreen(mode); }
void cv::viz::Viz3d::setBackgroundColor(const Color& color) { impl_->setBackgroundColor(color); }

void cv::viz::Viz3d::setRenderingProperty(const String &id, int property, double value) { getWidget(id).setRenderingProperty(property, value); }
double cv::viz::Viz3d::getRenderingProperty(const String &id, int property) { return getWidget(id).getRenderingProperty(property); }

void cv::viz::Viz3d::setDesiredUpdateRate(double rate) { impl_->setDesiredUpdateRate(rate); }
double cv::viz::Viz3d::getDesiredUpdateRate() { return impl_->getDesiredUpdateRate(); }

void cv::viz::Viz3d::setRepresentation(int representation) { impl_->setRepresentation(representation); }
