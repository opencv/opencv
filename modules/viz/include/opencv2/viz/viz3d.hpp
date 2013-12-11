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

#ifndef __OPENCV_VIZ_VIZ3D_HPP__
#define __OPENCV_VIZ_VIZ3D_HPP__

#if !defined YES_I_AGREE_THAT_VIZ_API_IS_NOT_STABLE_NOW_AND_BINARY_COMPARTIBILITY_WONT_BE_SUPPORTED && !defined CVAPI_EXPORTS
    //#error "Viz is in beta state now. Please define macro above to use it"
#endif

#include <opencv2/core.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>

namespace cv
{
    namespace viz
    {
        class CV_EXPORTS Viz3d
        {
        public:
            typedef void (*KeyboardCallback)(const KeyboardEvent&, void*);
            typedef void (*MouseCallback)(const MouseEvent&, void*);

            Viz3d(const String& window_name = String());
            Viz3d(const Viz3d&);
            Viz3d& operator=(const Viz3d&);
            ~Viz3d();

            void showWidget(const String &id, const Widget &widget, const Affine3f &pose = Affine3f::Identity());
            void removeWidget(const String &id);
            Widget getWidget(const String &id) const;
            void removeAllWidgets();

            void setWidgetPose(const String &id, const Affine3f &pose);
            void updateWidgetPose(const String &id, const Affine3f &pose);
            Affine3f getWidgetPose(const String &id) const;

            void setCamera(const Camera &camera);
            Camera getCamera() const;
            Affine3f getViewerPose();
            void setViewerPose(const Affine3f &pose);

            void resetCameraViewpoint(const String &id);
            void resetCamera();

            void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);
            void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);

            Size getWindowSize() const;
            void setWindowSize(const Size &window_size);
            String getWindowName() const;
            void saveScreenshot(const String &file);
            void setWindowPosition(int x, int y);
            void setFullScreen(bool mode);
            void setBackgroundColor(const Color& color = Color::black());

            void spin();
            void spinOnce(int time = 1, bool force_redraw = false);
            bool wasStopped() const;

            void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);
            void registerMouseCallback(MouseCallback callback, void* cookie = 0);

            void setRenderingProperty(const String &id, int property, double value);
            double getRenderingProperty(const String &id, int property);

            void setDesiredUpdateRate(double rate);
            double getDesiredUpdateRate();

            void setRepresentation(int representation);
        private:

            struct VizImpl;
            VizImpl* impl_;

            void create(const String &window_name);
            void release();

            friend class VizStorage;
        };

    } /* namespace viz */
} /* namespace cv */

#endif
