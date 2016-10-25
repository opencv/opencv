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
//M*/

#ifndef OPENCV_VIZ_VIZ3D_HPP
#define OPENCV_VIZ_VIZ3D_HPP

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

//! @addtogroup viz
//! @{

        /** @brief The Viz3d class represents a 3D visualizer window. This class is implicitly shared. :
        */
        class CV_EXPORTS Viz3d
        {
        public:
            typedef cv::viz::Color Color;
            typedef void (*KeyboardCallback)(const KeyboardEvent&, void*);
            typedef void (*MouseCallback)(const MouseEvent&, void*);

            /** @brief The constructors.

            @param window_name Name of the window.
             */
            Viz3d(const String& window_name = String());
            Viz3d(const Viz3d&);
            Viz3d& operator=(const Viz3d&);
            ~Viz3d();

            /** @brief Shows a widget in the window.

            @param id A unique id for the widget. @param widget The widget to be displayed in the window.
            @param pose Pose of the widget.
             */
            void showWidget(const String &id, const Widget &widget, const Affine3d &pose = Affine3d::Identity());

            /** @brief Removes a widget from the window.

            @param id The id of the widget that will be removed.
             */
            void removeWidget(const String &id);

            /** @brief Retrieves a widget from the window.

            A widget is implicitly shared; that is, if the returned widget is modified, the changes
            will be immediately visible in the window.

            @param id The id of the widget that will be returned.
             */
            Widget getWidget(const String &id) const;

            /** @brief Removes all widgets from the window.
            */
            void removeAllWidgets();

            /** @brief Removed all widgets and displays image scaled to whole window area.

            @param image Image to be displayed.
            @param window_size Size of Viz3d window. Default value means no change.
             */
            void showImage(InputArray image, const Size& window_size = Size(-1, -1));

            /** @brief Sets pose of a widget in the window.

            @param id The id of the widget whose pose will be set. @param pose The new pose of the widget.
             */
            void setWidgetPose(const String &id, const Affine3d &pose);

            /** @brief Updates pose of a widget in the window by pre-multiplying its current pose.

            @param id The id of the widget whose pose will be updated. @param pose The pose that the current
            pose of the widget will be pre-multiplied by.
             */
            void updateWidgetPose(const String &id, const Affine3d &pose);

            /** @brief Returns the current pose of a widget in the window.

            @param id The id of the widget whose pose will be returned.
             */
            Affine3d getWidgetPose(const String &id) const;

            /** @brief Sets the intrinsic parameters of the viewer using Camera.

            @param camera Camera object wrapping intrinsinc parameters.
             */
            void setCamera(const Camera &camera);

            /** @brief Returns a camera object that contains intrinsic parameters of the current viewer.
            */
            Camera getCamera() const;

            /** @brief Returns the current pose of the viewer.
            */
            Affine3d getViewerPose();

            /** @brief Sets pose of the viewer.

            @param pose The new pose of the viewer.
             */
            void setViewerPose(const Affine3d &pose);

            /** @brief Resets camera viewpoint to a 3D widget in the scene.

            @param id Id of a 3D widget.
             */
            void resetCameraViewpoint(const String &id);

            /** @brief Resets camera.
            */
            void resetCamera();

            /** @brief Transforms a point in world coordinate system to window coordinate system.

            @param pt Point in world coordinate system.
            @param window_coord Output point in window coordinate system.
             */
            void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);

            /** @brief Transforms a point in window coordinate system to a 3D ray in world coordinate system.

            @param window_coord Point in window coordinate system. @param origin Output origin of the ray.
            @param direction Output direction of the ray.
             */
            void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);

            /** @brief Returns the current size of the window.
            */
            Size getWindowSize() const;
            /** @brief Sets the size of the window.

            @param window_size New size of the window.
             */
            void setWindowSize(const Size &window_size);

            /** @brief Returns the name of the window which has been set in the constructor.
             */
            String getWindowName() const;

            /** @brief Returns the Mat screenshot of the current scene.
            */
            cv::Mat getScreenshot() const;

            /** @brief Saves screenshot of the current scene.

            @param file Name of the file.
             */
            void saveScreenshot(const String &file);

            /** @brief Sets the position of the window in the screen.

            @param window_position coordinates of the window
             */
            void setWindowPosition(const Point& window_position);

            /** @brief Sets or unsets full-screen rendering mode.

            @param mode If true, window will use full-screen mode.
             */
            void setFullScreen(bool mode = true);

            /** @brief Sets background color.
            */
            void setBackgroundColor(const Color& color = Color::black(), const Color& color2 = Color::not_set());
            void setBackgroundTexture(InputArray image = noArray());
            void setBackgroundMeshLab();

            /** @brief The window renders and starts the event loop.
            */
            void spin();

            /** @brief Starts the event loop for a given time.

            @param time Amount of time in milliseconds for the event loop to keep running.
            @param force_redraw If true, window renders.
             */
            void spinOnce(int time = 1, bool force_redraw = false);

            /** @brief Create a window in memory instead of on the screen.
             */
            void setOffScreenRendering();

            /** @brief Remove all lights from the current scene.
            */
            void removeAllLights();

            /** @brief Add a light in the scene.

            @param position The position of the light.
            @param focalPoint The point at which the light is shining
            @param color The color of the light
            @param diffuseColor The diffuse color of the light
            @param ambientColor The ambient color of the light
            @param specularColor The specular color of the light
             */
            void addLight(Vec3d position, Vec3d focalPoint = Vec3d(0, 0, 0), Color color = Color::white(),
                          Color diffuseColor = Color::white(), Color ambientColor = Color::black(), Color specularColor = Color::white());

            /** @brief Returns whether the event loop has been stopped.
            */
            bool wasStopped() const;
            void close();

            /** @brief Sets keyboard handler.

            @param callback Keyboard callback (void (\*KeyboardCallbackFunction(const
            KeyboardEvent&, void\*)).
            @param cookie The optional parameter passed to the callback.
             */
            void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);

            /** @brief Sets mouse handler.

            @param callback Mouse callback (void (\*MouseCallback)(const MouseEvent&, void\*)).
            @param cookie The optional parameter passed to the callback.
             */
            void registerMouseCallback(MouseCallback callback, void* cookie = 0);

            /** @brief Sets rendering property of a widget.

            @param id Id of the widget.
            @param property Property that will be modified.
            @param value The new value of the property.

            **Rendering property** can be one of the following:
            -   **POINT_SIZE**
            -   **OPACITY**
            -   **LINE_WIDTH**
            -   **FONT_SIZE**
            -
            **REPRESENTATION**: Expected values are
            -   **REPRESENTATION_POINTS**
            -   **REPRESENTATION_WIREFRAME**
            -   **REPRESENTATION_SURFACE**
            -
            **IMMEDIATE_RENDERING**:
            -   Turn on immediate rendering by setting the value to 1.
            -   Turn off immediate rendering by setting the value to 0.
            -
            **SHADING**: Expected values are
            -   **SHADING_FLAT**
            -   **SHADING_GOURAUD**
            -   **SHADING_PHONG**
             */
            void setRenderingProperty(const String &id, int property, double value);
            /** @brief Returns rendering property of a widget.

            @param id Id of the widget.
            @param property Property.

            **Rendering property** can be one of the following:
            -   **POINT_SIZE**
            -   **OPACITY**
            -   **LINE_WIDTH**
            -   **FONT_SIZE**
            -
            **REPRESENTATION**: Expected values are
            -   **REPRESENTATION_POINTS**
            -   **REPRESENTATION_WIREFRAME**
            -   **REPRESENTATION_SURFACE**
            -
            **IMMEDIATE_RENDERING**:
            -   Turn on immediate rendering by setting the value to 1.
            -   Turn off immediate rendering by setting the value to 0.
            -
            **SHADING**: Expected values are
            -   **SHADING_FLAT**
            -   **SHADING_GOURAUD**
            -   **SHADING_PHONG**
             */
            double getRenderingProperty(const String &id, int property);

            /** @brief Sets geometry representation of the widgets to surface, wireframe or points.

            @param representation Geometry representation which can be one of the following:
            -   **REPRESENTATION_POINTS**
            -   **REPRESENTATION_WIREFRAME**
            -   **REPRESENTATION_SURFACE**
             */
            void setRepresentation(int representation);

            void setGlobalWarnings(bool enabled = false);
        private:

            struct VizImpl;
            VizImpl* impl_;

            void create(const String &window_name);
            void release();

            friend class VizStorage;
        };

//! @}

    } /* namespace viz */
} /* namespace cv */

#endif
