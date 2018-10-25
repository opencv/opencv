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

#ifndef __OPENCV_VIZ_VIZ3D_IMPL_HPP__
#define __OPENCV_VIZ_VIZ3D_IMPL_HPP__

struct cv::viz::Viz3d::VizImpl
{
public:
    typedef Viz3d::KeyboardCallback KeyboardCallback;
    typedef Viz3d::MouseCallback MouseCallback;

    int ref_counter;

    VizImpl(const String &name);
    virtual ~VizImpl();

    bool wasStopped() const;
    void close();

    void spin();
    void spinOnce(int time = 1, bool force_redraw = false);
    void setOffScreenRendering();

    void removeAllLights();
    void addLight(Vec3d position, Vec3d focalPoint, const Color &color, const Color &diffuseColor,
                  const Color &ambientColor, const Color &specularColor);

    void showWidget(const String &id, const Widget &widget, const Affine3d &pose = Affine3d::Identity());
    void removeWidget(const String &id);
    Widget getWidget(const String &id) const;
    void removeAllWidgets();

    void showImage(InputArray image, const Size& window_size);

    void setWidgetPose(const String &id, const Affine3d &pose);
    void updateWidgetPose(const String &id, const Affine3d &pose);
    Affine3d getWidgetPose(const String &id) const;

    void setRepresentation(int representation);

    void setCamera(const Camera &camera);
    Camera getCamera() const;

    /** \brief Reset the camera to a given widget */
    void resetCameraViewpoint(const String& id);
    void resetCamera();

    void setViewerPose(const Affine3d &pose);
    Affine3d getViewerPose() const;

    void convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord);
    void converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction);

    Mat getScreenshot() const;
    void saveScreenshot(const String &file);
    void setWindowPosition(const Point& position);
    Size getWindowSize() const;
    void setWindowSize(const Size& window_size);
    void setFullScreen(bool mode);
    String getWindowName() const;
    void setBackgroundColor(const Color& color, const Color& color2);
    void setBackgroundTexture(InputArray image);
    void setBackgroundMeshLab();

    void registerKeyboardCallback(KeyboardCallback callback, void* cookie = 0);
    void registerMouseCallback(MouseCallback callback, void* cookie = 0);

private:
    struct TimerCallback : public vtkCommand
    {
        static TimerCallback* New() { return new TimerCallback; }
        virtual void Execute(vtkObject* caller, unsigned long event_id, void* cookie);
        int timer_id;
    };

    struct ExitCallback : public vtkCommand
    {
        static ExitCallback* New() { return new ExitCallback; }
        virtual void Execute(vtkObject*, unsigned long event_id, void*);
        VizImpl* viz;
    };

    mutable bool spin_once_state_;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    vtkSmartPointer<vtkRenderWindow> window_;
    String window_name_;
    Vec2i window_position_;

    vtkSmartPointer<TimerCallback> timer_callback_;
    vtkSmartPointer<ExitCallback> exit_callback_;

    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkVizInteractorStyle> style_;
    Ptr<WidgetActorMap> widget_actor_map_;

    bool offScreenMode_;

    bool removeActorFromRenderer(vtkSmartPointer<vtkProp> actor);
    void recreateRenderWindow();
};

#endif
