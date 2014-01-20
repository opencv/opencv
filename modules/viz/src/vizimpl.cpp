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

#include "precomp.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Viz3d::VizImpl::VizImpl(const String &name) : spin_once_state_(false),
    window_position_(Vec2i(std::numeric_limits<int>::min())), widget_actor_map_(new WidgetActorMap)
{
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    window_name_ = VizStorage::generateWindowName(name);

    // Create render window
    window_ = vtkSmartPointer<vtkRenderWindow>::New();
    cv::Vec2i window_size = cv::Vec2i(window_->GetScreenSize()) / 2;
    window_->SetSize(window_size.val);
    window_->AddRenderer(renderer_);

    // Create the interactor style
    style_ = vtkSmartPointer<InteractorStyle>::New();
    style_->setWidgetActorMap(widget_actor_map_);
    style_->UseTimersOn();
    style_->Initialize();

    timer_callback_ = vtkSmartPointer<TimerCallback>::New();
    exit_callback_ = vtkSmartPointer<ExitCallback>::New();
    exit_callback_->viz = this;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::TimerCallback::Execute(vtkObject* caller, unsigned long event_id, void* cookie)
{
    if (event_id == vtkCommand::TimerEvent && timer_id == *reinterpret_cast<int*>(cookie))
    {
        vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkRenderWindowInteractor::SafeDownCast(caller);
        interactor->TerminateApp();
    }
}

void cv::viz::Viz3d::VizImpl::ExitCallback::Execute(vtkObject*, unsigned long event_id, void*)
{
    if (event_id == vtkCommand::ExitEvent)
    {
        viz->interactor_->TerminateApp();
        viz->interactor_ = 0;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////

bool cv::viz::Viz3d::VizImpl::wasStopped() const
{
    bool stopped = spin_once_state_ ? interactor_ == 0 : false;
    spin_once_state_ &= !stopped;
    return stopped;
}

void cv::viz::Viz3d::VizImpl::close()
{
    if (!interactor_)
        return;
    interactor_->GetRenderWindow()->Finalize();
    interactor_->TerminateApp(); // This tends to close the window...
    interactor_ = 0;
}

void cv::viz::Viz3d::VizImpl::recreateRenderWindow()
{
#if !defined _MSC_VER
    //recreating is workaround for Ubuntu -- a crash in x-server
    Vec2i window_size(window_->GetSize());
    int fullscreen = window_->GetFullScreen();

    window_ = vtkSmartPointer<vtkRenderWindow>::New();
    if (window_position_[0] != std::numeric_limits<int>::min()) //also workaround
        window_->SetPosition(window_position_.val);

    window_->SetSize(window_size.val);
    window_->SetFullScreen(fullscreen);
    window_->AddRenderer(renderer_);
#endif
}


/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spin()
{
    recreateRenderWindow();
    interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor_->SetRenderWindow(window_);
    interactor_->SetInteractorStyle(style_);
    window_->AlphaBitPlanesOff();
    window_->PointSmoothingOff();
    window_->LineSmoothingOff();
    window_->PolygonSmoothingOff();
    window_->SwapBuffersOn();
    window_->SetStereoTypeToAnaglyph();
    window_->Render();
    window_->SetWindowName(window_name_.c_str());
    interactor_->Start();
    interactor_ = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spinOnce(int time, bool force_redraw)
{
    if (interactor_ == 0)
    {
        spin_once_state_ = true;
        recreateRenderWindow();
        interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        interactor_->SetRenderWindow(window_);
        interactor_->SetInteractorStyle(style_);
        interactor_->AddObserver(vtkCommand::TimerEvent, timer_callback_);
        interactor_->AddObserver(vtkCommand::ExitEvent, exit_callback_);
        window_->AlphaBitPlanesOff();
        window_->PointSmoothingOff();
        window_->LineSmoothingOff();
        window_->PolygonSmoothingOff();
        window_->SwapBuffersOn();
        window_->SetStereoTypeToAnaglyph();
        window_->Render();
        window_->SetWindowName(window_name_.c_str());
    }

    vtkSmartPointer<vtkRenderWindowInteractor> local = interactor_;

    if (force_redraw)
        local->Render();

    timer_callback_->timer_id = local->CreateRepeatingTimer(std::max(1, time));
    local->Start();
    local->DestroyTimer(timer_callback_->timer_id);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::showWidget(const String &id, const Widget &widget, const Affine3d &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    if (exists)
    {
        // Remove it if it exists and add it again
        removeActorFromRenderer(wam_itr->second);
    }
    // Get the actor and set the user matrix
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(widget));
    if (actor)
    {
        // If the actor is 3D, apply pose
        vtkSmartPointer<vtkMatrix4x4> matrix = vtkmatrix(pose.matrix);
        actor->SetUserMatrix(matrix);
        actor->Modified();
    }
    // If the actor is a vtkFollower, then it should always face the camera
    vtkFollower *follower = vtkFollower::SafeDownCast(actor);
    if (follower)
    {
        follower->SetCamera(renderer_->GetActiveCamera());
    }

    renderer_->AddActor(WidgetAccessor::getProp(widget));
    (*widget_actor_map_)[id] = WidgetAccessor::getProp(widget);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::removeWidget(const String &id)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);
    CV_Assert("Widget could not be removed." && removeActorFromRenderer(wam_itr->second));
    widget_actor_map_->erase(wam_itr);
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Widget cv::viz::Viz3d::VizImpl::getWidget(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    Widget widget;
    WidgetAccessor::setProp(widget, wam_itr->second);
    return widget;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setWidgetPose(const String &id, const Affine3d &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second);
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = vtkmatrix(pose.matrix);
    actor->SetUserMatrix(matrix);
    actor->Modified();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::updateWidgetPose(const String &id, const Affine3d &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second);
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    if (!matrix)
    {
        setWidgetPose(id, pose);
        return ;
    }
    Affine3d updated_pose = pose * Affine3d(*matrix->Element);
    matrix = vtkmatrix(updated_pose.matrix);

    actor->SetUserMatrix(matrix);
    actor->Modified();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::Affine3d cv::viz::Viz3d::VizImpl::getWidgetPose(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second);
    CV_Assert("Widget is not 3D." && actor);

    return Affine3d(*actor->GetUserMatrix()->Element);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::saveScreenshot(const String &file) { style_->saveScreenshot(file.c_str()); }

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::registerMouseCallback(MouseCallback callback, void* cookie)
{ style_->registerMouseCallback(callback, cookie); }

void cv::viz::Viz3d::VizImpl::registerKeyboardCallback(KeyboardCallback callback, void* cookie)
{ style_->registerKeyboardCallback(callback, cookie); }


//////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::removeAllWidgets()
{
    widget_actor_map_->clear();
    renderer_->RemoveAllViewProps();
}
/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::showImage(InputArray image, const Size& window_size)
{
    removeAllWidgets();
    if (window_size.width > 0 && window_size.height > 0)
        setWindowSize(window_size);

    showWidget("showImage", WImageOverlay(image, Rect(Point(0,0), getWindowSize())));
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeActorFromRenderer(vtkSmartPointer<vtkProp> actor)
{
    vtkPropCollection* actors = renderer_->GetViewProps();
    actors->InitTraversal();
    vtkProp* current_actor = NULL;
    while ((current_actor = actors->GetNextProp()) != NULL)
        if (current_actor == actor)
        {
            renderer_->RemoveActor(actor);
            return true;
        }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setBackgroundColor(const Color& color, const Color& color2)
{
    Color c = vtkcolor(color), c2 = vtkcolor(color2);
    bool gradient = color2[0] >= 0 && color2[1] >= 0 && color2[2] >= 0;

    if (gradient)
    {
        renderer_->SetBackground(c2.val);
        renderer_->SetBackground2(c.val);
        renderer_->GradientBackgroundOn();
    }
    else
    {
        renderer_->SetBackground(c.val);
        renderer_->GradientBackgroundOff();
    }
}

void cv::viz::Viz3d::VizImpl::setBackgroundMeshLab()
{ setBackgroundColor(Color(2, 1, 1), Color(240, 120, 120)); }

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setBackgroundTexture(InputArray image)
{
    if (image.empty())
    {
        renderer_->SetBackgroundTexture(0);
        renderer_->TexturedBackgroundOff();
        return;
    }

    vtkSmartPointer<vtkImageMatSource> source = vtkSmartPointer<vtkImageMatSource>::New();
    source->SetImage(image);

    vtkSmartPointer<vtkImageFlip> image_flip = vtkSmartPointer<vtkImageFlip>::New();
    image_flip->SetFilteredAxis(1); // Vertical flip
    image_flip->SetInputConnection(source->GetOutputPort());

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(image_flip->GetOutputPort());
    //texture->Update();

    renderer_->SetBackgroundTexture(texture);
    renderer_->TexturedBackgroundOn();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCamera(const Camera &camera)
{
    vtkSmartPointer<vtkCamera> active_camera = renderer_->GetActiveCamera();

    // Set the intrinsic parameters of the camera
    window_->SetSize(camera.getWindowSize().width, camera.getWindowSize().height);
    double aspect_ratio = static_cast<double>(camera.getWindowSize().width)/static_cast<double>(camera.getWindowSize().height);

    Matx44d proj_mat;
    camera.computeProjectionMatrix(proj_mat);

    // Use the intrinsic parameters of the camera to simulate more realistically
    vtkSmartPointer<vtkMatrix4x4> vtk_matrix = active_camera->GetProjectionTransformMatrix(aspect_ratio, -1.0, 1.0);
    Matx44d old_proj_mat(*vtk_matrix->Element);

    // This is a hack around not being able to set Projection Matrix
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->SetMatrix(vtkmatrix(proj_mat * old_proj_mat.inv()));
    active_camera->SetUserTransform(transform);

    renderer_->ResetCameraClippingRange();
    renderer_->Render();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Camera cv::viz::Viz3d::VizImpl::getCamera() const
{
    vtkSmartPointer<vtkCamera> active_camera = renderer_->GetActiveCamera();

    Size window_size(renderer_->GetRenderWindow()->GetSize()[0],
                     renderer_->GetRenderWindow()->GetSize()[1]);
    double aspect_ratio = window_size.width / (double)window_size.height;

    vtkSmartPointer<vtkMatrix4x4> proj_matrix = active_camera->GetProjectionTransformMatrix(aspect_ratio, -1.0f, 1.0f);
    return Camera(Matx44d(*proj_matrix->Element), window_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setViewerPose(const Affine3d &pose)
{
    vtkCamera& camera = *renderer_->GetActiveCamera();

    // Position = extrinsic translation
    cv::Vec3d pos_vec = pose.translation();

    // Rotate the view vector
    cv::Matx33d rotation = pose.rotation();
    cv::Vec3d y_axis(0.0, 1.0, 0.0);
    cv::Vec3d up_vec(rotation * y_axis);

    // Compute the new focal point
    cv::Vec3d z_axis(0.0, 0.0, 1.0);
    cv::Vec3d focal_vec = pos_vec + rotation * z_axis;

    camera.SetPosition(pos_vec.val);
    camera.SetFocalPoint(focal_vec.val);
    camera.SetViewUp(up_vec.val);

    renderer_->ResetCameraClippingRange();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::Affine3d cv::viz::Viz3d::VizImpl::getViewerPose()
{
    vtkCamera& camera = *renderer_->GetActiveCamera();

    Vec3d pos(camera.GetPosition());
    Vec3d view_up(camera.GetViewUp());
    Vec3d focal(camera.GetFocalPoint());

    Vec3d y_axis = normalized(view_up);
    Vec3d z_axis = normalized(focal - pos);
    Vec3d x_axis = normalized(y_axis.cross(z_axis));

    return makeTransformToGlobal(x_axis, y_axis, z_axis, pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::convertToWindowCoordinates(const Point3d &pt, Point3d &window_coord)
{
    Vec3d window_pt;
    vtkInteractorObserver::ComputeWorldToDisplay(renderer_, pt.x, pt.y, pt.z, window_pt.val);
    window_coord = window_pt;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::converTo3DRay(const Point3d &window_coord, Point3d &origin, Vec3d &direction)
{
    Vec4d world_pt;
    vtkInteractorObserver::ComputeDisplayToWorld(renderer_, window_coord.x, window_coord.y, window_coord.z, world_pt.val);
    Vec3d cam_pos(renderer_->GetActiveCamera()->GetPosition());
    origin = cam_pos;
    direction = normalize(Vec3d(world_pt.val) - cam_pos);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::resetCameraViewpoint(const String &id)
{
    vtkSmartPointer<vtkMatrix4x4> camera_pose;
    static WidgetActorMap::iterator it = widget_actor_map_->find(id);
    if (it != widget_actor_map_->end())
    {
        vtkProp3D *actor = vtkProp3D::SafeDownCast(it->second);
        CV_Assert("Widget is not 3D." && actor);
        camera_pose = actor->GetUserMatrix();
    }
    else
        return;

    // Prevent a segfault
    if (!camera_pose) return;

    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera();
    cam->SetPosition(camera_pose->GetElement(0, 3),
                     camera_pose->GetElement(1, 3),
                     camera_pose->GetElement(2, 3));

    cam->SetFocalPoint(camera_pose->GetElement(0, 3) - camera_pose->GetElement(0, 2),
                       camera_pose->GetElement(1, 3) - camera_pose->GetElement(1, 2),
                       camera_pose->GetElement(2, 3) - camera_pose->GetElement(2, 2));

    cam->SetViewUp(camera_pose->GetElement(0, 1),
                   camera_pose->GetElement(1, 1),
                   camera_pose->GetElement(2, 1));

    renderer_->SetActiveCamera(cam);
    renderer_->ResetCameraClippingRange();
    renderer_->Render();
}

///////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::resetCamera()
{
    renderer_->ResetCamera();
}

///////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setRepresentation(int representation)
{
    vtkActorCollection * actors = renderer_->GetActors();
    actors->InitTraversal();
    vtkActor * actor;
    switch (representation)
    {
        case REPRESENTATION_POINTS:
        {
            while ((actor = actors->GetNextActor()) != NULL)
                actor->GetProperty()->SetRepresentationToPoints();
            break;
        }
        case REPRESENTATION_SURFACE:
        {
            while ((actor = actors->GetNextActor()) != NULL)
                actor->GetProperty()->SetRepresentationToSurface();
            break;
        }
        case REPRESENTATION_WIREFRAME:
        {
            while ((actor = actors->GetNextActor()) != NULL)
                actor->GetProperty()->SetRepresentationToWireframe();
            break;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
cv::String cv::viz::Viz3d::VizImpl::getWindowName() const { return window_name_; }
void cv::viz::Viz3d::VizImpl::setFullScreen(bool mode) { window_->SetFullScreen(mode); }
void cv::viz::Viz3d::VizImpl::setWindowPosition(const Point& position) { window_position_ = position; window_->SetPosition(position.x, position.y); }
void cv::viz::Viz3d::VizImpl::setWindowSize(const Size& window_size) { window_->SetSize(window_size.width, window_size.height); }
cv::Size cv::viz::Viz3d::VizImpl::getWindowSize() const { return Size(Point(Vec2i(window_->GetSize()))); }
