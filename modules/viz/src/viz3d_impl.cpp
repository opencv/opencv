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

#if 1 || !defined __APPLE__
vtkRenderWindowInteractor* vtkRenderWindowInteractorFixNew()
{
  return vtkRenderWindowInteractor::New();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Viz3d::VizImpl::VizImpl(const String &name)
    :  style_(vtkSmartPointer<cv::viz::InteractorStyle>::New()) , widget_actor_map_(new WidgetActorMap), s_lastDone_(0.0)
{
    renderer_ = vtkSmartPointer<vtkRenderer>::New();

    // Create a RendererWindow
    window_ = vtkSmartPointer<vtkRenderWindow>::New();

    // Set the window size as 1/2 of the screen size
    cv::Vec2i window_size = cv::Vec2i(window_->GetScreenSize()) / 2;
    window_->SetSize(window_size.val);

    window_->AddRenderer(renderer_);

    // Create the interactor style
    style_->Initialize();
    style_->setRenderer(renderer_);
    style_->setWidgetActorMap(widget_actor_map_);
    style_->UseTimersOn();

    /////////////////////////////////////////////////
    interactor_ = vtkSmartPointer <vtkRenderWindowInteractor>::Take(vtkRenderWindowInteractorFixNew());

    window_->AlphaBitPlanesOff();
    window_->PointSmoothingOff();
    window_->LineSmoothingOff();
    window_->PolygonSmoothingOff();
    window_->SwapBuffersOn();
    window_->SetStereoTypeToAnaglyph();

    interactor_->SetRenderWindow(window_);
    interactor_->SetInteractorStyle(style_);
    interactor_->SetDesiredUpdateRate(30.0);

    // Initialize and create timer, also create window
    interactor_->Initialize();
    timer_id_ = interactor_->CreateRepeatingTimer(5000L);

    // Set a simple PointPicker
    vtkSmartPointer<vtkPointPicker> pp = vtkSmartPointer<vtkPointPicker>::New();
    pp->SetTolerance(pp->GetTolerance() * 2);
    interactor_->SetPicker(pp);

    exit_main_loop_timer_callback_ = vtkSmartPointer<ExitMainLoopTimerCallback>::New();
    exit_main_loop_timer_callback_->viz_ = this;
    exit_main_loop_timer_callback_->right_timer_id = -1;
    interactor_->AddObserver(vtkCommand::TimerEvent, exit_main_loop_timer_callback_);

    exit_callback_ = vtkSmartPointer<ExitCallback>::New();
    exit_callback_->viz_ = this;
    interactor_->AddObserver(vtkCommand::ExitEvent, exit_callback_);

    resetStoppedFlag();


    //////////////////////////////
    String window_name;
    VizAccessor::generateWindowName(name, window_name);
    window_->SetWindowName(window_name.c_str());
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Viz3d::VizImpl::~VizImpl()
{
    if (interactor_)
        interactor_->DestroyTimer(timer_id_);
    if (renderer_)
        renderer_->Clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::showWidget(const String &id, const Widget &widget, const Affine3f &pose)
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
        vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
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
void cv::viz::Viz3d::VizImpl::setWidgetPose(const String &id, const Affine3f &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second);
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
    actor->SetUserMatrix(matrix);
    actor->Modified();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::updateWidgetPose(const String &id, const Affine3f &pose)
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
    Matx44f matrix_cv = convertToMatx(matrix);
    Affine3f updated_pose = pose * Affine3f(matrix_cv);
    matrix = convertToVtkMatrix(updated_pose.matrix);

    actor->SetUserMatrix(matrix);
    actor->Modified();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::Affine3f cv::viz::Viz3d::VizImpl::getWidgetPose(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert("Widget does not exist." && exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second);
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    Matx44f matrix_cv = convertToMatx(matrix);
    return Affine3f(matrix_cv);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setDesiredUpdateRate(double rate)
{
    if (interactor_)
        interactor_->SetDesiredUpdateRate(rate);
}

/////////////////////////////////////////////////////////////////////////////////////////////
double cv::viz::Viz3d::VizImpl::getDesiredUpdateRate()
{
    if (interactor_)
        return interactor_->GetDesiredUpdateRate();
    return 0.0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::saveScreenshot(const String &file) { style_->saveScreenshot(file.c_str()); }

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::registerMouseCallback(MouseCallback callback, void* cookie)
{ style_->registerMouseCallback(callback, cookie); }

void cv::viz::Viz3d::VizImpl::registerKeyboardCallback(KeyboardCallback callback, void* cookie)
{ style_->registerKeyboardCallback(callback, cookie); }

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spin()
{
    resetStoppedFlag();
    window_->Render();
    interactor_->Start();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spinOnce(int time, bool force_redraw)
{
    resetStoppedFlag();

    if (time <= 0)
        time = 1;

    if (force_redraw)
        interactor_->Render();

    double s_now_ = cv::getTickCount() / cv::getTickFrequency();
    if (s_lastDone_ > s_now_)
      s_lastDone_ = s_now_;

    if ((s_now_ - s_lastDone_) > (1.0 / interactor_->GetDesiredUpdateRate()))
    {
        exit_main_loop_timer_callback_->right_timer_id = interactor_->CreateRepeatingTimer(time);
        interactor_->Start();
        interactor_->DestroyTimer(exit_main_loop_timer_callback_->right_timer_id);
        s_lastDone_ = s_now_;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::removeAllWidgets()
{
    widget_actor_map_->clear();
    renderer_->RemoveAllViewProps();
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeActorFromRenderer(const vtkSmartPointer<vtkProp> &actor)
{
    vtkProp* actor_to_remove = vtkProp::SafeDownCast(actor);

    vtkPropCollection* actors = renderer_->GetViewProps();
    actors->InitTraversal();
    vtkProp* current_actor = NULL;
    while ((current_actor = actors->GetNextProp()) != NULL)
    {
        if (current_actor != actor_to_remove)
            continue;
        renderer_->RemoveActor(actor);
        return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::createActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet> &data, vtkSmartPointer<vtkLODActor> &actor, bool use_scalars)
{
    if (!actor)
        actor = vtkSmartPointer<vtkLODActor>::New();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(data);
#else
    mapper->SetInputData(data);
#endif

    if (use_scalars)
    {
        vtkSmartPointer<vtkDataArray> scalars = data->GetPointData()->GetScalars();
        if (scalars)
        {
            cv::Vec3d minmax(scalars->GetRange());
            mapper->SetScalarRange(minmax.val);
            mapper->SetScalarModeToUsePointData();

            // interpolation OFF, if data is a vtkPolyData that contains only vertices, ON for anything else.
            vtkPolyData* polyData = vtkPolyData::SafeDownCast(data);
            bool interpolation = (polyData && polyData->GetNumberOfCells() != polyData->GetNumberOfVerts());

            mapper->SetInterpolateScalarsBeforeMapping(interpolation);
            mapper->ScalarVisibilityOn();
        }
    }
    mapper->ImmediateModeRenderingOff();

    actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();

    actor->SetMapper(mapper);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setBackgroundColor(const Color& color)
{
    Color c = vtkcolor(color);
    renderer_->SetBackground(c.val);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCamera(const Camera &camera)
{
    vtkCamera& active_camera = *renderer_->GetActiveCamera();

    // Set the intrinsic parameters of the camera
    window_->SetSize(camera.getWindowSize().width, camera.getWindowSize().height);
    double aspect_ratio = static_cast<double>(camera.getWindowSize().width)/static_cast<double>(camera.getWindowSize().height);

    Matx44f proj_mat;
    camera.computeProjectionMatrix(proj_mat);
    // Use the intrinsic parameters of the camera to simulate more realistically
    Matx44f old_proj_mat = convertToMatx(active_camera.GetProjectionTransformMatrix(aspect_ratio, -1.0, 1.0));
    vtkTransform *transform = vtkTransform::New();
    // This is a hack around not being able to set Projection Matrix
    transform->SetMatrix(convertToVtkMatrix(proj_mat * old_proj_mat.inv()));
    active_camera.SetUserTransform(transform);
    transform->Delete();

    renderer_->ResetCameraClippingRange();
    renderer_->Render();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Camera cv::viz::Viz3d::VizImpl::getCamera() const
{
    vtkCamera& active_camera = *renderer_->GetActiveCamera();

    Size window_size(renderer_->GetRenderWindow()->GetSize()[0],
                     renderer_->GetRenderWindow()->GetSize()[1]);
    double aspect_ratio = window_size.width / (double)window_size.height;

    Matx44f proj_matrix = convertToMatx(active_camera.GetProjectionTransformMatrix(aspect_ratio, -1.0f, 1.0f));
    Camera camera(proj_matrix, window_size);
    return camera;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setViewerPose(const Affine3f &pose)
{
    vtkCamera& camera = *renderer_->GetActiveCamera();

    // Position = extrinsic translation
    cv::Vec3f pos_vec = pose.translation();

    // Rotate the view vector
    cv::Matx33f rotation = pose.rotation();
    cv::Vec3f y_axis(0.f, 1.f, 0.f);
    cv::Vec3f up_vec(rotation * y_axis);

    // Compute the new focal point
    cv::Vec3f z_axis(0.f, 0.f, 1.f);
    cv::Vec3f focal_vec = pos_vec + rotation * z_axis;

    camera.SetPosition(pos_vec[0], pos_vec[1], pos_vec[2]);
    camera.SetFocalPoint(focal_vec[0], focal_vec[1], focal_vec[2]);
    camera.SetViewUp(up_vec[0], up_vec[1], up_vec[2]);

    renderer_->ResetCameraClippingRange();
    renderer_->Render();
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::Affine3f cv::viz::Viz3d::VizImpl::getViewerPose()
{
    vtkCamera& camera = *renderer_->GetActiveCamera();

    Vec3d pos(camera.GetPosition());
    Vec3d view_up(camera.GetViewUp());
    Vec3d focal(camera.GetFocalPoint());

    Vec3d y_axis = normalized(view_up);
    Vec3d z_axis = normalized(focal - pos);
    Vec3d x_axis = normalized(y_axis.cross(z_axis));

    cv::Matx33d R;
    R(0, 0) = x_axis[0];
    R(0, 1) = y_axis[0];
    R(0, 2) = z_axis[0];

    R(1, 0) = x_axis[1];
    R(1, 1) = y_axis[1];
    R(1, 2) = z_axis[1];

    R(2, 0) = x_axis[2];
    R(2, 1) = y_axis[2];
    R(2, 2) = z_axis[2];

    return cv::Affine3f(R, pos);
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

    vtkCamera &active_camera = *renderer_->GetActiveCamera();
    Vec3d cam_pos;
    active_camera.GetPosition(cam_pos.val);
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
void cv::viz::Viz3d::VizImpl::updateCells(vtkSmartPointer<vtkIdTypeArray> &cells, vtkSmartPointer<vtkIdTypeArray> &initcells, vtkIdType nr_points)
{
    // If no init cells and cells has not been initialized...
    if (!cells)
        cells = vtkSmartPointer<vtkIdTypeArray>::New();

    // If we have less values then we need to recreate the array
    if (cells->GetNumberOfTuples() < nr_points)
    {
        cells = vtkSmartPointer<vtkIdTypeArray>::New();

        // If init cells is given, and there's enough data in it, use it
        if (initcells && initcells->GetNumberOfTuples() >= nr_points)
        {
            cells->DeepCopy(initcells);
            cells->SetNumberOfComponents(2);
            cells->SetNumberOfTuples(nr_points);
        }
        else
        {
            // If the number of tuples is still too small, we need to recreate the array
            cells->SetNumberOfComponents(2);
            cells->SetNumberOfTuples(nr_points);
            vtkIdType *cell = cells->GetPointer(0);
            // Fill it with 1s
            std::fill_n(cell, nr_points * 2, 1);
            cell++;
            for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
                *cell = i;
            // Save the results in initcells
            initcells = vtkSmartPointer<vtkIdTypeArray>::New();
            initcells->DeepCopy(cells);
        }
    }
    else
    {
        // The assumption here is that the current set of cells has more data than needed
        cells->SetNumberOfComponents(2);
        cells->SetNumberOfTuples(nr_points);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setFullScreen(bool mode)
{
    if (window_)
        window_->SetFullScreen(mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
cv::String cv::viz::Viz3d::VizImpl::getWindowName() const
{
    return (window_ ? window_->GetWindowName() : "");
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setWindowPosition(int x, int y) { window_->SetPosition(x, y); }
void cv::viz::Viz3d::VizImpl::setWindowSize(int xw, int yw) { window_->SetSize(xw, yw); }
cv::Size cv::viz::Viz3d::VizImpl::getWindowSize() const { return Size(window_->GetSize()[0], window_->GetSize()[1]); }
