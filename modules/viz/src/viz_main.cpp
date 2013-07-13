#include "precomp.hpp"

#include <opencv2/calib3d.hpp>
#include "viz3d_impl.hpp"

#include <vtkRenderWindowInteractor.h>
#ifndef __APPLE__
vtkRenderWindowInteractor* vtkRenderWindowInteractorFixNew ()
{
  return vtkRenderWindowInteractor::New();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Viz3d::VizImpl::VizImpl (const std::string &name)
    :  style_ (vtkSmartPointer<cv::viz::InteractorStyle>::New ())
    , cloud_actor_map_ (new CloudActorMap)
    , shape_actor_map_ (new ShapeActorMap)
    , widget_actor_map_ (new WidgetActorMap)
    , s_lastDone_(0.0)
{
    renderer_ = vtkSmartPointer<vtkRenderer>::New ();

    // Create a RendererWindow
    window_ = vtkSmartPointer<vtkRenderWindow>::New ();

    // Set the window size as 1/2 of the screen size
    cv::Vec2i window_size = cv::Vec2i(window_->GetScreenSize()) / 2;
    window_->SetSize (window_size.val);

    window_->AddRenderer (renderer_);

    // Create the interactor style
    style_->Initialize ();
    style_->setRenderer (renderer_);
    style_->setCloudActorMap (cloud_actor_map_);
    style_->UseTimersOn ();

    /////////////////////////////////////////////////
    interactor_ = vtkSmartPointer <vtkRenderWindowInteractor>::Take (vtkRenderWindowInteractorFixNew ());

    //win_->PointSmoothingOn ();
    //win_->LineSmoothingOn ();
    //win_->PolygonSmoothingOn ();
    window_->AlphaBitPlanesOff ();
    window_->PointSmoothingOff ();
    window_->LineSmoothingOff ();
    window_->PolygonSmoothingOff ();
    window_->SwapBuffersOn ();
    window_->SetStereoTypeToAnaglyph ();

    interactor_->SetRenderWindow (window_);
    interactor_->SetInteractorStyle (style_);
    //interactor_->SetStillUpdateRate (30.0);
    interactor_->SetDesiredUpdateRate (30.0);

    // Initialize and create timer, also create window
    interactor_->Initialize ();
    timer_id_ = interactor_->CreateRepeatingTimer (5000L);

    // Set a simple PointPicker
    vtkSmartPointer<vtkPointPicker> pp = vtkSmartPointer<vtkPointPicker>::New ();
    pp->SetTolerance (pp->GetTolerance () * 2);
    interactor_->SetPicker (pp);

    exit_main_loop_timer_callback_ = vtkSmartPointer<ExitMainLoopTimerCallback>::New ();
    exit_main_loop_timer_callback_->viz_ = this;
    exit_main_loop_timer_callback_->right_timer_id = -1;
    interactor_->AddObserver (vtkCommand::TimerEvent, exit_main_loop_timer_callback_);

    exit_callback_ = vtkSmartPointer<ExitCallback>::New ();
    exit_callback_->viz_ = this;
    interactor_->AddObserver (vtkCommand::ExitEvent, exit_callback_);

    resetStoppedFlag ();


    //////////////////////////////

    String window_name("Viz");
    window_name = name.empty() ? window_name : window_name + " - " + name;
    window_->SetWindowName (window_name.c_str ());
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::viz::Viz3d::VizImpl::~VizImpl ()
{
    if (interactor_ != NULL)
        interactor_->DestroyTimer (timer_id_);

    renderer_->Clear();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::saveScreenshot (const std::string &file) { style_->saveScreenshot (file); }

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie)
{
    style_->registerMouseCallback(callback, cookie);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void* cookie)
{
    style_->registerKeyboardCallback(callback, cookie);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spin ()
{
    resetStoppedFlag ();
    window_->Render ();
    interactor_->Start ();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::spinOnce (int time, bool force_redraw)
{
    resetStoppedFlag ();

    if (time <= 0)
        time = 1;

    if (force_redraw)
        interactor_->Render ();

    double s_now_ = cv::getTickCount() / cv::getTickFrequency();
    if (s_lastDone_ > s_now_)
      s_lastDone_ = s_now_;

    if ((s_now_ - s_lastDone_) > (1.0 / interactor_->GetDesiredUpdateRate ()))
    {
        exit_main_loop_timer_callback_->right_timer_id = interactor_->CreateRepeatingTimer (time);
        interactor_->Start ();
        interactor_->DestroyTimer (exit_main_loop_timer_callback_->right_timer_id);
        s_lastDone_ = s_now_;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removePointCloud (const std::string &id)
{
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    if (am_it == cloud_actor_map_->end ())
        return false;

    if (removeActorFromRenderer (am_it->second.actor))
        return cloud_actor_map_->erase (am_it), true;

    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeShape (const std::string &id)
{
    // Check to see if the given ID entry exists
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    // Extra step: check if there is a cloud with the same ID
    CloudActorMap::iterator ca_it = cloud_actor_map_->find (id);

    bool shape = true;
    // Try to find a shape first
    if (am_it == shape_actor_map_->end ())
    {
        // There is no cloud or shape with this ID, so just exit
        if (ca_it == cloud_actor_map_->end ())
            return false;
        // Cloud found, set shape to false
        shape = false;
    }

    // Remove the pointer/ID pair to the global actor map
    if (shape)
    {
        if (removeActorFromRenderer (am_it->second))
        {
            shape_actor_map_->erase (am_it);
            return (true);
        }
    }
    else
    {
        if (removeActorFromRenderer (ca_it->second.actor))
        {
            cloud_actor_map_->erase (ca_it);
            return true;
        }
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeText3D (const std::string &id)
{
    // Check to see if the given ID entry exists
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it == shape_actor_map_->end ())
        return false;

    // Remove it from all renderers
    if (removeActorFromRenderer (am_it->second))
        return shape_actor_map_->erase (am_it), true;

    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeAllPointClouds ()
{
    // Check to see if the given ID entry exists
    CloudActorMap::iterator am_it = cloud_actor_map_->begin ();
    while (am_it != cloud_actor_map_->end () )
    {
        if (removePointCloud (am_it->first))
            am_it = cloud_actor_map_->begin ();
        else
            ++am_it;
    }
    return (true);
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeAllShapes ()
{
    // Check to see if the given ID entry exists
    ShapeActorMap::iterator am_it = shape_actor_map_->begin ();
    while (am_it != shape_actor_map_->end ())
    {
        if (removeShape (am_it->first))
            am_it = shape_actor_map_->begin ();
        else
            ++am_it;
    }
    return (true);
}


//////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeActorFromRenderer (const vtkSmartPointer<vtkLODActor> &actor)
{
    vtkLODActor* actor_to_remove = vtkLODActor::SafeDownCast (actor);



    // Iterate over all actors in this renderer
    vtkPropCollection* actors = renderer_->GetViewProps ();
    actors->InitTraversal ();

    vtkProp* current_actor = NULL;
    while ((current_actor = actors->GetNextProp ()) != NULL)
    {
        if (current_actor != actor_to_remove)
            continue;
        renderer_->RemoveActor (actor);
        //        renderer->Render ();
        // Found the correct viewport and removed the actor
        return (true);
    }

    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeActorFromRenderer (const vtkSmartPointer<vtkActor> &actor)
{
    vtkActor* actor_to_remove = vtkActor::SafeDownCast (actor);

    // Add it to all renderers
    //rens_->InitTraversal ();


        // Iterate over all actors in this renderer
    vtkPropCollection* actors = renderer_->GetViewProps ();
    actors->InitTraversal ();
    vtkProp* current_actor = NULL;
    while ((current_actor = actors->GetNextProp ()) != NULL)
    {
        if (current_actor != actor_to_remove)
            continue;
        renderer_->RemoveActor (actor);
        //        renderer->Render ();
        // Found the correct viewport and removed the actor
        return (true);
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::removeActorFromRenderer (const vtkSmartPointer<vtkProp> &actor)
{
    vtkProp* actor_to_remove = vtkProp::SafeDownCast(actor);

    vtkPropCollection* actors = renderer_->GetViewProps ();
    actors->InitTraversal ();
    vtkProp* current_actor = NULL;
    while ((current_actor = actors->GetNextProp ()) != NULL)
    {
        if (current_actor != actor_to_remove)
            continue;
        renderer_->RemoveActor (actor);
        return true;
    }
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::createActorFromVTKDataSet (const vtkSmartPointer<vtkDataSet> &data, vtkSmartPointer<vtkLODActor> &actor, bool use_scalars)
{
    if (!actor)
        actor = vtkSmartPointer<vtkLODActor>::New ();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (data);

    if (use_scalars)
    {
        vtkSmartPointer<vtkDataArray> scalars = data->GetPointData ()->GetScalars ();
        if (scalars)
        {
            cv::Vec3d minmax(scalars->GetRange());
            mapper->SetScalarRange(minmax.val);
            mapper->SetScalarModeToUsePointData ();

            // interpolation OFF, if data is a vtkPolyData that contains only vertices, ON for anything else.
            vtkPolyData* polyData = vtkPolyData::SafeDownCast (data);
            bool interpolation = (polyData && polyData->GetNumberOfCells () != polyData->GetNumberOfVerts ());

            mapper->SetInterpolateScalarsBeforeMapping (interpolation);
            mapper->ScalarVisibilityOn ();
        }
    }
    mapper->ImmediateModeRenderingOff ();

    actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, data->GetNumberOfPoints () / 10)));
    actor->GetProperty ()->SetInterpolationToFlat ();

    /// FIXME disabling backface culling due to known VTK bug: vtkTextActors are not
    /// shown when there is a vtkActor with backface culling on present in the scene
    /// Please see VTK bug tracker for more details: http://www.vtk.org/Bug/view.php?id=12588
    // actor->GetProperty ()->BackfaceCullingOn ();

    actor->SetMapper (mapper);
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setBackgroundColor (const Color& color)
{
    Color c = vtkcolor(color);
    renderer_->SetBackground (c.val);
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::getPointCloudRenderingProperties (int property, double &value, const std::string &id)
{
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    if (am_it == cloud_actor_map_->end ())
        return false;

    vtkLODActor* actor = vtkLODActor::SafeDownCast (am_it->second.actor);

    switch (property)
    {
        case VIZ_POINT_SIZE:
        {
            value = actor->GetProperty ()->GetPointSize ();
            actor->Modified ();
            break;
        }
        case VIZ_OPACITY:
        {
            value = actor->GetProperty ()->GetOpacity ();
            actor->Modified ();
            break;
        }
        case VIZ_LINE_WIDTH:
        {
            value = actor->GetProperty ()->GetLineWidth ();
            actor->Modified ();
            break;
        }
        default:
            CV_Assert("getPointCloudRenderingProperties: Unknown property");
    }

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::setPointCloudRenderingProperties (int property, double value, const std::string &id)
{
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    if (am_it == cloud_actor_map_->end ())
        return std::cout << "[setPointCloudRenderingProperties] Could not find any PointCloud datasets with id <" << id << ">!" << std::endl, false;

    vtkLODActor* actor = vtkLODActor::SafeDownCast (am_it->second.actor);

    switch (property)
    {
        case VIZ_POINT_SIZE:
        {
            actor->GetProperty ()->SetPointSize (float (value));
            actor->Modified ();
            break;
        }
        case VIZ_OPACITY:
        {
            actor->GetProperty ()->SetOpacity (value);
            actor->Modified ();
            break;
        }
            // Turn on/off flag to control whether data is rendered using immediate
            // mode or note. Immediate mode rendering tends to be slower but it can
            // handle larger datasets. The default value is immediate mode off. If you
            // are having problems rendering a large dataset you might want to consider
            // using immediate more rendering.
        case VIZ_IMMEDIATE_RENDERING:
        {
            actor->GetMapper ()->SetImmediateModeRendering (int (value));
            actor->Modified ();
            break;
        }
        case VIZ_LINE_WIDTH:
        {
            actor->GetProperty ()->SetLineWidth (float (value));
            actor->Modified ();
            break;
        }
        default:
            CV_Assert("setPointCloudRenderingProperties: Unknown property");
    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::setPointCloudSelected (const bool selected, const std::string &id)
{
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    if (am_it == cloud_actor_map_->end ())
        return std::cout << "[setPointCloudRenderingProperties] Could not find any PointCloud datasets with id <" << id << ">!" << std::endl, false;

    vtkLODActor* actor = vtkLODActor::SafeDownCast (am_it->second.actor);
    if (selected)
    {
        actor->GetProperty ()->EdgeVisibilityOn ();
        actor->GetProperty ()->SetEdgeColor (1.0, 0.0, 0.0);
        actor->Modified ();
    }
    else
    {
        actor->GetProperty ()->EdgeVisibilityOff ();
        actor->Modified ();
    }

    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::setShapeRenderingProperties (int property, double value, const std::string &id)
{
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it == shape_actor_map_->end ())
        return std::cout << "[setShapeRenderingProperties] Could not find any shape with id <" << id << ">!\n" << std::endl, false;

    vtkActor* actor = vtkActor::SafeDownCast (am_it->second);

    switch (property)
    {
    case VIZ_POINT_SIZE:
    {
        actor->GetProperty ()->SetPointSize (float (value));
        actor->Modified ();
        break;
    }
    case VIZ_OPACITY:
    {
        actor->GetProperty ()->SetOpacity (value);
        actor->Modified ();
        break;
    }
    case VIZ_LINE_WIDTH:
    {
        actor->GetProperty ()->SetLineWidth (float (value));
        actor->Modified ();
        break;
    }
    case VIZ_FONT_SIZE:
    {
        vtkTextActor* text_actor = vtkTextActor::SafeDownCast (am_it->second);
        vtkSmartPointer<vtkTextProperty> tprop = text_actor->GetTextProperty ();
        tprop->SetFontSize (int (value));
        text_actor->Modified ();
        break;
    }
    case VIZ_REPRESENTATION:
    {
        switch (int (value))
        {
            case REPRESENTATION_POINTS:    actor->GetProperty ()->SetRepresentationToPoints (); break;
            case REPRESENTATION_WIREFRAME: actor->GetProperty ()->SetRepresentationToWireframe (); break;
            case REPRESENTATION_SURFACE:   actor->GetProperty ()->SetRepresentationToSurface ();  break;
        }
        actor->Modified ();
        break;
    }
    case VIZ_SHADING:
    {
        switch (int (value))
        {
        case SHADING_FLAT: actor->GetProperty ()->SetInterpolationToFlat (); break;
        case SHADING_GOURAUD:
        {
            if (!actor->GetMapper ()->GetInput ()->GetPointData ()->GetNormals ())
            {
                std::cout << "[cv::viz::PCLVisualizer::setShapeRenderingProperties] Normals do not exist in the dataset, but Gouraud shading was requested. Estimating normals...\n" << std::endl;

                vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New ();
                normals->SetInput (actor->GetMapper ()->GetInput ());
                normals->Update ();
                vtkDataSetMapper::SafeDownCast (actor->GetMapper ())->SetInput (normals->GetOutput ());
            }
            actor->GetProperty ()->SetInterpolationToGouraud ();
            break;
        }
        case SHADING_PHONG:
        {
            if (!actor->GetMapper ()->GetInput ()->GetPointData ()->GetNormals ())
            {
                std::cout << "[cv::viz::PCLVisualizer::setShapeRenderingProperties] Normals do not exist in the dataset, but Phong shading was requested. Estimating normals...\n" << std::endl;
                vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New ();
                normals->SetInput (actor->GetMapper ()->GetInput ());
                normals->Update ();
                vtkDataSetMapper::SafeDownCast (actor->GetMapper ())->SetInput (normals->GetOutput ());
            }
            actor->GetProperty ()->SetInterpolationToPhong ();
            break;
        }
        }
        actor->Modified ();
        break;
    }
    default:
        CV_Assert("setShapeRenderingProperties: Unknown property");

    }
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::initCameraParameters ()
{
    Camera camera_temp;
    // Set default camera parameters to something meaningful
    camera_temp.clip = Vec2d(0.01, 1000.01);

    // Look straight along the z-axis
    camera_temp.focal = Vec3d(0.0, 0.0, 1.0);

    // Position the camera at the origin
    camera_temp.pos = Vec3d(0.0, 0.0, 0.0);

    // Set the up-vector of the camera to be the y-axis
    camera_temp.view_up = Vec3d(0.0, 1.0, 0.0);

    // Set the camera field of view to about
    camera_temp.fovy = 0.8575;
    camera_temp.window_size = Vec2i(window_->GetScreenSize()) / 2;
    camera_temp.window_pos = Vec2i(0, 0);

    setCameraParameters (camera_temp);
}

/////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::cameraParamsSet () const { return (camera_set_); }

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::updateCamera ()
{
    std::cout << "[cv::viz::PCLVisualizer::updateCamera()] This method was deprecated, just re-rendering all scenes now." << std::endl;
    //rens_->InitTraversal ();
    // Update the camera parameters

    renderer_->Render ();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::getCameras (cv::viz::Camera& camera)
{
    vtkCamera* active_camera = renderer_->GetActiveCamera ();

    camera.pos = cv::Vec3d(active_camera->GetPosition());
    camera.focal = cv::Vec3d(active_camera->GetFocalPoint());
    camera.clip = cv::Vec2d(active_camera->GetClippingRange());
    camera.view_up = cv::Vec3d(active_camera->GetViewUp());

    camera.fovy = active_camera->GetViewAngle()/ 180.0 * CV_PI;
    camera.window_size = cv::Vec2i(renderer_->GetRenderWindow()->GetSize());
    camera.window_pos = cv::Vec2d::all(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////
cv::Affine3f cv::viz::Viz3d::VizImpl::getViewerPose ()
{
    vtkCamera& camera = *renderer_->GetActiveCamera ();

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
void cv::viz::Viz3d::VizImpl::resetCamera ()
{
    renderer_->ResetCamera ();
}


/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraPosition (const cv::Vec3d& pos, const cv::Vec3d& view, const cv::Vec3d& up)
{

    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetPosition (pos[0], pos[1], pos[2]);
    cam->SetFocalPoint (view[0], view[1], view[2]);
    cam->SetViewUp (up[0], up[1], up[2]);
    renderer_->Render ();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraPosition (double pos_x, double pos_y, double pos_z, double up_x, double up_y, double up_z)
{
    //rens_->InitTraversal ();


    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetPosition (pos_x, pos_y, pos_z);
    cam->SetViewUp (up_x, up_y, up_z);
    renderer_->Render ();

}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraParameters (const cv::Matx33f& intrinsics, const cv::Affine3f& extrinsics)
{
    // Position = extrinsic translation
    cv::Vec3f pos_vec = extrinsics.translation();


    // Rotate the view vector
    cv::Matx33f rotation = extrinsics.rotation();
    cv::Vec3f y_axis (0.f, 1.f, 0.f);
    cv::Vec3f up_vec (rotation * y_axis);

    // Compute the new focal point
    cv::Vec3f z_axis (0.f, 0.f, 1.f);
    cv::Vec3f focal_vec = pos_vec + rotation * z_axis;

    // Get the width and height of the image - assume the calibrated centers are at the center of the image
    Eigen::Vector2i window_size;
    window_size[0] = static_cast<int> (intrinsics(0, 2));
    window_size[1] = static_cast<int> (intrinsics(1, 2));

    // Compute the vertical field of view based on the focal length and image heigh
    double fovy = 2 * atan (window_size[1] / (2. * intrinsics (1, 1))) * 180.0 / M_PI;

    //rens_->InitTraversal ();


    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetPosition (pos_vec[0], pos_vec[1], pos_vec[2]);
    cam->SetFocalPoint (focal_vec[0], focal_vec[1], focal_vec[2]);
    cam->SetViewUp (up_vec[0], up_vec[1], up_vec[2]);
    cam->SetUseHorizontalViewAngle (0);
    cam->SetViewAngle (fovy);
    cam->SetClippingRange (0.01, 1000.01);
    window_->SetSize (window_size[0], window_size[1]);

    renderer_->Render ();
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraParameters (const cv::viz::Camera &camera)
{
    //rens_->InitTraversal ();


    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetPosition (camera.pos[0], camera.pos[1], camera.pos[2]);
    cam->SetFocalPoint (camera.focal[0], camera.focal[1], camera.focal[2]);
    cam->SetViewUp (camera.view_up[0], camera.view_up[1], camera.view_up[2]);
    cam->SetClippingRange (camera.clip.val);
    cam->SetUseHorizontalViewAngle (0);
    cam->SetViewAngle (camera.fovy * 180.0 / M_PI);

    window_->SetSize (static_cast<int> (camera.window_size[0]), static_cast<int> (camera.window_size[1]));
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraClipDistances (double near, double far)
{
    //rens_->InitTraversal ();

    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetClippingRange (near, far);
}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setCameraFieldOfView (double fovy)
{
    //rens_->InitTraversal ();

    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetUseHorizontalViewAngle (0);
    cam->SetViewAngle (fovy * 180.0 / M_PI);

}

/////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::resetCameraViewpoint (const std::string &id)
{
    vtkSmartPointer<vtkMatrix4x4> camera_pose;
    static CloudActorMap::iterator it = cloud_actor_map_->find (id);
    if (it != cloud_actor_map_->end ())
        camera_pose = it->second.viewpoint_transformation_;
    else
        return;

    // Prevent a segfault
    if (!camera_pose)
        return;

    // set all renderer to this viewpoint
    //rens_->InitTraversal ();


    vtkSmartPointer<vtkCamera> cam = renderer_->GetActiveCamera ();
    cam->SetPosition (camera_pose->GetElement (0, 3),
                      camera_pose->GetElement (1, 3),
                      camera_pose->GetElement (2, 3));

    cam->SetFocalPoint (camera_pose->GetElement (0, 3) - camera_pose->GetElement (0, 2),
                        camera_pose->GetElement (1, 3) - camera_pose->GetElement (1, 2),
                        camera_pose->GetElement (2, 3) - camera_pose->GetElement (2, 2));

    cam->SetViewUp (camera_pose->GetElement (0, 1),
                    camera_pose->GetElement (1, 1),
                    camera_pose->GetElement (2, 1));

    renderer_->SetActiveCamera (cam);
    renderer_->ResetCameraClippingRange ();
    renderer_->Render ();
}

////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, const std::string & id)
{
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
    {
        std::cout << "[addModelFromPolyData] A shape with id <" << id << "> already exists! Please choose a different id and retry." << std::endl;
        return (false);
    }

    vtkSmartPointer<vtkLODActor> actor;
    createActorFromVTKDataSet (polydata, actor);
    actor->GetProperty ()->SetRepresentationToWireframe ();
    renderer_->AddActor(actor);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = actor;
    return (true);
}

////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addModelFromPolyData (vtkSmartPointer<vtkPolyData> polydata, vtkSmartPointer<vtkTransform> transform, const std::string & id)
{
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
    {
        std::cout << "[addModelFromPolyData] A shape with id <"<<id<<"> already exists! Please choose a different id and retry." << std::endl;
        return (false);
    }

    vtkSmartPointer <vtkTransformFilter> trans_filter = vtkSmartPointer<vtkTransformFilter>::New ();
    trans_filter->SetTransform (transform);
    trans_filter->SetInput (polydata);
    trans_filter->Update();

    // Create an Actor
    vtkSmartPointer <vtkLODActor> actor;
    createActorFromVTKDataSet (trans_filter->GetOutput (), actor);
    actor->GetProperty ()->SetRepresentationToWireframe ();
    renderer_->AddActor(actor);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = actor;
    return (true);
}


////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addModelFromPLYFile (const std::string &filename, const std::string &id)
{
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addModelFromPLYFile] A shape with id <"<<id<<"> already exists! Please choose a different id and retry.." << std::endl, false;

    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New ();
    reader->SetFileName (filename.c_str ());

    vtkSmartPointer<vtkLODActor> actor;
    createActorFromVTKDataSet (reader->GetOutput (), actor);
    actor->GetProperty ()->SetRepresentationToWireframe ();
    renderer_->AddActor(actor);

    (*shape_actor_map_)[id] = actor;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addModelFromPLYFile (const std::string &filename, vtkSmartPointer<vtkTransform> transform, const std::string &id)
{
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addModelFromPLYFile] A shape with id <"<<id<<"> already exists! Please choose a different id and retry." << std::endl, false;

    vtkSmartPointer <vtkPLYReader > reader = vtkSmartPointer<vtkPLYReader>::New ();
    reader->SetFileName (filename.c_str ());

    vtkSmartPointer <vtkTransformFilter> trans_filter = vtkSmartPointer<vtkTransformFilter>::New ();
    trans_filter->SetTransform (transform);
    trans_filter->SetInputConnection (reader->GetOutputPort ());

    vtkSmartPointer <vtkLODActor> actor;
    createActorFromVTKDataSet (trans_filter->GetOutput (), actor);
    actor->GetProperty ()->SetRepresentationToWireframe ();
    renderer_->AddActor(actor);

    (*shape_actor_map_)[id] = actor;
    return (true);
}

bool cv::viz::Viz3d::VizImpl::addPolylineFromPolygonMesh (const Mesh3d& mesh, const std::string &id)
{
    CV_Assert(mesh.cloud.rows == 1 && mesh.cloud.type() == CV_32FC3);

    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addPolylineFromPolygonMesh] A shape with id <"<< id << "> already exists! Please choose a different id and retry.\n" << std::endl, false;

    vtkSmartPointer<vtkPoints> poly_points = vtkSmartPointer<vtkPoints>::New ();
    poly_points->SetNumberOfPoints (mesh.cloud.size().area());

    const cv::Point3f *cdata = mesh.cloud.ptr<cv::Point3f>();
    for (int i = 0; i < mesh.cloud.cols; ++i)
        poly_points->InsertPoint (i, cdata[i].x, cdata[i].y,cdata[i].z);


    // Create a cell array to store the lines in and add the lines to it
    vtkSmartPointer <vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
    vtkSmartPointer <vtkPolyData> polyData;
    allocVtkPolyData (polyData);

    for (size_t i = 0; i < mesh.polygons.size (); i++)
    {
        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
        polyLine->GetPointIds()->SetNumberOfIds(mesh.polygons[i].vertices.size());
        for(unsigned int k = 0; k < mesh.polygons[i].vertices.size(); k++)
        {
            polyLine->GetPointIds()->SetId(k,mesh. polygons[i].vertices[k]);
        }

        cells->InsertNextCell (polyLine);
    }

    // Add the points to the dataset
    polyData->SetPoints (poly_points);

    // Add the lines to the dataset
    polyData->SetLines (cells);

    // Setup actor and mapper
    vtkSmartPointer < vtkPolyDataMapper > mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    mapper->SetInput (polyData);

    vtkSmartPointer <vtkActor> actor = vtkSmartPointer<vtkActor>::New ();
    actor->SetMapper (mapper);
    renderer_->AddActor(actor);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = actor;
    return (true);
}


///////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setRepresentationToSurfaceForAllActors ()
{
    vtkActorCollection * actors = renderer_->GetActors ();
    actors->InitTraversal ();
    vtkActor * actor;
    while ((actor = actors->GetNextActor ()) != NULL)
        actor->GetProperty ()->SetRepresentationToSurface ();
}

///////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setRepresentationToPointsForAllActors ()
{
    vtkActorCollection * actors = renderer_->GetActors ();
    actors->InitTraversal ();
    vtkActor * actor;
    while ((actor = actors->GetNextActor ()) != NULL)
        actor->GetProperty ()->SetRepresentationToPoints ();
}

///////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::setRepresentationToWireframeForAllActors ()
{
    vtkActorCollection * actors = renderer_->GetActors ();
    actors->InitTraversal ();
    vtkActor *actor;
    while ((actor = actors->GetNextActor ()) != NULL)
        actor->GetProperty ()->SetRepresentationToWireframe ();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::updateCells (vtkSmartPointer<vtkIdTypeArray> &cells, vtkSmartPointer<vtkIdTypeArray> &initcells, vtkIdType nr_points)
{
    // If no init cells and cells has not been initialized...
    if (!cells)
        cells = vtkSmartPointer<vtkIdTypeArray>::New ();

    // If we have less values then we need to recreate the array
    if (cells->GetNumberOfTuples () < nr_points)
    {
        cells = vtkSmartPointer<vtkIdTypeArray>::New ();

        // If init cells is given, and there's enough data in it, use it
        if (initcells && initcells->GetNumberOfTuples () >= nr_points)
        {
            cells->DeepCopy (initcells);
            cells->SetNumberOfComponents (2);
            cells->SetNumberOfTuples (nr_points);
        }
        else
        {
            // If the number of tuples is still too small, we need to recreate the array
            cells->SetNumberOfComponents (2);
            cells->SetNumberOfTuples (nr_points);
            vtkIdType *cell = cells->GetPointer (0);
            // Fill it with 1s
            std::fill_n (cell, nr_points * 2, 1);
            cell++;
            for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
                *cell = i;
            // Save the results in initcells
            initcells = vtkSmartPointer<vtkIdTypeArray>::New ();
            initcells->DeepCopy (cells);
        }
    }
    else
    {
        // The assumption here is that the current set of cells has more data than needed
        cells->SetNumberOfComponents (2);
        cells->SetNumberOfTuples (nr_points);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::allocVtkPolyData (vtkSmartPointer<vtkAppendPolyData> &polydata)
{
    polydata = vtkSmartPointer<vtkAppendPolyData>::New ();
}
//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::allocVtkPolyData (vtkSmartPointer<vtkPolyData> &polydata)
{
    polydata = vtkSmartPointer<vtkPolyData>::New ();
}
//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::Viz3d::VizImpl::allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata)
{
    polydata = vtkSmartPointer<vtkUnstructuredGrid>::New ();
}


//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::convertToVtkMatrix (const Eigen::Vector4f &origin, const Eigen::Quaternion<float> &orientation, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix)
{
    // set rotation
    Eigen::Matrix3f rot = orientation.toRotationMatrix ();
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            vtk_matrix->SetElement (i, k, rot (i, k));

    // set translation
    vtk_matrix->SetElement (0, 3, origin (0));
    vtk_matrix->SetElement (1, 3, origin (1));
    vtk_matrix->SetElement (2, 3, origin (2));
    vtk_matrix->SetElement (3, 3, 1.0f);
}

void cv::viz::convertToVtkMatrix (const Matx44f &m, vtkSmartPointer<vtkMatrix4x4> &vtk_matrix)
{
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            vtk_matrix->SetElement (i, k, m (i, k));
}

vtkSmartPointer<vtkMatrix4x4> cv::viz::convertToVtkMatrix (const cv::Matx44f &m)
{
    vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            vtk_matrix->SetElement(i, k, m(i, k));
    return vtk_matrix;
}

void cv::viz::convertToCvMatrix (const vtkSmartPointer<vtkMatrix4x4> &vtk_matrix, cv::Matx44f &m)
{
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            m(i,k) = vtk_matrix->GetElement (i, k);
}


cv::Matx44f cv::viz::convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
{
    cv::Matx44f m;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            m(i, k) = vtk_matrix->GetElement (i, k);
    return m;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::convertToEigenMatrix (const vtkSmartPointer<vtkMatrix4x4> &vtk_matrix, Eigen::Matrix4f &m)
{
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            m (i,k) = static_cast<float> (vtk_matrix->GetElement (i, k));
}


void cv::viz::Viz3d::VizImpl::setFullScreen (bool mode)
{
    if (window_)
        window_->SetFullScreen (mode);
}

void cv::viz::Viz3d::VizImpl::setWindowName (const std::string &name)
{
    if (window_)
        window_->SetWindowName (name.c_str ());
}

void cv::viz::Viz3d::VizImpl::setPosition (int x, int y) { window_->SetPosition (x, y); }
void cv::viz::Viz3d::VizImpl::setSize (int xw, int yw) { window_->SetSize (xw, yw); }

bool cv::viz::Viz3d::VizImpl::addPolygonMesh (const Mesh3d& mesh, const Mat& mask, const std::string &id)
{
    CV_Assert(mesh.cloud.type() == CV_32FC3 && mesh.cloud.rows == 1 && !mesh.polygons.empty ());
    CV_Assert(mesh.colors.empty() || (!mesh.colors.empty() && mesh.colors.size() == mesh.cloud.size() && mesh.colors.type() == CV_8UC3));
    CV_Assert(mask.empty() || (!mask.empty() && mask.size() == mesh.cloud.size() && mask.type() == CV_8U));

    if (cloud_actor_map_->find (id) != cloud_actor_map_->end ())
        return std::cout << "[addPolygonMesh] A shape with id <" << id << "> already exists! Please choose a different id and retry." << std::endl, false;

    //    int rgb_idx = -1;
    //    std::vector<sensor_msgs::PointField> fields;


    //    rgb_idx = cv::viz::getFieldIndex (*cloud, "rgb", fields);
    //    if (rgb_idx == -1)
    //      rgb_idx = cv::viz::getFieldIndex (*cloud, "rgba", fields);

    vtkSmartPointer<vtkUnsignedCharArray> colors_array;
#if 1
    if (!mesh.colors.empty())
    {
        colors_array = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        colors_array->SetNumberOfComponents (3);
        colors_array->SetName ("Colors");

        const unsigned char* data = mesh.colors.ptr<unsigned char>();

        //TODO check mask
        CV_Assert(mask.empty()); //because not implemented;

        for(int i = 0; i < mesh.colors.cols; ++i)
            colors_array->InsertNextTupleValue(&data[i*3]);

        //      cv::viz::RGB rgb_data;
        //      for (size_t i = 0; i < cloud->size (); ++i)
        //      {
        //        if (!isFinite (cloud->points[i]))
        //          continue;
        //        memcpy (&rgb_data, reinterpret_cast<const char*> (&cloud->points[i]) + fields[rgb_idx].offset, sizeof (cv::viz::RGB));
        //        unsigned char color[3];
        //        color[0] = rgb_data.r;
        //        color[1] = rgb_data.g;
        //        color[2] = rgb_data.b;
        //        colors->InsertNextTupleValue (color);
        //      }
    }
#endif

    // Create points from polyMesh.cloud
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
    vtkIdType nr_points = mesh.cloud.size().area();

    points->SetNumberOfPoints (nr_points);


    // Get a pointer to the beginning of the data array
    float *data = static_cast<vtkFloatArray*> (points->GetData ())->GetPointer (0);


    std::vector<int> lookup;
    // If the dataset is dense (no NaNs)
    if (mask.empty())
    {
        cv::Mat hdr(mesh.cloud.size(), CV_32FC3, (void*)data);
        mesh.cloud.copyTo(hdr);
    }
    else
    {
        lookup.resize (nr_points);

        const unsigned char *mdata = mask.ptr<unsigned char>();
        const cv::Point3f *cdata = mesh.cloud.ptr<cv::Point3f>();
        cv::Point3f* out = reinterpret_cast<cv::Point3f*>(data);

        int j = 0;    // true point index
        for (int i = 0; i < nr_points; ++i)
            if(mdata[i])
            {
                lookup[i] = j;
                out[j++] = cdata[i];
            }
        nr_points = j;
        points->SetNumberOfPoints (nr_points);
    }

    // Get the maximum size of a polygon
    int max_size_of_polygon = -1;
    for (size_t i = 0; i < mesh.polygons.size (); ++i)
        if (max_size_of_polygon < static_cast<int> (mesh.polygons[i].vertices.size ()))
            max_size_of_polygon = static_cast<int> (mesh.polygons[i].vertices.size ());

    vtkSmartPointer<vtkLODActor> actor;

    if (mesh.polygons.size () > 1)
    {
        // Create polys from polyMesh.polygons
        vtkSmartPointer<vtkCellArray> cell_array = vtkSmartPointer<vtkCellArray>::New ();
        vtkIdType *cell = cell_array->WritePointer (mesh.polygons.size (), mesh.polygons.size () * (max_size_of_polygon + 1));
        int idx = 0;
        if (lookup.size () > 0)
        {
            for (size_t i = 0; i < mesh.polygons.size (); ++i, ++idx)
            {
                size_t n_points = mesh.polygons[i].vertices.size ();
                *cell++ = n_points;
                //cell_array->InsertNextCell (n_points);
                for (size_t j = 0; j < n_points; j++, ++idx)
                    *cell++ = lookup[mesh.polygons[i].vertices[j]];
                //cell_array->InsertCellPoint (lookup[vertices[i].vertices[j]]);
            }
        }
        else
        {
            for (size_t i = 0; i < mesh.polygons.size (); ++i, ++idx)
            {
                size_t n_points = mesh.polygons[i].vertices.size ();
                *cell++ = n_points;
                //cell_array->InsertNextCell (n_points);
                for (size_t j = 0; j < n_points; j++, ++idx)
                    *cell++ = mesh.polygons[i].vertices[j];
                //cell_array->InsertCellPoint (vertices[i].vertices[j]);
            }
        }
        vtkSmartPointer<vtkPolyData> polydata;
        allocVtkPolyData (polydata);
        cell_array->GetData ()->SetNumberOfValues (idx);
        cell_array->Squeeze ();
        polydata->SetStrips (cell_array);
        polydata->SetPoints (points);

        if (colors_array)
            polydata->GetPointData ()->SetScalars (colors_array);

        createActorFromVTKDataSet (polydata, actor, false);
    }
    else
    {
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New ();
        size_t n_points = mesh.polygons[0].vertices.size ();
        polygon->GetPointIds ()->SetNumberOfIds (n_points - 1);

        if (lookup.size () > 0)
        {
            for (size_t j = 0; j < n_points - 1; ++j)
                polygon->GetPointIds ()->SetId (j, lookup[mesh.polygons[0].vertices[j]]);
        }
        else
        {
            for (size_t j = 0; j < n_points - 1; ++j)
                polygon->GetPointIds ()->SetId (j, mesh.polygons[0].vertices[j]);
        }
        vtkSmartPointer<vtkUnstructuredGrid> poly_grid;
        allocVtkUnstructuredGrid (poly_grid);
        poly_grid->Allocate (1, 1);
        poly_grid->InsertNextCell (polygon->GetCellType (), polygon->GetPointIds ());
        poly_grid->SetPoints (points);
        poly_grid->Update ();
        if (colors_array)
            poly_grid->GetPointData ()->SetScalars (colors_array);

        createActorFromVTKDataSet (poly_grid, actor, false);
    }
    renderer_->AddActor (actor);
    actor->GetProperty ()->SetRepresentationToSurface ();
    // Backface culling renders the visualization slower, but guarantees that we see all triangles
    actor->GetProperty ()->BackfaceCullingOff ();
    actor->GetProperty ()->SetInterpolationToFlat ();
    actor->GetProperty ()->EdgeVisibilityOff ();
    actor->GetProperty ()->ShadingOff ();

    // Save the pointer/ID pair to the global actor map
    (*cloud_actor_map_)[id].actor = actor;
    //if (vertices.size () > 1)
    //  (*cloud_actor_map_)[id].cells = static_cast<vtkPolyDataMapper*>(actor->GetMapper ())->GetInput ()->GetVerts ()->GetData ();

    const Eigen::Vector4f& sensor_origin = Eigen::Vector4f::Zero ();
    const Eigen::Quaternion<float>& sensor_orientation = Eigen::Quaternionf::Identity ();

    // Save the viewpoint transformation matrix to the global actor map
    vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New();
    convertToVtkMatrix (sensor_origin, sensor_orientation, transformation);
    (*cloud_actor_map_)[id].viewpoint_transformation_ = transformation;

    return (true);
}


bool cv::viz::Viz3d::VizImpl::updatePolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const std::string &id)
{
    CV_Assert(mesh.cloud.type() == CV_32FC3 && mesh.cloud.rows == 1 && !mesh.polygons.empty ());
    CV_Assert(mesh.colors.empty() || (!mesh.colors.empty() && mesh.colors.size() == mesh.cloud.size() && mesh.colors.type() == CV_8UC3));
    CV_Assert(mask.empty() || (!mask.empty() && mask.size() == mesh.cloud.size() && mask.type() == CV_8U));

    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    if (am_it == cloud_actor_map_->end ())
        return (false);

    // Get the current poly data
    vtkSmartPointer<vtkPolyData> polydata = static_cast<vtkPolyDataMapper*>(am_it->second.actor->GetMapper ())->GetInput ();
    if (!polydata)
        return (false);
    vtkSmartPointer<vtkCellArray> cells = polydata->GetStrips ();
    if (!cells)
        return (false);
    vtkSmartPointer<vtkPoints> points   = polydata->GetPoints ();
    // Copy the new point array in
    vtkIdType nr_points = mesh.cloud.size().area();
    points->SetNumberOfPoints (nr_points);

    // Get a pointer to the beginning of the data array
    float *data = (static_cast<vtkFloatArray*> (points->GetData ()))->GetPointer (0);

    int ptr = 0;
    std::vector<int> lookup;
    // If the dataset is dense (no NaNs)
    if (mask.empty())
    {
        cv::Mat hdr(mesh.cloud.size(), CV_32FC3, (void*)data);
        mesh.cloud.copyTo(hdr);

    }
    else
    {
        lookup.resize (nr_points);

        const unsigned char *mdata = mask.ptr<unsigned char>();
        const cv::Point3f *cdata = mesh.cloud.ptr<cv::Point3f>();
        cv::Point3f* out = reinterpret_cast<cv::Point3f*>(data);

        int j = 0;    // true point index
        for (int i = 0; i < nr_points; ++i)
            if(mdata[i])
            {
                lookup[i] = j;
                out[j++] = cdata[i];
            }
        nr_points = j;
        points->SetNumberOfPoints (nr_points);;
    }

    // Update colors
    vtkUnsignedCharArray* colors_array = vtkUnsignedCharArray::SafeDownCast (polydata->GetPointData ()->GetScalars ());

    if (!mesh.colors.empty() && colors_array)
    {
        if (mask.empty())
        {
            const unsigned char* data = mesh.colors.ptr<unsigned char>();
            for(int i = 0; i < mesh.colors.cols; ++i)
                colors_array->InsertNextTupleValue(&data[i*3]);
        }
        else
        {
            const unsigned char* color = mesh.colors.ptr<unsigned char>();
            const unsigned char* mdata = mask.ptr<unsigned char>();

            int j = 0;
            for(int i = 0; i < mesh.colors.cols; ++i)
                if (mdata[i])
                    colors_array->SetTupleValue (j++, &color[i*3]);

        }
    }

    // Get the maximum size of a polygon
    int max_size_of_polygon = -1;
    for (size_t i = 0; i < mesh.polygons.size (); ++i)
        if (max_size_of_polygon < static_cast<int> (mesh.polygons[i].vertices.size ()))
            max_size_of_polygon = static_cast<int> (mesh.polygons[i].vertices.size ());

    // Update the cells
    cells = vtkSmartPointer<vtkCellArray>::New ();
    vtkIdType *cell = cells->WritePointer (mesh.polygons.size (), mesh.polygons.size () * (max_size_of_polygon + 1));
    int idx = 0;
    if (lookup.size () > 0)
    {
        for (size_t i = 0; i < mesh.polygons.size (); ++i, ++idx)
        {
            size_t n_points = mesh.polygons[i].vertices.size ();
            *cell++ = n_points;
            for (size_t j = 0; j < n_points; j++, cell++, ++idx)
                *cell = lookup[mesh.polygons[i].vertices[j]];
        }
    }
    else
    {
        for (size_t i = 0; i < mesh.polygons.size (); ++i, ++idx)
        {
            size_t n_points = mesh.polygons[i].vertices.size ();
            *cell++ = n_points;
            for (size_t j = 0; j < n_points; j++, cell++, ++idx)
                *cell = mesh.polygons[i].vertices[j];
        }
    }
    cells->GetData ()->SetNumberOfValues (idx);
    cells->Squeeze ();
    // Set the the vertices
    polydata->SetStrips (cells);
    polydata->Update ();
    return (true);
}

////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addArrow (const cv::Point3f &p1, const cv::Point3f &p2, const Color& color, bool display_length, const std::string &id)
{
    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addArrow] A shape with id <" << id << "> already exists! Please choose a different id and retry." << std::endl, false;

    // Create an Actor
    vtkSmartPointer<vtkLeaderActor2D> leader = vtkSmartPointer<vtkLeaderActor2D>::New ();
    leader->GetPositionCoordinate()->SetCoordinateSystemToWorld ();
    leader->GetPositionCoordinate()->SetValue (p1.x, p1.y, p1.z);
    leader->GetPosition2Coordinate()->SetCoordinateSystemToWorld ();
    leader->GetPosition2Coordinate()->SetValue (p2.x, p2.y, p2.z);
    leader->SetArrowStyleToFilled();
    leader->SetArrowPlacementToPoint2 ();

    if (display_length)
        leader->AutoLabelOn ();
    else
        leader->AutoLabelOff ();

    Color c = vtkcolor(color);
    leader->GetProperty ()->SetColor (c.val);
    renderer_->AddActor (leader);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = leader;
    return (true);
}
////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::Viz3d::VizImpl::addArrow (const cv::Point3f &p1, const cv::Point3f &p2, const Color& color_line, const Color& color_text, const std::string &id)
{
    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
    {
        std::cout << "[addArrow] A shape with id <" << id << "> already exists! Please choose a different id and retry." << std::endl;
        return (false);
    }

    // Create an Actor
    vtkSmartPointer<vtkLeaderActor2D> leader = vtkSmartPointer<vtkLeaderActor2D>::New ();
    leader->GetPositionCoordinate ()->SetCoordinateSystemToWorld ();
    leader->GetPositionCoordinate ()->SetValue (p1.x, p1.y, p1.z);
    leader->GetPosition2Coordinate ()->SetCoordinateSystemToWorld ();
    leader->GetPosition2Coordinate ()->SetValue (p2.x, p2.y, p2.z);
    leader->SetArrowStyleToFilled ();
    leader->AutoLabelOn ();

    Color ct = vtkcolor(color_text);
    leader->GetLabelTextProperty()->SetColor(ct.val);

    Color cl = vtkcolor(color_line);
    leader->GetProperty ()->SetColor (cl.val);
    renderer_->AddActor (leader);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = leader;
    return (true);
}

bool cv::viz::Viz3d::VizImpl::addPolygon (const cv::Mat& cloud, const Color& color, const std::string &id)
{
    CV_Assert(cloud.type() == CV_32FC3 && cloud.rows == 1);

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
    vtkSmartPointer<vtkPolygon> polygon    = vtkSmartPointer<vtkPolygon>::New ();

    int total = cloud.size().area();
    points->SetNumberOfPoints (total);
    polygon->GetPointIds ()->SetNumberOfIds (total);

    for (int i = 0; i < total; ++i)
    {
        cv::Point3f p = cloud.ptr<cv::Point3f>()[i];
        points->SetPoint (i, p.x, p.y, p.z);
        polygon->GetPointIds ()->SetId (i, i);
    }

    vtkSmartPointer<vtkUnstructuredGrid> poly_grid;
    allocVtkUnstructuredGrid (poly_grid);
    poly_grid->Allocate (1, 1);
    poly_grid->InsertNextCell (polygon->GetCellType (), polygon->GetPointIds ());
    poly_grid->SetPoints (points);
    poly_grid->Update ();


    //////////////////////////////////////////////////////
    vtkSmartPointer<vtkDataSet> data = poly_grid;

    Color c = vtkcolor(color);

    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
    {
        vtkSmartPointer<vtkAppendPolyData> all_data = vtkSmartPointer<vtkAppendPolyData>::New ();

        // Add old data
        all_data->AddInput (reinterpret_cast<vtkPolyDataMapper*> ((vtkActor::SafeDownCast (am_it->second))->GetMapper ())->GetInput ());

        // Add new data
        vtkSmartPointer<vtkDataSetSurfaceFilter> surface_filter = vtkSmartPointer<vtkDataSetSurfaceFilter>::New ();
        surface_filter->SetInput (vtkUnstructuredGrid::SafeDownCast (data));
        vtkSmartPointer<vtkPolyData> poly_data = surface_filter->GetOutput ();
        all_data->AddInput (poly_data);

        // Create an Actor
        vtkSmartPointer<vtkLODActor> actor;
        createActorFromVTKDataSet (all_data->GetOutput (), actor);
        actor->GetProperty ()->SetRepresentationToWireframe ();
        actor->GetProperty ()->SetColor (c.val);
        actor->GetMapper ()->ScalarVisibilityOff ();
        actor->GetProperty ()->BackfaceCullingOff ();

        removeActorFromRenderer (am_it->second);
        renderer_->AddActor (actor);

        // Save the pointer/ID pair to the global actor map
        (*shape_actor_map_)[id] = actor;
    }
    else
    {
        // Create an Actor
        vtkSmartPointer<vtkLODActor> actor;
        createActorFromVTKDataSet (data, actor);
        actor->GetProperty ()->SetRepresentationToWireframe ();
        actor->GetProperty ()->SetColor (c.val);
        actor->GetMapper ()->ScalarVisibilityOff ();
        actor->GetProperty ()->BackfaceCullingOff ();
        renderer_->AddActor (actor);

        // Save the pointer/ID pair to the global actor map
        (*shape_actor_map_)[id] = actor;
    }

    return (true);
}

void cv::viz::Viz3d::VizImpl::showWidget(const String &id, const Widget &widget, const Affine3f &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    if (exists)
    {
        // Remove it if it exists and add it again
        removeActorFromRenderer(wam_itr->second.actor);
    }
    // Get the actor and set the user matrix
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(widget));
    if (actor)
    {
        // If the actor is 3D, apply pose
        vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
        actor->SetUserMatrix (matrix);
        actor->Modified();
    }
    // If the actor is a vtkFollower, then it should always face the camera
    vtkFollower *follower = vtkFollower::SafeDownCast(actor);
    if (follower)
    {
        follower->SetCamera(renderer_->GetActiveCamera());
    }

    renderer_->AddActor(WidgetAccessor::getProp(widget));
    (*widget_actor_map_)[id].actor = WidgetAccessor::getProp(widget);
}

void cv::viz::Viz3d::VizImpl::removeWidget(const String &id)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);
    CV_Assert(removeActorFromRenderer (wam_itr->second.actor));
    widget_actor_map_->erase(wam_itr);
}

cv::viz::Widget cv::viz::Viz3d::VizImpl::getWidget(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);

    Widget widget;
    WidgetAccessor::setProp(widget, wam_itr->second.actor);
    return widget;
}

void cv::viz::Viz3d::VizImpl::setWidgetPose(const String &id, const Affine3f &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second.actor);
    CV_Assert(actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
    actor->SetUserMatrix (matrix);
    actor->Modified ();
}

void cv::viz::Viz3d::VizImpl::updateWidgetPose(const String &id, const Affine3f &pose)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second.actor);
    CV_Assert(actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    if (!matrix)
    {
        setWidgetPose(id, pose);
        return ;
    }
    Matx44f matrix_cv = convertToMatx(matrix);
    Affine3f updated_pose = pose * Affine3f(matrix_cv);
    matrix = convertToVtkMatrix(updated_pose.matrix);

    actor->SetUserMatrix (matrix);
    actor->Modified ();
}

cv::Affine3f cv::viz::Viz3d::VizImpl::getWidgetPose(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);

    vtkProp3D *actor = vtkProp3D::SafeDownCast(wam_itr->second.actor);
    CV_Assert(actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    Matx44f matrix_cv = convertToMatx(matrix);
    return Affine3f(matrix_cv);
}
