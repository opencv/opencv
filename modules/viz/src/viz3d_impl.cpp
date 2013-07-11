#include "precomp.hpp"

namespace temp_viz
{
    template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);

    template<> Vec3f* vtkpoints_data<float>(vtkSmartPointer<vtkPoints>& points)
    {
        CV_Assert(points->GetDataType() == VTK_FLOAT);
        vtkDataArray *data = points->GetData();
        float *pointer = static_cast<vtkFloatArray*>(data)->GetPointer(0);
        return reinterpret_cast<Vec3f*>(pointer);
    }

    template<> Vec3d* vtkpoints_data<double>(vtkSmartPointer<vtkPoints>& points)
    {
        CV_Assert(points->GetDataType() == VTK_DOUBLE);
        vtkDataArray *data = points->GetData();
        double *pointer = static_cast<vtkDoubleArray*>(data)->GetPointer(0);
        return reinterpret_cast<Vec3d*>(pointer);
    }
}

void temp_viz::Viz3d::VizImpl::setFullScreen (bool mode)
{
    if (window_)
        window_->SetFullScreen (mode);
}

void temp_viz::Viz3d::VizImpl::setWindowName (const std::string &name)
{
    if (window_)
        window_->SetWindowName (name.c_str ());
}

void temp_viz::Viz3d::VizImpl::setPosition (int x, int y) { window_->SetPosition (x, y); }
void temp_viz::Viz3d::VizImpl::setSize (int xw, int yw) { window_->SetSize (xw, yw); }

bool temp_viz::Viz3d::VizImpl::addPolygonMesh (const Mesh3d& mesh, const Mat& mask, const std::string &id)
{
    CV_Assert(mesh.cloud.type() == CV_32FC3 && mesh.cloud.rows == 1 && !mesh.polygons.empty ());
    CV_Assert(mesh.colors.empty() || (!mesh.colors.empty() && mesh.colors.size() == mesh.cloud.size() && mesh.colors.type() == CV_8UC3));
    CV_Assert(mask.empty() || (!mask.empty() && mask.size() == mesh.cloud.size() && mask.type() == CV_8U));

    if (cloud_actor_map_->find (id) != cloud_actor_map_->end ())
        return std::cout << "[addPolygonMesh] A shape with id <" << id << "> already exists! Please choose a different id and retry." << std::endl, false;

    //    int rgb_idx = -1;
    //    std::vector<sensor_msgs::PointField> fields;


    //    rgb_idx = temp_viz::getFieldIndex (*cloud, "rgb", fields);
    //    if (rgb_idx == -1)
    //      rgb_idx = temp_viz::getFieldIndex (*cloud, "rgba", fields);

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

        //      temp_viz::RGB rgb_data;
        //      for (size_t i = 0; i < cloud->size (); ++i)
        //      {
        //        if (!isFinite (cloud->points[i]))
        //          continue;
        //        memcpy (&rgb_data, reinterpret_cast<const char*> (&cloud->points[i]) + fields[rgb_idx].offset, sizeof (temp_viz::RGB));
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


bool temp_viz::Viz3d::VizImpl::updatePolygonMesh (const Mesh3d& mesh, const cv::Mat& mask, const std::string &id)
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
bool temp_viz::Viz3d::VizImpl::addArrow (const cv::Point3f &p1, const cv::Point3f &p2, const Color& color, bool display_length, const std::string &id)
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
bool temp_viz::Viz3d::VizImpl::addArrow (const cv::Point3f &p1, const cv::Point3f &p2, const Color& color_line, const Color& color_text, const std::string &id)
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

bool temp_viz::Viz3d::VizImpl::addPolygon (const cv::Mat& cloud, const Color& color, const std::string &id)
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

void temp_viz::Viz3d::VizImpl::showWidget(const String &id, const Widget &widget, const Affine3f &pose)
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

void temp_viz::Viz3d::VizImpl::removeWidget(const String &id)
{
    WidgetActorMap::iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);
    CV_Assert(removeActorFromRenderer (wam_itr->second.actor));
    widget_actor_map_->erase(wam_itr);
}

temp_viz::Widget temp_viz::Viz3d::VizImpl::getWidget(const String &id) const
{
    WidgetActorMap::const_iterator wam_itr = widget_actor_map_->find(id);
    bool exists = wam_itr != widget_actor_map_->end();
    CV_Assert(exists);
    
    Widget widget;
    WidgetAccessor::setProp(widget, wam_itr->second.actor);
    return widget;
}

void temp_viz::Viz3d::VizImpl::setWidgetPose(const String &id, const Affine3f &pose)
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

void temp_viz::Viz3d::VizImpl::updateWidgetPose(const String &id, const Affine3f &pose)
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

temp_viz::Affine3f temp_viz::Viz3d::VizImpl::getWidgetPose(const String &id) const
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