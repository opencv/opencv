#include "precomp.hpp"
#include <q/shapes.h>
#include <q/viz3d_impl.hpp>

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

void temp_viz::Viz3d::VizImpl::showPointCloud(const String& id, InputArray _cloud, InputArray _colors, const Affine3f& pose)
{
    Mat cloud = _cloud.getMat();
    Mat colors = _colors.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    CV_Assert(colors.type() == CV_8UC3 && cloud.size() == colors.size());
        
    vtkSmartPointer<vtkPolyData> polydata;
    vtkSmartPointer<vtkCellArray> vertices;
    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkIdTypeArray> initcells;
    vtkIdType nr_points = cloud.total();

    // If the cloud already exists, update otherwise create new one
    CloudActorMap::iterator am_it = cloud_actor_map_->find (id);
    bool exist = (am_it == cloud_actor_map_->end());
    if (exist)
    {
        // Add as new cloud
        allocVtkPolyData(polydata);
        //polydata = vtkSmartPointer<vtkPolyData>::New ();
        vertices = vtkSmartPointer<vtkCellArray>::New ();
        polydata->SetVerts (vertices);

        points = polydata->GetPoints ();

        if (!points)
        {
            points = vtkSmartPointer<vtkPoints>::New ();
            if (cloud.depth() == CV_32F)
                points->SetDataTypeToFloat();
            else if (cloud.depth() == CV_64F)
                points->SetDataTypeToDouble();
            polydata->SetPoints (points);
        }
        points->SetNumberOfPoints (nr_points);
    }
    else
    {
        // Update the cloud
        // Get the current poly data
        polydata = reinterpret_cast<vtkPolyDataMapper*>(am_it->second.actor->GetMapper ())->GetInput ();
        vertices = polydata->GetVerts ();
        points = polydata->GetPoints ();
        // Update the point data type based on the cloud
        if (cloud.depth() == CV_32F)
            points->SetDataTypeToFloat ();
        else if (cloud.depth() == CV_64F)
            points->SetDataTypeToDouble ();

        points->SetNumberOfPoints (nr_points);
    }

    if (cloud.depth() == CV_32F)
    {
        // Get a pointer to the beginning of the data array
        Vec3f *data_beg = vtkpoints_data<float>(points);
        Vec3f *data_end = NanFilter::copy(cloud, data_beg, cloud);
        std::transform(data_beg, data_end, data_beg, ApplyAffine(pose));
        nr_points = data_end - data_beg;

    }
    else if (cloud.depth() == CV_64F)
    {
        // Get a pointer to the beginning of the data array
        Vec3d *data_beg = vtkpoints_data<double>(points);
        Vec3d *data_end = NanFilter::copy(cloud, data_beg, cloud);
        std::transform(data_beg, data_end, data_beg, ApplyAffine(pose));
        nr_points = data_end - data_beg;
    }

    points->SetNumberOfPoints (nr_points);

    vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData ();

    if (exist)
        updateCells (cells, initcells, nr_points);
    else
        updateCells (cells, am_it->second.cells, nr_points);

    // Set the cells and the vertices
    vertices->SetCells (nr_points, cells);

    // Get a random color
    Vec3b* colors_data = new Vec3b[nr_points];
    NanFilter::copy(colors, colors_data, cloud);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    scalars->SetNumberOfComponents (3);
    scalars->SetNumberOfTuples (nr_points);
    scalars->SetArray (colors_data->val, 3 * nr_points, 0);

    // Assign the colors
    Vec2d minmax;
    polydata->GetPointData ()->SetScalars (scalars);
    scalars->GetRange (minmax.val);

    // If this is the new point cloud, a new actor is created
    if (exist)
    {
        vtkSmartPointer<vtkLODActor> actor;
        createActorFromVTKDataSet (polydata, actor);

        actor->GetMapper ()->SetScalarRange (minmax.val);

        // Add it to all renderers
        renderer_->AddActor (actor);

        // Save the pointer/ID pair to the global actor map
        (*cloud_actor_map_)[id].actor = actor;
        (*cloud_actor_map_)[id].cells = initcells;

        const Eigen::Vector4f sensor_origin = Eigen::Vector4f::Zero ();
        const Eigen::Quaternionf sensor_orientation = Eigen::Quaternionf::Identity ();

        // Save the viewpoint transformation matrix to the global actor map
        vtkSmartPointer<vtkMatrix4x4> transformation = vtkSmartPointer<vtkMatrix4x4>::New();
        convertToVtkMatrix (sensor_origin, sensor_orientation, transformation);

        (*cloud_actor_map_)[id].viewpoint_transformation_ = transformation;
    }
    else
    {
        // Update the mapper
        reinterpret_cast<vtkPolyDataMapper*>(am_it->second.actor->GetMapper ())->SetInput (polydata);
    }
}

void temp_viz::Viz3d::VizImpl::showPointCloud(const String& id, InputArray _cloud, const Color& color, const Affine3f& pose)
{
    // Generate an array of colors from single color
    Mat colors(_cloud.size(), CV_8UC3, color);
    showPointCloud(id, _cloud, colors, pose);
}

bool temp_viz::Viz3d::VizImpl::addPointCloudNormals (const cv::Mat &cloud, const cv::Mat& normals, int level, float scale, const std::string &id)
{
    CV_Assert(cloud.size() == normals.size() && cloud.type() == CV_32FC3 && normals.type() == CV_32FC3);

    if (cloud_actor_map_->find (id) != cloud_actor_map_->end ())
        return (false);

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    points->SetDataTypeToFloat ();
    vtkSmartPointer<vtkFloatArray> data = vtkSmartPointer<vtkFloatArray>::New ();
    data->SetNumberOfComponents (3);

    vtkIdType nr_normals = 0;
    float* pts = 0;

    // If the cloud is organized, then distribute the normal step in both directions
    if (cloud.cols > 1 && cloud.rows > 1)
    {
        vtkIdType point_step = static_cast<vtkIdType> (sqrt (double (level)));
        nr_normals = (static_cast<vtkIdType> ((cloud.cols - 1)/ point_step) + 1) *
                (static_cast<vtkIdType> ((cloud.rows - 1) / point_step) + 1);
        pts = new float[2 * nr_normals * 3];

        vtkIdType cell_count = 0;
        for (vtkIdType y = 0; y < cloud.rows; y += point_step)
            for (vtkIdType x = 0; x < cloud.cols; x += point_step)
            {
                cv::Point3f p = cloud.at<cv::Point3f>(y, x);
                cv::Point3f n = normals.at<cv::Point3f>(y, x) * scale;

                pts[2 * cell_count * 3 + 0] = p.x;
                pts[2 * cell_count * 3 + 1] = p.y;
                pts[2 * cell_count * 3 + 2] = p.z;
                pts[2 * cell_count * 3 + 3] = p.x + n.x;
                pts[2 * cell_count * 3 + 4] = p.y + n.y;
                pts[2 * cell_count * 3 + 5] = p.z + n.z;

                lines->InsertNextCell (2);
                lines->InsertCellPoint (2 * cell_count);
                lines->InsertCellPoint (2 * cell_count + 1);
                cell_count++;
            }
    }
    else
    {
        nr_normals = (cloud.size().area() - 1) / level + 1 ;
        pts = new float[2 * nr_normals * 3];

        for (vtkIdType i = 0, j = 0; j < nr_normals; j++, i = j * level)
        {
            cv::Point3f p = cloud.ptr<cv::Point3f>()[i];
            cv::Point3f n = normals.ptr<cv::Point3f>()[i] * scale;

            pts[2 * j * 3 + 0] = p.x;
            pts[2 * j * 3 + 1] = p.y;
            pts[2 * j * 3 + 2] = p.z;
            pts[2 * j * 3 + 3] = p.x + n.x;
            pts[2 * j * 3 + 4] = p.y + n.y;
            pts[2 * j * 3 + 5] = p.z + n.z;

            lines->InsertNextCell (2);
            lines->InsertCellPoint (2 * j);
            lines->InsertCellPoint (2 * j + 1);
        }
    }

    data->SetArray (&pts[0], 2 * nr_normals * 3, 0);
    points->SetData (data);

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints (points);
    polyData->SetLines (lines);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (polyData);
    mapper->SetColorModeToMapScalars();
    mapper->SetScalarModeToUsePointData();

    // create actor
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New ();
    actor->SetMapper (mapper);

    // Add it to all renderers
    renderer_->AddActor (actor);

    // Save the pointer/ID pair to the global actor map
    (*cloud_actor_map_)[id].actor = actor;
    return (true);
}


////////////////////////////////////////////////////////////////////////////////////////////
bool temp_viz::Viz3d::VizImpl::addLine (const cv::Point3f &pt1, const cv::Point3f &pt2, const Color& color, const std::string &id)
{
    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addLine] A shape with id <" << id <<  "> already exists! Please choose a different id and retry." << std::endl, false;

    vtkSmartPointer<vtkDataSet> data = createLine (pt1, pt2);

    // Create an Actor
    vtkSmartPointer<vtkLODActor> actor;
    createActorFromVTKDataSet (data, actor);
    actor->GetProperty ()->SetRepresentationToWireframe ();

    Color c = vtkcolor(color);
    actor->GetProperty ()->SetColor (c.val);
    actor->GetMapper ()->ScalarVisibilityOff ();
    renderer_->AddActor (actor);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = actor;
    return (true);
}



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

#include <vtkSphereSource.h>
////////////////////////////////////////////////////////////////////////////////////////////
bool temp_viz::Viz3d::VizImpl::addSphere (const cv::Point3f& center, float radius, const Color& color, const std::string &id)
{
    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addSphere] A shape with id <"<<id << "> already exists! Please choose a different id and retry." << std::endl, false;

    //vtkSmartPointer<vtkDataSet> data = createSphere (center.getVector4fMap (), radius);
    vtkSmartPointer<vtkSphereSource> data = vtkSmartPointer<vtkSphereSource>::New ();
    data->SetRadius (radius);
    data->SetCenter (center.x, center.y, center.z);
    data->SetPhiResolution (10);
    data->SetThetaResolution (10);
    data->LatLongTessellationOff ();
    data->Update ();

    // Setup actor and mapper
    vtkSmartPointer <vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    mapper->SetInputConnection (data->GetOutputPort ());

    // Create an Actor
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New ();
    actor->SetMapper (mapper);
    //createActorFromVTKDataSet (data, actor);
    actor->GetProperty ()->SetRepresentationToSurface ();
    actor->GetProperty ()->SetInterpolationToFlat ();

    Color c = vtkcolor(color);
    actor->GetProperty ()->SetColor (c.val);
    actor->GetMapper ()->ImmediateModeRenderingOn ();
    actor->GetMapper ()->StaticOn ();
    actor->GetMapper ()->ScalarVisibilityOff ();
    actor->GetMapper ()->Update ();
    renderer_->AddActor (actor);

    // Save the pointer/ID pair to the global actor map
    (*shape_actor_map_)[id] = actor;
    return (true);
}

////////////////////////////////////////////////////////////////////////////////////////////
bool temp_viz::Viz3d::VizImpl::updateSphere (const cv::Point3f &center, float radius, const Color& color, const std::string &id)
{
    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
    if (am_it == shape_actor_map_->end ())
        return (false);

    //////////////////////////////////////////////////////////////////////////
    // Get the actor pointer
    vtkLODActor* actor = vtkLODActor::SafeDownCast (am_it->second);
    vtkAlgorithm *algo = actor->GetMapper ()->GetInput ()->GetProducerPort ()->GetProducer ();
    vtkSphereSource *src = vtkSphereSource::SafeDownCast (algo);

    src->SetCenter(center.x, center.y, center.z);
    src->SetRadius(radius);
    src->Update ();
    Color c = vtkcolor(color);
    actor->GetProperty ()->SetColor (c.val);
    actor->Modified ();

    return (true);
}

//////////////////////////////////////////////////
bool temp_viz::Viz3d::VizImpl::addText3D (const std::string &text, const cv::Point3f& position, const Color& color, double textScale, const std::string &id)
{
    std::string tid;
    if (id.empty ())
        tid = text;
    else
        tid = id;

    // Check to see if this ID entry already exists (has it been already added to the visualizer?)
    ShapeActorMap::iterator am_it = shape_actor_map_->find (tid);
    if (am_it != shape_actor_map_->end ())
        return std::cout << "[addText3d] A text with id <" << tid << "> already exists! Please choose a different id and retry." << std::endl, false;

    vtkSmartPointer<vtkVectorText> textSource = vtkSmartPointer<vtkVectorText>::New ();
    textSource->SetText (text.c_str());
    textSource->Update ();

    vtkSmartPointer<vtkPolyDataMapper> textMapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    textMapper->SetInputConnection (textSource->GetOutputPort ());

    // Since each follower may follow a different camera, we need different followers
    vtkRenderer* renderer = renderer_;

    vtkSmartPointer<vtkFollower> textActor = vtkSmartPointer<vtkFollower>::New ();
    textActor->SetMapper (textMapper);
    textActor->SetPosition (position.x, position.y, position.z);
    textActor->SetScale (textScale);

    Color c = vtkcolor(color);
    textActor->GetProperty ()->SetColor (c.val);
    textActor->SetCamera (renderer->GetActiveCamera ());

    renderer->AddActor (textActor);
    renderer->Render ();

    // Save the pointer/ID pair to the global actor map. If we are saving multiple vtkFollowers
    // for multiple viewport
    (*shape_actor_map_)[tid] = textActor;


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
