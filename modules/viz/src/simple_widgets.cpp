#include "precomp.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
temp_viz::LineWidget::LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color)
{   
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1 (pt1.x, pt1.y, pt1.z);
    line->SetPoint2 (pt2.x, pt2.y, pt2.z);
    line->Update ();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(line->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

void temp_viz::LineWidget::setLineWidth(float line_width)
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    actor->GetProperty()->SetLineWidth(line_width);
}

float temp_viz::LineWidget::getLineWidth()
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    return actor->GetProperty()->GetLineWidth();
}

template<> temp_viz::LineWidget temp_viz::Widget::cast<temp_viz::LineWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<LineWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// plane widget implementation

temp_viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, double size, const Color &color)
{    
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    plane->SetNormal (coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm(cv::Vec3f(coefs.val));
    plane->Push (-coefs[3] / norm);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->SetScale(size);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

temp_viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    cv::Point3f coefs3(coefs[0], coefs[1], coefs[2]);
    double norm_sqr = 1.0 / coefs3.dot (coefs3);
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot(pt) + coefs[3];
    cv::Vec3f p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter (p_center[0], p_center[1], p_center[2]);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->SetScale(size);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::PlaneWidget temp_viz::Widget::cast<temp_viz::PlaneWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<PlaneWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// sphere widget implementation

temp_viz::SphereWidget::SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution, const Color &color)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New ();
    sphere->SetRadius (radius);
    sphere->SetCenter (center.x, center.y, center.z);
    sphere->SetPhiResolution (sphere_resolution);
    sphere->SetThetaResolution (sphere_resolution);
    sphere->LatLongTessellationOff ();
    sphere->Update ();
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(sphere->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::SphereWidget temp_viz::Widget::cast<temp_viz::SphereWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<SphereWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// arrow widget implementation

temp_viz::ArrowWidget::ArrowWidget(const Point3f& pt1, const Point3f& pt2, const Color &color)
{
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New ();
    
    float startPoint[3], endPoint[3];
    startPoint[0] = pt1.x;
    startPoint[1] = pt1.y;
    startPoint[2] = pt1.z;
    endPoint[0] = pt2.x;
    endPoint[1] = pt2.y;
    endPoint[2] = pt2.z;
    float normalizedX[3], normalizedY[3], normalizedZ[3];
    
    // The X axis is a vector from start to end
    vtkMath::Subtract(endPoint, startPoint, normalizedX);
    float length = vtkMath::Norm(normalizedX);
    vtkMath::Normalize(normalizedX);

    // The Z axis is an arbitrary vecotr cross X
    float arbitrary[3];
    arbitrary[0] = vtkMath::Random(-10,10);
    arbitrary[1] = vtkMath::Random(-10,10);
    arbitrary[2] = vtkMath::Random(-10,10);
    vtkMath::Cross(normalizedX, arbitrary, normalizedZ);
    vtkMath::Normalize(normalizedZ);

    // The Y axis is Z cross X
    vtkMath::Cross(normalizedZ, normalizedX, normalizedY);
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();

    // Create the direction cosine matrix
    matrix->Identity();
    for (unsigned int i = 0; i < 3; i++)
    {
        matrix->SetElement(i, 0, normalizedX[i]);
        matrix->SetElement(i, 1, normalizedY[i]);
        matrix->SetElement(i, 2, normalizedZ[i]);
    }    

    // Apply the transforms
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(startPoint);
    transform->Concatenate(matrix);
    transform->Scale(length, length, length);

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(arrowSource->GetOutputPort());
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(transformPD->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::ArrowWidget temp_viz::Widget::cast<temp_viz::ArrowWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<ArrowWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// circle widget implementation

temp_viz::CircleWidget::CircleWidget(const temp_viz::Point3f& pt, double radius, double thickness, const temp_viz::Color& color)
{
    vtkSmartPointer<vtkDiskSource> disk = vtkSmartPointer<vtkDiskSource>::New ();
    // Maybe the resolution should be lower e.g. 50 or 25
    disk->SetCircumferentialResolution (50);
    disk->SetInnerRadius (radius - thickness);
    disk->SetOuterRadius (radius + thickness);

    // Set the circle origin
    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New ();
    t->Identity ();
    t->Translate (pt.x, pt.y, pt.z);

    vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New ();
    tf->SetTransform (t);
    tf->SetInputConnection (disk->GetOutputPort ());
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(tf->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::CircleWidget temp_viz::Widget::cast<temp_viz::CircleWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CircleWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

temp_viz::CylinderWidget::CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides, const Color &color)
{   
    const cv::Point3f pt2 = pt_on_axis + axis_direction;
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New ();
    line->SetPoint1 (pt_on_axis.x, pt_on_axis.y, pt_on_axis.z);
    line->SetPoint2 (pt2.x, pt2.y, pt2.z);
    
    vtkSmartPointer<vtkTubeFilter> tuber = vtkSmartPointer<vtkTubeFilter>::New ();
    tuber->SetInputConnection (line->GetOutputPort ());
    tuber->SetRadius (radius);
    tuber->SetNumberOfSides (numsides);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(tuber->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New ();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::CylinderWidget temp_viz::Widget::cast<temp_viz::CylinderWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CylinderWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

temp_viz::CubeWidget::CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame, const Color &color)
{
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
    cube->SetBounds (pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(cube->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    
    if (wire_frame)
        actor->GetProperty ()->SetRepresentationToWireframe ();
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::CubeWidget temp_viz::Widget::cast<temp_viz::CubeWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CubeWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// coordinate system widget implementation

temp_viz::CoordinateSystemWidget::CoordinateSystemWidget(double scale, const Affine3f& affine)
{
    vtkSmartPointer<vtkAxes> axes = vtkSmartPointer<vtkAxes>::New ();
    axes->SetOrigin (0, 0, 0);
    axes->SetScaleFactor (scale);

    vtkSmartPointer<vtkFloatArray> axes_colors = vtkSmartPointer<vtkFloatArray>::New ();
    axes_colors->Allocate (6);
    axes_colors->InsertNextValue (0.0);
    axes_colors->InsertNextValue (0.0);
    axes_colors->InsertNextValue (0.5);
    axes_colors->InsertNextValue (0.5);
    axes_colors->InsertNextValue (1.0);
    axes_colors->InsertNextValue (1.0);

    vtkSmartPointer<vtkPolyData> axes_data = axes->GetOutput ();
    axes_data->Update ();
    axes_data->GetPointData ()->SetScalars (axes_colors);

    vtkSmartPointer<vtkTubeFilter> axes_tubes = vtkSmartPointer<vtkTubeFilter>::New ();
    axes_tubes->SetInput (axes_data);
    axes_tubes->SetRadius (axes->GetScaleFactor () / 50.0);
    axes_tubes->SetNumberOfSides (6);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetScalarModeToUsePointData ();
    mapper->SetInput(axes_tubes->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    cv::Vec3d t = affine.translation();
    actor->SetPosition (t[0], t[1], t[2]);

    cv::Matx33f m = affine.rotation();

    cv::Vec3f rvec;
    cv::Rodrigues(m, rvec);

    float r_angle = cv::norm(rvec);
    rvec *= 1.f/r_angle;

    actor->SetOrientation(0,0,0);
    actor->RotateWXYZ(r_angle*180/CV_PI,rvec[0], rvec[1], rvec[2]);
    
    WidgetAccessor::setProp(*this, actor);
}

template<> temp_viz::CoordinateSystemWidget temp_viz::Widget::cast<temp_viz::CoordinateSystemWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CoordinateSystemWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// polyline widget implementation

struct temp_viz::PolyLineWidget::CopyImpl
{    
    template<typename _Tp>
    static void copy(const Mat& source, Vec<_Tp, 3> *output, vtkSmartPointer<vtkPolyLine> polyLine)
    {
        int s_chs = source.channels();

        for(int y = 0, id = 0; y < source.rows; ++y)
        {
            const _Tp* srow = source.ptr<_Tp>(y);

            for(int x = 0; x < source.cols; ++x, srow += s_chs, ++id)
            {
                *output++ = Vec<_Tp, 3>(srow);
                polyLine->GetPointIds()->SetId(id,id);
            }
        }
    }
};

temp_viz::PolyLineWidget::PolyLineWidget(InputArray _pointData, const Color &color)
{
    Mat pointData = _pointData.getMat();
    CV_Assert(pointData.type() == CV_32FC3 || pointData.type() == CV_32FC4 || pointData.type() == CV_64FC3 || pointData.type() == CV_64FC4);
    vtkIdType nr_points = pointData.total();    
    
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();
    
    if (pointData.depth() == CV_32F)
        points->SetDataTypeToFloat();
    else
        points->SetDataTypeToDouble();
    
    points->SetNumberOfPoints(nr_points);
    polyLine->GetPointIds()->SetNumberOfIds(nr_points);
    
    if (pointData.depth() == CV_32F)
    {
        // Get a pointer to the beginning of the data array
        Vec3f *data_beg = vtkpoints_data<float>(points);
        CopyImpl::copy(pointData, data_beg, polyLine);
    }
    else if (pointData.depth() == CV_64F)
    {
        // Get a pointer to the beginning of the data array
        Vec3d *data_beg = vtkpoints_data<double>(points);
        CopyImpl::copy(pointData, data_beg, polyLine);
    }
    
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
    cells->InsertNextCell(polyLine);
    
    polyData->SetPoints(points);
    polyData->SetLines(cells);
    
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInput(polyData);
    
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::PolyLineWidget temp_viz::Widget::cast<temp_viz::PolyLineWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<PolyLineWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// grid widget implementation

temp_viz::GridWidget::GridWidget(Vec2i dimensions, Vec2d spacing, const Color &color)
{
    // Create the grid using image data
    vtkSmartPointer<vtkImageData> grid = vtkSmartPointer<vtkImageData>::New();
    
    // Add 1 to dimensions because in ImageData dimensions is the number of lines
    // - however here it means number of cells
    grid->SetDimensions(dimensions[0]+1, dimensions[1]+1, 1);
    grid->SetSpacing(spacing[0], spacing[1], 0.);
    
    // Set origin of the grid to be the middle of the grid
    grid->SetOrigin(dimensions[0] * spacing[0] * (-0.5), dimensions[1] * spacing[1] * (-0.5), 0);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInput(grid);
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    // Show it as wireframe
    actor->GetProperty ()->SetRepresentationToWireframe ();
    WidgetAccessor::setProp(*this, actor);
}

template<> temp_viz::GridWidget temp_viz::Widget::cast<temp_viz::GridWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<GridWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text3D widget implementation

temp_viz::Text3DWidget::Text3DWidget(const String &text, const Point3f &position, double text_scale, const Color &color)
{
    vtkSmartPointer<vtkVectorText> textSource = vtkSmartPointer<vtkVectorText>::New ();
    textSource->SetText (text.c_str());
    textSource->Update ();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
    mapper->SetInputConnection (textSource->GetOutputPort ());
    
    vtkSmartPointer<vtkFollower> actor = vtkSmartPointer<vtkFollower>::New ();
    actor->SetMapper (mapper);
    actor->SetPosition (position.x, position.y, position.z);
    actor->SetScale (text_scale);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

void temp_viz::Text3DWidget::setText(const String &text)
{
    vtkFollower *actor = vtkFollower::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    
    // Update text source
    vtkPolyDataMapper *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    vtkVectorText * textSource = vtkVectorText::SafeDownCast(mapper->GetInputConnection(0,0)->GetProducer());
    CV_Assert(textSource);
    
    textSource->SetText(text.c_str());
    textSource->Update();
}

temp_viz::String temp_viz::Text3DWidget::getText() const
{
    vtkFollower *actor = vtkFollower::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    
    vtkPolyDataMapper *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    vtkVectorText * textSource = vtkVectorText::SafeDownCast(mapper->GetInputConnection(0,0)->GetProducer());
    CV_Assert(textSource);
    
    return textSource->GetText();
}

template<> temp_viz::Text3DWidget temp_viz::Widget::cast<temp_viz::Text3DWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<Text3DWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text widget implementation

temp_viz::TextWidget::TextWidget(const String &text, const Point2i &pos, int font_size, const Color &color)
{
    vtkSmartPointer<vtkTextActor> actor = vtkSmartPointer<vtkTextActor>::New();
    actor->SetPosition (pos.x, pos.y);
    actor->SetInput (text.c_str ());

    vtkSmartPointer<vtkTextProperty> tprop = actor->GetTextProperty ();
    tprop->SetFontSize (font_size);
    tprop->SetFontFamilyToArial ();
    tprop->SetJustificationToLeft ();
    tprop->BoldOn ();

    Color c = vtkcolor(color);
    tprop->SetColor (c.val);
    
    WidgetAccessor::setProp(*this, actor);
}

template<> temp_viz::TextWidget temp_viz::Widget::cast<temp_viz::TextWidget>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<TextWidget&>(widget);
}

void temp_viz::TextWidget::setText(const String &text)
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    actor->SetInput(text.c_str());
}

temp_viz::String temp_viz::TextWidget::getText() const
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    return actor->GetInput();
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// point cloud widget implementation

struct temp_viz::CloudWidget::CreateCloudWidget
{
    static inline vtkSmartPointer<vtkPolyData> create(const Mat &cloud, vtkIdType &nr_points)
    {
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New ();
        vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New ();
        
        polydata->SetVerts (vertices);
        
        vtkSmartPointer<vtkPoints> points = polydata->GetPoints();
        vtkSmartPointer<vtkIdTypeArray> initcells;
        nr_points = cloud.total();

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
        
        if (cloud.depth() == CV_32F)
        {
            // Get a pointer to the beginning of the data array
            Vec3f *data_beg = vtkpoints_data<float>(points);
            Vec3f *data_end = NanFilter::copy(cloud, data_beg, cloud);
            nr_points = data_end - data_beg;
        }
        else if (cloud.depth() == CV_64F)
        {
            // Get a pointer to the beginning of the data array
            Vec3d *data_beg = vtkpoints_data<double>(points);
            Vec3d *data_end = NanFilter::copy(cloud, data_beg, cloud);
            nr_points = data_end - data_beg;
        }
        points->SetNumberOfPoints (nr_points);
        
        // Update cells
        vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData ();
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
        
        // Set the cells and the vertices
        vertices->SetCells (nr_points, cells);
        return polydata;
    }
};

temp_viz::CloudWidget::CloudWidget(InputArray _cloud, InputArray _colors)
{
    Mat cloud = _cloud.getMat();
    Mat colors = _colors.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    CV_Assert(colors.type() == CV_8UC3 && cloud.size() == colors.size());

    if (cloud.isContinuous() && colors.isContinuous())
    {
        cloud.reshape(cloud.channels(), 1);
        colors.reshape(colors.channels(), 1);
    }

    vtkIdType nr_points;
    vtkSmartPointer<vtkPolyData> polydata = CreateCloudWidget::create(cloud, nr_points);

    // Filter colors
    Vec3b* colors_data = new Vec3b[nr_points];
    NanFilter::copy(colors, colors_data, cloud);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
    scalars->SetNumberOfComponents (3);
    scalars->SetNumberOfTuples (nr_points);
    scalars->SetArray (colors_data->val, 3 * nr_points, 0);

    // Assign the colors
    polydata->GetPointData ()->SetScalars (scalars);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (polydata);

    cv::Vec3d minmax(scalars->GetRange());
    mapper->SetScalarRange(minmax.val);
    mapper->SetScalarModeToUsePointData ();

    bool interpolation = (polydata && polydata->GetNumberOfCells () != polydata->GetNumberOfVerts ());

    mapper->SetInterpolateScalarsBeforeMapping (interpolation);
    mapper->ScalarVisibilityOn ();
        
    mapper->ImmediateModeRenderingOff ();
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, polydata->GetNumberOfPoints () / 10)));
    actor->GetProperty ()->SetInterpolationToFlat ();
    actor->GetProperty ()->BackfaceCullingOn ();
    actor->SetMapper (mapper);
    
    WidgetAccessor::setProp(*this, actor);
}

temp_viz::CloudWidget::CloudWidget(InputArray _cloud, const Color &color)
{
    Mat cloud = _cloud.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    

    vtkIdType nr_points;
    vtkSmartPointer<vtkPolyData> polydata = CreateCloudWidget::create(cloud, nr_points);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (polydata);

    bool interpolation = (polydata && polydata->GetNumberOfCells () != polydata->GetNumberOfVerts ());

    mapper->SetInterpolateScalarsBeforeMapping (interpolation);
    mapper->ScalarVisibilityOff ();
        
    mapper->ImmediateModeRenderingOff ();
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetNumberOfCloudPoints (int (std::max<vtkIdType> (1, polydata->GetNumberOfPoints () / 10)));
    actor->GetProperty ()->SetInterpolationToFlat ();
    actor->GetProperty ()->BackfaceCullingOn ();
    actor->SetMapper (mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::CloudWidget temp_viz::Widget::cast<temp_viz::CloudWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CloudWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cloud normals widget implementation

struct temp_viz::CloudNormalsWidget::ApplyCloudNormals
{
    template<typename _Tp>
    struct Impl 
    {
        static vtkSmartPointer<vtkCellArray> applyOrganized(const cv::Mat &cloud, const cv::Mat& normals, 
                                                            int level, float scale, _Tp *&pts, vtkIdType &nr_normals)
        {
            vtkIdType point_step = static_cast<vtkIdType> (sqrt (double (level)));
            nr_normals = (static_cast<vtkIdType> ((cloud.cols - 1)/ point_step) + 1) *
                    (static_cast<vtkIdType> ((cloud.rows - 1) / point_step) + 1);
            vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
            
            pts = new _Tp[2 * nr_normals * 3];

            int cch = cloud.channels();
            vtkIdType cell_count = 0;    
            for (vtkIdType y = 0; y < cloud.rows; y += point_step)
            {
                const _Tp *prow = cloud.ptr<_Tp>(y);
                const _Tp *nrow = normals.ptr<_Tp>(y);
                for (vtkIdType x = 0; x < cloud.cols; x += point_step * cch)
                {
                    pts[2 * cell_count * 3 + 0] = prow[x];
                    pts[2 * cell_count * 3 + 1] = prow[x+1];
                    pts[2 * cell_count * 3 + 2] = prow[x+2];
                    pts[2 * cell_count * 3 + 3] = prow[x] + nrow[x] * scale;
                    pts[2 * cell_count * 3 + 4] = prow[x+1] + nrow[x+1] * scale;
                    pts[2 * cell_count * 3 + 5] = prow[x+2] + nrow[x+2] * scale;

                    lines->InsertNextCell (2);
                    lines->InsertCellPoint (2 * cell_count);
                    lines->InsertCellPoint (2 * cell_count + 1);
                    cell_count++;
                }
            }
            return lines;
        }
        
        static vtkSmartPointer<vtkCellArray> applyUnorganized(const cv::Mat &cloud, const cv::Mat& normals, 
                                                              int level, float scale, _Tp *&pts, vtkIdType &nr_normals)
        {
            vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
            nr_normals = (cloud.size().area() - 1) / level + 1 ;
            pts = new _Tp[2 * nr_normals * 3];

            int cch = cloud.channels();
            const _Tp *p = cloud.ptr<_Tp>();
            const _Tp *n = normals.ptr<_Tp>();
            for (vtkIdType i = 0, j = 0; j < nr_normals; j++, i = j * level * cch)
            {
                
                pts[2 * j * 3 + 0] = p[i];
                pts[2 * j * 3 + 1] = p[i+1];
                pts[2 * j * 3 + 2] = p[i+2];
                pts[2 * j * 3 + 3] = p[i] + n[i] * scale;
                pts[2 * j * 3 + 4] = p[i+1] + n[i+1] * scale;
                pts[2 * j * 3 + 5] = p[i+2] + n[i+2] * scale;

                lines->InsertNextCell (2);
                lines->InsertCellPoint (2 * j);
                lines->InsertCellPoint (2 * j + 1);
            }
            return lines;
        }
    };
    
    template<typename _Tp>
    static inline vtkSmartPointer<vtkCellArray> apply(const cv::Mat &cloud, const cv::Mat& normals, 
                                                               int level, float scale, _Tp *&pts, vtkIdType &nr_normals)
    {
        if (cloud.cols > 1 && cloud.rows > 1)
            return ApplyCloudNormals::Impl<_Tp>::applyOrganized(cloud, normals, level, scale, pts, nr_normals);
        else
            return ApplyCloudNormals::Impl<_Tp>::applyUnorganized(cloud, normals, level, scale, pts, nr_normals);
    }
};

temp_viz::CloudNormalsWidget::CloudNormalsWidget(InputArray _cloud, InputArray _normals, int level, float scale, const Color &color)
{
    Mat cloud = _cloud.getMat();
    Mat normals = _normals.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    CV_Assert(cloud.size() == normals.size() && cloud.type() == normals.type());
  
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
    vtkIdType nr_normals = 0;
    
    if (cloud.depth() == CV_32F)
    {
        points->SetDataTypeToFloat();
    
        vtkSmartPointer<vtkFloatArray> data = vtkSmartPointer<vtkFloatArray>::New ();
        data->SetNumberOfComponents (3);
        
        float* pts = 0;
        lines = ApplyCloudNormals::apply(cloud, normals, level, scale, pts, nr_normals);
        data->SetArray (&pts[0], 2 * nr_normals * 3, 0);
        points->SetData (data);
    }
    else
    {
        points->SetDataTypeToDouble();
    
        vtkSmartPointer<vtkDoubleArray> data = vtkSmartPointer<vtkDoubleArray>::New ();
        data->SetNumberOfComponents (3);
        
        double* pts = 0;
        lines = ApplyCloudNormals::apply(cloud, normals, level, scale, pts, nr_normals);
        data->SetArray (&pts[0], 2 * nr_normals * 3, 0);
        points->SetData (data);
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints (points);
    polyData->SetLines (lines);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput (polyData);
    mapper->SetColorModeToMapScalars();
    mapper->SetScalarModeToUsePointData();
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> temp_viz::CloudNormalsWidget temp_viz::Widget::cast<temp_viz::CloudNormalsWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CloudNormalsWidget&>(widget);
}


