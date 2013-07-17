#include "precomp.hpp"

namespace cv
{
    namespace viz
    {
        template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
cv::viz::LineWidget::LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color)
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

void cv::viz::LineWidget::setLineWidth(float line_width)
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    actor->GetProperty()->SetLineWidth(line_width);
}

float cv::viz::LineWidget::getLineWidth()
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    return actor->GetProperty()->GetLineWidth();
}

template<> cv::viz::LineWidget cv::viz::Widget::cast<cv::viz::LineWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<LineWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// plane widget implementation

cv::viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, double size, const Color &color)
{    
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    plane->SetNormal (coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm(Vec3f(coefs.val));
    plane->Push (-coefs[3] / norm);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->SetScale(size);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::PlaneWidget::PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
    Point3f coefs3(coefs[0], coefs[1], coefs[2]);
    double norm_sqr = 1.0 / coefs3.dot (coefs3);
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot(pt) + coefs[3];
    Vec3f p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter (p_center[0], p_center[1], p_center[2]);
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();
    mapper->SetInput(plane->GetOutput ());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    actor->SetScale(size);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::PlaneWidget cv::viz::Widget::cast<cv::viz::PlaneWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<PlaneWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// sphere widget implementation

cv::viz::SphereWidget::SphereWidget(const Point3f &center, float radius, int sphere_resolution, const Color &color)
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

template<> cv::viz::SphereWidget cv::viz::Widget::cast<cv::viz::SphereWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<SphereWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// arrow widget implementation

cv::viz::ArrowWidget::ArrowWidget(const Point3f& pt1, const Point3f& pt2, double thickness, const Color &color)
{
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New ();
    arrowSource->SetShaftRadius(thickness);
    // The thickness and radius of the tip are adjusted based on the thickness of the arrow
    arrowSource->SetTipRadius(thickness * 3.0);
    arrowSource->SetTipLength(thickness * 10.0);
    
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

template<> cv::viz::ArrowWidget cv::viz::Widget::cast<cv::viz::ArrowWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<ArrowWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// circle widget implementation

cv::viz::CircleWidget::CircleWidget(const Point3f& pt, double radius, double thickness, const Color& color)
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

template<> cv::viz::CircleWidget cv::viz::Widget::cast<cv::viz::CircleWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CircleWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

cv::viz::CylinderWidget::CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides, const Color &color)
{   
    const Point3f pt2 = pt_on_axis + axis_direction;
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

template<> cv::viz::CylinderWidget cv::viz::Widget::cast<cv::viz::CylinderWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CylinderWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

cv::viz::CubeWidget::CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame, const Color &color)
{   
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New ();   
    if (wire_frame)
    {
        vtkSmartPointer<vtkOutlineSource> cube = vtkSmartPointer<vtkOutlineSource>::New();
        cube->SetBounds (pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
        mapper->SetInput(cube->GetOutput ());
    }
    else
    {
        vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New ();
        cube->SetBounds (pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
        mapper->SetInput(cube->GetOutput ());
    }
    
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::CubeWidget cv::viz::Widget::cast<cv::viz::CubeWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CubeWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// coordinate system widget implementation

cv::viz::CoordinateSystemWidget::CoordinateSystemWidget(double scale)
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
    
    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::CoordinateSystemWidget cv::viz::Widget::cast<cv::viz::CoordinateSystemWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<CoordinateSystemWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// polyline widget implementation

struct cv::viz::PolyLineWidget::CopyImpl
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

cv::viz::PolyLineWidget::PolyLineWidget(InputArray _pointData, const Color &color)
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

template<> cv::viz::PolyLineWidget cv::viz::Widget::cast<cv::viz::PolyLineWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<PolyLineWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// grid widget implementation

cv::viz::GridWidget::GridWidget(Vec2i dimensions, Vec2d spacing, const Color &color)
{
    // Create the grid using image data
    vtkSmartPointer<vtkImageData> grid = vtkSmartPointer<vtkImageData>::New();
    
    // Add 1 to dimensions because in ImageData dimensions is the number of lines
    // - however here it means number of cells
    grid->SetDimensions(dimensions[0]+1, dimensions[1]+1, 1);
    grid->SetSpacing(spacing[0], spacing[1], 0.);
    
    // Set origin of the grid to be the middle of the grid
    grid->SetOrigin(dimensions[0] * spacing[0] * (-0.5), dimensions[1] * spacing[1] * (-0.5), 0);
    
    // Extract the edges so we have the grid
    vtkSmartPointer<vtkExtractEdges> filter = vtkSmartPointer<vtkExtractEdges>::New();
    filter->SetInputConnection(grid->GetProducerPort());
    filter->Update();
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInput(filter->GetOutput());
    
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::GridWidget cv::viz::Widget::cast<cv::viz::GridWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<GridWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text3D widget implementation

cv::viz::Text3DWidget::Text3DWidget(const String &text, const Point3f &position, double text_scale, const Color &color)
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

void cv::viz::Text3DWidget::setText(const String &text)
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

cv::String cv::viz::Text3DWidget::getText() const
{
    vtkFollower *actor = vtkFollower::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    
    vtkPolyDataMapper *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    vtkVectorText * textSource = vtkVectorText::SafeDownCast(mapper->GetInputConnection(0,0)->GetProducer());
    CV_Assert(textSource);
    
    return textSource->GetText();
}

template<> cv::viz::Text3DWidget cv::viz::Widget::cast<cv::viz::Text3DWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<Text3DWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text widget implementation

cv::viz::TextWidget::TextWidget(const String &text, const Point2i &pos, int font_size, const Color &color)
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

template<> cv::viz::TextWidget cv::viz::Widget::cast<cv::viz::TextWidget>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<TextWidget&>(widget);
}

void cv::viz::TextWidget::setText(const String &text)
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    actor->SetInput(text.c_str());
}

cv::String cv::viz::TextWidget::getText() const
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    return actor->GetInput();
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// image overlay widget implementation

struct cv::viz::ImageOverlayWidget::CopyImpl
{
    struct Impl
    {
        static void copyImageMultiChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
        {
            int i_chs = image.channels();
    
            for (int i = 0; i < image.rows; ++i)
            {
                const unsigned char * irows = image.ptr<unsigned char>(i);
                for (int j = 0; j < image.cols; ++j, irows += i_chs)
                {
                    unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                    memcpy(vrows, irows, i_chs);
                    std::swap(vrows[0], vrows[2]); // BGR -> RGB
                }
            }
            output->Modified();
        }
        
        static void copyImageSingleChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
        {
            for (int i = 0; i < image.rows; ++i)
            {
                const unsigned char * irows = image.ptr<unsigned char>(i);
                for (int j = 0; j < image.cols; ++j, ++irows)
                {
                    unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                    *vrows = *irows;
                }
            }
            output->Modified();
        }
    };
    
    static void copyImage(const Mat &image, vtkSmartPointer<vtkImageData> output)
    {
        int i_chs = image.channels();
        if (i_chs > 1)
        {
            // Multi channel images are handled differently because of BGR <-> RGB
            Impl::copyImageMultiChannel(image, output);
        }
        else
        {
            Impl::copyImageSingleChannel(image, output);
        }
    }
};

cv::viz::ImageOverlayWidget::ImageOverlayWidget(const Mat &image, const Point2i &pos)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    
    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    vtk_image->SetDimensions(image.cols, image.rows, 1);
    vtk_image->SetNumberOfScalarComponents(image.channels());
    vtk_image->SetScalarTypeToUnsignedChar();
    vtk_image->AllocateScalars();
    
    CopyImpl::copyImage(image, vtk_image);
    
    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
    flipFilter->Update();
    
    vtkSmartPointer<vtkImageMapper> imageMapper = vtkSmartPointer<vtkImageMapper>::New();
    imageMapper->SetInputConnection(flipFilter->GetOutputPort());
    imageMapper->SetColorWindow(255); // OpenCV color
    imageMapper->SetColorLevel(127.5);  
    
    vtkSmartPointer<vtkActor2D> actor = vtkSmartPointer<vtkActor2D>::New();
    actor->SetMapper(imageMapper);
    actor->SetPosition(pos.x, pos.y);
    
    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::ImageOverlayWidget::setImage(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    
    vtkActor2D *actor = vtkActor2D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    
    vtkImageMapper *mapper = vtkImageMapper::SafeDownCast(actor->GetMapper());
    CV_Assert(mapper);
    
    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    vtk_image->SetDimensions(image.cols, image.rows, 1);
    vtk_image->SetNumberOfScalarComponents(image.channels());
    vtk_image->SetScalarTypeToUnsignedChar();
    vtk_image->AllocateScalars();
    
    CopyImpl::copyImage(image, vtk_image);
    
    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
    flipFilter->Update();
    
    mapper->SetInputConnection(flipFilter->GetOutputPort());
}

template<> cv::viz::ImageOverlayWidget cv::viz::Widget::cast<cv::viz::ImageOverlayWidget>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<ImageOverlayWidget&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// image 3D widget implementation

struct cv::viz::Image3DWidget::CopyImpl
{
    struct Impl
    {
        static void copyImageMultiChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
        {
            int i_chs = image.channels();
    
            for (int i = 0; i < image.rows; ++i)
            {
                const unsigned char * irows = image.ptr<unsigned char>(i);
                for (int j = 0; j < image.cols; ++j, irows += i_chs)
                {
                    unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                    memcpy(vrows, irows, i_chs);
                    std::swap(vrows[0], vrows[2]); // BGR -> RGB
                }
            }
            output->Modified();
        }
        
        static void copyImageSingleChannel(const Mat &image, vtkSmartPointer<vtkImageData> output)
        {
            for (int i = 0; i < image.rows; ++i)
            {
                const unsigned char * irows = image.ptr<unsigned char>(i);
                for (int j = 0; j < image.cols; ++j, ++irows)
                {
                    unsigned char * vrows = static_cast<unsigned char *>(output->GetScalarPointer(j,i,0));
                    *vrows = *irows;
                }
            }
            output->Modified();
        }
    };
    
    static void copyImage(const Mat &image, vtkSmartPointer<vtkImageData> output)
    {
        int i_chs = image.channels();
        if (i_chs > 1)
        {
            // Multi channel images are handled differently because of BGR <-> RGB
            Impl::copyImageMultiChannel(image, output);
        }
        else
        {
            Impl::copyImageSingleChannel(image, output);
        }
    }
};

cv::viz::Image3DWidget::Image3DWidget(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    
    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    vtk_image->SetDimensions(image.cols, image.rows, 1);
    vtk_image->SetNumberOfScalarComponents(image.channels());
    vtk_image->SetScalarTypeToUnsignedChar();
    vtk_image->AllocateScalars();
    
    CopyImpl::copyImage(image, vtk_image);
    
    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
    flipFilter->Update();
    
    vtkSmartPointer<vtkImageActor> actor = vtkSmartPointer<vtkImageActor>::New();
    actor->SetInput(flipFilter->GetOutput());
    
    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::Image3DWidget::setImage(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    
    vtkImageActor *actor = vtkImageActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert(actor);
    
    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    vtk_image->SetDimensions(image.cols, image.rows, 1);
    vtk_image->SetNumberOfScalarComponents(image.channels());
    vtk_image->SetScalarTypeToUnsignedChar();
    vtk_image->AllocateScalars();
    
    CopyImpl::copyImage(image, vtk_image);
    
    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
    flipFilter->Update();
    
    actor->SetInput(flipFilter->GetOutput());
}

template<> cv::viz::Image3DWidget cv::viz::Widget::cast<cv::viz::Image3DWidget>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<Image3DWidget&>(widget);
}