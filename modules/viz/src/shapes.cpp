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

namespace cv
{
    namespace viz
    {
        template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
cv::viz::WLine::WLine(const Point3d &pt1, const Point3d &pt2, const Color &color)
{
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(pt1.x, pt1.y, pt1.z);
    line->SetPoint2(pt2.x, pt2.y, pt2.z);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(line->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WLine cv::viz::Widget::cast<cv::viz::WLine>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WLine&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// plane widget implementation

namespace cv { namespace viz { namespace
{
    struct PlaneUtils
    {
        template<typename _Tp>
        static vtkSmartPointer<vtkTransformPolyDataFilter> setSize(const Vec<_Tp, 3> &center, vtkSmartPointer<vtkAlgorithmOutput> poly_data_port, double size)
        {
            vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
            transform->PreMultiply();
            transform->Translate(center[0], center[1], center[2]);
            transform->Scale(size, size, size);
            transform->Translate(-center[0], -center[1], -center[2]);

            vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
            transform_filter->SetInputConnection(poly_data_port);
            transform_filter->SetTransform(transform);
            transform_filter->Update();

            return transform_filter;
        }
    };
}}}

cv::viz::WPlane::WPlane(const Vec4d& coefs, double size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm(Vec3d(coefs.val));
    plane->Push(-coefs[3] / norm);

    Vec3d p_center;
    plane->GetOrigin(p_center.val);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(PlaneUtils::setSize(p_center, plane->GetOutputPort(), size)->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WPlane::WPlane(const Vec4d& coefs, const Point3d& pt, double size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    Point3d coefs3(coefs[0], coefs[1], coefs[2]);
    double norm_sqr = 1.0 / coefs3.dot(coefs3);
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot(pt) + coefs[3];
    Vec3d p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter(p_center[0], p_center[1], p_center[2]);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(PlaneUtils::setSize(p_center, plane->GetOutputPort(), size)->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WPlane cv::viz::Widget::cast<cv::viz::WPlane>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WPlane&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// sphere widget implementation

cv::viz::WSphere::WSphere(const Point3d &center, double radius, int sphere_resolution, const Color &color)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetRadius(radius);
    sphere->SetCenter(center.x, center.y, center.z);
    sphere->SetPhiResolution(sphere_resolution);
    sphere->SetThetaResolution(sphere_resolution);
    sphere->LatLongTessellationOff();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(sphere->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WSphere cv::viz::Widget::cast<cv::viz::WSphere>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WSphere&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// arrow widget implementation

cv::viz::WArrow::WArrow(const Point3d& pt1, const Point3d& pt2, double thickness, const Color &color)
{
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();
    arrowSource->SetShaftRadius(thickness);
    // The thickness and radius of the tip are adjusted based on the thickness of the arrow
    arrowSource->SetTipRadius(thickness * 3.0);
    arrowSource->SetTipLength(thickness * 10.0);

    RNG rng = theRNG();
    Vec3d arbitrary(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
    Vec3d startPoint(pt1.x, pt1.y, pt1.z), endPoint(pt2.x, pt2.y, pt2.z);

    double length = cv::norm(endPoint - startPoint);

    Vec3d xvec = normalized(endPoint - startPoint);
    Vec3d zvec = normalized(xvec.cross(arbitrary));
    Vec3d yvec = zvec.cross(xvec);

    Affine3d pose = makeTransformToGlobal(xvec, yvec, zvec);

    // Apply the transforms
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Translate(startPoint.val);
    transform->Concatenate(vtkmatrix(pose.matrix));
    transform->Scale(length, length, length);

    // Transform the polydata
    vtkSmartPointer<vtkTransformPolyDataFilter> transformPD = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transformPD->SetTransform(transform);
    transformPD->SetInputConnection(arrowSource->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(transformPD->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WArrow cv::viz::Widget::cast<cv::viz::WArrow>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WArrow&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// circle widget implementation

cv::viz::WCircle::WCircle(const Point3d& pt, double radius, double thickness, const Color& color)
{
    vtkSmartPointer<vtkDiskSource> disk = vtkSmartPointer<vtkDiskSource>::New();
    // Maybe the resolution should be lower e.g. 50 or 25
    disk->SetCircumferentialResolution(50);
    disk->SetInnerRadius(radius - thickness);
    disk->SetOuterRadius(radius + thickness);

    // Set the circle origin
    vtkSmartPointer<vtkTransform> t = vtkSmartPointer<vtkTransform>::New();
    t->Identity();
    t->Translate(pt.x, pt.y, pt.z);

    vtkSmartPointer<vtkTransformPolyDataFilter> tf = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    tf->SetTransform(t);
    tf->SetInputConnection(disk->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(tf->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WCircle cv::viz::Widget::cast<cv::viz::WCircle>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCircle&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

cv::viz::WCylinder::WCylinder(const Point3d& pt_on_axis, const Point3d& axis_direction, double radius, int numsides, const Color &color)
{
    const Point3d pt2 = pt_on_axis + axis_direction;
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(pt_on_axis.x, pt_on_axis.y, pt_on_axis.z);
    line->SetPoint2(pt2.x, pt2.y, pt2.z);

    vtkSmartPointer<vtkTubeFilter> tuber = vtkSmartPointer<vtkTubeFilter>::New();
    tuber->SetInputConnection(line->GetOutputPort());
    tuber->SetRadius(radius);
    tuber->SetNumberOfSides(numsides);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(tuber->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WCylinder cv::viz::Widget::cast<cv::viz::WCylinder>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCylinder&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// cylinder widget implementation

cv::viz::WCube::WCube(const Point3d& pt_min, const Point3d& pt_max, bool wire_frame, const Color &color)
{
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    if (wire_frame)
    {
        vtkSmartPointer<vtkOutlineSource> cube = vtkSmartPointer<vtkOutlineSource>::New();
        cube->SetBounds(pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
        mapper->SetInputConnection(cube->GetOutputPort());
    }
    else
    {
        vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
        cube->SetBounds(pt_min.x, pt_max.x, pt_min.y, pt_max.y, pt_min.z, pt_max.z);
        mapper->SetInputConnection(cube->GetOutputPort());
    }

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WCube cv::viz::Widget::cast<cv::viz::WCube>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCube&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// coordinate system widget implementation

cv::viz::WCoordinateSystem::WCoordinateSystem(double scale)
{
    vtkSmartPointer<vtkAxes> axes = vtkSmartPointer<vtkAxes>::New();
    axes->SetOrigin(0, 0, 0);
    axes->SetScaleFactor(scale);
    axes->Update();

    vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->InsertNextTuple3(255, 0, 0);
    colors->InsertNextTuple3(255, 0, 0);
    colors->InsertNextTuple3(0, 255, 0);
    colors->InsertNextTuple3(0, 255, 0);
    colors->InsertNextTuple3(0, 0, 255);
    colors->InsertNextTuple3(0, 0, 255);

    vtkSmartPointer<vtkPolyData> polydata = axes->GetOutput();
    polydata->GetPointData()->SetScalars(colors);

    vtkSmartPointer<vtkTubeFilter> tube_filter = vtkSmartPointer<vtkTubeFilter>::New();
    VtkUtils::SetInputData(tube_filter, polydata);
    tube_filter->SetRadius(axes->GetScaleFactor() / 50.0);
    tube_filter->SetNumberOfSides(6);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUsePointData();
    VtkUtils::SetInputData(mapper, tube_filter->GetOutput());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WCoordinateSystem cv::viz::Widget::cast<cv::viz::WCoordinateSystem>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCoordinateSystem&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// polyline widget implementation

cv::viz::WPolyLine::WPolyLine(InputArray _points, const Color &color)
{
    CV_Assert(_points.type() == CV_32FC3 || _points.type() == CV_32FC4 || _points.type() == CV_64FC3 || _points.type() == CV_64FC4);

    const float *fpoints = _points.getMat().ptr<float>();
    const double *dpoints = _points.getMat().ptr<double>();
    size_t total = _points.total();
    int s_chs = _points.channels();

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataType(_points.depth() == CV_32F ? VTK_FLOAT : VTK_DOUBLE);
    points->SetNumberOfPoints(total);

    if (_points.depth() == CV_32F)
        for(size_t i = 0; i < total; ++i, fpoints += s_chs)
            points->SetPoint(i, fpoints);

    if (_points.depth() == CV_64F)
        for(size_t i = 0; i < total; ++i, dpoints += s_chs)
            points->SetPoint(i, dpoints);

    vtkSmartPointer<vtkCellArray> cell_array = vtkSmartPointer<vtkCellArray>::New();
    cell_array->Allocate(cell_array->EstimateSize(1, total));
    cell_array->InsertNextCell(total);
    for(size_t i = 0; i < total; ++i)
        cell_array->InsertCellPoint(i);

    vtkSmartPointer<vtkUnsignedCharArray> scalars =  VtkUtils::FillScalars(total, color);

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetLines(cell_array);
    polydata->GetPointData()->SetScalars(scalars);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(polydata->GetProducerPort());
    mapper->SetScalarRange(0, 255);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WPolyLine cv::viz::Widget::cast<cv::viz::WPolyLine>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WPolyLine&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// grid widget implementation

namespace cv { namespace viz { namespace
{
    struct GridUtils
    {
        static vtkSmartPointer<vtkPolyData> createGrid(const Vec2i &dimensions, const Vec2f &spacing)
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
#if VTK_MAJOR_VERSION <= 5
            filter->SetInputConnection(grid->GetProducerPort());
#else
            filter->SetInputData(grid);
#endif
            filter->Update();
            return filter->GetOutput();
        }
    };
}}}

cv::viz::WGrid::WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color)
{
    vtkSmartPointer<vtkPolyData> grid = GridUtils::createGrid(dimensions, spacing);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInputConnection(grid->GetProducerPort());
#else
    mapper->SetInputData(grid);
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WGrid::WGrid(const Vec4d &coefs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color)
{
    vtkSmartPointer<vtkPolyData> grid = GridUtils::createGrid(dimensions, spacing);

    // Estimate the transform to set the normal based on the coefficients
    Vec3d normal(coefs[0], coefs[1], coefs[2]);
    Vec3d up_vector(0.0, 1.0, 0.0); // Just set as default
    double push_distance = -coefs[3]/cv::norm(Vec3d(coefs.val));
    Vec3d n = normalize(normal);
    Vec3d u = normalize(up_vector.cross(n));
    Vec3d v = n.cross(u);

    Affine3d pose = makeTransformToGlobal(u, v, n, n * push_distance);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(vtkmatrix(pose.matrix));

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
#if VTK_MAJOR_VERSION <= 5
    transform_filter->SetInputConnection(grid->GetProducerPort());
#else
    transform_filter->SetInputData(grid);
#endif
    transform_filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(transform_filter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WGrid cv::viz::Widget::cast<cv::viz::WGrid>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WGrid&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text3D widget implementation

cv::viz::WText3D::WText3D(const String &text, const Point3d &position, double text_scale, bool face_camera, const Color &color)
{
    vtkSmartPointer<vtkVectorText> textSource = vtkSmartPointer<vtkVectorText>::New();
    textSource->SetText(text.c_str());
    textSource->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(textSource->GetOutputPort());

    if (face_camera)
    {
        vtkSmartPointer<vtkFollower> actor = vtkSmartPointer<vtkFollower>::New();
        actor->SetMapper(mapper);
        actor->SetPosition(position.x, position.y, position.z);
        actor->SetScale(text_scale);
        WidgetAccessor::setProp(*this, actor);
    }
    else
    {
        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->SetPosition(position.x, position.y, position.z);
        actor->SetScale(text_scale);
        WidgetAccessor::setProp(*this, actor);
    }

    setColor(color);
}

void cv::viz::WText3D::setText(const String &text)
{
    vtkFollower *actor = vtkFollower::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support text." && actor);

    // Update text source
    vtkPolyDataMapper *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    vtkVectorText * textSource = vtkVectorText::SafeDownCast(mapper->GetInputConnection(0,0)->GetProducer());
    CV_Assert("This widget does not support text." && textSource);

    textSource->SetText(text.c_str());
    textSource->Update();
}

cv::String cv::viz::WText3D::getText() const
{
    vtkFollower *actor = vtkFollower::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support text." && actor);

    vtkPolyDataMapper *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    vtkVectorText * textSource = vtkVectorText::SafeDownCast(mapper->GetInputConnection(0,0)->GetProducer());
    CV_Assert("This widget does not support text." && textSource);

    return textSource->GetText();
}

template<> cv::viz::WText3D cv::viz::Widget::cast<cv::viz::WText3D>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WText3D&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// text widget implementation

cv::viz::WText::WText(const String &text, const Point2i &pos, int font_size, const Color &color)
{
    vtkSmartPointer<vtkTextActor> actor = vtkSmartPointer<vtkTextActor>::New();
    actor->SetPosition(pos.x, pos.y);
    actor->SetInput(text.c_str());

    vtkSmartPointer<vtkTextProperty> tprop = actor->GetTextProperty();
    tprop->SetFontSize(font_size);
    tprop->SetFontFamilyToArial();
    tprop->SetJustificationToLeft();
    tprop->BoldOn();

    Color c = vtkcolor(color);
    tprop->SetColor(c.val);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WText cv::viz::Widget::cast<cv::viz::WText>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<WText&>(widget);
}

void cv::viz::WText::setText(const String &text)
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support text." && actor);
    actor->SetInput(text.c_str());
}

cv::String cv::viz::WText::getText() const
{
    vtkTextActor *actor = vtkTextActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support text." && actor);
    return actor->GetInput();
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// image overlay widget implementation

cv::viz::WImageOverlay::WImageOverlay(const Mat &image, const Rect &rect)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    ConvertToVtkImage::convert(image, vtk_image);

    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
    flipFilter->SetInputData(vtk_image);
#endif
    flipFilter->Update();

    // Scale the image based on the Rect
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Scale(double(image.cols)/rect.width,double(image.rows)/rect.height,1.0);

    vtkSmartPointer<vtkImageReslice> image_reslice = vtkSmartPointer<vtkImageReslice>::New();
    image_reslice->SetResliceTransform(transform);
    image_reslice->SetInputConnection(flipFilter->GetOutputPort());
    image_reslice->SetOutputDimensionality(2);
    image_reslice->InterpolateOn();
    image_reslice->AutoCropOutputOn();

    vtkSmartPointer<vtkImageMapper> imageMapper = vtkSmartPointer<vtkImageMapper>::New();
    imageMapper->SetInputConnection(image_reslice->GetOutputPort());
    imageMapper->SetColorWindow(255); // OpenCV color
    imageMapper->SetColorLevel(127.5);

    vtkSmartPointer<vtkActor2D> actor = vtkSmartPointer<vtkActor2D>::New();
    actor->SetMapper(imageMapper);
    actor->SetPosition(rect.x, rect.y);

    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::WImageOverlay::setImage(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    vtkActor2D *actor = vtkActor2D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support overlay image." && actor);

    vtkImageMapper *mapper = vtkImageMapper::SafeDownCast(actor->GetMapper());
    CV_Assert("This widget does not support overlay image." && mapper);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    ConvertToVtkImage::convert(image, vtk_image);

    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
    flipFilter->SetInputData(vtk_image);
#endif
    flipFilter->Update();

    mapper->SetInputConnection(flipFilter->GetOutputPort());
}

template<> cv::viz::WImageOverlay cv::viz::Widget::cast<cv::viz::WImageOverlay>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<WImageOverlay&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// image 3D widget implementation

cv::viz::WImage3D::WImage3D(const Mat &image, const Size &size)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    ConvertToVtkImage::convert(image, vtk_image);

    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
    flipFilter->SetInputData(vtk_image);
#endif
    flipFilter->Update();

    Vec3d plane_center(size.width * 0.5, size.height * 0.5, 0.0);

    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    plane->SetCenter(plane_center[0], plane_center[1], plane_center[2]);
    plane->SetNormal(0.0, 0.0, 1.0);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->Translate(plane_center[0], plane_center[1], plane_center[2]);
    transform->Scale(size.width, size.height, 1.0);
    transform->Translate(-plane_center[0], -plane_center[1], -plane_center[2]);

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
    transform_filter->SetInputConnection(plane->GetOutputPort());
    transform_filter->Update();

    // Apply the texture
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(flipFilter->GetOutputPort());

    vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
    texturePlane->SetInputConnection(transform_filter->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    planeMapper->SetInputConnection(texturePlane->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(planeMapper);
    actor->SetTexture(texture);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WImage3D::WImage3D(const Vec3d &position, const Vec3d &normal, const Vec3d &up_vector, const Mat &image, const Size &size)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    ConvertToVtkImage::convert(image, vtk_image);

    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
    flipFilter->SetInputData(vtk_image);
#endif
    flipFilter->Update();

    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    plane->SetCenter(0.0, 0.0, 0.0);
    plane->SetNormal(0.0, 0.0, 1.0);

    // Compute the transformation matrix for drawing the camera frame in a scene
    Vec3d n = normalize(normal);
    Vec3d u = normalize(up_vector.cross(n));
    Vec3d v = n.cross(u);

    Affine3d pose = makeTransformToGlobal(u, v, n, position);

    // Apply the texture
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(flipFilter->GetOutputPort());

    vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
    texturePlane->SetInputConnection(plane->GetOutputPort());

    // Apply the transform after texture mapping
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(vtkmatrix(pose.matrix));
    transform->Scale(size.width, size.height, 1.0);

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
    transform_filter->SetInputConnection(texturePlane->GetOutputPort());
    transform_filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    planeMapper->SetInputConnection(transform_filter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(planeMapper);
    actor->SetTexture(texture);

    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::WImage3D::setImage(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support 3D image." && actor);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
    ConvertToVtkImage::convert(image, vtk_image);

    // Need to flip the image as the coordinates are different in OpenCV and VTK
    vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
    flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
    flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
    flipFilter->SetInputData(vtk_image);
#endif
    flipFilter->Update();

    // Apply the texture
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(flipFilter->GetOutputPort());

    actor->SetTexture(texture);
}

template<> cv::viz::WImage3D cv::viz::Widget::cast<cv::viz::WImage3D>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WImage3D&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// camera position widget implementation

namespace  cv  { namespace viz { namespace
{
    struct CameraPositionUtils
    {
        static void projectImage(double fovy, double far_end_height, const Mat &image,
                                 double scale, const Color &color, vtkSmartPointer<vtkActor> actor)
        {
            // Create a camera
            vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
            float aspect_ratio = float(image.cols)/float(image.rows);

            // Create the vtk image
            vtkSmartPointer<vtkImageData> vtk_image = vtkSmartPointer<vtkImageData>::New();
            ConvertToVtkImage::convert(image, vtk_image);

            // Adjust a pixel of the vtk_image
            vtk_image->SetScalarComponentFromDouble(0, image.rows-1, 0, 0, color[2]);
            vtk_image->SetScalarComponentFromDouble(0, image.rows-1, 0, 1, color[1]);
            vtk_image->SetScalarComponentFromDouble(0, image.rows-1, 0, 2, color[0]);

            // Need to flip the image as the coordinates are different in OpenCV and VTK
            vtkSmartPointer<vtkImageFlip> flipFilter = vtkSmartPointer<vtkImageFlip>::New();
            flipFilter->SetFilteredAxis(1); // Vertical flip
#if VTK_MAJOR_VERSION <= 5
            flipFilter->SetInputConnection(vtk_image->GetProducerPort());
#else
            flipFilter->SetInputData(vtk_image);
#endif
            flipFilter->Update();

            Vec3d plane_center(0.0, 0.0, scale);

            vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
            plane->SetCenter(plane_center[0], plane_center[1], plane_center[2]);
            plane->SetNormal(0.0, 0.0, 1.0);

            vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
            transform->PreMultiply();
            transform->Translate(plane_center[0], plane_center[1], plane_center[2]);
            transform->Scale(far_end_height*aspect_ratio, far_end_height, 1.0);
            transform->RotateY(180.0);
            transform->Translate(-plane_center[0], -plane_center[1], -plane_center[2]);

            // Apply the texture
            vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
            texture->SetInputConnection(flipFilter->GetOutputPort());

            vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
            texturePlane->SetInputConnection(plane->GetOutputPort());

            vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
            transform_filter->SetTransform(transform);
            transform_filter->SetInputConnection(texturePlane->GetOutputPort());
            transform_filter->Update();

            // Create frustum
            camera->SetViewAngle(fovy);
            camera->SetPosition(0.0, 0.0, 0.0);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetFocalPoint(0.0, 0.0, 1.0);
            camera->SetClippingRange(0.01, scale);

            double planesArray[24];
            camera->GetFrustumPlanes(aspect_ratio, planesArray);

            vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
            planes->SetFrustumPlanes(planesArray);

            vtkSmartPointer<vtkFrustumSource> frustumSource =
            vtkSmartPointer<vtkFrustumSource>::New();
            frustumSource->SetPlanes(planes);
            frustumSource->Update();

            vtkSmartPointer<vtkExtractEdges> filter = vtkSmartPointer<vtkExtractEdges>::New();
            filter->SetInputConnection(frustumSource->GetOutputPort());
            filter->Update();

            // Frustum needs to be textured or else it can't be combined with image
            vtkSmartPointer<vtkTextureMapToPlane> frustum_texture = vtkSmartPointer<vtkTextureMapToPlane>::New();
            frustum_texture->SetInputConnection(filter->GetOutputPort());
            // Texture mapping with only one pixel from the image to have constant color
            frustum_texture->SetSRange(0.0, 0.0);
            frustum_texture->SetTRange(0.0, 0.0);

            vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
            appendFilter->AddInputConnection(frustum_texture->GetOutputPort());
            appendFilter->AddInputConnection(transform_filter->GetOutputPort());

            vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            planeMapper->SetInputConnection(appendFilter->GetOutputPort());

            actor->SetMapper(planeMapper);
            actor->SetTexture(texture);
        }

        static vtkSmartPointer<vtkPolyData> createFrustrum(double aspect_ratio, double fovy, double scale)
        {
            vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
            camera->SetViewAngle(fovy);
            camera->SetPosition(0.0, 0.0, 0.0);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetFocalPoint(0.0, 0.0, 1.0);
            camera->SetClippingRange(0.01, scale);

            double planesArray[24];
            camera->GetFrustumPlanes(aspect_ratio, planesArray);

            vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
            planes->SetFrustumPlanes(planesArray);

            vtkSmartPointer<vtkFrustumSource> frustumSource = vtkSmartPointer<vtkFrustumSource>::New();
            frustumSource->SetPlanes(planes);

            vtkSmartPointer<vtkExtractEdges> extract_edges = vtkSmartPointer<vtkExtractEdges>::New();
            extract_edges->SetInputConnection(frustumSource->GetOutputPort());
            extract_edges->Update();

            return extract_edges->GetOutput();
        }

    };
}}}

cv::viz::WCameraPosition::WCameraPosition(double scale)
{
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();

    VtkUtils::SetInputData(mapper, getPolyData(WCoordinateSystem(scale)));
    mapper->SetScalarModeToUsePointData();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCameraPosition::WCameraPosition(const Matx33d &K, double scale, const Color &color)
{
    double f_x = K(0,0), f_y = K(1,1), c_y = K(1,2);

    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    double fovy = 2.0 * atan2(c_y, f_y) * 180 / CV_PI;
    double aspect_ratio = f_y / f_x;

    vtkSmartPointer<vtkPolyData> polydata = CameraPositionUtils::createFrustrum(aspect_ratio, fovy, scale);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(polydata->GetProducerPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WCameraPosition::WCameraPosition(const Vec2d &fov, double scale, const Color &color)
{
    double aspect_ratio = tan(fov[0] * 0.5) / tan(fov[1] * 0.5);
    double fovy = fov[1] * 180 / CV_PI;

    vtkSmartPointer<vtkPolyData> polydata = CameraPositionUtils::createFrustrum(aspect_ratio, fovy, scale);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(polydata->GetProducerPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WCameraPosition::WCameraPosition(const Matx33d &K, const Mat &image, double scale, const Color &color)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    double f_y = K(1,1), c_y = K(1,2);

    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    double fovy = 2.0 * atan2(c_y, f_y) * 180.0 / CV_PI;
    double far_end_height = 2.00 * c_y * scale / f_y;

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    CameraPositionUtils::projectImage(fovy, far_end_height, image, scale, color, actor);
    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCameraPosition::WCameraPosition(const Vec2d &fov, const Mat &image, double scale, const Color &color)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    double fovy = fov[1] * 180.0 / CV_PI;
    double far_end_height = 2.0 * scale * tan(fov[1] * 0.5);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    CameraPositionUtils::projectImage(fovy, far_end_height, image, scale, color, actor);
    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WCameraPosition cv::viz::Widget::cast<cv::viz::WCameraPosition>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCameraPosition&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// trajectory widget implementation

namespace cv { namespace viz { namespace
{
    struct TrajectoryUtils
    {
        static void applyPath(vtkSmartPointer<vtkPolyData> poly_data, vtkSmartPointer<vtkAppendPolyData> append_filter, const std::vector<Affine3d> &path)
        {
            vtkIdType nr_points = path.size();

            for (vtkIdType i = 0; i < nr_points; ++i)
            {
                vtkSmartPointer<vtkPolyData> new_data = vtkSmartPointer<vtkPolyData>::New();
                new_data->DeepCopy(poly_data);

                // Transform the default coordinate frame
                vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
                transform->PreMultiply();
                vtkSmartPointer<vtkMatrix4x4> mat_trans = vtkSmartPointer<vtkMatrix4x4>::New();
                mat_trans = vtkmatrix(path[i].matrix);
                transform->SetMatrix(mat_trans);

                vtkSmartPointer<vtkTransformPolyDataFilter> filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
#if VTK_MAJOR_VERSION <= 5
                filter->SetInput(new_data);
#else
                filter->SetInputData(new_data);
#endif
                filter->SetTransform(transform);
                filter->Update();

                append_filter->AddInputConnection(filter->GetOutputPort());
            }
        }
    };
}}}

cv::viz::WTrajectory::WTrajectory(InputArray _path, int display_mode, double scale, const Color &color)
{
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

    // Bitwise and with 3 in order to limit the domain to 2 bits
    if (display_mode & WTrajectory::PATH)
    {
        Mat points = vtkTrajectorySource::ExtractPoints(_path);
        vtkSmartPointer<vtkPolyData> polydata = getPolyData(WPolyLine(points, color));
        appendFilter->AddInputConnection(polydata->GetProducerPort());
    }

    vtkSmartPointer<vtkTensorGlyph> tensor_glyph;
    if (display_mode & WTrajectory::FRAMES)
    {
        vtkSmartPointer<vtkTrajectorySource> source = vtkSmartPointer<vtkTrajectorySource>::New();
        source->SetTrajectory(_path);

        vtkSmartPointer<vtkPolyData> glyph = getPolyData(WCoordinateSystem(scale));

        tensor_glyph = vtkSmartPointer<vtkTensorGlyph>::New();
        tensor_glyph->SetInputConnection(source->GetOutputPort());
        tensor_glyph->SetSourceConnection(glyph->GetProducerPort());
        tensor_glyph->ExtractEigenvaluesOff();  // Treat as a rotation matrix, not as something with eigenvalues
        tensor_glyph->ThreeGlyphsOff();
        tensor_glyph->SymmetricOff();
        tensor_glyph->ColorGlyphsOff();

        appendFilter->AddInputConnection(tensor_glyph->GetOutputPort());
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, appendFilter->GetOutput());
    mapper->SetScalarModeToUsePointData();
    mapper->SetScalarRange(0, 255);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WTrajectory cv::viz::Widget::cast<cv::viz::WTrajectory>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WTrajectory&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// WTrajectoryFrustums widget implementation

cv::viz::WTrajectoryFrustums::WTrajectoryFrustums(const std::vector<Affine3d> &path, const Matx33d &K, double scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    double f_x = K(0,0);
    double f_y = K(1,1);
    double c_y = K(1,2);
    double aspect_ratio = f_y / f_x;
    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    double fovy = 2.0 * atan2(c_y, f_y) * 180 / CV_PI;

    camera->SetViewAngle(fovy);
    camera->SetPosition(0.0, 0.0, 0.0);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetFocalPoint(0.0, 0.0, 1.0);
    camera->SetClippingRange(0.01, scale);

    double planesArray[24];
    camera->GetFrustumPlanes(aspect_ratio, planesArray);

    vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
    planes->SetFrustumPlanes(planesArray);

    vtkSmartPointer<vtkFrustumSource> frustumSource = vtkSmartPointer<vtkFrustumSource>::New();
    frustumSource->SetPlanes(planes);
    frustumSource->Update();

    // Extract the edges
    vtkSmartPointer<vtkExtractEdges> filter = vtkSmartPointer<vtkExtractEdges>::New();
    filter->SetInputConnection(frustumSource->GetOutputPort());
    filter->Update();

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    TrajectoryUtils::applyPath(filter->GetOutput(), appendFilter, path);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WTrajectoryFrustums::WTrajectoryFrustums(const std::vector<Affine3d> &path, const Vec2d &fov, double scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();

    camera->SetViewAngle(fov[1] * 180 / CV_PI); // Vertical field of view
    camera->SetPosition(0.0, 0.0, 0.0);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetFocalPoint(0.0, 0.0, 1.0);
    camera->SetClippingRange(0.01, scale);

    double aspect_ratio = tan(fov[0] * 0.5) / tan(fov[1] * 0.5);

    double planesArray[24];
    camera->GetFrustumPlanes(aspect_ratio, planesArray);

    vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
    planes->SetFrustumPlanes(planesArray);

    vtkSmartPointer<vtkFrustumSource> frustumSource = vtkSmartPointer<vtkFrustumSource>::New();
    frustumSource->SetPlanes(planes);
    frustumSource->Update();

    // Extract the edges
    vtkSmartPointer<vtkExtractEdges> filter = vtkSmartPointer<vtkExtractEdges>::New();
    filter->SetInputConnection(frustumSource->GetOutputPort());
    filter->Update();

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    TrajectoryUtils::applyPath(filter->GetOutput(), appendFilter, path);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WTrajectoryFrustums cv::viz::Widget::cast<cv::viz::WTrajectoryFrustums>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WTrajectoryFrustums&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// WTrajectorySpheres widget implementation

cv::viz::WTrajectorySpheres::WTrajectorySpheres(const std::vector<Affine3d> &path, double line_length, double init_sphere_radius, double sphere_radius,
                                                          const Color &line_color, const Color &sphere_color)
{
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    vtkIdType nr_poses = path.size();

    // Create color arrays
    vtkSmartPointer<vtkUnsignedCharArray> line_scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    line_scalars->SetNumberOfComponents(3);
    line_scalars->InsertNextTuple3(line_color[2], line_color[1], line_color[0]);

    // Create color array for sphere
    vtkSmartPointer<vtkSphereSource> dummy_sphere = vtkSmartPointer<vtkSphereSource>::New();
    // Create the array for big sphere
    dummy_sphere->SetRadius(init_sphere_radius);
    dummy_sphere->Update();
    vtkIdType nr_points = dummy_sphere->GetOutput()->GetNumberOfCells();
    vtkSmartPointer<vtkUnsignedCharArray> sphere_scalars_init = VtkUtils::FillScalars(nr_points, sphere_color);

    // Create the array for small sphere
    dummy_sphere->SetRadius(sphere_radius);
    dummy_sphere->Update();
    nr_points = dummy_sphere->GetOutput()->GetNumberOfCells();
    vtkSmartPointer<vtkUnsignedCharArray> sphere_scalars = VtkUtils::FillScalars(nr_points, sphere_color);


    for (vtkIdType i = 0; i < nr_poses; ++i)
    {
        Point3f new_pos = path[i].translation();

        vtkSmartPointer<vtkSphereSource> sphere_source = vtkSmartPointer<vtkSphereSource>::New();
        sphere_source->SetCenter(new_pos.x, new_pos.y, new_pos.z);
        if (i == 0)
        {
            sphere_source->SetRadius(init_sphere_radius);
            sphere_source->Update();
            sphere_source->GetOutput()->GetCellData()->SetScalars(sphere_scalars_init);
            appendFilter->AddInputConnection(sphere_source->GetOutputPort());
            continue;
        }
        else
        {
            sphere_source->SetRadius(sphere_radius);
            sphere_source->Update();
            sphere_source->GetOutput()->GetCellData()->SetScalars(sphere_scalars);
            appendFilter->AddInputConnection(sphere_source->GetOutputPort());
        }


        Affine3d relativeAffine = path[i].inv() * path[i-1];
        Vec3d v = path[i].rotation() * relativeAffine.translation();
        v = normalize(v) * line_length;

        vtkSmartPointer<vtkLineSource> line_source = vtkSmartPointer<vtkLineSource>::New();
        line_source->SetPoint1(new_pos.x + v[0], new_pos.y + v[1], new_pos.z + v[2]);
        line_source->SetPoint2(new_pos.x, new_pos.y, new_pos.z);
        line_source->Update();
        line_source->GetOutput()->GetCellData()->SetScalars(line_scalars);

        appendFilter->AddInputConnection(line_source->GetOutputPort());
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUseCellData();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WTrajectorySpheres cv::viz::Widget::cast<cv::viz::WTrajectorySpheres>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WTrajectorySpheres&>(widget);
}
