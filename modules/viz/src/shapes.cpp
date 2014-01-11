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

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
cv::viz::WLine::WLine(const Point3d &pt1, const Point3d &pt2, const Color &color)
{
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(pt1.x, pt1.y, pt1.z);
    line->SetPoint2(pt2.x, pt2.y, pt2.z);
    line->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, line->GetOutput());

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
/// sphere widget implementation

cv::viz::WSphere::WSphere(const Point3d &center, double radius, int sphere_resolution, const Color &color)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetRadius(radius);
    sphere->SetCenter(center.x, center.y, center.z);
    sphere->SetPhiResolution(sphere_resolution);
    sphere->SetThetaResolution(sphere_resolution);
    sphere->LatLongTessellationOff();
    sphere->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, sphere->GetOutput());

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
/// arrow widget implementation

cv::viz::WArrow::WArrow(const Point3d& pt1, const Point3d& pt2, double thickness, const Color &color)
{
    vtkSmartPointer<vtkArrowSource> arrow_source = vtkSmartPointer<vtkArrowSource>::New();
    arrow_source->SetShaftRadius(thickness);
    arrow_source->SetTipRadius(thickness * 3.0);
    arrow_source->SetTipLength(thickness * 10.0);

    RNG rng = theRNG();
    Vec3d arbitrary(rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0));
    Vec3d startPoint(pt1.x, pt1.y, pt1.z), endPoint(pt2.x, pt2.y, pt2.z);

    double length = norm(endPoint - startPoint);

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
    transformPD->SetInputConnection(arrow_source->GetOutputPort());
    transformPD->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, transformPD->GetOutput());

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
    tf->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, tf->GetOutput());

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
    tuber->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, tuber->GetOutput());

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

cv::viz::WCube::WCube(const Point3d& min_point, const Point3d& max_point, bool wire_frame, const Color &color)
{
    vtkSmartPointer<vtkPolyDataAlgorithm> cube;
    if (wire_frame)
    {
        cube = vtkSmartPointer<vtkOutlineSource>::New();
        vtkOutlineSource::SafeDownCast(cube)->SetBounds(min_point.x, max_point.x, min_point.y, max_point.y, min_point.z, max_point.z);
    }
    else
    {
        cube = vtkSmartPointer<vtkCubeSource>::New();
        vtkCubeSource::SafeDownCast(cube)->SetBounds(min_point.x, max_point.x, min_point.y, max_point.y, min_point.z, max_point.z);
    }
    cube->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, cube->GetOutput());

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
    tube_filter->SetInputConnection(polydata->GetProducerPort());
    tube_filter->SetRadius(axes->GetScaleFactor() / 50.0);
    tube_filter->SetNumberOfSides(6);
    tube_filter->Update();

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
    VtkUtils::SetInputData(mapper, polydata);
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
            filter->SetInputConnection(grid->GetProducerPort());
            filter->Update();
            return filter->GetOutput();
        }
    };
}}}

cv::viz::WGrid::WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color)
{
    vtkSmartPointer<vtkPolyData> grid = GridUtils::createGrid(dimensions, spacing);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, grid);

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
    transform_filter->SetInputConnection(grid->GetProducerPort());
    transform_filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, transform_filter->GetOutput());

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
    textSource->Modified();
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
    vtkSmartPointer<vtkImageMatSource> source = vtkSmartPointer<vtkImageMatSource>::New();
    source->SetImage(image);

    // Scale the image based on the Rect, and flip to match y-ais orientation
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Scale(image.cols/(double)rect.width, image.rows/(double)rect.height, 1.0);
    transform->RotateX(180);

    vtkSmartPointer<vtkImageReslice> image_reslice = vtkSmartPointer<vtkImageReslice>::New();
    image_reslice->SetResliceTransform(transform);
    image_reslice->SetInputConnection(source->GetOutputPort());
    image_reslice->SetOutputDimensionality(2);
    image_reslice->InterpolateOn();
    image_reslice->AutoCropOutputOn();
    image_reslice->Update();

    vtkSmartPointer<vtkImageMapper> image_mapper = vtkSmartPointer<vtkImageMapper>::New();
    image_mapper->SetInputConnection(image_reslice->GetOutputPort());
    image_mapper->SetColorWindow(255); // OpenCV color
    image_mapper->SetColorLevel(127.5);

    vtkSmartPointer<vtkActor2D> actor = vtkSmartPointer<vtkActor2D>::New();
    actor->SetMapper(image_mapper);
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
    \
    Vec6i extent;
    mapper->GetInput()->GetExtent(extent.val);
    Size size(extent[1], extent[3]);

    // Create the vtk image and set its parameters based on input image
    vtkSmartPointer<vtkImageMatSource> source = vtkSmartPointer<vtkImageMatSource>::New();
    source->SetImage(image);

    // Scale the image based on the Rect, and flip to match y-ais orientation
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->Scale(image.cols/(double)size.width, image.rows/(double)size.height, 1.0);
    transform->RotateX(180);

    vtkSmartPointer<vtkImageReslice> image_reslice = vtkSmartPointer<vtkImageReslice>::New();
    image_reslice->SetResliceTransform(transform);
    image_reslice->SetInputConnection(source->GetOutputPort());
    image_reslice->SetOutputDimensionality(2);
    image_reslice->InterpolateOn();
    image_reslice->AutoCropOutputOn();
    image_reslice->Update();

    mapper->SetInputConnection(image_reslice->GetOutputPort());
}

template<> cv::viz::WImageOverlay cv::viz::Widget::cast<cv::viz::WImageOverlay>()
{
    Widget2D widget = this->cast<Widget2D>();
    return static_cast<WImageOverlay&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// image 3D widget implementation

cv::viz::WImage3D::WImage3D(const Mat &image, const Size2d &size)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    vtkSmartPointer<vtkImageMatSource> source = vtkSmartPointer<vtkImageMatSource>::New();
    source->SetImage(image);

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(source->GetOutputPort());

    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    plane->SetOrigin(-0.5 * size.width, -0.5 * size.height, 0.0);
    plane->SetPoint1( 0.5 * size.width, -0.5 * size.height, 0.0);
    plane->SetPoint2(-0.5 * size.width,  0.5 * size.height, 0.0);

    vtkSmartPointer<vtkTextureMapToPlane> textured_plane = vtkSmartPointer<vtkTextureMapToPlane>::New();
    textured_plane->SetInputConnection(plane->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(textured_plane->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->SetTexture(texture);
    actor->GetProperty()->ShadingOff();
    actor->GetProperty()->LightingOff();

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WImage3D::WImage3D(const Mat &image, const Size2d &size, const Vec3d &center, const Vec3d &normal, const Vec3d &up_vector)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    // Compute the transformation matrix for drawing the camera frame in a scene
    Vec3d n = normalize(normal);
    Vec3d u = normalize(up_vector.cross(n));
    Vec3d v = n.cross(u);
    Affine3d pose = makeTransformToGlobal(u, v, n, center);

    WImage3D image3d(image, size);
    image3d.applyTransform(pose);
    *this = image3d;
}

void cv::viz::WImage3D::setImage(const Mat &image)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);

    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("This widget does not support 3D image." && actor);

    vtkSmartPointer<vtkImageMatSource> source = vtkSmartPointer<vtkImageMatSource>::New();
    source->SetImage(image);

    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(source->GetOutputPort());

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
        static vtkSmartPointer<vtkPolyData> createFrustum(double aspect_ratio, double fovy, double scale)
        {
            vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
            camera->SetViewAngle(fovy);
            camera->SetPosition(0.0, 0.0, 0.0);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetFocalPoint(0.0, 0.0, 1.0);
            camera->SetClippingRange(1e-9, scale);

            double planes_array[24];
            camera->GetFrustumPlanes(aspect_ratio, planes_array);

            vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
            planes->SetFrustumPlanes(planes_array);

            vtkSmartPointer<vtkFrustumSource> frustumSource = vtkSmartPointer<vtkFrustumSource>::New();
            frustumSource->SetPlanes(planes);

            vtkSmartPointer<vtkExtractEdges> extract_edges = vtkSmartPointer<vtkExtractEdges>::New();
            extract_edges->SetInputConnection(frustumSource->GetOutputPort());
            extract_edges->Update();

            return extract_edges->GetOutput();
        }

        static Mat ensureColorImage(InputArray image)
        {
            Mat color(image.size(), CV_8UC3);
            if (image.channels() == 1)
            {
                Vec3b *drow = color.ptr<Vec3b>();
                for(int y = 0; y < color.rows; ++y)
                {
                    const unsigned char *srow = image.getMat().ptr<unsigned char>(y);
                    const unsigned char *send = srow + color.cols;
                    for(;srow < send;)
                        *drow++ = Vec3b::all(*srow++);
                }
            }
            else
                image.copyTo(color);
            return color;
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

    vtkSmartPointer<vtkPolyData> polydata = CameraPositionUtils::createFrustum(aspect_ratio, fovy, scale);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WCameraPosition::WCameraPosition(const Vec2d &fov, double scale, const Color &color)
{
    double aspect_ratio = tan(fov[0] * 0.5) / tan(fov[1] * 0.5);
    double fovy = fov[1] * 180 / CV_PI;

    vtkSmartPointer<vtkPolyData> polydata = CameraPositionUtils::createFrustum(aspect_ratio, fovy, scale);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WCameraPosition::WCameraPosition(const Matx33d &K, InputArray _image, double scale, const Color &color)
{
    CV_Assert(!_image.empty() && _image.depth() == CV_8U);
    Mat image = CameraPositionUtils::ensureColorImage(_image);
    image.at<Vec3b>(0, 0) = Vec3d(color.val); //workaround of VTK limitation

    double f_y = K(1,1), c_y = K(1,2);
    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    double fovy = 2.0 * atan2(c_y, f_y) * 180.0 / CV_PI;
    double far_end_height = 2.00 * c_y * scale / f_y;
    double aspect_ratio = image.cols/(double)image.rows;
    double image_scale = far_end_height/image.rows;

    WImage3D image_widget(image, Size2d(image.size()) * image_scale);
    image_widget.applyTransform(Affine3d().translate(Vec3d(0, 0, scale)));
    vtkSmartPointer<vtkPolyData> plane = getPolyData(image_widget);

    vtkSmartPointer<vtkPolyData> frustum = CameraPositionUtils::createFrustum(aspect_ratio, fovy, scale);

    // Frustum needs to be textured or else it can't be combined with image
    vtkSmartPointer<vtkTextureMapToPlane> frustum_texture = vtkSmartPointer<vtkTextureMapToPlane>::New();
    frustum_texture->SetInputConnection(frustum->GetProducerPort());
    frustum_texture->SetSRange(0.0, 0.0); // Texture mapping with only one pixel
    frustum_texture->SetTRange(0.0, 0.0); // from the image to have constant color

    vtkSmartPointer<vtkAppendPolyData> append_filter = vtkSmartPointer<vtkAppendPolyData>::New();
    append_filter->AddInputConnection(frustum_texture->GetOutputPort());
    append_filter->AddInputConnection(plane->GetProducerPort());

    vtkSmartPointer<vtkActor> actor = getActor(image_widget);
    actor->GetMapper()->SetInputConnection(append_filter->GetOutputPort());
    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCameraPosition::WCameraPosition(const Vec2d &fov, InputArray _image, double scale, const Color &color)
{
    CV_Assert(!_image.empty() && _image.depth() == CV_8U);
    Mat image = CameraPositionUtils::ensureColorImage(_image);
    image.at<Vec3b>(0, 0) = Vec3d(color.val); //workaround of VTK limitation

    double fovy = fov[1] * 180.0 / CV_PI;
    double far_end_height = 2.0 * scale * tan(fov[1] * 0.5);
    double aspect_ratio = image.cols/(double)image.rows;
    double image_scale = far_end_height/image.rows;

    WImage3D image_widget(image, Size2d(image.size()) * image_scale);
    image_widget.applyTransform(Affine3d().translate(Vec3d(0, 0, scale)));
    vtkSmartPointer<vtkPolyData> plane = getPolyData(image_widget);

    vtkSmartPointer<vtkPolyData> frustum = CameraPositionUtils::createFrustum(aspect_ratio, fovy, scale);

    // Frustum needs to be textured or else it can't be combined with image
    vtkSmartPointer<vtkTextureMapToPlane> frustum_texture = vtkSmartPointer<vtkTextureMapToPlane>::New();
    frustum_texture->SetInputConnection(frustum->GetProducerPort());
    frustum_texture->SetSRange(0.0, 0.0); // Texture mapping with only one pixel
    frustum_texture->SetTRange(0.0, 0.0); // from the image to have constant color

    vtkSmartPointer<vtkAppendPolyData> append_filter = vtkSmartPointer<vtkAppendPolyData>::New();
    append_filter->AddInputConnection(frustum_texture->GetOutputPort());
    append_filter->AddInputConnection(plane->GetProducerPort());

    vtkSmartPointer<vtkActor> actor = getActor(image_widget);
    actor->GetMapper()->SetInputConnection(append_filter->GetOutputPort());
    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WCameraPosition cv::viz::Widget::cast<cv::viz::WCameraPosition>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCameraPosition&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// trajectory widget implementation

cv::viz::WTrajectory::WTrajectory(InputArray _path, int display_mode, double scale, const Color &color)
{
    vtkSmartPointer<vtkAppendPolyData> append_filter = vtkSmartPointer<vtkAppendPolyData>::New();

    // Bitwise and with 3 in order to limit the domain to 2 bits
    if (display_mode & WTrajectory::PATH)
    {
        Mat points = vtkTrajectorySource::ExtractPoints(_path);
        vtkSmartPointer<vtkPolyData> polydata = getPolyData(WPolyLine(points, color));
        append_filter->AddInputConnection(polydata->GetProducerPort());
    }

    if (display_mode & WTrajectory::FRAMES)
    {
        vtkSmartPointer<vtkTrajectorySource> source = vtkSmartPointer<vtkTrajectorySource>::New();
        source->SetTrajectory(_path);

        vtkSmartPointer<vtkPolyData> glyph = getPolyData(WCoordinateSystem(scale));

        vtkSmartPointer<vtkTensorGlyph> tensor_glyph = vtkSmartPointer<vtkTensorGlyph>::New();
        tensor_glyph->SetInputConnection(source->GetOutputPort());
        tensor_glyph->SetSourceConnection(glyph->GetProducerPort());
        tensor_glyph->ExtractEigenvaluesOff();  // Treat as a rotation matrix, not as something with eigenvalues
        tensor_glyph->ThreeGlyphsOff();
        tensor_glyph->SymmetricOff();
        tensor_glyph->ColorGlyphsOff();

        append_filter->AddInputConnection(tensor_glyph->GetOutputPort());
    }
    append_filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, append_filter->GetOutput());
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

cv::viz::WTrajectoryFrustums::WTrajectoryFrustums(InputArray _path, const Matx33d &K, double scale, const Color &color)
{
    vtkSmartPointer<vtkTrajectorySource> source = vtkSmartPointer<vtkTrajectorySource>::New();
    source->SetTrajectory(_path);

    vtkSmartPointer<vtkPolyData> glyph = getPolyData(WCameraPosition(K, scale));

    vtkSmartPointer<vtkTensorGlyph> tensor_glyph = vtkSmartPointer<vtkTensorGlyph>::New();
    tensor_glyph->SetInputConnection(source->GetOutputPort());
    tensor_glyph->SetSourceConnection(glyph->GetProducerPort());
    tensor_glyph->ExtractEigenvaluesOff();  // Treat as a rotation matrix, not as something with eigenvalues
    tensor_glyph->ThreeGlyphsOff();
    tensor_glyph->SymmetricOff();
    tensor_glyph->ColorGlyphsOff();
    tensor_glyph->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, tensor_glyph->GetOutput());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WTrajectoryFrustums::WTrajectoryFrustums(InputArray _path, const Vec2d &fov, double scale, const Color &color)
{
    vtkSmartPointer<vtkTrajectorySource> source = vtkSmartPointer<vtkTrajectorySource>::New();
    source->SetTrajectory(_path);

    vtkSmartPointer<vtkPolyData> glyph = getPolyData(WCameraPosition(fov, scale));

    vtkSmartPointer<vtkTensorGlyph> tensor_glyph = vtkSmartPointer<vtkTensorGlyph>::New();
    tensor_glyph->SetInputConnection(source->GetOutputPort());
    tensor_glyph->SetSourceConnection(glyph->GetProducerPort());
    tensor_glyph->ExtractEigenvaluesOff();  // Treat as a rotation matrix, not as something with eigenvalues
    tensor_glyph->ThreeGlyphsOff();
    tensor_glyph->SymmetricOff();
    tensor_glyph->ColorGlyphsOff();
    tensor_glyph->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, tensor_glyph->GetOutput());

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

cv::viz::WTrajectorySpheres::WTrajectorySpheres(InputArray _path, double line_length, double radius, const Color &from, const Color &to)
{
    CV_Assert(_path.kind() == _InputArray::STD_VECTOR || _path.kind() == _InputArray::MAT);
    CV_Assert(_path.type() == CV_32FC(16) || _path.type() == CV_64FC(16));

    Mat path64;
    _path.getMat().convertTo(path64, CV_64F);
    Affine3d *traj = path64.ptr<Affine3d>();
    size_t total = path64.total();

    vtkSmartPointer<vtkAppendPolyData> append_filter = vtkSmartPointer<vtkAppendPolyData>::New();

    for(size_t i = 0; i < total; ++i)
    {
        Vec3d curr = traj[i].translation();

        vtkSmartPointer<vtkSphereSource> sphere_source = vtkSmartPointer<vtkSphereSource>::New();
        sphere_source->SetCenter(curr.val);
        sphere_source->SetRadius( (i == 0) ? 2 * radius : radius );
        sphere_source->Update();

        double alpha = static_cast<double>(i)/total;
        Color c = from * (1 - alpha) + to * alpha;

        vtkSmartPointer<vtkPolyData> polydata = sphere_source->GetOutput();
        polydata->GetCellData()->SetScalars(VtkUtils::FillScalars(polydata->GetNumberOfCells(), c));
        append_filter->AddInputConnection(polydata->GetProducerPort());

        if (i > 0)
        {
            Vec3d prev = traj[i-1].translation();
            Vec3d lvec = prev - curr;

            if(norm(lvec) > line_length)
                lvec = normalize(lvec) * line_length;

            Vec3d lend = curr + lvec;

            vtkSmartPointer<vtkLineSource> line_source = vtkSmartPointer<vtkLineSource>::New();
            line_source->SetPoint1(curr.val);
            line_source->SetPoint2(lend.val);
            line_source->Update();
            vtkSmartPointer<vtkPolyData> polydata = line_source->GetOutput();
            polydata->GetCellData()->SetScalars(VtkUtils::FillScalars(polydata->GetNumberOfCells(), c));
            append_filter->AddInputConnection(polydata->GetProducerPort());
        }
    }
    append_filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUseCellData();
    VtkUtils::SetInputData(mapper, append_filter->GetOutput());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WTrajectorySpheres cv::viz::Widget::cast<cv::viz::WTrajectorySpheres>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WTrajectorySpheres&>(widget);
}
