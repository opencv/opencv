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

namespace cv
{
    namespace viz
    {
        template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// line widget implementation
cv::viz::WLine::WLine(const Point3f &pt1, const Point3f &pt2, const Color &color)
{
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(pt1.x, pt1.y, pt1.z);
    line->SetPoint2(pt2.x, pt2.y, pt2.z);
    line->Update();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(line->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

struct cv::viz::WPlane::SetSizeImpl
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

cv::viz::WPlane::WPlane(const Vec4f& coefs, float size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);
    double norm = cv::norm(Vec3f(coefs.val));
    plane->Push(-coefs[3] / norm);

    Vec3d p_center;
    plane->GetOrigin(p_center.val);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(SetSizeImpl::setSize(p_center, plane->GetOutputPort(), size)->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WPlane::WPlane(const Vec4f& coefs, const Point3f& pt, float size, const Color &color)
{
    vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
    Point3f coefs3(coefs[0], coefs[1], coefs[2]);
    double norm_sqr = 1.0 / coefs3.dot(coefs3);
    plane->SetNormal(coefs[0], coefs[1], coefs[2]);

    double t = coefs3.dot(pt) + coefs[3];
    Vec3f p_center = pt - coefs3 * t * norm_sqr;
    plane->SetCenter(p_center[0], p_center[1], p_center[2]);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(SetSizeImpl::setSize(p_center, plane->GetOutputPort(), size)->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WSphere::WSphere(const Point3f &center, float radius, int sphere_resolution, const Color &color)
{
    vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetRadius(radius);
    sphere->SetCenter(center.x, center.y, center.z);
    sphere->SetPhiResolution(sphere_resolution);
    sphere->SetThetaResolution(sphere_resolution);
    sphere->LatLongTessellationOff();
    sphere->Update();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(sphere->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WArrow::WArrow(const Point3f& pt1, const Point3f& pt2, float thickness, const Color &color)
{
    vtkSmartPointer<vtkArrowSource> arrowSource = vtkSmartPointer<vtkArrowSource>::New();
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

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(transformPD->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WCircle::WCircle(const Point3f& pt, float radius, float thickness, const Color& color)
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

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(tf->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WCylinder::WCylinder(const Point3f& pt_on_axis, const Point3f& axis_direction, float radius, int numsides, const Color &color)
{
    const Point3f pt2 = pt_on_axis + axis_direction;
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    line->SetPoint1(pt_on_axis.x, pt_on_axis.y, pt_on_axis.z);
    line->SetPoint2(pt2.x, pt2.y, pt2.z);

    vtkSmartPointer<vtkTubeFilter> tuber = vtkSmartPointer<vtkTubeFilter>::New();
    tuber->SetInputConnection(line->GetOutputPort());
    tuber->SetRadius(radius);
    tuber->SetNumberOfSides(numsides);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(tuber->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WCube::WCube(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame, const Color &color)
{
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
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

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

cv::viz::WCoordinateSystem::WCoordinateSystem(float scale)
{
    vtkSmartPointer<vtkAxes> axes = vtkSmartPointer<vtkAxes>::New();
    axes->SetOrigin(0, 0, 0);
    axes->SetScaleFactor(scale);

    vtkSmartPointer<vtkFloatArray> axes_colors = vtkSmartPointer<vtkFloatArray>::New();
    axes_colors->Allocate(6);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(1.0);
    axes_colors->InsertNextValue(1.0);

    vtkSmartPointer<vtkPolyData> axes_data = axes->GetOutput();
#if VTK_MAJOR_VERSION <= 5
    axes_data->Update();
#else
    axes->Update();
#endif
    axes_data->GetPointData()->SetScalars(axes_colors);

    vtkSmartPointer<vtkTubeFilter> axes_tubes = vtkSmartPointer<vtkTubeFilter>::New();
#if VTK_MAJOR_VERSION <= 5
    axes_tubes->SetInput(axes_data);
#else
    axes_tubes->SetInputData(axes_data);
#endif
    axes_tubes->SetRadius(axes->GetScaleFactor() / 50.0);
    axes_tubes->SetNumberOfSides(6);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->SetInputConnection(axes_tubes->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

struct cv::viz::WPolyLine::CopyImpl
{
    template<typename _Tp>
    static void copy(const Mat& source, Vec<_Tp, 3> *output, vtkSmartPointer<vtkPolyLine> polyLine)
    {
        int s_chs = source.channels();

        for (int y = 0, id = 0; y < source.rows; ++y)
        {
            const _Tp* srow = source.ptr<_Tp>(y);

            for (int x = 0; x < source.cols; ++x, srow += s_chs, ++id)
            {
                *output++ = Vec<_Tp, 3>(srow);
                polyLine->GetPointIds()->SetId(id,id);
            }
        }
    }
};

cv::viz::WPolyLine::WPolyLine(InputArray _pointData, const Color &color)
{
    Mat pointData = _pointData.getMat();
    CV_Assert(pointData.type() == CV_32FC3 || pointData.type() == CV_32FC4 || pointData.type() == CV_64FC3 || pointData.type() == CV_64FC4);
    vtkIdType nr_points = pointData.total();

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

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
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polyData);
#else
    mapper->SetInputData(polyData);
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WPolyLine cv::viz::Widget::cast<cv::viz::WPolyLine>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WPolyLine&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// grid widget implementation

struct cv::viz::WGrid::GridImpl
{
    static vtkSmartPointer<vtkPolyData> createGrid(const Vec2i &dimensions, const Vec2d &spacing)
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

cv::viz::WGrid::WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color)
{
    vtkSmartPointer<vtkPolyData> grid = GridImpl::createGrid(dimensions, spacing);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
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

cv::viz::WGrid::WGrid(const Vec4f &coefs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color)
{
    vtkSmartPointer<vtkPolyData> grid = GridImpl::createGrid(dimensions, spacing);

    // Estimate the transform to set the normal based on the coefficients
    Vec3f normal(coefs[0], coefs[1], coefs[2]);
    Vec3f up_vector(0.0f, 1.0f, 0.0f); // Just set as default
    double push_distance = -coefs[3]/cv::norm(Vec3f(coefs.val));
    Vec3f u,v,n;
    n = normalize(normal);
    u = normalize(up_vector.cross(n));
    v = n.cross(u);

    vtkSmartPointer<vtkMatrix4x4> mat_trans = vtkSmartPointer<vtkMatrix4x4>::New();
    mat_trans->SetElement(0,0,u[0]);
    mat_trans->SetElement(0,1,u[1]);
    mat_trans->SetElement(0,2,u[2]);
    mat_trans->SetElement(1,0,v[0]);
    mat_trans->SetElement(1,1,v[1]);
    mat_trans->SetElement(1,2,v[2]);
    mat_trans->SetElement(2,0,n[0]);
    mat_trans->SetElement(2,1,n[1]);
    mat_trans->SetElement(2,2,n[2]);
    // Inverse rotation (orthogonal, so just take transpose)
    mat_trans->Transpose();
    mat_trans->SetElement(0,3,n[0] * push_distance);
    mat_trans->SetElement(1,3,n[1] * push_distance);
    mat_trans->SetElement(2,3,n[2] * push_distance);
    mat_trans->SetElement(3,3,1);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(mat_trans);

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
#if VTK_MAJOR_VERSION <= 5
    transform_filter->SetInputConnection(grid->GetProducerPort());
#else
    transform_filter->SetInputData(grid);
#endif
    transform_filter->Update();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
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

cv::viz::WText3D::WText3D(const String &text, const Point3f &position, float text_scale, bool face_camera, const Color &color)
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

cv::viz::WImage3D::WImage3D(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size)
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
    Vec3f u,v,n;
    n = normalize(normal);
    u = normalize(up_vector.cross(n));
    v = n.cross(u);

    vtkSmartPointer<vtkMatrix4x4> mat_trans = vtkSmartPointer<vtkMatrix4x4>::New();
    mat_trans->SetElement(0,0,u[0]);
    mat_trans->SetElement(0,1,u[1]);
    mat_trans->SetElement(0,2,u[2]);
    mat_trans->SetElement(1,0,v[0]);
    mat_trans->SetElement(1,1,v[1]);
    mat_trans->SetElement(1,2,v[2]);
    mat_trans->SetElement(2,0,n[0]);
    mat_trans->SetElement(2,1,n[1]);
    mat_trans->SetElement(2,2,n[2]);
    // Inverse rotation (orthogonal, so just take transpose)
    mat_trans->Transpose();
    // Then translate the coordinate frame to camera position
    mat_trans->SetElement(0,3,position[0]);
    mat_trans->SetElement(1,3,position[1]);
    mat_trans->SetElement(2,3,position[2]);
    mat_trans->SetElement(3,3,1);

    // Apply the texture
    vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
    texture->SetInputConnection(flipFilter->GetOutputPort());

    vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
    texturePlane->SetInputConnection(plane->GetOutputPort());

    // Apply the transform after texture mapping
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(mat_trans);
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

struct cv::viz::WCameraPosition::ProjectImage
{
    static void projectImage(float fovy, float far_end_height, const Mat &image,
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
        camera->SetPosition(0.0,0.0,0.0);
        camera->SetViewUp(0.0,1.0,0.0);
        camera->SetFocalPoint(0.0,0.0,1.0);
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
};

cv::viz::WCameraPosition::WCameraPosition(float scale)
{
    vtkSmartPointer<vtkAxes> axes = vtkSmartPointer<vtkAxes>::New();
    axes->SetOrigin(0, 0, 0);
    axes->SetScaleFactor(scale);

    vtkSmartPointer<vtkFloatArray> axes_colors = vtkSmartPointer<vtkFloatArray>::New();
    axes_colors->Allocate(6);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.0);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(0.5);
    axes_colors->InsertNextValue(1.0);
    axes_colors->InsertNextValue(1.0);

    vtkSmartPointer<vtkPolyData> axes_data = axes->GetOutput();
#if VTK_MAJOR_VERSION <= 5
    axes_data->Update();
#else
    axes->Update();
#endif
    axes_data->GetPointData()->SetScalars(axes_colors);

    vtkSmartPointer<vtkTubeFilter> axes_tubes = vtkSmartPointer<vtkTubeFilter>::New();
#if VTK_MAJOR_VERSION <= 5
    axes_tubes->SetInput(axes_data);
#else
    axes_tubes->SetInputData(axes_data);
#endif
    axes_tubes->SetRadius(axes->GetScaleFactor() / 50.0);
    axes_tubes->SetNumberOfSides(6);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->SetInputConnection(axes_tubes->GetOutputPort());

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCameraPosition::WCameraPosition(const Matx33f &K, float scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    float f_x = K(0,0);
    float f_y = K(1,1);
    float c_y = K(1,2);
    float aspect_ratio = f_y / f_x;
    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    float fovy = 2.0f * atan2(c_y,f_y) * 180 / CV_PI;

    camera->SetViewAngle(fovy);
    camera->SetPosition(0.0,0.0,0.0);
    camera->SetViewUp(0.0,1.0,0.0);
    camera->SetFocalPoint(0.0,0.0,1.0);
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

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(filter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}


cv::viz::WCameraPosition::WCameraPosition(const Vec2f &fov, float scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();

    camera->SetViewAngle(fov[1] * 180 / CV_PI); // Vertical field of view
    camera->SetPosition(0.0,0.0,0.0);
    camera->SetViewUp(0.0,1.0,0.0);
    camera->SetFocalPoint(0.0,0.0,1.0);
    camera->SetClippingRange(0.01, scale);

    double aspect_ratio = tan(fov[0] * 0.5) / tan(fov[1] * 0.5);

    double planesArray[24];
    camera->GetFrustumPlanes(aspect_ratio, planesArray);

    vtkSmartPointer<vtkPlanes> planes = vtkSmartPointer<vtkPlanes>::New();
    planes->SetFrustumPlanes(planesArray);

    vtkSmartPointer<vtkFrustumSource> frustumSource =
    vtkSmartPointer<vtkFrustumSource>::New();
    frustumSource->SetPlanes(planes);
    frustumSource->Update();

    // Extract the edges so we have the grid
    vtkSmartPointer<vtkExtractEdges> filter = vtkSmartPointer<vtkExtractEdges>::New();
    filter->SetInputConnection(frustumSource->GetOutputPort());
    filter->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(filter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WCameraPosition::WCameraPosition(const Matx33f &K, const Mat &image, float scale, const Color &color)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    float f_y = K(1,1);
    float c_y = K(1,2);
    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    float fovy = 2.0f * atan2(c_y,f_y) * 180.0f / CV_PI;
    float far_end_height = 2.0f * c_y * scale / f_y;

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    ProjectImage::projectImage(fovy, far_end_height, image, scale, color, actor);
    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCameraPosition::WCameraPosition(const Vec2f &fov, const Mat &image, float scale, const Color &color)
{
    CV_Assert(!image.empty() && image.depth() == CV_8U);
    float fovy = fov[1] * 180.0f / CV_PI;
    float far_end_height = 2.0 * scale * tan(fov[1] * 0.5);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    ProjectImage::projectImage(fovy, far_end_height, image, scale, color, actor);
    WidgetAccessor::setProp(*this, actor);
}

template<> cv::viz::WCameraPosition cv::viz::Widget::cast<cv::viz::WCameraPosition>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCameraPosition&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// trajectory widget implementation

struct cv::viz::WTrajectory::ApplyPath
{
    static void applyPath(vtkSmartPointer<vtkPolyData> poly_data, vtkSmartPointer<vtkAppendPolyData> append_filter, const std::vector<Affine3f> &path)
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
            mat_trans = convertToVtkMatrix(path[i].matrix);
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

cv::viz::WTrajectory::WTrajectory(const std::vector<Affine3f> &path, int display_mode, const Color &color, float scale)
{
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

    // Bitwise and with 3 in order to limit the domain to 2 bits
    if ((~display_mode & 3) ^ WTrajectory::DISPLAY_PATH)
    {
        // Create a poly line along the path
        vtkIdType nr_points = path.size();

        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();

        points->SetDataTypeToFloat();
        points->SetNumberOfPoints(nr_points);
        polyLine->GetPointIds()->SetNumberOfIds(nr_points);

        Vec3f *data_beg = vtkpoints_data<float>(points);

        for (vtkIdType i = 0; i < nr_points; ++i)
        {
            Vec3f cam_pose = path[i].translation();
            *data_beg++ = cam_pose;
            polyLine->GetPointIds()->SetId(i,i);
        }

        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
        cells->InsertNextCell(polyLine);

        polyData->SetPoints(points);
        polyData->SetLines(cells);

        // Set the color for polyData
        vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetNumberOfTuples(nr_points);
        colors->FillComponent(0, color[2]);
        colors->FillComponent(1, color[1]);
        colors->FillComponent(2, color[0]);

        polyData->GetPointData()->SetScalars(colors);
#if VTK_MAJOR_VERSION <= 5
        appendFilter->AddInputConnection(polyData->GetProducerPort());
#else
        appendFilter->AddInputData(polyData);
#endif
    }

    if ((~display_mode & 3) ^ WTrajectory::DISPLAY_FRAMES)
    {
        // Create frames and transform along the path
        vtkSmartPointer<vtkAxes> axes = vtkSmartPointer<vtkAxes>::New();
        axes->SetOrigin(0, 0, 0);
        axes->SetScaleFactor(scale);

        vtkSmartPointer<vtkUnsignedCharArray> axes_colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        axes_colors->SetNumberOfComponents(3);
        axes_colors->InsertNextTuple3(255,0,0);
        axes_colors->InsertNextTuple3(255,0,0);
        axes_colors->InsertNextTuple3(0,255,0);
        axes_colors->InsertNextTuple3(0,255,0);
        axes_colors->InsertNextTuple3(0,0,255);
        axes_colors->InsertNextTuple3(0,0,255);

        vtkSmartPointer<vtkPolyData> axes_data = axes->GetOutput();
#if VTK_MAJOR_VERSION <= 5
        axes_data->Update();
#else
        axes->Update();
#endif
        axes_data->GetPointData()->SetScalars(axes_colors);

        vtkSmartPointer<vtkTubeFilter> axes_tubes = vtkSmartPointer<vtkTubeFilter>::New();
#if VTK_MAJOR_VERSION <= 5
        axes_tubes->SetInput(axes_data);
#else
        axes_tubes->SetInputData(axes_data);
#endif
        axes_tubes->SetRadius(axes->GetScaleFactor() / 50.0);
        axes_tubes->SetNumberOfSides(6);
        axes_tubes->Update();

        ApplyPath::applyPath(axes_tubes->GetOutput(), appendFilter, path);
    }

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WTrajectory::WTrajectory(const std::vector<Affine3f> &path, const Matx33f &K, float scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
    float f_x = K(0,0);
    float f_y = K(1,1);
    float c_y = K(1,2);
    float aspect_ratio = f_y / f_x;
    // Assuming that this is an ideal camera (c_y and c_x are at the center of the image)
    float fovy = 2.0f * atan2(c_y,f_y) * 180 / CV_PI;

    camera->SetViewAngle(fovy);
    camera->SetPosition(0.0,0.0,0.0);
    camera->SetViewUp(0.0,1.0,0.0);
    camera->SetFocalPoint(0.0,0.0,1.0);
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
    ApplyPath::applyPath(filter->GetOutput(), appendFilter, path);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

cv::viz::WTrajectory::WTrajectory(const std::vector<Affine3f> &path, const Vec2f &fov, float scale, const Color &color)
{
    vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();

    camera->SetViewAngle(fov[1] * 180 / CV_PI); // Vertical field of view
    camera->SetPosition(0.0,0.0,0.0);
    camera->SetViewUp(0.0,1.0,0.0);
    camera->SetFocalPoint(0.0,0.0,1.0);
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
    ApplyPath::applyPath(filter->GetOutput(), appendFilter, path);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WTrajectory cv::viz::Widget::cast<cv::viz::WTrajectory>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WTrajectory&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// spheres trajectory widget implementation

cv::viz::WSpheresTrajectory::WSpheresTrajectory(const std::vector<Affine3f> &path, float line_length, float init_sphere_radius, float sphere_radius,
                                                          const Color &line_color, const Color &sphere_color)
{
    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    vtkIdType nr_poses = path.size();

    // Create color arrays
    vtkSmartPointer<vtkUnsignedCharArray> line_scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    line_scalars->SetNumberOfComponents(3);
    line_scalars->InsertNextTuple3(line_color[2], line_color[1], line_color[0]);

    // Create color array for sphere
    vtkSphereSource * dummy_sphere = vtkSphereSource::New();
    // Create the array for big sphere
    dummy_sphere->SetRadius(init_sphere_radius);
    dummy_sphere->Update();
    vtkIdType nr_points = dummy_sphere->GetOutput()->GetNumberOfCells();
    vtkSmartPointer<vtkUnsignedCharArray> sphere_scalars_init = vtkSmartPointer<vtkUnsignedCharArray>::New();
    sphere_scalars_init->SetNumberOfComponents(3);
    sphere_scalars_init->SetNumberOfTuples(nr_points);
    sphere_scalars_init->FillComponent(0, sphere_color[2]);
    sphere_scalars_init->FillComponent(1, sphere_color[1]);
    sphere_scalars_init->FillComponent(2, sphere_color[0]);
    // Create the array for small sphere
    dummy_sphere->SetRadius(sphere_radius);
    dummy_sphere->Update();
    nr_points = dummy_sphere->GetOutput()->GetNumberOfCells();
    vtkSmartPointer<vtkUnsignedCharArray> sphere_scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    sphere_scalars->SetNumberOfComponents(3);
    sphere_scalars->SetNumberOfTuples(nr_points);
    sphere_scalars->FillComponent(0, sphere_color[2]);
    sphere_scalars->FillComponent(1, sphere_color[1]);
    sphere_scalars->FillComponent(2, sphere_color[0]);
    dummy_sphere->Delete();

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


        Affine3f relativeAffine = path[i].inv() * path[i-1];
        Vec3f v = path[i].rotation() * relativeAffine.translation();
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

template<> cv::viz::WSpheresTrajectory cv::viz::Widget::cast<cv::viz::WSpheresTrajectory>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WSpheresTrajectory&>(widget);
}
