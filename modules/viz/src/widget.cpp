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
/// widget implementation

class cv::viz::Widget::Impl
{
public:
    vtkSmartPointer<vtkProp> prop;
    Impl() : prop(0) {}
};

cv::viz::Widget::Widget() : impl_( new Impl() ) { }

cv::viz::Widget::Widget(const Widget& other) : impl_( new Impl() )
{
    if (other.impl_ && other.impl_->prop)
        impl_->prop = other.impl_->prop;
}

cv::viz::Widget& cv::viz::Widget::operator=(const Widget& other)
{
    if (!impl_)
        impl_ = new Impl();

    if (other.impl_)
        impl_->prop = other.impl_->prop;
    return *this;
}

cv::viz::Widget::~Widget()
{
    if (impl_)
    {
        delete impl_;
        impl_ = 0;
    }
}

cv::viz::Widget cv::viz::Widget::fromPlyFile(const String &file_name)
{
    CV_Assert(vtkPLYReader::CanReadFile(file_name.c_str()));

    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(file_name.c_str());

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection( reader->GetOutputPort() );
#if VTK_MAJOR_VERSION < 8
    mapper->ImmediateModeRenderingOff();
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();
    actor->SetMapper(mapper);

    Widget widget;
    WidgetAccessor::setProp(widget, actor);
    return widget;
}

void cv::viz::Widget::setRenderingProperty(int property, double value)
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);

    switch (property)
    {
        case POINT_SIZE:          actor->GetProperty()->SetPointSize(float(value)); break;
        case OPACITY:             actor->GetProperty()->SetOpacity(value);          break;
        case LINE_WIDTH:          actor->GetProperty()->SetLineWidth(float(value)); break;
#if VTK_MAJOR_VERSION < 8
        case IMMEDIATE_RENDERING: actor->GetMapper()->SetImmediateModeRendering(int(value)); break;
#else
        case IMMEDIATE_RENDERING: std::cerr << "this property has no effect" << std::endl; break;
#endif
        case AMBIENT:             actor->GetProperty()->SetAmbient(float(value)); break;
        case LIGHTING:
        {
            if (value == 0)
                actor->GetProperty()->LightingOff();
            else
                actor->GetProperty()->LightingOn();
            break;
        }
        case FONT_SIZE:
        {
            vtkTextActor* text_actor = vtkTextActor::SafeDownCast(actor);
            CV_Assert("Widget does not have text content." && text_actor);
            text_actor->GetTextProperty()->SetFontSize(int(value));
            break;
        }
        case REPRESENTATION:
        {
            switch (int(value))
            {
                case REPRESENTATION_POINTS:    actor->GetProperty()->SetRepresentationToPoints(); break;
                case REPRESENTATION_WIREFRAME: actor->GetProperty()->SetRepresentationToWireframe(); break;
                case REPRESENTATION_SURFACE:   actor->GetProperty()->SetRepresentationToSurface();  break;
            }
            break;
        }
        case SHADING:
        {
            switch (int(value))
            {
                case SHADING_FLAT: actor->GetProperty()->SetInterpolationToFlat(); break;
                case SHADING_GOURAUD:
                {
                    if (!actor->GetMapper()->GetInput()->GetPointData()->GetNormals())
                    {
                        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
                        CV_Assert("Can't set shading property for such type of widget" && mapper);

                        vtkSmartPointer<vtkPolyData> with_normals = VtkUtils::ComputeNormals(mapper->GetInput());
                        VtkUtils::SetInputData(mapper, with_normals);
                    }
                    actor->GetProperty()->SetInterpolationToGouraud();
                    break;
                }
                case SHADING_PHONG:
                {
                    if (!actor->GetMapper()->GetInput()->GetPointData()->GetNormals())
                    {
                        vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
                        CV_Assert("Can't set shading property for such type of widget" && mapper);

                        vtkSmartPointer<vtkPolyData> with_normals = VtkUtils::ComputeNormals(mapper->GetInput());
                        VtkUtils::SetInputData(mapper, with_normals);
                    }
                    actor->GetProperty()->SetInterpolationToPhong();
                    break;
                }
            }
            break;
        }
        default:
            CV_Assert("setRenderingProperty: Unknown property");
    }
    actor->Modified();
}

double cv::viz::Widget::getRenderingProperty(int property) const
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);

    double value = 0.0;
    switch (property)
    {
        case POINT_SIZE: value = actor->GetProperty()->GetPointSize(); break;
        case OPACITY:    value = actor->GetProperty()->GetOpacity();   break;
        case LINE_WIDTH: value = actor->GetProperty()->GetLineWidth(); break;
#if VTK_MAJOR_VERSION < 8
        case IMMEDIATE_RENDERING:  value = actor->GetMapper()->GetImmediateModeRendering();  break;
#else
        case IMMEDIATE_RENDERING: std::cerr << "this property has no effect" << std::endl; break;
#endif
        case AMBIENT: value = actor->GetProperty()->GetAmbient(); break;
        case LIGHTING: value = actor->GetProperty()->GetLighting(); break;
        case FONT_SIZE:
        {
            vtkTextActor* text_actor = vtkTextActor::SafeDownCast(actor);
            CV_Assert("Widget does not have text content." && text_actor);
            value = text_actor->GetTextProperty()->GetFontSize();;
            break;
        }
        case REPRESENTATION:
        {
            switch (actor->GetProperty()->GetRepresentation())
            {
                case VTK_POINTS:    value = REPRESENTATION_POINTS; break;
                case VTK_WIREFRAME: value = REPRESENTATION_WIREFRAME; break;
                case VTK_SURFACE:   value = REPRESENTATION_SURFACE; break;
            }
            break;
        }
        case SHADING:
        {
            switch (actor->GetProperty()->GetInterpolation())
            {
                case VTK_FLAT:      value = SHADING_FLAT; break;
                case VTK_GOURAUD:   value = SHADING_GOURAUD; break;
                case VTK_PHONG:     value = SHADING_PHONG; break;
            }
            break;
        }
        default:
            CV_Assert("getRenderingProperty: Unknown property");
    }
    return value;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget accessor implementation

vtkSmartPointer<vtkProp> cv::viz::WidgetAccessor::getProp(const Widget& widget)
{
    return widget.impl_->prop;
}

void cv::viz::WidgetAccessor::setProp(Widget& widget, vtkSmartPointer<vtkProp> prop)
{
    widget.impl_->prop = prop;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget3D implementation

void cv::viz::Widget3D::setPose(const Affine3d &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = vtkmatrix(pose.matrix);
    actor->SetUserMatrix(matrix);
    actor->Modified();
}

void cv::viz::Widget3D::updatePose(const Affine3d &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    if (!matrix)
    {
        setPose(pose);
        return;
    }

    Affine3d updated_pose = pose * Affine3d(*matrix->Element);
    matrix = vtkmatrix(updated_pose.matrix);

    actor->SetUserMatrix(matrix);
    actor->Modified();
}

cv::Affine3d cv::viz::Widget3D::getPose() const
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);
    if (!actor->GetUserMatrix())
    {
        return Affine3d(); // empty user matrix, return an identity transform.
    }
    return Affine3d(*actor->GetUserMatrix()->Element);
}

void cv::viz::Widget3D::applyTransform(const Affine3d &transform)
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D actor." && actor);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    CV_Assert("Widget doesn't have a polydata mapper" && mapper);

    VtkUtils::SetInputData(mapper, VtkUtils::TransformPolydata(mapper->GetInput(), transform));
    mapper->Update();
}

void cv::viz::Widget3D::setColor(const Color &color)
{
    // Cast to actor instead of prop3d since prop3d doesn't provide getproperty
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);

    Color c = vtkcolor(color);
    actor->GetMapper()->ScalarVisibilityOff();
    actor->GetProperty()->SetColor(c.val);
    actor->GetProperty()->SetEdgeColor(c.val);
    actor->Modified();
}

template<> cv::viz::Widget3D cv::viz::Widget::cast<cv::viz::Widget3D>()
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget cannot be cast." && actor);

    Widget3D widget;
    WidgetAccessor::setProp(widget, actor);
    return widget;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget2D implementation

void cv::viz::Widget2D::setColor(const Color &color)
{
    vtkActor2D *actor = vtkActor2D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);
    Color c = vtkcolor(color);
    actor->GetProperty()->SetColor(c.val);
    actor->Modified();
}

template<> cv::viz::Widget2D cv::viz::Widget::cast<cv::viz::Widget2D>()
{
    vtkActor2D *actor = vtkActor2D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget cannot be cast." && actor);

    Widget2D widget;
    WidgetAccessor::setProp(widget, actor);
    return widget;
}
