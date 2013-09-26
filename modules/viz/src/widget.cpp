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
    if (other.impl_ && other.impl_->prop) impl_->prop = other.impl_->prop;
}

cv::viz::Widget& cv::viz::Widget::operator=(const Widget& other)
{
    if (!impl_) impl_ = new Impl();
    if (other.impl_) impl_->prop = other.impl_->prop;
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
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(file_name.c_str());

    vtkSmartPointer<vtkDataSet> data = reader->GetOutput();
    CV_Assert("File does not exist or file format is not supported." && data);

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(data);
#else
    mapper->SetInputData(data);
#endif

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
    mapper->ImmediateModeRenderingOff();

    actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, data->GetNumberOfPoints() / 10)));
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();

    actor->SetMapper(mapper);

    Widget widget;
    widget.impl_->prop = actor;
    return widget;
}

void cv::viz::Widget::setRenderingProperty(int property, double value)
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);

    switch (property)
    {
        case POINT_SIZE:
        {
            actor->GetProperty()->SetPointSize(float(value));
            actor->Modified();
            break;
        }
        case OPACITY:
        {
            actor->GetProperty()->SetOpacity(value);
            actor->Modified();
            break;
        }
        case IMMEDIATE_RENDERING:
        {
            actor->GetMapper()->SetImmediateModeRendering(int(value));
            actor->Modified();
            break;
        }
        case LINE_WIDTH:
        {
            actor->GetProperty()->SetLineWidth(float(value));
            actor->Modified();
            break;
        }
        case FONT_SIZE:
        {
            vtkTextActor* text_actor = vtkTextActor::SafeDownCast(actor);
            CV_Assert("Widget does not have text content." && text_actor);
            vtkSmartPointer<vtkTextProperty> tprop = text_actor->GetTextProperty();
            tprop->SetFontSize(int(value));
            text_actor->Modified();
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
            actor->Modified();
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
                        vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
#if VTK_MAJOR_VERSION <= 5
                        normals->SetInput(actor->GetMapper()->GetInput());
#else
                        normals->SetInputData(actor->GetMapper()->GetInput());
#endif
                        normals->Update();
                        vtkDataSetMapper::SafeDownCast(actor->GetMapper())->SetInputConnection(normals->GetOutputPort());
                    }
                    actor->GetProperty()->SetInterpolationToGouraud();
                    break;
                }
                case SHADING_PHONG:
                {
                    if (!actor->GetMapper()->GetInput()->GetPointData()->GetNormals())
                    {
                        vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
#if VTK_MAJOR_VERSION <= 5
                        normals->SetInput(actor->GetMapper()->GetInput());
#else
                        normals->SetInputData(actor->GetMapper()->GetInput());
#endif
                        normals->Update();
                        vtkDataSetMapper::SafeDownCast(actor->GetMapper())->SetInputConnection(normals->GetOutputPort());
                    }
                    actor->GetProperty()->SetInterpolationToPhong();
                    break;
                }
            }
            actor->Modified();
            break;
        }


        default:
            CV_Assert("setPointCloudRenderingProperties: Unknown property");
    }
}

double cv::viz::Widget::getRenderingProperty(int property) const
{
    vtkActor *actor = vtkActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget type is not supported." && actor);

    double value = 0.0;
    switch (property)
    {
        case POINT_SIZE:
        {
            value = actor->GetProperty()->GetPointSize();
            break;
        }
        case OPACITY:
        {
            value = actor->GetProperty()->GetOpacity();
            break;
        }
        case IMMEDIATE_RENDERING:
        {
            value = actor->GetMapper()->GetImmediateModeRendering();
            break;
        }
        case LINE_WIDTH:
        {
            value = actor->GetProperty()->GetLineWidth();
            break;
        }
        case FONT_SIZE:
        {
            vtkTextActor* text_actor = vtkTextActor::SafeDownCast(actor);
            CV_Assert("Widget does not have text content." && text_actor);
            vtkSmartPointer<vtkTextProperty> tprop = text_actor->GetTextProperty();
            value = tprop->GetFontSize();
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
            CV_Assert("getPointCloudRenderingProperties: Unknown property");
    }
    return value;
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// widget accessor implementaion

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

struct cv::viz::Widget3D::MatrixConverter
{
    static Matx44f convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
    {
        Matx44f m;
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                m(i, k) = vtk_matrix->GetElement(i, k);
        return m;
    }

    static vtkSmartPointer<vtkMatrix4x4> convertToVtkMatrix(const Matx44f& m)
    {
        vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
                vtk_matrix->SetElement(i, k, m(i, k));
        return vtk_matrix;
    }
};

void cv::viz::Widget3D::setPose(const Affine3f &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = convertToVtkMatrix(pose.matrix);
    actor->SetUserMatrix(matrix);
    actor->Modified();
}

void cv::viz::Widget3D::updatePose(const Affine3f &pose)
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    if (!matrix)
    {
        setPose(pose);
        return ;
    }
    Matx44f matrix_cv = MatrixConverter::convertToMatx(matrix);

    Affine3f updated_pose = pose * Affine3f(matrix_cv);
    matrix = MatrixConverter::convertToVtkMatrix(updated_pose.matrix);

    actor->SetUserMatrix(matrix);
    actor->Modified();
}

cv::Affine3f cv::viz::Widget3D::getPose() const
{
    vtkProp3D *actor = vtkProp3D::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Widget is not 3D." && actor);

    vtkSmartPointer<vtkMatrix4x4> matrix = actor->GetUserMatrix();
    Matx44f matrix_cv = MatrixConverter::convertToMatx(matrix);
    return Affine3f(matrix_cv);
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
