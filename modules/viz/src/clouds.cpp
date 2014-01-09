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
/// Point Cloud Widget implementation

cv::viz::WCloud::WCloud(InputArray cloud, InputArray colors)
{
    CV_Assert(!cloud.empty() && !colors.empty());

    vtkSmartPointer<vtkCloudMatSource> cloud_source = vtkSmartPointer<vtkCloudMatSource>::New();
    cloud_source->SetColorCloud(cloud, colors);
    cloud_source->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, cloud_source->GetOutput());
    mapper->SetScalarModeToUsePointData();
    mapper->ImmediateModeRenderingOff();
    mapper->SetScalarRange(0, 255);
    mapper->ScalarVisibilityOn();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCloud::WCloud(InputArray cloud, const Color &color)
{
    vtkSmartPointer<vtkCloudMatSource> cloud_source = vtkSmartPointer<vtkCloudMatSource>::New();
    cloud_source->SetCloud(cloud);
    cloud_source->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    VtkUtils::SetInputData(mapper, cloud_source->GetOutput());
    mapper->ImmediateModeRenderingOff();
    mapper->ScalarVisibilityOff();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WCloud cv::viz::Widget::cast<cv::viz::WCloud>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCloud&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Cloud Collection Widget implementation

cv::viz::WCloudCollection::WCloudCollection()
{
    // Just create the actor
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::WCloudCollection::addCloud(InputArray cloud, InputArray colors, const Affine3d &pose)
{
    vtkSmartPointer<vtkCloudMatSource> source = vtkSmartPointer<vtkCloudMatSource>::New();
    source->SetColorCloud(cloud, colors);

    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->SetMatrix(pose.matrix.val);

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetInputConnection(source->GetOutputPort());
    transform_filter->SetTransform(transform);
    transform_filter->Update();

    vtkSmartPointer<vtkPolyData> polydata = transform_filter->GetOutput();
    vtkSmartPointer<vtkLODActor> actor = vtkLODActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Incompatible widget type." && actor);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
    if (!mapper)
    {
        // This is the first cloud
        mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetScalarRange(0, 255);
        mapper->SetScalarModeToUsePointData();
        mapper->ScalarVisibilityOn();
        mapper->ImmediateModeRenderingOff();
        VtkUtils::SetInputData(mapper, polydata);

        actor->SetNumberOfCloudPoints(std::max(1, polydata->GetNumberOfPoints()/10));
        actor->GetProperty()->SetInterpolationToFlat();
        actor->GetProperty()->BackfaceCullingOn();
        actor->SetMapper(mapper);
        return;
    }

    vtkPolyData *currdata = vtkPolyData::SafeDownCast(mapper->GetInput());
    CV_Assert("Cloud Widget without data" && currdata);

    vtkSmartPointer<vtkAppendPolyData> append_filter = vtkSmartPointer<vtkAppendPolyData>::New();
    append_filter->AddInputConnection(currdata->GetProducerPort());
    append_filter->AddInputConnection(polydata->GetProducerPort());
    append_filter->Update();

    VtkUtils::SetInputData(mapper, append_filter->GetOutput());

    actor->SetNumberOfCloudPoints(std::max(1, actor->GetNumberOfCloudPoints() + polydata->GetNumberOfPoints()/10));
}

void cv::viz::WCloudCollection::addCloud(InputArray cloud, const Color &color, const Affine3d &pose)
{
    addCloud(cloud, Mat(cloud.size(), CV_8UC3, color), pose);
}

template<> cv::viz::WCloudCollection cv::viz::Widget::cast<cv::viz::WCloudCollection>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCloudCollection&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Cloud Normals Widget implementation

cv::viz::WCloudNormals::WCloudNormals(InputArray _cloud, InputArray _normals, int level, double scale, const Color &color)
{
    Mat cloud = _cloud.getMat();
    Mat normals = _normals.getMat();

    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);
    CV_Assert(cloud.size() == normals.size() && cloud.type() == normals.type());

    int sqlevel = (int)std::sqrt((double)level);
    int ystep = (cloud.cols > 1 && cloud.rows > 1) ? sqlevel : 1;
    int xstep = (cloud.cols > 1 && cloud.rows > 1) ? sqlevel : level;

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataType(cloud.depth() == CV_32F ? VTK_FLOAT : VTK_DOUBLE);

    vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();

    int s_chs = cloud.channels();
    int n_chs = normals.channels();
    int total = 0;

    for(int y = 0; y < cloud.rows; y += ystep)
    {
        if (cloud.depth() == CV_32F)
        {
            const float *srow = cloud.ptr<float>(y);
            const float *send = srow + cloud.cols * s_chs;
            const float *nrow = normals.ptr<float>(y);

            for (; srow < send; srow += xstep * s_chs, nrow += xstep * n_chs)
                if (!isNan(srow) && !isNan(nrow))
                {
                    Vec3f endp = Vec3f(srow) + Vec3f(nrow) * (float)scale;

                    points->InsertNextPoint(srow);
                    points->InsertNextPoint(endp.val);

                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(total++);
                    lines->InsertCellPoint(total++);
                }
        }
        else
        {
            const double *srow = cloud.ptr<double>(y);
            const double *send = srow + cloud.cols * s_chs;
            const double *nrow = normals.ptr<double>(y);

            for (; srow < send; srow += xstep * s_chs, nrow += xstep * n_chs)
                if (!isNan(srow) && !isNan(nrow))
                {
                    Vec3d endp = Vec3d(srow) + Vec3d(nrow) * (double)scale;

                    points->InsertNextPoint(srow);
                    points->InsertNextPoint(endp.val);

                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(total++);
                    lines->InsertCellPoint(total++);
                }
        }
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetLines(lines);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetColorModeToMapScalars();
    mapper->SetScalarModeToUsePointData();
    VtkUtils::SetInputData(mapper, polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
    setColor(color);
}

template<> cv::viz::WCloudNormals cv::viz::Widget::cast<cv::viz::WCloudNormals>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCloudNormals&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Mesh Widget implementation

cv::viz::WMesh::WMesh(const Mesh &mesh)
{
    CV_Assert(mesh.cloud.rows == 1 && mesh.polygons.type() == CV_32SC1);

    vtkSmartPointer<vtkCloudMatSource> source = vtkSmartPointer<vtkCloudMatSource>::New();
    source->SetColorCloud(mesh.cloud, mesh.colors);
    source->Update();

    Mat lookup_buffer(1, mesh.cloud.total(), CV_32SC1);
    int *lookup = lookup_buffer.ptr<int>();
    for(int y = 0, index = 0; y < mesh.cloud.rows; ++y)
    {
        int s_chs = mesh.cloud.channels();

        if (mesh.cloud.depth() == CV_32F)
        {
            const float* srow = mesh.cloud.ptr<float>(y);
            const float* send = srow + mesh.cloud.cols * s_chs;

            for (; srow != send; srow += s_chs, ++lookup)
                if (!isNan(srow[0]) && !isNan(srow[1]) && !isNan(srow[2]))
                    *lookup = index++;
        }

        if (mesh.cloud.depth() == CV_64F)
        {
            const double* srow = mesh.cloud.ptr<double>(y);
            const double* send = srow + mesh.cloud.cols * s_chs;

            for (; srow != send; srow += s_chs, ++lookup)
                if (!isNan(srow[0]) && !isNan(srow[1]) && !isNan(srow[2]))
                    *lookup = index++;
        }
    }
    lookup = lookup_buffer.ptr<int>();

    vtkSmartPointer<vtkPolyData> polydata = source->GetOutput();
    polydata->SetVerts(0);

    const int * polygons = mesh.polygons.ptr<int>();
    vtkSmartPointer<vtkCellArray> cell_array = vtkSmartPointer<vtkCellArray>::New();

    int idx = 0;
    size_t polygons_size = mesh.polygons.total();
    for (size_t i = 0; i < polygons_size; ++idx)
    {
        int n_points = polygons[i++];

        cell_array->InsertNextCell(n_points);
        for (int j = 0; j < n_points; ++j, ++idx)
            cell_array->InsertCellPoint(lookup[polygons[i++]]);
    }
    cell_array->GetData()->SetNumberOfValues(idx);
    cell_array->Squeeze();
    polydata->SetStrips(cell_array);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->ImmediateModeRenderingOff();
    VtkUtils::SetInputData(mapper, polydata);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    //actor->SetNumberOfCloudPoints(std::max(1, polydata->GetNumberOfPoints() / 10));
    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->BackfaceCullingOff(); // Backface culling is off for higher efficiency
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->EdgeVisibilityOff();
    actor->GetProperty()->ShadingOff();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WMesh::WMesh(InputArray cloud, InputArray polygons, InputArray colors, InputArray normals)
{
    Mesh mesh;
    mesh.cloud = cloud.getMat();
    mesh.colors = colors.getMat();
    mesh.normals = normals.getMat();
    mesh.polygons = polygons.getMat();
    *this = WMesh(mesh);
}

template<> CV_EXPORTS cv::viz::WMesh cv::viz::Widget::cast<cv::viz::WMesh>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WMesh&>(widget);
}
