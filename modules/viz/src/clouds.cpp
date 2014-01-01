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
/// Point Cloud Widget implementation

cv::viz::WCloud::WCloud(InputArray cloud, InputArray colors)
{
    CV_Assert(!cloud.empty() && !colors.empty());

    vtkSmartPointer<vtkCloudMatSource> cloud_source = vtkSmartPointer<vtkCloudMatSource>::New();
    cloud_source->SetColorCloud(cloud, colors);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(cloud_source->GetOutputPort());
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

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(cloud_source->GetOutputPort());
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
#if VTK_MAJOR_VERSION <= 5
        mapper->SetInput(polydata);
#else
        mapper->SetInputData(polydata);
#endif
        mapper->SetScalarRange(0, 255);
        mapper->SetScalarModeToUsePointData();
        mapper->ScalarVisibilityOn();
        mapper->ImmediateModeRenderingOff();

        actor->SetNumberOfCloudPoints(std::max(1, polydata->GetNumberOfPoints()/10));
        actor->GetProperty()->SetInterpolationToFlat();
        actor->GetProperty()->BackfaceCullingOn();
        actor->SetMapper(mapper);
        return;
    }

    vtkPolyData *currdata = vtkPolyData::SafeDownCast(mapper->GetInput());
    CV_Assert("Cloud Widget without data" && currdata);

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
#if VTK_MAJOR_VERSION <= 5
    appendFilter->AddInput(currdata);
    appendFilter->AddInput(polydata);
    mapper->SetInput(appendFilter->GetOutput());
#else
    appendFilter->AddInputData(currdata);
    appendFilter->AddInputData(polydata);
    mapper->SetInputData(appendFilter->GetOutput());
#endif

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

namespace cv { namespace viz { namespace
{
    struct CloudNormalsUtils
    {
        template<typename _Tp>
        struct Impl
        {
            static vtkSmartPointer<vtkCellArray> applyOrganized(const Mat &cloud, const Mat& normals, double level, float scale, _Tp *&pts, vtkIdType &nr_normals)
            {
                vtkIdType point_step = static_cast<vtkIdType>(std::sqrt(level));
                nr_normals = (static_cast<vtkIdType>((cloud.cols - 1) / point_step) + 1) *
                             (static_cast<vtkIdType>((cloud.rows - 1) / point_step) + 1);
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

                        lines->InsertNextCell(2);
                        lines->InsertCellPoint(2 * cell_count);
                        lines->InsertCellPoint(2 * cell_count + 1);
                        cell_count++;
                    }
                }
                return lines;
            }

            static vtkSmartPointer<vtkCellArray> applyUnorganized(const Mat &cloud, const Mat& normals, int level, float scale, _Tp *&pts, vtkIdType &nr_normals)
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

                    lines->InsertNextCell(2);
                    lines->InsertCellPoint(2 * j);
                    lines->InsertCellPoint(2 * j + 1);
                }
                return lines;
            }
        };

        template<typename _Tp>
        static inline vtkSmartPointer<vtkCellArray> apply(const Mat &cloud, const Mat& normals, int level, float scale, _Tp *&pts, vtkIdType &nr_normals)
        {
            if (cloud.cols > 1 && cloud.rows > 1)
                return CloudNormalsUtils::Impl<_Tp>::applyOrganized(cloud, normals, level, scale, pts, nr_normals);
            else
                return CloudNormalsUtils::Impl<_Tp>::applyUnorganized(cloud, normals, level, scale, pts, nr_normals);
        }
    };

}}}

cv::viz::WCloudNormals::WCloudNormals(InputArray _cloud, InputArray _normals, int level, float scale, const Color &color)
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

        vtkSmartPointer<vtkFloatArray> data = vtkSmartPointer<vtkFloatArray>::New();
        data->SetNumberOfComponents(3);

        float* pts = 0;
        lines = CloudNormalsUtils::apply(cloud, normals, level, scale, pts, nr_normals);
        data->SetArray(&pts[0], 2 * nr_normals * 3, 0);
        points->SetData(data);
    }
    else
    {
        points->SetDataTypeToDouble();

        vtkSmartPointer<vtkDoubleArray> data = vtkSmartPointer<vtkDoubleArray>::New();
        data->SetNumberOfComponents(3);

        double* pts = 0;
        lines = CloudNormalsUtils::apply(cloud, normals, level, scale, pts, nr_normals);
        data->SetArray(&pts[0], 2 * nr_normals * 3, 0);
        points->SetData(data);
    }

    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->SetPoints(points);
    polyData->SetLines(lines);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polyData);
#else
    mapper->SetInputData(polyData);
#endif
    mapper->SetColorModeToMapScalars();
    mapper->SetScalarModeToUsePointData();

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

cv::viz::WMesh::WMesh(const Mesh3d &mesh)
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
    int poly_size = mesh.polygons.total();
    for (int i = 0; i < poly_size; ++idx)
    {
        int n_points = polygons[i++];

        cell_array->InsertNextCell(n_points);
        for (int j = 0; j < n_points; ++j, ++idx)
            cell_array->InsertCellPoint(lookup[polygons[i++]]);
    }
    cell_array->GetData()->SetNumberOfValues(idx);
    cell_array->Squeeze();
    polydata->SetStrips(cell_array);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetScalarModeToUsePointData();
    mapper->ImmediateModeRenderingOff();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polydata);
#else
    mapper->SetInputData(polydata);
#endif

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

template<> CV_EXPORTS cv::viz::WMesh cv::viz::Widget::cast<cv::viz::WMesh>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WMesh&>(widget);
}
