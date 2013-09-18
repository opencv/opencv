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
/// Point Cloud Widget implementation

struct cv::viz::WCloud::CreateCloudWidget
{
    static inline vtkSmartPointer<vtkPolyData> create(const Mat &cloud, vtkIdType &nr_points)
    {
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();

        polydata->SetVerts(vertices);

        vtkSmartPointer<vtkPoints> points = polydata->GetPoints();
        vtkSmartPointer<vtkIdTypeArray> initcells;
        nr_points = cloud.total();

        if (!points)
        {
            points = vtkSmartPointer<vtkPoints>::New();
            if (cloud.depth() == CV_32F)
                points->SetDataTypeToFloat();
            else if (cloud.depth() == CV_64F)
                points->SetDataTypeToDouble();
            polydata->SetPoints(points);
        }
        points->SetNumberOfPoints(nr_points);

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
        points->SetNumberOfPoints(nr_points);

        // Update cells
        vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData();
        // If no init cells and cells has not been initialized...
        if (!cells)
            cells = vtkSmartPointer<vtkIdTypeArray>::New();

        // If we have less values then we need to recreate the array
        if (cells->GetNumberOfTuples() < nr_points)
        {
            cells = vtkSmartPointer<vtkIdTypeArray>::New();

            // If init cells is given, and there's enough data in it, use it
            if (initcells && initcells->GetNumberOfTuples() >= nr_points)
            {
                cells->DeepCopy(initcells);
                cells->SetNumberOfComponents(2);
                cells->SetNumberOfTuples(nr_points);
            }
            else
            {
                // If the number of tuples is still too small, we need to recreate the array
                cells->SetNumberOfComponents(2);
                cells->SetNumberOfTuples(nr_points);
                vtkIdType *cell = cells->GetPointer(0);
                // Fill it with 1s
                std::fill_n(cell, nr_points * 2, 1);
                cell++;
                for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
                    *cell = i;
                // Save the results in initcells
                initcells = vtkSmartPointer<vtkIdTypeArray>::New();
                initcells->DeepCopy(cells);
            }
        }
        else
        {
            // The assumption here is that the current set of cells has more data than needed
            cells->SetNumberOfComponents(2);
            cells->SetNumberOfTuples(nr_points);
        }

        // Set the cells and the vertices
        vertices->SetCells(nr_points, cells);
        return polydata;
    }
};

cv::viz::WCloud::WCloud(InputArray _cloud, InputArray _colors)
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
    NanFilter::copyColor(colors, colors_data, cloud);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetNumberOfTuples(nr_points);
    scalars->SetArray(colors_data->val, 3 * nr_points, 0);

    // Assign the colors
    polydata->GetPointData()->SetScalars(scalars);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polydata);
#else
    mapper->SetInputData(polydata);
#endif

    Vec3d minmax(scalars->GetRange());
    mapper->SetScalarRange(minmax.val);
    mapper->SetScalarModeToUsePointData();

    bool interpolation = (polydata && polydata->GetNumberOfCells() != polydata->GetNumberOfVerts());

    mapper->SetInterpolateScalarsBeforeMapping(interpolation);
    mapper->ScalarVisibilityOn();

    mapper->ImmediateModeRenderingOff();

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, polydata->GetNumberOfPoints() / 10)));
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->BackfaceCullingOn();
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

cv::viz::WCloud::WCloud(InputArray _cloud, const Color &color)
{
    Mat cloud = _cloud.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);

    vtkIdType nr_points;
    vtkSmartPointer<vtkPolyData> polydata = CreateCloudWidget::create(cloud, nr_points);

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(polydata);
#else
    mapper->SetInputData(polydata);
#endif

    bool interpolation = (polydata && polydata->GetNumberOfCells() != polydata->GetNumberOfVerts());

    mapper->SetInterpolateScalarsBeforeMapping(interpolation);
    mapper->ScalarVisibilityOff();

    mapper->ImmediateModeRenderingOff();

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, polydata->GetNumberOfPoints() / 10)));
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

struct cv::viz::WCloudCollection::CreateCloudWidget
{
    static inline vtkSmartPointer<vtkPolyData> create(const Mat &cloud, vtkIdType &nr_points)
    {
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();

        polydata->SetVerts(vertices);

        vtkSmartPointer<vtkPoints> points = polydata->GetPoints();
        vtkSmartPointer<vtkIdTypeArray> initcells;
        nr_points = cloud.total();

        if (!points)
        {
            points = vtkSmartPointer<vtkPoints>::New();
            if (cloud.depth() == CV_32F)
                points->SetDataTypeToFloat();
            else if (cloud.depth() == CV_64F)
                points->SetDataTypeToDouble();
            polydata->SetPoints(points);
        }
        points->SetNumberOfPoints(nr_points);

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
        points->SetNumberOfPoints(nr_points);

        // Update cells
        vtkSmartPointer<vtkIdTypeArray> cells = vertices->GetData();
        // If no init cells and cells has not been initialized...
        if (!cells)
            cells = vtkSmartPointer<vtkIdTypeArray>::New();

        // If we have less values then we need to recreate the array
        if (cells->GetNumberOfTuples() < nr_points)
        {
            cells = vtkSmartPointer<vtkIdTypeArray>::New();

            // If init cells is given, and there's enough data in it, use it
            if (initcells && initcells->GetNumberOfTuples() >= nr_points)
            {
                cells->DeepCopy(initcells);
                cells->SetNumberOfComponents(2);
                cells->SetNumberOfTuples(nr_points);
            }
            else
            {
                // If the number of tuples is still too small, we need to recreate the array
                cells->SetNumberOfComponents(2);
                cells->SetNumberOfTuples(nr_points);
                vtkIdType *cell = cells->GetPointer(0);
                // Fill it with 1s
                std::fill_n(cell, nr_points * 2, 1);
                cell++;
                for (vtkIdType i = 0; i < nr_points; ++i, cell += 2)
                    *cell = i;
                // Save the results in initcells
                initcells = vtkSmartPointer<vtkIdTypeArray>::New();
                initcells->DeepCopy(cells);
            }
        }
        else
        {
            // The assumption here is that the current set of cells has more data than needed
            cells->SetNumberOfComponents(2);
            cells->SetNumberOfTuples(nr_points);
        }

        // Set the cells and the vertices
        vertices->SetCells(nr_points, cells);
        return polydata;
    }

    static void createMapper(vtkSmartPointer<vtkLODActor> actor, vtkSmartPointer<vtkPolyData> poly_data, Vec3d& minmax)
    {
        vtkDataSetMapper *mapper = vtkDataSetMapper::SafeDownCast(actor->GetMapper());
        if (!mapper)
        {
            // This is the first cloud
            vtkSmartPointer<vtkDataSetMapper> mapper_new = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
            mapper_new->SetInputConnection(poly_data->GetProducerPort());
#else
            mapper_new->SetInputData(poly_data);
#endif

            mapper_new->SetScalarRange(minmax.val);
            mapper_new->SetScalarModeToUsePointData();

            bool interpolation = (poly_data && poly_data->GetNumberOfCells() != poly_data->GetNumberOfVerts());

            mapper_new->SetInterpolateScalarsBeforeMapping(interpolation);
            mapper_new->ScalarVisibilityOn();
            mapper_new->ImmediateModeRenderingOff();

            actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, poly_data->GetNumberOfPoints() / 10)));
            actor->GetProperty()->SetInterpolationToFlat();
            actor->GetProperty()->BackfaceCullingOn();
            actor->SetMapper(mapper_new);
            return ;
        }

        vtkPolyData *data = vtkPolyData::SafeDownCast(mapper->GetInput());
        CV_Assert("Cloud Widget without data" && data);

        vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
#if VTK_MAJOR_VERSION <= 5
        appendFilter->AddInputConnection(mapper->GetInput()->GetProducerPort());
        appendFilter->AddInputConnection(poly_data->GetProducerPort());
#else
        appendFilter->AddInputData(data);
        appendFilter->AddInputData(poly_data);
#endif
        mapper->SetInputConnection(appendFilter->GetOutputPort());

        // Update the number of cloud points
        vtkIdType old_cloud_points = actor->GetNumberOfCloudPoints();
        actor->SetNumberOfCloudPoints(int(std::max<vtkIdType>(1, old_cloud_points+poly_data->GetNumberOfPoints() / 10)));
    }
};

cv::viz::WCloudCollection::WCloudCollection()
{
    // Just create the actor
    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
    WidgetAccessor::setProp(*this, actor);
}

void cv::viz::WCloudCollection::addCloud(InputArray _cloud, InputArray _colors, const Affine3f &pose)
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
    vtkSmartPointer<vtkPolyData> polydata =  CreateCloudWidget::create(cloud, nr_points);

    // Filter colors
    Vec3b* colors_data = new Vec3b[nr_points];
    NanFilter::copyColor(colors, colors_data, cloud);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetNumberOfTuples(nr_points);
    scalars->SetArray(colors_data->val, 3 * nr_points, 0);

    // Assign the colors
    polydata->GetPointData()->SetScalars(scalars);

    // Transform the poly data based on the pose
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(convertToVtkMatrix(pose.matrix));

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
#if VTK_MAJOR_VERSION <= 5
    transform_filter->SetInputConnection(polydata->GetProducerPort());
#else
    transform_filter->SetInputData(polydata);
#endif
    transform_filter->Update();

    vtkLODActor *actor = vtkLODActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Incompatible widget type." && actor);

    Vec3d minmax(scalars->GetRange());
    CreateCloudWidget::createMapper(actor, transform_filter->GetOutput(), minmax);
}

void cv::viz::WCloudCollection::addCloud(InputArray _cloud, const Color &color, const Affine3f &pose)
{
    Mat cloud = _cloud.getMat();
    CV_Assert(cloud.type() == CV_32FC3 || cloud.type() == CV_64FC3 || cloud.type() == CV_32FC4 || cloud.type() == CV_64FC4);

    vtkIdType nr_points;
    vtkSmartPointer<vtkPolyData> polydata =  CreateCloudWidget::create(cloud, nr_points);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetNumberOfTuples(nr_points);
    scalars->FillComponent(0, color[2]);
    scalars->FillComponent(1, color[1]);
    scalars->FillComponent(2, color[0]);

    // Assign the colors
    polydata->GetPointData()->SetScalars(scalars);

    // Transform the poly data based on the pose
    vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
    transform->PreMultiply();
    transform->SetMatrix(convertToVtkMatrix(pose.matrix));

    vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
    transform_filter->SetTransform(transform);
#if VTK_MAJOR_VERSION <= 5
    transform_filter->SetInputConnection(polydata->GetProducerPort());
#else
    transform_filter->SetInputData(polydata);
#endif
    transform_filter->Update();

    vtkLODActor *actor = vtkLODActor::SafeDownCast(WidgetAccessor::getProp(*this));
    CV_Assert("Incompatible widget type." && actor);

    Vec3d minmax(scalars->GetRange());
    CreateCloudWidget::createMapper(actor, transform_filter->GetOutput(), minmax);
}

template<> cv::viz::WCloudCollection cv::viz::Widget::cast<cv::viz::WCloudCollection>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WCloudCollection&>(widget);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Cloud Normals Widget implementation

struct cv::viz::WCloudNormals::ApplyCloudNormals
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
            return ApplyCloudNormals::Impl<_Tp>::applyOrganized(cloud, normals, level, scale, pts, nr_normals);
        else
            return ApplyCloudNormals::Impl<_Tp>::applyUnorganized(cloud, normals, level, scale, pts, nr_normals);
    }
};

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
        lines = ApplyCloudNormals::apply(cloud, normals, level, scale, pts, nr_normals);
        data->SetArray(&pts[0], 2 * nr_normals * 3, 0);
        points->SetData(data);
    }
    else
    {
        points->SetDataTypeToDouble();

        vtkSmartPointer<vtkDoubleArray> data = vtkSmartPointer<vtkDoubleArray>::New();
        data->SetNumberOfComponents(3);

        double* pts = 0;
        lines = ApplyCloudNormals::apply(cloud, normals, level, scale, pts, nr_normals);
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

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();
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

struct cv::viz::WMesh::CopyImpl
{
    template<typename _Tp>
    static Vec<_Tp, 3> * copy(const Mat &source, Vec<_Tp, 3> *output, int *look_up, const Mat &nan_mask)
    {
        CV_Assert(DataDepth<_Tp>::value == source.depth() && source.size() == nan_mask.size());
        CV_Assert(nan_mask.channels() == 3 || nan_mask.channels() == 4);
        CV_DbgAssert(DataDepth<_Tp>::value == nan_mask.depth());

        int s_chs = source.channels();
        int m_chs = nan_mask.channels();

        int index = 0;
        const _Tp* srow = source.ptr<_Tp>(0);
        const _Tp* mrow = nan_mask.ptr<_Tp>(0);

        for (int x = 0; x < source.cols; ++x, srow += s_chs, mrow += m_chs)
        {
            if (!isNan(mrow[0]) && !isNan(mrow[1]) && !isNan(mrow[2]))
            {
                look_up[x] = index;
                *output++ = Vec<_Tp, 3>(srow);
                ++index;
            }
        }
        return output;
    }
};

cv::viz::WMesh::WMesh(const Mesh3d &mesh)
{
    CV_Assert(mesh.cloud.rows == 1 && (mesh.cloud.type() == CV_32FC3 || mesh.cloud.type() == CV_64FC3 || mesh.cloud.type() == CV_32FC4 || mesh.cloud.type() == CV_64FC4));
    CV_Assert(mesh.colors.empty() || (mesh.colors.type() == CV_8UC3 && mesh.cloud.size() == mesh.colors.size()));
    CV_Assert(!mesh.polygons.empty() && mesh.polygons.type() == CV_32SC1);

    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkIdType nr_points = mesh.cloud.total();
    Mat look_up_mat(1, nr_points, CV_32SC1);
    int * look_up = look_up_mat.ptr<int>();
    points->SetNumberOfPoints(nr_points);

    // Copy data from cloud to vtkPoints
    if (mesh.cloud.depth() == CV_32F)
    {
        points->SetDataTypeToFloat();
        Vec3f *data_beg = vtkpoints_data<float>(points);
        Vec3f *data_end = CopyImpl::copy(mesh.cloud, data_beg, look_up, mesh.cloud);
        nr_points = data_end - data_beg;
    }
    else
    {
        points->SetDataTypeToDouble();
        Vec3d *data_beg = vtkpoints_data<double>(points);
        Vec3d *data_end = CopyImpl::copy(mesh.cloud, data_beg, look_up, mesh.cloud);
        nr_points = data_end - data_beg;
    }

    vtkSmartPointer<vtkUnsignedCharArray> scalars;

    if (!mesh.colors.empty())
    {
        Vec3b * colors_data = 0;
        colors_data = new Vec3b[nr_points];
        NanFilter::copyColor(mesh.colors, colors_data, mesh.cloud);

        scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetNumberOfTuples(nr_points);
        scalars->SetArray(colors_data->val, 3 * nr_points, 0);
    }

    points->SetNumberOfPoints(nr_points);

    vtkSmartPointer<vtkPointSet> data;

    if (mesh.polygons.size().area() > 1)
    {
        vtkSmartPointer<vtkCellArray> cell_array = vtkSmartPointer<vtkCellArray>::New();
        const int * polygons = mesh.polygons.ptr<int>();

        int idx = 0;
        int poly_size = mesh.polygons.total();
        for (int i = 0; i < poly_size; ++idx)
        {
            int n_points = polygons[i++];

            cell_array->InsertNextCell(n_points);
            for (int j = 0; j < n_points; ++j, ++idx)
                cell_array->InsertCellPoint(look_up[polygons[i++]]);
        }
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        cell_array->GetData()->SetNumberOfValues(idx);
        cell_array->Squeeze();
        polydata->SetStrips(cell_array);
        polydata->SetPoints(points);

        if (scalars)
            polydata->GetPointData()->SetScalars(scalars);

        data = polydata;
    }
    else
    {
        // Only one polygon
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
        const int * polygons = mesh.polygons.ptr<int>();
        int n_points = polygons[0];

        polygon->GetPointIds()->SetNumberOfIds(n_points);

        for (int j = 1; j < n_points+1; ++j)
            polygon->GetPointIds()->SetId(j, look_up[polygons[j]]);

        vtkSmartPointer<vtkUnstructuredGrid> poly_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        poly_grid->Allocate(1, 1);
        poly_grid->InsertNextCell(polygon->GetCellType(), polygon->GetPointIds());
        poly_grid->SetPoints(points);

        if (scalars)
            poly_grid->GetPointData()->SetScalars(scalars);

        data = poly_grid;
    }

    vtkSmartPointer<vtkLODActor> actor = vtkSmartPointer<vtkLODActor>::New();

    actor->GetProperty()->SetRepresentationToSurface();
    actor->GetProperty()->BackfaceCullingOff(); // Backface culling is off for higher efficiency
    actor->GetProperty()->SetInterpolationToFlat();
    actor->GetProperty()->EdgeVisibilityOff();
    actor->GetProperty()->ShadingOff();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInput(data);
#else
    mapper->SetInputData(data);
#endif
    mapper->ImmediateModeRenderingOff();

    vtkIdType numberOfCloudPoints = nr_points * 0.1;
    actor->SetNumberOfCloudPoints(int(numberOfCloudPoints > 1 ? numberOfCloudPoints : 1));
    actor->SetMapper(mapper);

    WidgetAccessor::setProp(*this, actor);
}

template<> CV_EXPORTS cv::viz::WMesh cv::viz::Widget::cast<cv::viz::WMesh>()
{
    Widget3D widget = this->cast<Widget3D>();
    return static_cast<WMesh&>(widget);
}
