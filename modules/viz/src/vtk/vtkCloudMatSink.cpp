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
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//M*/

#include "../precomp.hpp"

namespace cv { namespace viz
{
    vtkStandardNewMacro(vtkCloudMatSink);
}}

cv::viz::vtkCloudMatSink::vtkCloudMatSink() {}
cv::viz::vtkCloudMatSink::~vtkCloudMatSink() {}

void cv::viz::vtkCloudMatSink::SetOutput(OutputArray _cloud, OutputArray _colors, OutputArray _normals, OutputArray _tcoords)
{
    cloud = _cloud;
    colors = _colors;
    normals = _normals;
    tcoords = _tcoords;
}

void cv::viz::vtkCloudMatSink::WriteData()
{
    vtkPolyData *input = this->GetInput();
    if (!input)
        return;

    vtkSmartPointer<vtkPoints> points_Data = input->GetPoints();

    if (cloud.needed() && points_Data)
    {
        int vtktype = points_Data->GetDataType();
        CV_Assert(vtktype == VTK_FLOAT || vtktype == VTK_DOUBLE);

        cloud.create(1, points_Data->GetNumberOfPoints(), vtktype == VTK_FLOAT ? CV_32FC3 : CV_64FC3);
        Vec3d *ddata = cloud.getMat().ptr<Vec3d>();
        Vec3f *fdata = cloud.getMat().ptr<Vec3f>();

        if (cloud.depth() == CV_32F)
            for(size_t i = 0; i < cloud.total(); ++i)
                *fdata++ = Vec3d(points_Data->GetPoint((vtkIdType)i));

        if (cloud.depth() == CV_64F)
            for(size_t i = 0; i < cloud.total(); ++i)
                *ddata++ = Vec3d(points_Data->GetPoint((vtkIdType)i));
    }
    else
        cloud.release();

    vtkSmartPointer<vtkDataArray> scalars_data = input->GetPointData() ? input->GetPointData()->GetScalars() : 0;

    if (colors.needed() && scalars_data)
    {
        int channels = scalars_data->GetNumberOfComponents();
        int vtktype = scalars_data->GetDataType();

        CV_Assert((channels == 3 || channels == 4) && "Only 3- or 4-channel color data support is implemented");
        CV_Assert(cloud.total() == (size_t)scalars_data->GetNumberOfTuples());

        Mat buffer(cloud.size(), CV_64FC(channels));
        Vec3d *cptr = buffer.ptr<Vec3d>();
        for(size_t i = 0; i < buffer.total(); ++i)
            *cptr++ = Vec3d(scalars_data->GetTuple((vtkIdType)i));

        buffer.convertTo(colors, CV_8U, vtktype == VTK_FLOAT || VTK_FLOAT == VTK_DOUBLE ?  255.0 : 1.0);
    }
    else
        colors.release();

    vtkSmartPointer<vtkDataArray> normals_data = input->GetPointData() ? input->GetPointData()->GetNormals() : 0;

    if (normals.needed() && normals_data)
    {
        int channels = normals_data->GetNumberOfComponents();
        int vtktype = normals_data->GetDataType();

        CV_Assert((vtktype == VTK_FLOAT || VTK_FLOAT == VTK_DOUBLE) && (channels == 3 || channels == 4));
        CV_Assert(cloud.total() == (size_t)normals_data->GetNumberOfTuples());

        Mat buffer(cloud.size(), CV_64FC(channels));
        Vec3d *cptr = buffer.ptr<Vec3d>();
        for(size_t i = 0; i < buffer.total(); ++i)
            *cptr++ = Vec3d(normals_data->GetTuple((vtkIdType)i));

        buffer.convertTo(normals, vtktype == VTK_FLOAT ? CV_32F : CV_64F);
    }
    else
        normals.release();

    vtkSmartPointer<vtkDataArray> coords_data = input->GetPointData() ? input->GetPointData()->GetTCoords() : 0;

    if (tcoords.needed() && coords_data)
    {
        int vtktype = coords_data->GetDataType();

        CV_Assert(vtktype == VTK_FLOAT || VTK_FLOAT == VTK_DOUBLE);
        CV_Assert(cloud.total() == (size_t)coords_data->GetNumberOfTuples());

        Mat buffer(cloud.size(), CV_64FC2);
        Vec2d *cptr = buffer.ptr<Vec2d>();
        for(size_t i = 0; i < buffer.total(); ++i)
            *cptr++ = Vec2d(coords_data->GetTuple((vtkIdType)i));

        buffer.convertTo(tcoords, vtktype == VTK_FLOAT ? CV_32F : CV_64F);

    }
    else
        tcoords.release();
}

void cv::viz::vtkCloudMatSink::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Cloud: " << cloud.needed() << "\n";
  os << indent << "Colors: " << colors.needed() << "\n";
  os << indent << "Normals: " << normals.needed() << "\n";
}

int cv::viz::vtkCloudMatSink::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
}

vtkPolyData* cv::viz::vtkCloudMatSink::GetInput()
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput());
}

vtkPolyData* cv::viz::vtkCloudMatSink::GetInput(int port)
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput(port));
}
