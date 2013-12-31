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

#include "precomp.hpp"

namespace cv { namespace viz
{
    vtkStandardNewMacro(vtkCloudMatSource);

    template<typename _Tp> bool isNan(const _Tp* data)
    {
        return isNan(data[0]) || isNan(data[1]) || isNan(data[2]);
    }

    template<typename _Tp> struct VtkDepthTraits;

    template<> struct VtkDepthTraits<float>
    {
        const static int data_type = VTK_FLOAT;
        typedef vtkFloatArray array_type;
    };

    template<> struct VtkDepthTraits<double>
    {
        const static int data_type = VTK_DOUBLE;
        typedef vtkDoubleArray array_type;
    };
}}

cv::viz::vtkCloudMatSource::vtkCloudMatSource() { SetNumberOfInputPorts(0); }
cv::viz::vtkCloudMatSource::~vtkCloudMatSource() {}

int cv::viz::vtkCloudMatSource::SetCloud(const Mat& cloud)
{
    CV_Assert(cloud.depth() == CV_32F || cloud.depth() == CV_64F);
    CV_Assert(cloud.channels() == 3 || cloud.channels() == 4);

    int total = cloud.depth() == CV_32F ? filterNanCopy<float>(cloud) : filterNanCopy<double>(cloud);

    vertices = vtkSmartPointer<vtkCellArray>::New();
    vertices->Allocate(vertices->EstimateSize(1, total));
    vertices->InsertNextCell(total);
    for(int i = 0; i < total; ++i)
        vertices->InsertCellPoint(i);

    return total;
}

int cv::viz::vtkCloudMatSource::SetColorCloud(const Mat &cloud, const Mat &colors)
{
    int total = SetCloud(cloud);

    if (colors.empty())
        return total;

    CV_Assert(colors.depth() == CV_8U && colors.channels() <= 4 && colors.channels() != 2);
    CV_Assert(colors.size() == cloud.size());

    if (cloud.depth() == CV_32F)
        filterNanColorsCopy<float>(colors, cloud, total);
    else if (cloud.depth() == CV_64F)
        filterNanColorsCopy<double>(colors, cloud, total);

    return total;
}

int cv::viz::vtkCloudMatSource::SetColorCloudNormals(const Mat &cloud, const Mat &colors, const Mat &normals)
{
    int total = SetColorCloud(cloud, colors);

    if (normals.empty())
        return total;

    CV_Assert(normals.depth() == CV_32F || normals.depth() == CV_64F);
    CV_Assert(normals.channels() == 3 || normals.channels() == 4);
    CV_Assert(normals.size() == cloud.size());

    if (normals.depth() == CV_32F && cloud.depth() == CV_32F)
        filterNanNormalsCopy<float, float>(colors, cloud, total);
    else if (normals.depth() == CV_32F && cloud.depth() == CV_64F)
        filterNanNormalsCopy<float, double>(colors, cloud, total);
    else if (normals.depth() == CV_64F && cloud.depth() == CV_32F)
        filterNanNormalsCopy<double, float>(colors, cloud, total);
    else if (normals.depth() == CV_64F && cloud.depth() == CV_64F)
        filterNanNormalsCopy<double, double>(colors, cloud, total);
    else
        CV_Assert(!"Unsupported normals type");

    return total;
}

int cv::viz::vtkCloudMatSource::RequestData(vtkInformation *vtkNotUsed(request), vtkInformationVector **vtkNotUsed(inputVector), vtkInformationVector *outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    output->SetPoints(points);
    output->SetVerts(vertices);
    if (scalars)
        output->GetPointData()->SetScalars(scalars);

    if (normals)
        output->GetPointData()->SetNormals(normals);

    return 1;
}

template<typename _Tp>
int cv::viz::vtkCloudMatSource::filterNanCopy(const Mat& cloud)
{
    CV_DbgAssert(DataType<_Tp>::depth == cloud.depth());
    points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataType(VtkDepthTraits<_Tp>::data_type);
    points->Allocate(cloud.total());
    points->SetNumberOfPoints(cloud.total());

    int cn = cloud.channels();
    int total = 0;
    for (int y = 0; y < cloud.rows; ++y)
    {
        const _Tp* srow = cloud.ptr<_Tp>(y);
        const _Tp* send = srow + cloud.cols * cn;

        for (; srow != send; srow += cn)
            if (!isNan(srow))
                points->SetPoint(total++, srow);
    }
    points->SetNumberOfPoints(total);
    points->Squeeze();
    return total;
}

template<typename _Msk>
void cv::viz::vtkCloudMatSource::filterNanColorsCopy(const Mat& cloud_colors, const Mat& mask, int total)
{
    Vec3b* array = new Vec3b[total];
    Vec3b* pos = array;

    int s_chs = cloud_colors.channels();
    int m_chs = mask.channels();
    for (int y = 0; y < cloud_colors.rows; ++y)
    {
        const unsigned char* srow = cloud_colors.ptr<unsigned char>(y);
        const unsigned char* send = srow + cloud_colors.cols * cloud_colors.channels();
        const _Msk* mrow = mask.ptr<_Msk>(y);

        if (cloud_colors.channels() == 1)
        {
            for (; srow != send; srow += s_chs, mrow += m_chs)
                if (!isNan(mrow))
                    *pos++ = Vec3b(srow[0], srow[0], srow[0]);
        }
        else
            for (; srow != send; srow += s_chs, mrow += m_chs)
                if (!isNan(mrow))
                    *pos++ = Vec3b(srow[2], srow[1], srow[0]);

    }

    scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetName("Colors");
    scalars->SetNumberOfComponents(3);
    scalars->SetNumberOfTuples(total);
    scalars->SetArray(array->val, total * 3, 0);
}

template<typename _Tn, typename _Msk>
void cv::viz::vtkCloudMatSource::filterNanNormalsCopy(const Mat& cloud_normals, const Mat& mask, int total)
{
    normals = vtkSmartPointer< VtkDepthTraits<_Tn>::array_type >::New();
    normals->SetNumberOfComponents(3);
    normals->SetNumberOfTuples(total);

    int s_chs = cloud_normals.channels();
    int m_chs = mask.channels();

    int pos = 0;
    for (int y = 0; y < cloud_normals.rows; ++y)
    {
        const _Tn* srow = cloud_normals.ptr<_Tn>(y);
        const _Tn* send = srow + cloud_normals.cols * s_chs;

        const _Msk* mrow = mask.ptr<_Msk>(y);

        for (; srow != send; srow += s_chs, mrow += m_chs)
            if (!isNan(mrow))
                normals->SetTuple(pos++, srow);
    }
}
