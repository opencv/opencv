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
}}

cv::viz::vtkCloudMatSource::vtkCloudMatSource() { SetNumberOfInputPorts(0); }
cv::viz::vtkCloudMatSource::~vtkCloudMatSource() {}

void cv::viz::vtkCloudMatSource::SetCloud(const Mat& cloud)
{
    CV_Assert(cloud.depth() == CV_32F || cloud.depth() == CV_64F);
    CV_Assert(cloud.channels() == 3 || cloud.channels() == 4);

    int total = cloud.depth() == CV_32F ? filterNanCopy<float >(cloud, VTK_FLOAT)
                                        : filterNanCopy<double>(cloud, VTK_DOUBLE);

    vertices = vtkSmartPointer<vtkCellArray>::New();
    vertices->Allocate(vertices->EstimateSize(1, total));
    vertices->InsertNextCell(total);
    for(int i = 0; i < total; ++i)
        vertices->InsertCellPoint(i);
}

int cv::viz::vtkCloudMatSource::RequestData(vtkInformation *vtkNotUsed(request), vtkInformationVector **vtkNotUsed(inputVector), vtkInformationVector *outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    output->SetPoints(points);
    output->SetVerts(vertices);
    return 1;
}

template<typename _Tp>
int cv::viz::vtkCloudMatSource::filterNanCopy(const Mat& source, int dataType)
{
    CV_DbgAssert(DataType<_Tp>::depth == source.depth());
    points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataType(dataType);
    points->Allocate(source.total());
    points->SetNumberOfPoints(source.total());

    int cn = source.channels();
    int total = 0;
    for (int y = 0; y < source.rows; ++y)
    {
        const _Tp* srow = source.ptr<_Tp>(y);
        const _Tp* send = srow + source.cols * cn;

        for (; srow != send; srow += cn)
            if (!isNan(srow[0]) && !isNan(srow[1]) && !isNan(srow[2]))
                points->SetPoint(total++, srow);
    }
    points->SetNumberOfPoints(total);
    points->Squeeze();
    return total;
}


