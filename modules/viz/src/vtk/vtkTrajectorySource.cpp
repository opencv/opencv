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
    vtkStandardNewMacro(vtkTrajectorySource);
}}

cv::viz::vtkTrajectorySource::vtkTrajectorySource() { SetNumberOfInputPorts(0); }
cv::viz::vtkTrajectorySource::~vtkTrajectorySource() {}

void cv::viz::vtkTrajectorySource::SetTrajectory(InputArray _traj)
{
    CV_Assert(_traj.kind() == _InputArray::STD_VECTOR || _traj.kind() == _InputArray::MAT);
    CV_Assert(_traj.type() == CV_32FC(16) || _traj.type() == CV_64FC(16));

    Mat traj;
    _traj.getMat().convertTo(traj, CV_64F);
    const Affine3d* dpath = traj.ptr<Affine3d>();
    size_t total = traj.total();

    points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataType(VTK_DOUBLE);
    points->SetNumberOfPoints((vtkIdType)total);

    tensors = vtkSmartPointer<vtkDoubleArray>::New();
    tensors->SetNumberOfComponents(9);
    tensors->SetNumberOfTuples((vtkIdType)total);

    for(size_t i = 0; i < total; ++i, ++dpath)
    {
        Matx33d R = dpath->rotation().t();  // transposed because of
        tensors->SetTuple((vtkIdType)i, R.val);        // column major order

        Vec3d p = dpath->translation();
        points->SetPoint((vtkIdType)i, p.val);
    }
}

cv::Mat cv::viz::vtkTrajectorySource::ExtractPoints(InputArray _traj)
{
    CV_Assert(_traj.kind() == _InputArray::STD_VECTOR || _traj.kind() == _InputArray::MAT);
    CV_Assert(_traj.type() == CV_32FC(16) || _traj.type() == CV_64FC(16));

    Mat points(1, (int)_traj.total(), CV_MAKETYPE(_traj.depth(), 3));
    const Affine3d* dpath = _traj.getMat().ptr<Affine3d>();
    const Affine3f* fpath = _traj.getMat().ptr<Affine3f>();

    if (_traj.depth() == CV_32F)
        for(int i = 0; i < points.cols; ++i)
            points.at<Vec3f>(i) = fpath[i].translation();

    if (_traj.depth() == CV_64F)
        for(int i = 0; i < points.cols; ++i)
            points.at<Vec3d>(i) = dpath[i].translation();

    return points;
}

int cv::viz::vtkTrajectorySource::RequestData(vtkInformation *vtkNotUsed(request), vtkInformationVector **vtkNotUsed(inputVector), vtkInformationVector *outputVector)
{
    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
    output->SetPoints(points);
    output->GetPointData()->SetTensors(tensors);
    return 1;
}
