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
    vtkStandardNewMacro(vtkXYZReader);
}}


cv::viz::vtkXYZReader::vtkXYZReader()
{
    this->FileName = 0;
    this->SetNumberOfInputPorts(0);
}

cv::viz::vtkXYZReader::~vtkXYZReader()
{
    this->SetFileName(0);
}

void cv::viz::vtkXYZReader::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);
    os << indent << "FileName: " << (this->FileName ? this->FileName : "(none)") << "\n";
}

int cv::viz::vtkXYZReader::RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector* outputVector)
{
    // Make sure we have a file to read.
    if(!this->FileName)
    {
        vtkErrorMacro("A FileName must be specified.");
        return 0;
    }

    // Open the input file.
    ifstream fin(this->FileName);
    if(!fin)
    {
        vtkErrorMacro("Error opening file " << this->FileName);
        return 0;
    }

    // Allocate objects to hold points and vertex cells.
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> verts = vtkSmartPointer<vtkCellArray>::New();

    // Read points from the file.
    vtkDebugMacro("Reading points from file " << this->FileName);
    double x[3];
    while(fin >> x[0] >> x[1] >> x[2])
    {
        vtkIdType id = points->InsertNextPoint(x);
        verts->InsertNextCell(1, &id);
    }
    vtkDebugMacro("Read " << points->GetNumberOfPoints() << " points.");

    // Store the points and cells in the output data object.
    vtkPolyData* output = vtkPolyData::GetData(outputVector);
    output->SetPoints(points);
    output->SetVerts(verts);

    return 1;
}
