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
    vtkStandardNewMacro(vtkXYZWriter);
}}

cv::viz::vtkXYZWriter::vtkXYZWriter()
{
    std::ofstream fout; // only used to extract the default precision
    this->DecimalPrecision = fout.precision();
}

void cv::viz::vtkXYZWriter::WriteData()
{
    vtkPolyData *input = this->GetInput();
    if (!input)
        return;

    if (!this->FileName )
    {
        vtkErrorMacro(<< "No FileName specified! Can't write!");
        this->SetErrorCode(vtkErrorCode::NoFileNameError);
        return;
    }

    vtkDebugMacro(<<"Opening vtk file for writing...");
    ostream *outfilep = new ofstream(this->FileName, ios::out);
    if (outfilep->fail())
    {
        vtkErrorMacro(<< "Unable to open file: "<< this->FileName);
        this->SetErrorCode(vtkErrorCode::CannotOpenFileError);
        delete outfilep;
        return;
    }

    ostream &outfile = *outfilep;

    for(vtkIdType i = 0; i < input->GetNumberOfPoints(); ++i)
    {
        Vec3d p;
        input->GetPoint(i, p.val);
        outfile << std::setprecision(this->DecimalPrecision) << p[0] << " " << p[1] << " " << p[2] << std::endl;
    }

    // Close the file
    vtkDebugMacro(<<"Closing vtk file\n");
    delete outfilep;

    // Delete the file if an error occurred
    if (this->ErrorCode == vtkErrorCode::OutOfDiskSpaceError)
    {
        vtkErrorMacro("Ran out of disk space; deleting file: " << this->FileName);
        unlink(this->FileName);
    }
}

int cv::viz::vtkXYZWriter::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
}

void cv::viz::vtkXYZWriter::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);
    os << indent << "DecimalPrecision: " << this->DecimalPrecision << "\n";
}

vtkPolyData* cv::viz::vtkXYZWriter::GetInput()
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput());
}

vtkPolyData* cv::viz::vtkXYZWriter::GetInput(int port)
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput(port));
}
