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
    vtkStandardNewMacro(vtkOBJWriter);
}}

cv::viz::vtkOBJWriter::vtkOBJWriter()
{
    std::ofstream fout; // only used to extract the default precision
    this->DecimalPrecision = fout.precision();
    this->FileName = NULL;
}

cv::viz::vtkOBJWriter::~vtkOBJWriter(){}

void cv::viz::vtkOBJWriter::WriteData()
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

    std::ostream& outfile = *outfilep;

    //write header
    outfile << "# wavefront obj file written by opencv viz module" << std::endl << std::endl;
    outfile << "mtllib NONE" << std::endl << std::endl;

    // write out the points
    for (int i = 0; i < input->GetNumberOfPoints(); i++)
    {
        Vec3d p;
        input->GetPoint(i, p.val);
        outfile << std::setprecision(this->DecimalPrecision) << "v " << p[0] << " " << p[1] << " " << p[2] << std::endl;
    }

    const int idStart = 1;

    // write out the point data
    vtkSmartPointer<vtkDataArray> normals = input->GetPointData()->GetNormals();
    if(normals)
    {
        for (int i = 0; i < normals->GetNumberOfTuples(); i++)
        {
            Vec3d p;
            normals->GetTuple(i, p.val);
            outfile << std::setprecision(this->DecimalPrecision) << "vn " << p[0] << " " << p[1] << " " << p[2] << std::endl;
        }
    }

    vtkSmartPointer<vtkDataArray> tcoords = input->GetPointData()->GetTCoords();
    if (tcoords)
    {
        for (int i = 0; i < tcoords->GetNumberOfTuples(); i++)
        {
            Vec2d p;
            tcoords->GetTuple(i, p.val);
            outfile << std::setprecision(this->DecimalPrecision) << "vt " << p[0] << " " << p[1] << std::endl;
        }
    }

    // write out a group name and material
    outfile << std::endl << "g grp" << idStart << std::endl;
    outfile << "usemtl mtlNONE" << std::endl;

    // write out verts if any
    if (input->GetNumberOfVerts() > 0)
    {
        vtkIdType npts = 0, *index = 0;
        vtkCellArray *cells = input->GetVerts();
        for (cells->InitTraversal(); cells->GetNextCell(npts, index); )
        {
            outfile << "p ";
            for (int i = 0; i < npts; i++)
                outfile << index[i] + idStart << " ";
            outfile << std::endl;
        }
    }

    // write out lines if any
    if (input->GetNumberOfLines() > 0)
    {
        vtkIdType npts = 0, *index = 0;
        vtkCellArray *cells = input->GetLines();
        for (cells->InitTraversal(); cells->GetNextCell(npts, index); )
        {
            outfile << "l ";
            if (tcoords)
            {
                for (int i = 0; i < npts; i++)
                    outfile << index[i] + idStart << "/" << index[i] + idStart << " ";
            }
            else
                for (int i = 0; i < npts; i++)
                    outfile << index[i] + idStart << " ";

            outfile << std::endl;
        }
    }

    // write out polys if any
    if (input->GetNumberOfPolys() > 0)
    {
        vtkIdType npts = 0, *index = 0;
        vtkCellArray *cells = input->GetPolys();
        for (cells->InitTraversal(); cells->GetNextCell(npts, index); )
        {
            outfile << "f ";
            for (int i = 0; i < npts; i++)
            {
                if (normals)
                {
                    if (tcoords)
                        outfile << index[i] + idStart << "/"  << index[i] + idStart << "/" << index[i] + idStart << " ";
                    else
                        outfile << index[i] + idStart << "//" << index[i] + idStart << " ";
                }
                else
                {
                    if (tcoords)
                        outfile << index[i] + idStart << " " << index[i] + idStart << " ";
                    else
                        outfile << index[i] + idStart << " ";
                }
            }
            outfile << std::endl;
        }
    }

    // write out tstrips if any
    if (input->GetNumberOfStrips() > 0)
    {
        vtkIdType npts = 0, *index = 0;
        vtkCellArray *cells = input->GetStrips();
        for (cells->InitTraversal(); cells->GetNextCell(npts, index); )
        {
            for (int i = 2, i1, i2; i < npts; ++i)
            {
                if (i % 2)
                {
                    i1 = i - 1;
                    i2 = i - 2;
                }
                else
                {
                    i1 = i - 1;
                    i2 = i - 2;
                }

                if(normals)
                {
                    if (tcoords)
                    {
                        outfile << "f " << index[i1] + idStart << "/" << index[i1] + idStart << "/" << index[i1] + idStart << " "
                            << index[i2]+ idStart << "/" << index[i2] + idStart << "/" << index[i2] + idStart << " "
                            << index[i] + idStart << "/" << index[i]  + idStart << "/" << index[i]  + idStart << std::endl;
                    }
                    else
                    {
                        outfile << "f " << index[i1] + idStart << "//" << index[i1] + idStart << " " << index[i2] + idStart
                                << "//" << index[i2] + idStart << " "  << index[i]  + idStart << "//" << index[i] + idStart << std::endl;
                    }
                }
                else
                {
                    if (tcoords)
                    {
                        outfile << "f " << index[i1] + idStart << "/" << index[i1] + idStart << " " << index[i2] + idStart
                                << "/" << index[i2] + idStart << " "  << index[i]  + idStart << "/" << index[i]  + idStart << std::endl;
                    }
                    else
                        outfile << "f " << index[i1] + idStart << " " << index[i2] + idStart << " " << index[i] + idStart << std::endl;
                }
            } /* for (int i = 2; i < npts; ++i) */
        }
    } /* if (input->GetNumberOfStrips() > 0) */

    vtkDebugMacro(<<"Closing vtk file\n");
    delete outfilep;

    // Delete the file if an error occurred
    if (this->ErrorCode == vtkErrorCode::OutOfDiskSpaceError)
    {
        vtkErrorMacro("Ran out of disk space; deleting file: " << this->FileName);
        unlink(this->FileName);
    }
}

void cv::viz::vtkOBJWriter::PrintSelf(ostream& os, vtkIndent indent)
{
    Superclass::PrintSelf(os, indent);
    os << indent << "DecimalPrecision: " << DecimalPrecision << "\n";
}

int cv::viz::vtkOBJWriter::FillInputPortInformation(int, vtkInformation *info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData");
    return 1;
}

vtkPolyData* cv::viz::vtkOBJWriter::GetInput()
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput());
}

vtkPolyData* cv::viz::vtkOBJWriter::GetInput(int port)
{
    return vtkPolyData::SafeDownCast(this->Superclass::GetInput(port));
}
