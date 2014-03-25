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

#ifndef __vtkCloudMatSource_h
#define __vtkCloudMatSource_h

#include <opencv2/core.hpp>
#include <vtkPolyDataAlgorithm.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>

namespace cv
{
    namespace viz
    {
        class vtkCloudMatSource : public vtkPolyDataAlgorithm
        {
        public:
            static vtkCloudMatSource *New();
            vtkTypeMacro(vtkCloudMatSource,vtkPolyDataAlgorithm)

            virtual int SetCloud(InputArray cloud);
            virtual int SetColorCloud(InputArray cloud, InputArray colors);
            virtual int SetColorCloudNormals(InputArray cloud, InputArray colors, InputArray normals);
            virtual int SetColorCloudNormalsTCoords(InputArray cloud, InputArray colors, InputArray normals, InputArray tcoords);

        protected:
            vtkCloudMatSource();
            ~vtkCloudMatSource();

            int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

            vtkSmartPointer<vtkPoints> points;
            vtkSmartPointer<vtkCellArray> vertices;
            vtkSmartPointer<vtkUnsignedCharArray> scalars;
            vtkSmartPointer<vtkDataArray> normals;
            vtkSmartPointer<vtkDataArray> tcoords;
        private:
            vtkCloudMatSource(const vtkCloudMatSource&);  // Not implemented.
            void operator=(const vtkCloudMatSource&);  // Not implemented.

            template<typename _Tp> int filterNanCopy(const Mat& cloud);
            template<typename _Msk> void filterNanColorsCopy(const Mat& cloud_colors, const Mat& mask, int total);

            template<typename _Tn, typename _Msk>
            void filterNanNormalsCopy(const Mat& cloud_normals, const Mat& mask, int total);

            template<typename _Tn, typename _Msk>
            void filterNanTCoordsCopy(const Mat& tcoords, const Mat& mask, int total);
        };
    }
}

#endif
