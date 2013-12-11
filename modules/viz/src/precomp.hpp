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

#ifndef __OPENCV_VIZ_PRECOMP_HPP__
#define __OPENCV_VIZ_PRECOMP_HPP__

#include <map>
#include <ctime>
#include <list>
#include <vector>

#include <vtkAppendPolyData.h>
#include <vtkAssemblyPath.h>
#include <vtkCellData.h>
#include <vtkLineSource.h>
#include <vtkPropPicker.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>
#include <vtkPolygon.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDiskSource.h>
#include <vtkPlaneSource.h>
#include <vtkSphereSource.h>
#include <vtkArrowSource.h>
#include <vtkOutlineSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTubeFilter.h>
#include <vtkCubeSource.h>
#include <vtkAxes.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkPLYReader.h>
#include <vtkPolyLine.h>
#include <vtkVectorText.h>
#include <vtkFollower.h>
#include <vtkInteractorStyle.h>
#include <vtkUnsignedCharArray.h>
#include <vtkRendererCollection.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkObjectFactory.h>
#include <vtkPlanes.h>
#include <vtkImageFlip.h>
#include <vtkRenderWindow.h>
#include <vtkTextProperty.h>
#include <vtkProperty2D.h>
#include <vtkLODActor.h>
#include <vtkTextActor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkMath.h>
#include <vtkExtractEdges.h>
#include <vtkFrustumSource.h>
#include <vtkTextureMapToPlane.h>
#include <vtkPolyDataNormals.h>
#include <vtkAlgorithmOutput.h>
#include <vtkImageMapper.h>

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#include <opencv2/core/utility.hpp>

namespace cv
{
    namespace viz
    {
        typedef std::map<String, vtkSmartPointer<vtkProp> > WidgetActorMap;
        typedef std::map<String, Viz3d> VizMap;

        class VizStorage
        {
        public:
            static void unregisterAll();

            //! window names automatically have Viz - prefix even though not provided by the users
            static String generateWindowName(const String &window_name);

        private:
            VizStorage(); // Static
            ~VizStorage();

            static void add(const Viz3d& window);
            static Viz3d& get(const String &window_name);
            static void remove(const String &window_name);
            static bool windowExists(const String &window_name);
            static void removeUnreferenced();

            static VizMap storage;
            friend class Viz3d;
        };
    }
}

#include "interactor_style.hpp"
#include "viz3d_impl.hpp"


#endif
