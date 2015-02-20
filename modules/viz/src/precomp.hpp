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
//M*/

#ifndef __OPENCV_VIZ_PRECOMP_HPP__
#define __OPENCV_VIZ_PRECOMP_HPP__

#include <map>
#include <ctime>
#include <list>
#include <vector>
#include <iomanip>
#include <limits>

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
#include <vtkPlanes.h>
#include <vtkImageFlip.h>
#include <vtkRenderWindow.h>
#include <vtkTextProperty.h>
#include <vtkProperty2D.h>
#include <vtkLODActor.h>
#include <vtkActor.h>
#include <vtkTextActor.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkMath.h>
#include <vtkExtractEdges.h>
#include <vtkFrustumSource.h>
#include <vtkTextureMapToPlane.h>
#include <vtkPolyDataNormals.h>
#include <vtkAlgorithmOutput.h>
#include <vtkImageMapper.h>
#include <vtkPoints.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkObjectFactory.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkMergeFilter.h>
#include <vtkErrorCode.h>
#include <vtkPLYWriter.h>
#include <vtkSTLWriter.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkSTLReader.h>
#include <vtkPNGReader.h>
#include <vtkOBJExporter.h>
#include <vtkVRMLExporter.h>
#include <vtkTensorGlyph.h>
#include <vtkImageAlgorithm.h>
#include <vtkTransformFilter.h>
#include <vtkConeSource.h>
#include <vtkElevationFilter.h>
#include <vtkColorTransferFunction.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include "vtkCallbackCommand.h"

#if !defined(_WIN32) || defined(__CYGWIN__)
# include <unistd.h> /* unlink */
#else
# include <io.h> /* unlink */
#endif

#include <vtk/vtkOBJWriter.h>
#include <vtk/vtkXYZWriter.h>
#include <vtk/vtkXYZReader.h>
#include <vtk/vtkCloudMatSink.h>
#include <vtk/vtkCloudMatSource.h>
#include <vtk/vtkTrajectorySource.h>
#include <vtk/vtkImageMatSource.h>


#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>
#include <opencv2/core/utility.hpp>


namespace cv
{
    namespace viz
    {
        typedef std::map<String, vtkSmartPointer<vtkProp> > WidgetActorMap;

        struct VizMap
        {
            typedef std::map<String, Viz3d> type;
            typedef type::iterator iterator;

            type m;
            ~VizMap();
            void replace_clear();
        };

        class VizStorage
        {
        public:
            static void unregisterAll();

            //! window names automatically have Viz - prefix even though not provided by the users
            static String generateWindowName(const String &window_name);

        private:
            VizStorage(); // Static

            static void add(const Viz3d& window);
            static Viz3d& get(const String &window_name);
            static void remove(const String &window_name);
            static bool windowExists(const String &window_name);
            static void removeUnreferenced();

            static VizMap storage;
            friend class Viz3d;

            static VizStorage init;
        };

        template<typename _Tp> inline _Tp normalized(const _Tp& v) { return v * 1/norm(v); }

        template<typename _Tp> inline bool isNan(const _Tp* data)
        {
            return isNan(data[0]) || isNan(data[1]) || isNan(data[2]);
        }

        inline vtkSmartPointer<vtkActor> getActor(const Widget3D& widget)
        {
            return vtkActor::SafeDownCast(WidgetAccessor::getProp(widget));
        }

        inline vtkSmartPointer<vtkPolyData> getPolyData(const Widget3D& widget)
        {
            vtkSmartPointer<vtkMapper> mapper = getActor(widget)->GetMapper();
            return vtkPolyData::SafeDownCast(mapper->GetInput());
        }

        inline vtkSmartPointer<vtkMatrix4x4> vtkmatrix(const cv::Matx44d &matrix)
        {
            vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
            vtk_matrix->DeepCopy(matrix.val);
            return vtk_matrix;
        }

        inline Color vtkcolor(const Color& color)
        {
            Color scaled_color = color * (1.0/255.0);
            std::swap(scaled_color[0], scaled_color[2]);
            return scaled_color;
        }

        inline Vec3d get_random_vec(double from = -10.0, double to = 10.0)
        {
            RNG& rng = theRNG();
            return Vec3d(rng.uniform(from, to), rng.uniform(from, to), rng.uniform(from, to));
        }

        struct VtkUtils
        {
            template<class Filter>
            static void SetInputData(vtkSmartPointer<Filter> filter, vtkPolyData* polydata)
            {
            #if VTK_MAJOR_VERSION <= 5
                filter->SetInput(polydata);
            #else
                filter->SetInputData(polydata);
            #endif
            }
            template<class Filter>
            static void SetSourceData(vtkSmartPointer<Filter> filter, vtkPolyData* polydata)
            {
            #if VTK_MAJOR_VERSION <= 5
                filter->SetSource(polydata);
            #else
                filter->SetSourceData(polydata);
            #endif
            }

            template<class Filter>
            static void SetInputData(vtkSmartPointer<Filter> filter, vtkImageData* polydata)
            {
            #if VTK_MAJOR_VERSION <= 5
                filter->SetInput(polydata);
            #else
                filter->SetInputData(polydata);
            #endif
            }

            template<class Filter>
            static void AddInputData(vtkSmartPointer<Filter> filter, vtkPolyData *polydata)
            {
            #if VTK_MAJOR_VERSION <= 5
                filter->AddInput(polydata);
            #else
                filter->AddInputData(polydata);
            #endif
            }

            static vtkSmartPointer<vtkUnsignedCharArray> FillScalars(size_t size, const Color& color)
            {
                Vec3b rgb = Vec3d(color[2], color[1], color[0]);
                Vec3b* color_data = new Vec3b[size];
                std::fill(color_data, color_data + size, rgb);

                vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
                scalars->SetName("Colors");
                scalars->SetNumberOfComponents(3);
                scalars->SetNumberOfTuples((vtkIdType)size);
                scalars->SetArray(color_data->val, (vtkIdType)(size * 3), 0);
                return scalars;
            }

            static vtkSmartPointer<vtkPolyData> FillScalars(vtkSmartPointer<vtkPolyData> polydata, const Color& color)
            {
                return polydata->GetPointData()->SetScalars(FillScalars(polydata->GetNumberOfPoints(), color)), polydata;
            }

            static vtkSmartPointer<vtkPolyData> ComputeNormals(vtkSmartPointer<vtkPolyData> polydata)
            {
                vtkSmartPointer<vtkPolyDataNormals> normals_generator = vtkSmartPointer<vtkPolyDataNormals>::New();
                normals_generator->ComputePointNormalsOn();
                normals_generator->ComputeCellNormalsOff();
                normals_generator->SetFeatureAngle(0.1);
                normals_generator->SetSplitting(0);
                normals_generator->SetConsistency(1);
                normals_generator->SetAutoOrientNormals(0);
                normals_generator->SetFlipNormals(0);
                normals_generator->SetNonManifoldTraversal(1);
                VtkUtils::SetInputData(normals_generator, polydata);
                normals_generator->Update();
                return normals_generator->GetOutput();
            }

            static vtkSmartPointer<vtkPolyData> TransformPolydata(vtkSmartPointer<vtkAlgorithmOutput> algorithm_output_port, const Affine3d& pose)
            {
                vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
                transform->SetMatrix(vtkmatrix(pose.matrix));

                vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                transform_filter->SetTransform(transform);
                transform_filter->SetInputConnection(algorithm_output_port);
                transform_filter->Update();
                return transform_filter->GetOutput();
            }

            static vtkSmartPointer<vtkPolyData> TransformPolydata(vtkSmartPointer<vtkPolyData> polydata, const Affine3d& pose)
            {
                vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
                transform->SetMatrix(vtkmatrix(pose.matrix));

                vtkSmartPointer<vtkTransformPolyDataFilter> transform_filter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
                VtkUtils::SetInputData(transform_filter, polydata);
                transform_filter->SetTransform(transform);
                transform_filter->Update();
                return transform_filter->GetOutput();
            }
        };

        vtkSmartPointer<vtkRenderWindowInteractor> vtkCocoaRenderWindowInteractorNew();
    }
}

#include "vtk/vtkVizInteractorStyle.hpp"
#include "vizimpl.hpp"

#endif
