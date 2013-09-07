#pragma once

#include <map>
#include <ctime>
#include <list>
#include <vector>

#if defined __GNUC__
#pragma GCC system_header
#ifdef __DEPRECATED
#undef __DEPRECATED
#define __DEPRECATED_DISABLED__
#endif
#endif

#include <vtkAppendPolyData.h>
#include <vtkAssemblyPath.h>
#include <vtkCellData.h>
#include <vtkLineSource.h>
#include <vtkPropPicker.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>
#include <vtkPolygon.h>
#include <vtkPointPicker.h>
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
#include <vtkImageViewer.h>
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

#if defined __GNUC__ && defined __DEPRECATED_DISABLED__
#define __DEPRECATED
#undef __DEPRECATED_DISABLED__
#endif

namespace cv
{
    namespace viz
    {
        typedef std::map<std::string, vtkSmartPointer<vtkProp> > WidgetActorMap;
    }
}

#include "viz3d_impl.hpp"
#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/types.hpp>
#include "opencv2/viz/widget_accessor.hpp"
