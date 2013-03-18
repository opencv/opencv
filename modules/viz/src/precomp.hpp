#pragma once

#include <opencv2/core.hpp>
#include <map>
#include <vector>


#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/signals2.hpp>
#include <boost/thread.hpp>

#include <Eigen/Geometry>

#if defined __GNUC__
#pragma GCC system_header
#ifdef __DEPRECATED
#undef __DEPRECATED
#define __DEPRECATED_DISABLED__
#endif
#endif

#include <vtkAppendPolyData.h>
#include <vtkAssemblyPath.h>
#include <vtkAxesActor.h>
#include <vtkActor.h>
#include <vtkBoxRepresentation.h>
#include <vtkBoxWidget.h>
#include <vtkBoxWidget2.h>
#include <vtkCellData.h>
#include <vtkMath.h>
#include <vtkLoopSubdivisionFilter.h>
#include <vtkLineSource.h>
#include <vtkLegendScaleActor.h>
#include <vtkLightKit.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkPropPicker.h>
#include <vtkGeneralTransform.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkExecutive.h>
#include <vtkPolygon.h>
#include <vtkPointPicker.h>
#include <vtkUnstructuredGrid.h>
#include <vtkConeSource.h>
#include <vtkDiskSource.h>
#include <vtkPlaneSource.h>
#include <vtkSphereSource.h>
#include <vtkIdentityTransform.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTubeFilter.h>
#include <vtkCubeSource.h>
#include <vtkAxes.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkCellLocator.h>
#include <vtkPLYReader.h>
#include <vtkTransformFilter.h>
#include <vtkPolyLine.h>
#include <vtkVectorText.h>
#include <vtkFollower.h>
#include <vtkCallbackCommand.h>
#include <vtkInteractorStyle.h>
#include <vtkInformationVector.h>
#include <vtkDataArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPoints.h>
#include <vtkRendererCollection.h>
#include <vtkPNGWriter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkProperty.h>
#include <vtkCamera.h>
#include <vtkObjectFactory.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarsToColors.h>
#include <vtkClipPolyData.h>
#include <vtkPlanes.h>
#include <vtkImageImport.h>
#include <vtkImageViewer.h>
#include <vtkInteractorStyleImage.h>
#include <vtkImageFlip.h>
#include <vtkTIFFWriter.h>
#include <vtkBMPWriter.h>
#include <vtkJPEGWriter.h>
#include <vtkImageViewer2.h>
#include <vtkRenderWindow.h>
#include <vtkXYPlotActor.h>
#include <vtkTextProperty.h>
#include <vtkProperty2D.h>
#include <vtkFieldData.h>
#include <vtkDoubleArray.h>
#include <vtkLODActor.h>
#include <vtkPolyDataWriter.h>
#include <vtkTextActor.h>
#include <vtkCleanPolyData.h>
#include <vtkRenderer.h>
#include <vtkObject.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkImageReslice.h>
#include <vtkImageChangeInformation.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkImageBlend.h>
#include <vtkImageStencilData.h>

#include <vtkRenderWindowInteractor.h>
#include <vtkChartXY.h>
#include <vtkPlot.h>
#include <vtkTable.h>
#include <vtkContextView.h>
#include <vtkContextScene.h>
#include <vtkColorSeries.h>
#include <vtkAxis.h>
#include <vtkSelection.h>
#include <vtkHardwareSelector.h>
#include <vtkTriangle.h>
#include <vtkWorldPointPicker.h>
#include <vtkInteractorStyleRubberBandPick.h>
#include <vtkInteractorStyleTrackballActor.h>
#include <vtkAreaPicker.h>
#include <vtkExtractGeometry.h>
#include <vtkExtractPolyDataGeometry.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkIdFilter.h>
#include <vtkIdTypeArray.h>
#include <vtkImageReader2Factory.h>
#include <vtkImageReader2.h>
#include <vtkImageData.h>



#include <vtkPolyDataNormals.h>
#include <vtkMapper.h>
#include <vtkSelectionNode.h>



#include <vtkAbstractPicker.h>
#include <vtkAbstractPropPicker.h>
#include <vtkPointPicker.h>
#include <vtkMatrix4x4.h>
#include <vtkInteractorObserver.h>



#if defined __GNUC__ && defined __DEPRECATED_DISABLED__
#define __DEPRECATED
#undef __DEPRECATED_DISABLED__
#endif
