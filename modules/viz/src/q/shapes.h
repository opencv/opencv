#pragma once

#include <Eigen/Core>
#include <opencv2/viz/types.hpp>
#include "precomp.hpp"

namespace temp_viz
{
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createLine (const cv::Point3f& pt1, const cv::Point3f& pt2);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createSphere (const cv::Point3f &center, float radius, int sphere_resolution = 10);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCylinder (const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs, const Point3f& pt);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> create2DCircle (const Point3f& pt, double radius);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createCube(const Point3f& pt_min, const Point3f& pt_max);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createSphere (const Point3f& pt, double radius);
    CV_EXPORTS vtkSmartPointer<vtkDataSet> createArrow (const Point3f& pt1, const Point3f& pt2);
    /** \brief Allocate a new unstructured grid smartpointer. For internal use only.
      * \param[out] polydata the resultant unstructured grid. 
      */
    CV_EXPORTS void allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata);
}
