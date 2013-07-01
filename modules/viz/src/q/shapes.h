#pragma once

#include <opencv2/viz/types.hpp>

namespace temp_viz
{
    vtkSmartPointer<vtkDataSet> createLine (const cv::Point3f& pt1, const cv::Point3f& pt2);
    vtkSmartPointer<vtkDataSet> createSphere (const cv::Point3f &center, float radius, int sphere_resolution = 10);
    vtkSmartPointer<vtkDataSet> createCylinder (const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30);
    vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs);
    vtkSmartPointer<vtkDataSet> createPlane (const Vec4f& coefs, const Point3f& pt);
    vtkSmartPointer<vtkDataSet> create2DCircle (const Point3f& pt, double radius);
    vtkSmartPointer<vtkDataSet> createCube(const Point3f& pt_min, const Point3f& pt_max);
    vtkSmartPointer<vtkDataSet> createSphere (const Point3f& pt, double radius);
    vtkSmartPointer<vtkDataSet> createArrow (const Point3f& pt1, const Point3f& pt2);

    //brief Allocate a new unstructured grid smartpointer. For internal use only.
    void allocVtkUnstructuredGrid (vtkSmartPointer<vtkUnstructuredGrid> &polydata);
}
