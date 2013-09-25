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

#include "precomp.hpp"

cv::Affine3f cv::viz::makeTransformToGlobal(const Vec3f& axis_x, const Vec3f& axis_y, const Vec3f& axis_z, const Vec3f& origin)
{
    Affine3f::Mat3 R;
    R.val[0] = axis_x.val[0];
    R.val[3] = axis_x.val[1];
    R.val[6] = axis_x.val[2];

    R.val[1] = axis_y.val[0];
    R.val[4] = axis_y.val[1];
    R.val[7] = axis_y.val[2];

    R.val[2] = axis_z.val[0];
    R.val[5] = axis_z.val[1];
    R.val[8] = axis_z.val[2];

    return Affine3f(R, origin);
}

cv::Affine3f cv::viz::makeCameraPose(const Vec3f& position, const Vec3f& focal_point, const Vec3f& y_dir)
{
    // Compute the transformation matrix for drawing the camera frame in a scene
    Vec3f n = normalize(focal_point - position);
    Vec3f u = normalize(y_dir.cross(n));
    Vec3f v = n.cross(u);

    Matx44f pose_mat = Matx44f::zeros();
    pose_mat(0,0) = u[0];
    pose_mat(0,1) = u[1];
    pose_mat(0,2) = u[2];
    pose_mat(1,0) = v[0];
    pose_mat(1,1) = v[1];
    pose_mat(1,2) = v[2];
    pose_mat(2,0) = n[0];
    pose_mat(2,1) = n[1];
    pose_mat(2,2) = n[2];
    pose_mat(3,0) = position[0];
    pose_mat(3,1) = position[1];
    pose_mat(3,2) = position[2];
    pose_mat(3,3) = 1.0f;
    pose_mat = pose_mat.t();
    return pose_mat;
}

vtkSmartPointer<vtkMatrix4x4> cv::viz::convertToVtkMatrix(const cv::Matx44f &m)
{
    vtkSmartPointer<vtkMatrix4x4> vtk_matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            vtk_matrix->SetElement(i, k, m(i, k));
    return vtk_matrix;
}

cv::Matx44f cv::viz::convertToMatx(const vtkSmartPointer<vtkMatrix4x4>& vtk_matrix)
{
    cv::Matx44f m;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
            m(i, k) = vtk_matrix->GetElement(i, k);
    return m;
}

namespace cv
{
    namespace viz
    {
        template<typename _Tp> Vec<_Tp, 3>* vtkpoints_data(vtkSmartPointer<vtkPoints>& points);

        template<> Vec3f* vtkpoints_data<float>(vtkSmartPointer<vtkPoints>& points)
        {
            CV_Assert(points->GetDataType() == VTK_FLOAT);
            vtkDataArray *data = points->GetData();
            float *pointer = static_cast<vtkFloatArray*>(data)->GetPointer(0);
            return reinterpret_cast<Vec3f*>(pointer);
        }

        template<> Vec3d* vtkpoints_data<double>(vtkSmartPointer<vtkPoints>& points)
        {
            CV_Assert(points->GetDataType() == VTK_DOUBLE);
            vtkDataArray *data = points->GetData();
            double *pointer = static_cast<vtkDoubleArray*>(data)->GetPointer(0);
            return reinterpret_cast<Vec3d*>(pointer);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Viz accessor implementation

cv::viz::VizAccessor * cv::viz::VizAccessor::instance_ = 0;
bool cv::viz::VizAccessor::is_instantiated_ = false;
cv::viz::VizAccessor::VizAccessorImpl * cv::viz::VizAccessor::impl_ = 0;

struct cv::viz::VizAccessor::VizAccessorImpl
{
    cv::viz::VizMap viz_map;
};

cv::viz::VizAccessor::VizAccessor() { impl_ = new cv::viz::VizAccessor::VizAccessorImpl;}

cv::viz::VizAccessor::~VizAccessor()
{
    if(impl_)
    {
        delete impl_;
        impl_ = 0;
    }
}

cv::viz::VizAccessor & cv::viz::VizAccessor::getInstance()
{
    if (!is_instantiated_)
    {
        instance_ = new VizAccessor();
        is_instantiated_ = true;
    }
    return *instance_;
}

void cv::viz::VizAccessor::release()
{
    if (is_instantiated_)
    {
        delete instance_;
        instance_ = 0;
        is_instantiated_ = false;
    }
}

cv::viz::Viz3d cv::viz::VizAccessor::get(const String & window_name)
{
    // Add the prefix Viz
    String name;
    generateWindowName(window_name, name);

    VizMap::iterator vm_itr = impl_->viz_map.find(name);
    bool exists = vm_itr != impl_->viz_map.end();
    if (exists) return vm_itr->second;
    else return Viz3d(window_name);
}

void cv::viz::VizAccessor::add(Viz3d window)
{
    String window_name = window.getWindowName();
    VizMap::iterator vm_itr = impl_->viz_map.find(window_name);
    bool exists = vm_itr != impl_->viz_map.end();
    if (exists) return ;
    impl_->viz_map.insert(VizPair(window_name, window));
}

void cv::viz::VizAccessor::remove(const String &window_name)
{
    // Add the prefix Viz
    String name;
    generateWindowName(window_name, name);

    VizMap::iterator vm_itr = impl_->viz_map.find(name);
    bool exists = vm_itr != impl_->viz_map.end();
    if (!exists) return ;
    impl_->viz_map.erase(vm_itr);
}

void cv::viz::VizAccessor::generateWindowName(const String &window_name, String &output)
{
    output = "Viz";
    // Already is Viz
    if (window_name == output) return;

    String prefixed = output + " - ";
    if (window_name.substr(0, prefixed.length()) == prefixed) output = window_name; // Already has "Viz - "
    else if (window_name.substr(0, output.length()) == output) output = prefixed + window_name; // Doesn't have prefix
    else output = (window_name == "" ? output : prefixed + window_name);
}

cv::viz::Viz3d cv::viz::get(const String &window_name)
{
    return cv::viz::VizAccessor::getInstance().get(window_name);
}
