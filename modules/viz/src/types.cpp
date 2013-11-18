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

////////////////////////////////////////////////////////////////////
/// cv::viz::KeyboardEvent

cv::viz::KeyboardEvent::KeyboardEvent(bool _action, const String& _key_sym, unsigned char key, bool alt, bool ctrl, bool shift)
  : action_(_action), modifiers_(0), key_code_(key), key_sym_(_key_sym)
{
  if (alt)
    modifiers_ = Alt;

  if (ctrl)
    modifiers_ |= Ctrl;

  if (shift)
    modifiers_ |= Shift;
}

bool cv::viz::KeyboardEvent::isAltPressed() const { return (modifiers_ & Alt) != 0; }
bool cv::viz::KeyboardEvent::isCtrlPressed() const { return (modifiers_ & Ctrl) != 0; }
bool cv::viz::KeyboardEvent::isShiftPressed() const { return (modifiers_ & Shift) != 0; }
unsigned char cv::viz::KeyboardEvent::getKeyCode() const { return key_code_; }
const cv::String& cv::viz::KeyboardEvent::getKeySym() const { return key_sym_; }
bool cv::viz::KeyboardEvent::keyDown() const { return action_; }
bool cv::viz::KeyboardEvent::keyUp() const { return !action_; }

////////////////////////////////////////////////////////////////////
/// cv::viz::MouseEvent

cv::viz::MouseEvent::MouseEvent(const Type& _type, const MouseButton& _button, const Point& _p,  bool alt, bool ctrl, bool shift)
    : type(_type), button(_button), pointer(_p), key_state(0)
{
    if (alt)
        key_state = KeyboardEvent::Alt;

    if (ctrl)
        key_state |= KeyboardEvent::Ctrl;

    if (shift)
        key_state |= KeyboardEvent::Shift;
}

////////////////////////////////////////////////////////////////////
/// cv::viz::Mesh3d

struct cv::viz::Mesh3d::loadMeshImpl
{
    static cv::viz::Mesh3d loadMesh(const String &file)
    {
        Mesh3d mesh;

        vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
        reader->SetFileName(file.c_str());
        reader->Update();

        vtkSmartPointer<vtkPolyData> poly_data = reader->GetOutput();
        CV_Assert("File does not exist or file format is not supported." && poly_data);

        vtkSmartPointer<vtkPoints> mesh_points = poly_data->GetPoints();
        vtkIdType nr_points = mesh_points->GetNumberOfPoints();

        mesh.cloud.create(1, nr_points, CV_32FC3);

        Vec3f *mesh_cloud = mesh.cloud.ptr<Vec3f>();
        for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints(); i++)
        {
            Vec3d point;
            mesh_points->GetPoint(i, point.val);
            mesh_cloud[i] = point;
        }

        // Then the color information, if any
        vtkUnsignedCharArray* poly_colors = 0;
        if (poly_data->GetPointData())
            poly_colors = vtkUnsignedCharArray::SafeDownCast(poly_data->GetPointData()->GetScalars());

        if (poly_colors && (poly_colors->GetNumberOfComponents() == 3))
        {
            mesh.colors.create(1, nr_points, CV_8UC3);
            Vec3b *mesh_colors = mesh.colors.ptr<cv::Vec3b>();

            for (vtkIdType i = 0; i < mesh_points->GetNumberOfPoints(); i++)
            {
                Vec3b point_color;
                poly_colors->GetTupleValue(i, point_color.val);

                std::swap(point_color[0], point_color[2]); // RGB -> BGR
                mesh_colors[i] = point_color;
            }
        }
        else
            mesh.colors.release();

        // Now handle the polygons
        vtkIdType* cell_points;
        vtkIdType nr_cell_points;
        vtkCellArray * mesh_polygons = poly_data->GetPolys();
        mesh_polygons->InitTraversal();

        mesh.polygons.create(1, mesh_polygons->GetSize(), CV_32SC1);

        int* polygons = mesh.polygons.ptr<int>();
        while (mesh_polygons->GetNextCell(nr_cell_points, cell_points))
        {
            *polygons++ = nr_cell_points;
            for (int i = 0; i < nr_cell_points; ++i)
                *polygons++ = static_cast<int>(cell_points[i]);
        }

        return mesh;
    }
};

cv::viz::Mesh3d cv::viz::Mesh3d::loadMesh(const String& file)
{
    return loadMeshImpl::loadMesh(file);
}

////////////////////////////////////////////////////////////////////
/// Camera implementation

cv::viz::Camera::Camera(float f_x, float f_y, float c_x, float c_y, const Size &window_size)
{
    init(f_x, f_y, c_x, c_y, window_size);
}

cv::viz::Camera::Camera(const Vec2f &fov, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01)); // Default clipping
    setFov(fov);
    window_size_ = window_size;
    // Principal point at the center
    principal_point_ = Vec2f(static_cast<float>(window_size.width)*0.5f, static_cast<float>(window_size.height)*0.5f);
    focal_ = Vec2f(principal_point_[0] / tan(fov_[0]*0.5f), principal_point_[1] / tan(fov_[1]*0.5f));
}

cv::viz::Camera::Camera(const cv::Matx33f & K, const Size &window_size)
{
    float f_x = K(0,0);
    float f_y = K(1,1);
    float c_x = K(0,2);
    float c_y = K(1,2);
    init(f_x, f_y, c_x, c_y, window_size);
}

cv::viz::Camera::Camera(const Matx44f &proj, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);

    double near = proj(2,3) / (proj(2,2) - 1.0);
    double far = near * (proj(2,2) - 1.0) / (proj(2,2) + 1.0);
    double left = near * (proj(0,2)-1) / proj(0,0);
    double right = 2.0 * near / proj(0,0) + left;
    double bottom = near * (proj(1,2)-1) / proj(1,1);
    double top = 2.0 * near / proj(1,1) + bottom;

    double epsilon = 2.2204460492503131e-16;

    if (fabs(left-right) < epsilon) principal_point_[0] = static_cast<float>(window_size.width) * 0.5f;
    else principal_point_[0] = (left * static_cast<float>(window_size.width)) / (left - right);
    focal_[0] = -near * principal_point_[0] / left;

    if (fabs(top-bottom) < epsilon) principal_point_[1] = static_cast<float>(window_size.height) * 0.5f;
    else principal_point_[1] = (top * static_cast<float>(window_size.height)) / (top - bottom);
    focal_[1] = near * principal_point_[1] / top;

    setClip(Vec2d(near, far));
    fov_[0] = (atan2(principal_point_[0],focal_[0]) + atan2(window_size.width-principal_point_[0],focal_[0]));
    fov_[1] = (atan2(principal_point_[1],focal_[1]) + atan2(window_size.height-principal_point_[1],focal_[1]));

    window_size_ = window_size;
}

void cv::viz::Camera::init(float f_x, float f_y, float c_x, float c_y, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01));// Default clipping

    fov_[0] = (atan2(c_x,f_x) + atan2(window_size.width-c_x,f_x));
    fov_[1] = (atan2(c_y,f_y) + atan2(window_size.height-c_y,f_y));

    principal_point_[0] = c_x;
    principal_point_[1] = c_y;

    focal_[0] = f_x;
    focal_[1] = f_y;

    window_size_ = window_size;
}

void cv::viz::Camera::setWindowSize(const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);

    // Get the scale factor and update the principal points
    float scalex = static_cast<float>(window_size.width) / static_cast<float>(window_size_.width);
    float scaley = static_cast<float>(window_size.height) / static_cast<float>(window_size_.height);

    principal_point_[0] *= scalex;
    principal_point_[1] *= scaley;
    focal_ *= scaley;
    // Vertical field of view is fixed!  Update horizontal field of view
    fov_[0] = (atan2(principal_point_[0],focal_[0]) + atan2(window_size.width-principal_point_[0],focal_[0]));

    window_size_ = window_size;
}

void cv::viz::Camera::computeProjectionMatrix(Matx44f &proj) const
{
    double top = clip_[0] * principal_point_[1] / focal_[1];
    double left = -clip_[0] * principal_point_[0] / focal_[0];
    double right = clip_[0] * (window_size_.width - principal_point_[0]) / focal_[0];
    double bottom = -clip_[0] * (window_size_.height - principal_point_[1]) / focal_[1];

    double temp1 = 2.0 * clip_[0];
    double temp2 = 1.0 / (right - left);
    double temp3 = 1.0 / (top - bottom);
    double temp4 = 1.0 / (clip_[0] - clip_[1]);

    proj = Matx44d::zeros();
    proj(0,0) = temp1 * temp2;
    proj(1,1) = temp1 * temp3;
    proj(0,2) = (right + left) * temp2;
    proj(1,2) = (top + bottom) * temp3;
    proj(2,2) = (clip_[1]+clip_[0]) * temp4;
    proj(3,2) = -1.0;
    proj(2,3) = (temp1 * clip_[1]) * temp4;
}

cv::viz::Camera cv::viz::Camera::KinectCamera(const Size &window_size)
{
    // Without distortion, RGB Camera
    // Received from http://nicolas.burrus.name/index.php/Research/KinectCalibration
    Matx33f K = Matx33f::zeros();
    K(0,0) = 5.2921508098293293e+02;
    K(0,2) = 3.2894272028759258e+02;
    K(1,1) = 5.2556393630057437e+02;
    K(1,2) = 2.6748068171871557e+02;
    K(2,2) = 1.0f;
    return Camera(K, window_size);
}
