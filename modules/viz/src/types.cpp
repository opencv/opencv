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

#include "precomp.hpp"

////////////////////////////////////////////////////////////////////
/// Events

cv::viz::KeyboardEvent::KeyboardEvent(Action _action, const String& _symbol, unsigned char _code, int _modifiers)
  : action(_action), symbol(_symbol), code(_code), modifiers(_modifiers) {}

cv::viz::MouseEvent::MouseEvent(const Type& _type, const MouseButton& _button, const Point& _pointer, int _modifiers)
    : type(_type), button(_button), pointer(_pointer), modifiers(_modifiers) {}

////////////////////////////////////////////////////////////////////
/// cv::viz::Mesh3d

namespace cv { namespace viz { namespace
{
    struct MeshUtils
    {
        static Mesh3d loadMesh(const String &file)
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
}}}

cv::viz::Mesh3d cv::viz::Mesh3d::load(const String& file)
{
    return MeshUtils::loadMesh(file);
}

////////////////////////////////////////////////////////////////////
/// Camera implementation

cv::viz::Camera::Camera(double fx, double fy, double cx, double cy, const Size &window_size)
{
    init(fx, fy, cx, cy, window_size);
}

cv::viz::Camera::Camera(const Vec2d &fov, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01)); // Default clipping
    setFov(fov);
    window_size_ = window_size;
    // Principal point at the center
    principal_point_ = Vec2f(static_cast<float>(window_size.width)*0.5f, static_cast<float>(window_size.height)*0.5f);
    focal_ = Vec2f(principal_point_[0] / tan(fov_[0]*0.5f), principal_point_[1] / tan(fov_[1]*0.5f));
}

cv::viz::Camera::Camera(const cv::Matx33d & K, const Size &window_size)
{
    double f_x = K(0,0);
    double f_y = K(1,1);
    double c_x = K(0,2);
    double c_y = K(1,2);
    init(f_x, f_y, c_x, c_y, window_size);
}

cv::viz::Camera::Camera(const Matx44d &proj, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);

    double near = proj(2,3) / (proj(2,2) - 1.0);
    double far = near * (proj(2,2) - 1.0) / (proj(2,2) + 1.0);
    double left = near * (proj(0,2)-1) / proj(0,0);
    double right = 2.0 * near / proj(0,0) + left;
    double bottom = near * (proj(1,2)-1) / proj(1,1);
    double top = 2.0 * near / proj(1,1) + bottom;

    double epsilon = 2.2204460492503131e-16;

    principal_point_[0] = fabs(left-right) < epsilon ? window_size.width  * 0.5 : (left * window_size.width) / (left - right);
    principal_point_[1] = fabs(top-bottom) < epsilon ? window_size.height * 0.5 : (top * window_size.height) / (top - bottom);

    focal_[0] = -near * principal_point_[0] / left;
    focal_[1] =  near * principal_point_[1] / top;

    setClip(Vec2d(near, far));
    fov_[0] = atan2(principal_point_[0], focal_[0]) + atan2(window_size.width-principal_point_[0],  focal_[0]);
    fov_[1] = atan2(principal_point_[1], focal_[1]) + atan2(window_size.height-principal_point_[1], focal_[1]);

    window_size_ = window_size;
}

void cv::viz::Camera::init(double fx, double fy, double cx, double cy, const Size &window_size)
{
    CV_Assert(window_size.width > 0 && window_size.height > 0);
    setClip(Vec2d(0.01, 1000.01));// Default clipping

    fov_[0] = atan2(cx, fx) + atan2(window_size.width  - cx, fx);
    fov_[1] = atan2(cy, fy) + atan2(window_size.height - cy, fy);

    principal_point_[0] = cx;
    principal_point_[1] = cy;

    focal_[0] = fx;
    focal_[1] = fy;

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

void cv::viz::Camera::computeProjectionMatrix(Matx44d &proj) const
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
    Matx33d K(525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0) ;
    return Camera(K, window_size);
}
