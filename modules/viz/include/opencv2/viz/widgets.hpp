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

#ifndef __OPENCV_VIZ_WIDGETS_HPP__
#define __OPENCV_VIZ_WIDGETS_HPP__

#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
    {
        /////////////////////////////////////////////////////////////////////////////
        /// Widget rendering properties
        enum RenderingProperties
        {
            POINT_SIZE,
            OPACITY,
            LINE_WIDTH,
            FONT_SIZE,
            REPRESENTATION,
            IMMEDIATE_RENDERING,
            SHADING
        };

        enum RepresentationValues
        {
            REPRESENTATION_POINTS,
            REPRESENTATION_WIREFRAME,
            REPRESENTATION_SURFACE
        };

        enum ShadingValues
        {
            SHADING_FLAT,
            SHADING_GOURAUD,
            SHADING_PHONG
        };

        /////////////////////////////////////////////////////////////////////////////
        /// The base class for all widgets
        class CV_EXPORTS Widget
        {
        public:
            Widget();
            Widget(const Widget& other);
            Widget& operator=(const Widget& other);
            ~Widget();

            //! Create a widget directly from ply file
            static Widget fromPlyFile(const String &file_name);

            //! Rendering properties of this particular widget
            void setRenderingProperty(int property, double value);
            double getRenderingProperty(int property) const;

            //! Casting between widgets
            template<typename _W> _W cast();
        private:
            class Impl;
            Impl *impl_;
            friend struct WidgetAccessor;
        };

        /////////////////////////////////////////////////////////////////////////////
        /// The base class for all 3D widgets
        class CV_EXPORTS Widget3D : public Widget
        {
        public:
            Widget3D() {}

            //! widget position manipulation, i.e. place where it is rendered
            void setPose(const Affine3d &pose);
            void updatePose(const Affine3d &pose);
            Affine3d getPose() const;

            //! update internal widget data, i.e. points, normals, etc.
            void applyTransform(const Affine3d &transform);

            void setColor(const Color &color);

        };

        /////////////////////////////////////////////////////////////////////////////
        /// The base class for all 2D widgets
        class CV_EXPORTS Widget2D : public Widget
        {
        public:
            Widget2D() {}

            void setColor(const Color &color);
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Simple widgets

        class CV_EXPORTS WLine : public Widget3D
        {
        public:
            WLine(const Point3d &pt1, const Point3d &pt2, const Color &color = Color::white());
        };

        class CV_EXPORTS WPlane : public Widget3D
        {
        public:
            //! created default plane with center point at origin and normal oriented along z-axis
            WPlane(const Size2d& size = Size2d(1.0, 1.0), const Color &color = Color::white());

            //! repositioned plane
            WPlane(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
                   const Size2d& size = Size2d(1.0, 1.0), const Color &color = Color::white());
        };

        class CV_EXPORTS WSphere : public Widget3D
        {
        public:
            WSphere(const cv::Point3d &center, double radius, int sphere_resolution = 10, const Color &color = Color::white());
        };

        class CV_EXPORTS WArrow : public Widget3D
        {
        public:
            WArrow(const Point3d& pt1, const Point3d& pt2, double thickness = 0.03, const Color &color = Color::white());
        };

        class CV_EXPORTS WCircle : public Widget3D
        {
        public:
            //! creates default planar circle centred at origin with plane normal along z-axis
            WCircle(double radius, double thickness = 0.01, const Color &color = Color::white());

            //! creates repositioned circle
            WCircle(double radius, const Point3d& center, const Vec3d& normal, double thickness = 0.01, const Color &color = Color::white());
        };

        class CV_EXPORTS WCone : public Widget3D
        {
        public:
            //! create default cone, oriented along x-axis with center of its base located at origin
            WCone(double length, double radius, int resolution = 6.0, const Color &color = Color::white());

            //! creates repositioned cone
            WCone(double radius, const Point3d& center, const Point3d& tip, int resolution = 6.0, const Color &color = Color::white());
        };

        class CV_EXPORTS WCylinder : public Widget3D
        {
        public:
            WCylinder(const Point3d& axis_point1, const Point3d& axis_point2, double radius, int numsides = 30, const Color &color = Color::white());
        };

        class CV_EXPORTS WCube : public Widget3D
        {
        public:
            WCube(const Point3d& min_point = Vec3d::all(-0.5), const Point3d& max_point = Vec3d::all(0.5),
                  bool wire_frame = true, const Color &color = Color::white());
        };

        class CV_EXPORTS WPolyLine : public Widget3D
        {
        public:
            WPolyLine(InputArray points, const Color &color = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Text and image widgets

        class CV_EXPORTS WText : public Widget2D
        {
        public:
            WText(const String &text, const Point &pos, int font_size = 20, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };

        class CV_EXPORTS WText3D : public Widget3D
        {
        public:
            //! creates text label in 3D. If face_camera = false, text plane normal is oriented along z-axis. Use widget pose to orient it properly
            WText3D(const String &text, const Point3d &position, double text_scale = 1., bool face_camera = true, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };

        class CV_EXPORTS WImageOverlay : public Widget2D
        {
        public:
            WImageOverlay(InputArray image, const Rect &rect);
            void setImage(InputArray image);
        };

        class CV_EXPORTS WImage3D : public Widget3D
        {
        public:
            //! Creates 3D image in a plane centered at the origin with normal orientaion along z-axis,
            //! image x- and y-axes are oriented along x- and y-axes of 3d world
            WImage3D(InputArray image, const Size2d &size);

            //! Creates 3D image at a given position, pointing in the direction of the normal, and having the up_vector orientation
            WImage3D(InputArray image, const Size2d &size, const Vec3d &center, const Vec3d &normal, const Vec3d &up_vector);

            void setImage(InputArray image);
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Compond widgets

        class CV_EXPORTS WCoordinateSystem : public Widget3D
        {
        public:
            WCoordinateSystem(double scale = 1.0);
        };

        class CV_EXPORTS WGrid : public Widget3D
        {
        public:
            //! Creates grid at the origin and normal oriented along z-axis
            WGrid(const Vec2i &cells = Vec2i::all(10), const Vec2d &cells_spacing = Vec2d::all(1.0), const Color &color = Color::white());

            //! Creates repositioned grid
            WGrid(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
                  const Vec2i &cells = Vec2i::all(10), const Vec2d &cells_spacing = Vec2d::all(1.0), const Color &color = Color::white());
        };

        class CV_EXPORTS WCameraPosition : public Widget3D
        {
        public:
            //! Creates camera coordinate frame (axes) at the origin
            WCameraPosition(double scale = 1.0);
            //! Creates frustum based on the intrinsic marix K at the origin
            WCameraPosition(const Matx33d &K, double scale = 1.0, const Color &color = Color::white());
            //! Creates frustum based on the field of view at the origin
            WCameraPosition(const Vec2d &fov, double scale = 1.0, const Color &color = Color::white());
            //! Creates frustum and display given image at the far plane
            WCameraPosition(const Matx33d &K, InputArray image, double scale = 1.0, const Color &color = Color::white());
            //! Creates frustum and display given image at the far plane
            WCameraPosition(const Vec2d &fov, InputArray image, double scale = 1.0, const Color &color = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Trajectories

        class CV_EXPORTS WTrajectory : public Widget3D
        {
        public:
            enum {FRAMES = 1, PATH = 2, BOTH = FRAMES + PATH };

            //! Takes vector<Affine3<T>> and displays trajectory of the given path either by coordinate frames or polyline
            WTrajectory(InputArray path, int display_mode = WTrajectory::PATH, double scale = 1.0, const Color &color = Color::white());
        };

        class CV_EXPORTS WTrajectoryFrustums : public Widget3D
        {
        public:
            //! Takes vector<Affine3<T>> and displays trajectory of the given path by frustums
            WTrajectoryFrustums(InputArray path, const Matx33d &K, double scale = 1., const Color &color = Color::white());

            //! Takes vector<Affine3<T>> and displays trajectory of the given path by frustums
            WTrajectoryFrustums(InputArray path, const Vec2d &fov, double scale = 1., const Color &color = Color::white());
        };

        class CV_EXPORTS WTrajectorySpheres: public Widget3D
        {
        public:
            //! Takes vector<Affine3<T>> and displays trajectory of the given path
            WTrajectorySpheres(InputArray path, double line_length = 0.05, double radius = 0.007,
                               const Color &from = Color::red(), const Color &to = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Clouds

        class CV_EXPORTS WCloud: public Widget3D
        {
        public:
            //! Each point in cloud is mapped to a color in colors
            WCloud(InputArray cloud, InputArray colors);
            //! All points in cloud have the same color
            WCloud(InputArray cloud, const Color &color = Color::white());
        };

        class CV_EXPORTS WPaintedCloud: public Widget3D
        {
        public:
            //! Paint cloud with default gradient between cloud bounds points
            WPaintedCloud(InputArray cloud);

            //! Paint cloud with default gradient between given points
            WPaintedCloud(InputArray cloud, const Point3d& p1, const Point3d& p2);

            //! Paint cloud with gradient specified by given colors between given points
            WPaintedCloud(InputArray cloud, const Point3d& p1, const Point3d& p2, const Color& c1, const Color c2);
        };

        class CV_EXPORTS WCloudCollection : public Widget3D
        {
        public:
            WCloudCollection();

            //! Each point in cloud is mapped to a color in colors
            void addCloud(InputArray cloud, InputArray colors, const Affine3d &pose = Affine3d::Identity());
            //! All points in cloud have the same color
            void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3d &pose = Affine3d::Identity());
        };

        class CV_EXPORTS WCloudNormals : public Widget3D
        {
        public:
            WCloudNormals(InputArray cloud, InputArray normals, int level = 64, double scale = 0.1, const Color &color = Color::white());
        };

        class CV_EXPORTS WMesh : public Widget3D
        {
        public:
            WMesh(const Mesh &mesh);
            WMesh(InputArray cloud, InputArray polygons, InputArray colors = noArray(), InputArray normals = noArray());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Utility exports

        template<> CV_EXPORTS Widget2D Widget::cast<Widget2D>();
        template<> CV_EXPORTS Widget3D Widget::cast<Widget3D>();
        template<> CV_EXPORTS WLine Widget::cast<WLine>();
        template<> CV_EXPORTS WPlane Widget::cast<WPlane>();
        template<> CV_EXPORTS WSphere Widget::cast<WSphere>();
        template<> CV_EXPORTS WCylinder Widget::cast<WCylinder>();
        template<> CV_EXPORTS WArrow Widget::cast<WArrow>();
        template<> CV_EXPORTS WCircle Widget::cast<WCircle>();
        template<> CV_EXPORTS WCone Widget::cast<WCone>();
        template<> CV_EXPORTS WCube Widget::cast<WCube>();
        template<> CV_EXPORTS WCoordinateSystem Widget::cast<WCoordinateSystem>();
        template<> CV_EXPORTS WPolyLine Widget::cast<WPolyLine>();
        template<> CV_EXPORTS WGrid Widget::cast<WGrid>();
        template<> CV_EXPORTS WText3D Widget::cast<WText3D>();
        template<> CV_EXPORTS WText Widget::cast<WText>();
        template<> CV_EXPORTS WImageOverlay Widget::cast<WImageOverlay>();
        template<> CV_EXPORTS WImage3D Widget::cast<WImage3D>();
        template<> CV_EXPORTS WCameraPosition Widget::cast<WCameraPosition>();
        template<> CV_EXPORTS WTrajectory Widget::cast<WTrajectory>();
        template<> CV_EXPORTS WTrajectoryFrustums Widget::cast<WTrajectoryFrustums>();
        template<> CV_EXPORTS WTrajectorySpheres Widget::cast<WTrajectorySpheres>();
        template<> CV_EXPORTS WCloud Widget::cast<WCloud>();
        template<> CV_EXPORTS WPaintedCloud Widget::cast<WPaintedCloud>();
        template<> CV_EXPORTS WCloudCollection Widget::cast<WCloudCollection>();
        template<> CV_EXPORTS WCloudNormals Widget::cast<WCloudNormals>();
        template<> CV_EXPORTS WMesh Widget::cast<WMesh>();

    } /* namespace viz */
} /* namespace cv */

#endif
