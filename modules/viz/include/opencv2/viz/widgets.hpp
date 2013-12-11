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

        enum RenderingRepresentationProperties
        {
            REPRESENTATION_POINTS,
            REPRESENTATION_WIREFRAME,
            REPRESENTATION_SURFACE
        };

        enum ShadingRepresentationProperties
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

            void setPose(const Affine3f &pose);
            void updatePose(const Affine3f &pose);
            Affine3f getPose() const;

            void setColor(const Color &color);
        private:
            struct MatrixConverter;

        };

        /////////////////////////////////////////////////////////////////////////////
        /// The base class for all 2D widgets
        class CV_EXPORTS Widget2D : public Widget
        {
        public:
            Widget2D() {}

            void setColor(const Color &color);
        };

        class CV_EXPORTS WLine : public Widget3D
        {
        public:
            WLine(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
        };

        class CV_EXPORTS WPlane : public Widget3D
        {
        public:
            WPlane(const Vec4f& coefs, float size = 1.f, const Color &color = Color::white());
            WPlane(const Vec4f& coefs, const Point3f& pt, float size = 1.f, const Color &color = Color::white());
        private:
            struct SetSizeImpl;
        };

        class CV_EXPORTS WSphere : public Widget3D
        {
        public:
            WSphere(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white());
        };

        class CV_EXPORTS WArrow : public Widget3D
        {
        public:
            WArrow(const Point3f& pt1, const Point3f& pt2, float thickness = 0.03f, const Color &color = Color::white());
        };

        class CV_EXPORTS WCircle : public Widget3D
        {
        public:
            WCircle(const Point3f& pt, float radius, float thickness = 0.01f, const Color &color = Color::white());
        };

        class CV_EXPORTS WCylinder : public Widget3D
        {
        public:
            WCylinder(const Point3f& pt_on_axis, const Point3f& axis_direction, float radius, int numsides = 30, const Color &color = Color::white());
        };

        class CV_EXPORTS WCube : public Widget3D
        {
        public:
            WCube(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white());
        };

        class CV_EXPORTS WCoordinateSystem : public Widget3D
        {
        public:
            WCoordinateSystem(float scale = 1.f);
        };

        class CV_EXPORTS WPolyLine : public Widget3D
        {
        public:
            WPolyLine(InputArray points, const Color &color = Color::white());

        private:
            struct CopyImpl;
        };

        class CV_EXPORTS WGrid : public Widget3D
        {
        public:
            //! Creates grid at the origin
            WGrid(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
            //! Creates grid based on the plane equation
            WGrid(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());

        private:
            struct GridImpl;

        };

        class CV_EXPORTS WText3D : public Widget3D
        {
        public:
            WText3D(const String &text, const Point3f &position, float text_scale = 1.f, bool face_camera = true, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };

        class CV_EXPORTS WText : public Widget2D
        {
        public:
            WText(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };

        class CV_EXPORTS WImageOverlay : public Widget2D
        {
        public:
            WImageOverlay(const Mat &image, const Rect &rect);

            void setImage(const Mat &image);
        };

        class CV_EXPORTS WImage3D : public Widget3D
        {
        public:
            //! Creates 3D image at the origin
            WImage3D(const Mat &image, const Size &size);
            //! Creates 3D image at a given position, pointing in the direction of the normal, and having the up_vector orientation
            WImage3D(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size);

            void setImage(const Mat &image);
        };

        class CV_EXPORTS WCameraPosition : public Widget3D
        {
        public:
            //! Creates camera coordinate frame (axes) at the origin
            WCameraPosition(float scale = 1.f);
            //! Creates frustum based on the intrinsic marix K at the origin
            WCameraPosition(const Matx33f &K, float scale = 1.f, const Color &color = Color::white());
            //! Creates frustum based on the field of view at the origin
            WCameraPosition(const Vec2f &fov, float scale = 1.f, const Color &color = Color::white());
            //! Creates frustum and display given image at the far plane
            WCameraPosition(const Matx33f &K, const Mat &img, float scale = 1.f, const Color &color = Color::white());
            //! Creates frustum and display given image at the far plane
            WCameraPosition(const Vec2f &fov, const Mat &img, float scale = 1.f, const Color &color = Color::white());

        private:
            struct ProjectImage;
        };

        class CV_EXPORTS WTrajectory : public Widget3D
        {
        public:
            enum {DISPLAY_FRAMES = 1, DISPLAY_PATH = 2};

            //! Displays trajectory of the given path either by coordinate frames or polyline
            WTrajectory(const std::vector<Affine3f> &path, int display_mode = WTrajectory::DISPLAY_PATH, const Color &color = Color::white(), float scale = 1.f);
            //! Displays trajectory of the given path by frustums
            WTrajectory(const std::vector<Affine3f> &path, const Matx33f &K, float scale = 1.f, const Color &color = Color::white());
            //! Displays trajectory of the given path by frustums
            WTrajectory(const std::vector<Affine3f> &path, const Vec2f &fov, float scale = 1.f, const Color &color = Color::white());

        private:
            struct ApplyPath;
        };

        class CV_EXPORTS WSpheresTrajectory: public Widget3D
        {
        public:
            WSpheresTrajectory(const std::vector<Affine3f> &path, float line_length = 0.05f, float init_sphere_radius = 0.021f,
                                    float sphere_radius = 0.007f, const Color &line_color = Color::white(), const Color &sphere_color = Color::white());
        };

        class CV_EXPORTS WCloud: public Widget3D
        {
        public:
            //! Each point in cloud is mapped to a color in colors
            WCloud(InputArray cloud, InputArray colors);
            //! All points in cloud have the same color
            WCloud(InputArray cloud, const Color &color = Color::white());

        private:
            struct CreateCloudWidget;
        };

        class CV_EXPORTS WCloudCollection : public Widget3D
        {
        public:
            WCloudCollection();

            //! Each point in cloud is mapped to a color in colors
            void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity());
            //! All points in cloud have the same color
            void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3f &pose = Affine3f::Identity());

        private:
            struct CreateCloudWidget;
        };

        class CV_EXPORTS WCloudNormals : public Widget3D
        {
        public:
            WCloudNormals(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white());

        private:
            struct ApplyCloudNormals;
        };

        class CV_EXPORTS WMesh : public Widget3D
        {
        public:
            WMesh(const Mesh3d &mesh);

        private:
            struct CopyImpl;
        };

        template<> CV_EXPORTS Widget2D Widget::cast<Widget2D>();
        template<> CV_EXPORTS Widget3D Widget::cast<Widget3D>();
        template<> CV_EXPORTS WLine Widget::cast<WLine>();
        template<> CV_EXPORTS WPlane Widget::cast<WPlane>();
        template<> CV_EXPORTS WSphere Widget::cast<WSphere>();
        template<> CV_EXPORTS WCylinder Widget::cast<WCylinder>();
        template<> CV_EXPORTS WArrow Widget::cast<WArrow>();
        template<> CV_EXPORTS WCircle Widget::cast<WCircle>();
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
        template<> CV_EXPORTS WSpheresTrajectory Widget::cast<WSpheresTrajectory>();
        template<> CV_EXPORTS WCloud Widget::cast<WCloud>();
        template<> CV_EXPORTS WCloudCollection Widget::cast<WCloudCollection>();
        template<> CV_EXPORTS WCloudNormals Widget::cast<WCloudNormals>();
        template<> CV_EXPORTS WMesh Widget::cast<WMesh>();

    } /* namespace viz */
} /* namespace cv */

#endif
