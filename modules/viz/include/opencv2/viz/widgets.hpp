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

//! @addtogroup viz_widget
//! @{

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
            SHADING,
            AMBIENT
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

        /** @brief Base class of all widgets. Widget is implicitly shared. :
        */
        class CV_EXPORTS Widget
        {
        public:
            Widget();
            Widget(const Widget& other);
            Widget& operator=(const Widget& other);
            ~Widget();

            /** @brief Creates a widget from ply file.

            @param file_name Ply file name.
             */
            static Widget fromPlyFile(const String &file_name);

            /** @brief Sets rendering property of the widget.

            @param property Property that will be modified.
            @param value The new value of the property.

            **Rendering property** can be one of the following:
            -   **POINT_SIZE**
            -   **OPACITY**
            -   **LINE_WIDTH**
            -   **FONT_SIZE**
            -
            **REPRESENTATION**: Expected values are
            -   **REPRESENTATION_POINTS**
            -   **REPRESENTATION_WIREFRAME**
            -   **REPRESENTATION_SURFACE**
            -
            **IMMEDIATE_RENDERING**:
            -   Turn on immediate rendering by setting the value to 1.
            -   Turn off immediate rendering by setting the value to 0.
            -
            **SHADING**: Expected values are
            -   **SHADING_FLAT**
            -   **SHADING_GOURAUD**
            -   **SHADING_PHONG**
             */
            void setRenderingProperty(int property, double value);
            /** @brief Returns rendering property of the widget.

            @param property Property.

            **Rendering property** can be one of the following:
            -   **POINT_SIZE**
            -   **OPACITY**
            -   **LINE_WIDTH**
            -   **FONT_SIZE**
            -   **AMBIENT**
            -
            **REPRESENTATION**: Expected values are
            :   -   **REPRESENTATION_POINTS**
            -   **REPRESENTATION_WIREFRAME**
            -   **REPRESENTATION_SURFACE**
            -
            **IMMEDIATE_RENDERING**:
            :   -   Turn on immediate rendering by setting the value to 1.
            -   Turn off immediate rendering by setting the value to 0.
            -
            **SHADING**: Expected values are
            :   -   **SHADING_FLAT**
            -   **SHADING_GOURAUD**
            -   **SHADING_PHONG**
             */
            double getRenderingProperty(int property) const;

            /** @brief Casts a widget to another.

            @code
            // Create a sphere widget
            viz::WSphere sw(Point3f(0.0f,0.0f,0.0f), 0.5f);
            // Cast sphere widget to cloud widget
            viz::WCloud cw = sw.cast<viz::WCloud>();
            @endcode

            @note 3D Widgets can only be cast to 3D Widgets. 2D Widgets can only be cast to 2D Widgets.
             */
            template<typename _W> _W cast();
        private:
            class Impl;
            Impl *impl_;
            friend struct WidgetAccessor;
        };

        /////////////////////////////////////////////////////////////////////////////

        /** @brief Base class of all 3D widgets.
         */
        class CV_EXPORTS Widget3D : public Widget
        {
        public:
            Widget3D() {}

            /** @brief Sets pose of the widget.

            @param pose The new pose of the widget.
             */
            void setPose(const Affine3d &pose);
            /** @brief Updates pose of the widget by pre-multiplying its current pose.

            @param pose The pose that the current pose of the widget will be pre-multiplied by.
             */
            void updatePose(const Affine3d &pose);
            /** @brief Returns the current pose of the widget.
             */
            Affine3d getPose() const;

            /** @brief Transforms internal widget data (i.e. points, normals) using the given transform.

            @param transform Specified transformation to apply.
             */
            void applyTransform(const Affine3d &transform);

            /** @brief Sets the color of the widget.

            @param color color of type Color
             */
            void setColor(const Color &color);

        };

        /////////////////////////////////////////////////////////////////////////////

        /** @brief Base class of all 2D widgets.
        */
        class CV_EXPORTS Widget2D : public Widget
        {
        public:
            Widget2D() {}

            /** @brief Sets the color of the widget.

            @param color color of type Color
             */
            void setColor(const Color &color);
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Simple widgets

        /** @brief This 3D Widget defines a finite line.
        */
        class CV_EXPORTS WLine : public Widget3D
        {
        public:
            /** @brief Constructs a WLine.

            @param pt1 Start point of the line.
            @param pt2 End point of the line.
            @param color Color of the line.
             */
            WLine(const Point3d &pt1, const Point3d &pt2, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a finite plane.
        */
        class CV_EXPORTS WPlane : public Widget3D
        {
        public:
            /** @brief Constructs a default plane with center point at origin and normal oriented along z-axis.

            @param size Size of the plane
            @param color Color of the plane.
             */
            WPlane(const Size2d& size = Size2d(1.0, 1.0), const Color &color = Color::white());

            /** @brief Constructs a repositioned plane

            @param center Center of the plane
            @param normal Plane normal orientation
            @param new_yaxis Up-vector. New orientation of plane y-axis.
            @param size
            @param color Color of the plane.
             */
            WPlane(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
                   const Size2d& size = Size2d(1.0, 1.0), const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a sphere. :
        */
        class CV_EXPORTS WSphere : public Widget3D
        {
        public:
            /** @brief Constructs a WSphere.

            @param center Center of the sphere.
            @param radius Radius of the sphere.
            @param sphere_resolution Resolution of the sphere.
            @param color Color of the sphere.
             */
            WSphere(const cv::Point3d &center, double radius, int sphere_resolution = 10, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines an arrow.
        */
        class CV_EXPORTS WArrow : public Widget3D
        {
        public:
            /** @brief Constructs an WArrow.

            @param pt1 Start point of the arrow.
            @param pt2 End point of the arrow.
            @param thickness Thickness of the arrow. Thickness of arrow head is also adjusted
            accordingly.
            @param color Color of the arrow.

            Arrow head is located at the end point of the arrow.
             */
            WArrow(const Point3d& pt1, const Point3d& pt2, double thickness = 0.03, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a circle.
        */
        class CV_EXPORTS WCircle : public Widget3D
        {
        public:
            /** @brief Constructs default planar circle centred at origin with plane normal along z-axis

            @param radius Radius of the circle.
            @param thickness Thickness of the circle.
            @param color Color of the circle.
             */
            WCircle(double radius, double thickness = 0.01, const Color &color = Color::white());

            /** @brief Constructs repositioned planar circle.

            @param radius Radius of the circle.
            @param center Center of the circle.
            @param normal Normal of the plane in which the circle lies.
            @param thickness Thickness of the circle.
            @param color Color of the circle.
             */
            WCircle(double radius, const Point3d& center, const Vec3d& normal, double thickness = 0.01, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a cone. :
        */
        class CV_EXPORTS WCone : public Widget3D
        {
        public:
            /** @brief Constructs default cone oriented along x-axis with center of its base located at origin

            @param length Length of the cone.
            @param radius Radius of the cone.
            @param resolution Resolution of the cone.
            @param color Color of the cone.
             */
            WCone(double length, double radius, int resolution = 6.0, const Color &color = Color::white());

            /** @brief Constructs repositioned planar cone.

            @param radius Radius of the cone.
            @param center Center of the cone base.
            @param tip Tip of the cone.
            @param resolution Resolution of the cone.
            @param color Color of the cone.

             */
            WCone(double radius, const Point3d& center, const Point3d& tip, int resolution = 6.0, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a cylinder. :
        */
        class CV_EXPORTS WCylinder : public Widget3D
        {
        public:
            /** @brief Constructs a WCylinder.

            @param axis_point1 A point1 on the axis of the cylinder.
            @param axis_point2 A point2 on the axis of the cylinder.
            @param radius Radius of the cylinder.
            @param numsides Resolution of the cylinder.
            @param color Color of the cylinder.
             */
            WCylinder(const Point3d& axis_point1, const Point3d& axis_point2, double radius, int numsides = 30, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a cube.
         */
        class CV_EXPORTS WCube : public Widget3D
        {
        public:
            /** @brief Constructs a WCube.

            @param min_point Specifies minimum point of the bounding box.
            @param max_point Specifies maximum point of the bounding box.
            @param wire_frame If true, cube is represented as wireframe.
            @param color Color of the cube.

            ![Cube Widget](images/cube_widget.png)
             */
            WCube(const Point3d& min_point = Vec3d::all(-0.5), const Point3d& max_point = Vec3d::all(0.5),
                  bool wire_frame = true, const Color &color = Color::white());
        };

        /** @brief This 3D Widget defines a poly line. :
        */
        class CV_EXPORTS WPolyLine : public Widget3D
        {
        public:
            WPolyLine(InputArray points, InputArray colors);
            /** @brief Constructs a WPolyLine.

            @param points Point set.
            @param color Color of the poly line.
             */
            WPolyLine(InputArray points, const Color &color = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Text and image widgets

        /** @brief This 2D Widget represents text overlay.
        */
        class CV_EXPORTS WText : public Widget2D
        {
        public:
            /** @brief Constructs a WText.

            @param text Text content of the widget.
            @param pos Position of the text.
            @param font_size Font size.
            @param color Color of the text.
             */
            WText(const String &text, const Point &pos, int font_size = 20, const Color &color = Color::white());

            /** @brief Sets the text content of the widget.

            @param text Text content of the widget.
             */
            void setText(const String &text);
            /** @brief Returns the current text content of the widget.
            */
            String getText() const;
        };

        /** @brief This 3D Widget represents 3D text. The text always faces the camera.
        */
        class CV_EXPORTS WText3D : public Widget3D
        {
        public:
            /** @brief Constructs a WText3D.

            @param text Text content of the widget.
            @param position Position of the text.
            @param text_scale Size of the text.
            @param face_camera If true, text always faces the camera.
            @param color Color of the text.
             */
            WText3D(const String &text, const Point3d &position, double text_scale = 1., bool face_camera = true, const Color &color = Color::white());

            /** @brief Sets the text content of the widget.

            @param text Text content of the widget.

             */
            void setText(const String &text);
            /** @brief Returns the current text content of the widget.
            */
            String getText() const;
        };

        /** @brief This 2D Widget represents an image overlay. :
        */
        class CV_EXPORTS WImageOverlay : public Widget2D
        {
        public:
            /** @brief Constructs an WImageOverlay.

            @param image BGR or Gray-Scale image.
            @param rect Image is scaled and positioned based on rect.
             */
            WImageOverlay(InputArray image, const Rect &rect);
            /** @brief Sets the image content of the widget.

            @param image BGR or Gray-Scale image.
             */
            void setImage(InputArray image);
        };

        /** @brief This 3D Widget represents an image in 3D space. :
        */
        class CV_EXPORTS WImage3D : public Widget3D
        {
        public:
            /** @brief Constructs an WImage3D.

            @param image BGR or Gray-Scale image.
            @param size Size of the image.
             */
            WImage3D(InputArray image, const Size2d &size);

            /** @brief Constructs an WImage3D.

            @param image BGR or Gray-Scale image.
            @param size Size of the image.
            @param center Position of the image.
            @param normal Normal of the plane that represents the image.
            @param up_vector Determines orientation of the image.
             */
            WImage3D(InputArray image, const Size2d &size, const Vec3d &center, const Vec3d &normal, const Vec3d &up_vector);

            /** @brief Sets the image content of the widget.

            @param image BGR or Gray-Scale image.
             */
            void setImage(InputArray image);

            /** @brief Sets the image size of the widget.

            @param size the new size of the image.
             */
            void setSize(const Size& size);
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Compond widgets

        /** @brief This 3D Widget represents a coordinate system. :
        */
        class CV_EXPORTS WCoordinateSystem : public Widget3D
        {
        public:
            /** @brief Constructs a WCoordinateSystem.

            @param scale Determines the size of the axes.
             */
            WCoordinateSystem(double scale = 1.0);
        };

        /** @brief This 3D Widget defines a grid. :
         */
        class CV_EXPORTS WGrid : public Widget3D
        {
        public:
            /** @brief Constructs a WGrid.

            @param cells Number of cell columns and rows, respectively.
            @param cells_spacing Size of each cell, respectively.
            @param color Color of the grid.
             */
            WGrid(const Vec2i &cells = Vec2i::all(10), const Vec2d &cells_spacing = Vec2d::all(1.0), const Color &color = Color::white());

            //! Creates repositioned grid
            WGrid(const Point3d& center, const Vec3d& normal, const Vec3d& new_yaxis,
                  const Vec2i &cells = Vec2i::all(10), const Vec2d &cells_spacing = Vec2d::all(1.0), const Color &color = Color::white());
        };

        /** @brief This 3D Widget represents camera position in a scene by its axes or viewing frustum. :
        */
        class CV_EXPORTS WCameraPosition : public Widget3D
        {
        public:
            /** @brief Creates camera coordinate frame at the origin.

            ![Camera coordinate frame](images/cpw1.png)
             */
            WCameraPosition(double scale = 1.0);
            /** @brief Display the viewing frustum
            @param K Intrinsic matrix of the camera.
            @param scale Scale of the frustum.
            @param color Color of the frustum.

            Creates viewing frustum of the camera based on its intrinsic matrix K.

            ![Camera viewing frustum](images/cpw2.png)
            */
            WCameraPosition(const Matx33d &K, double scale = 1.0, const Color &color = Color::white());
            /** @brief Display the viewing frustum
            @param fov Field of view of the camera (horizontal, vertical).
            @param scale Scale of the frustum.
            @param color Color of the frustum.

            Creates viewing frustum of the camera based on its field of view fov.

            ![Camera viewing frustum](images/cpw2.png)
             */
            WCameraPosition(const Vec2d &fov, double scale = 1.0, const Color &color = Color::white());
            /** @brief Display image on the far plane of the viewing frustum

            @param K Intrinsic matrix of the camera.
            @param image BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
            @param scale Scale of the frustum and image.
            @param color Color of the frustum.

            Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on
            the far end plane.

            ![Camera viewing frustum with image](images/cpw3.png)
             */
            WCameraPosition(const Matx33d &K, InputArray image, double scale = 1.0, const Color &color = Color::white());
            /** @brief  Display image on the far plane of the viewing frustum

            @param fov Field of view of the camera (horizontal, vertical).
            @param image BGR or Gray-Scale image that is going to be displayed on the far plane of the frustum.
            @param scale Scale of the frustum and image.
            @param color Color of the frustum.

            Creates viewing frustum of the camera based on its intrinsic matrix K, and displays image on
            the far end plane.

            ![Camera viewing frustum with image](images/cpw3.png)
             */
            WCameraPosition(const Vec2d &fov, InputArray image, double scale = 1.0, const Color &color = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Trajectories

        /** @brief This 3D Widget represents a trajectory. :
        */
        class CV_EXPORTS WTrajectory : public Widget3D
        {
        public:
            enum {FRAMES = 1, PATH = 2, BOTH = FRAMES + PATH };

            /** @brief Constructs a WTrajectory.

            @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
            @param display_mode Display mode. This can be PATH, FRAMES, and BOTH.
            @param scale Scale of the frames. Polyline is not affected.
            @param color Color of the polyline that represents path.

            Frames are not affected.
            Displays trajectory of the given path as follows:
            -   PATH : Displays a poly line that represents the path.
            -   FRAMES : Displays coordinate frames at each pose.
            -   PATH & FRAMES : Displays both poly line and coordinate frames.
             */
            WTrajectory(InputArray path, int display_mode = WTrajectory::PATH, double scale = 1.0, const Color &color = Color::white());
        };

        /** @brief This 3D Widget represents a trajectory. :
        */
        class CV_EXPORTS WTrajectoryFrustums : public Widget3D
        {
        public:
            /** @brief Constructs a WTrajectoryFrustums.

            @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
            @param K Intrinsic matrix of the camera.
            @param scale Scale of the frustums.
            @param color Color of the frustums.

            Displays frustums at each pose of the trajectory.
             */
            WTrajectoryFrustums(InputArray path, const Matx33d &K, double scale = 1., const Color &color = Color::white());

            /** @brief Constructs a WTrajectoryFrustums.

            @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
            @param fov Field of view of the camera (horizontal, vertical).
            @param scale Scale of the frustums.
            @param color Color of the frustums.

            Displays frustums at each pose of the trajectory.
             */
            WTrajectoryFrustums(InputArray path, const Vec2d &fov, double scale = 1., const Color &color = Color::white());
        };

        /** @brief This 3D Widget represents a trajectory using spheres and lines

        where spheres represent the positions of the camera, and lines represent the direction from
        previous position to the current. :
         */
        class CV_EXPORTS WTrajectorySpheres: public Widget3D
        {
        public:
            /** @brief Constructs a WTrajectorySpheres.

            @param path List of poses on a trajectory. Takes std::vector\<Affine\<T\>\> with T == [float | double]
            @param line_length Max length of the lines which point to previous position
            @param radius Radius of the spheres.
            @param from Color for first sphere.
            @param to Color for last sphere. Intermediate spheres will have interpolated color.
             */
            WTrajectorySpheres(InputArray path, double line_length = 0.05, double radius = 0.007,
                               const Color &from = Color::red(), const Color &to = Color::white());
        };

        /////////////////////////////////////////////////////////////////////////////
        /// Clouds

        /** @brief This 3D Widget defines a point cloud. :

        @note In case there are four channels in the cloud, fourth channel is ignored.
        */
        class CV_EXPORTS WCloud: public Widget3D
        {
        public:
            /** @brief Constructs a WCloud.

            @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param colors Set of colors. It has to be of the same size with cloud.

            Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            WCloud(InputArray cloud, InputArray colors);

            /** @brief Constructs a WCloud.
            @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param color A single Color for the whole cloud.

            Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            WCloud(InputArray cloud, const Color &color = Color::white());

            /** @brief Constructs a WCloud.
            @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param colors Set of colors. It has to be of the same size with cloud.
            @param normals Normals for each point in cloud. Size and type should match with the cloud parameter.

            Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            WCloud(InputArray cloud, InputArray colors, InputArray normals);

            /** @brief Constructs a WCloud.
            @param cloud Set of points which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param color A single Color for the whole cloud.
            @param normals Normals for each point in cloud.

            Size and type should match with the cloud parameter.
            Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            WCloud(InputArray cloud, const Color &color, InputArray normals);
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

        /** @brief This 3D Widget defines a collection of clouds. :
        @note In case there are four channels in the cloud, fourth channel is ignored.
        */
        class CV_EXPORTS WCloudCollection : public Widget3D
        {
        public:
            WCloudCollection();

            /** @brief Adds a cloud to the collection.

            @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param colors Set of colors. It has to be of the same size with cloud.
            @param pose Pose of the cloud. Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            void addCloud(InputArray cloud, InputArray colors, const Affine3d &pose = Affine3d::Identity());
            /** @brief Adds a cloud to the collection.

            @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param color A single Color for the whole cloud.
            @param pose Pose of the cloud. Points in the cloud belong to mask when they are set to (NaN, NaN, NaN).
             */
            void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3d &pose = Affine3d::Identity());
            /** @brief Finalizes cloud data by repacking to single cloud.

            Useful for large cloud collections to reduce memory usage
            */
            void finalize();
        };

        /** @brief This 3D Widget represents normals of a point cloud. :
        */
        class CV_EXPORTS WCloudNormals : public Widget3D
        {
        public:
            /** @brief Constructs a WCloudNormals.

            @param cloud Point set which can be of type: CV_32FC3, CV_32FC4, CV_64FC3, CV_64FC4.
            @param normals A set of normals that has to be of same type with cloud.
            @param level Display only every level th normal.
            @param scale Scale of the arrows that represent normals.
            @param color Color of the arrows that represent normals.

            @note In case there are four channels in the cloud, fourth channel is ignored.
             */
            WCloudNormals(InputArray cloud, InputArray normals, int level = 64, double scale = 0.1, const Color &color = Color::white());
        };

        /** @brief Constructs a WMesh.

        @param mesh Mesh object that will be displayed.
        @param cloud Points of the mesh object.
        @param polygons Points of the mesh object.
        @param colors Point colors.
        @param normals Point normals.
         */
        class CV_EXPORTS WMesh : public Widget3D
        {
        public:
            WMesh(const Mesh &mesh);
            WMesh(InputArray cloud, InputArray polygons, InputArray colors = noArray(), InputArray normals = noArray());
        };

        /** @brief This class allows to merge several widgets to single one.

        It has quite limited functionality and can't merge widgets with different attributes. For
        instance, if widgetA has color array and widgetB has only global color defined, then result
        of merge won't have color at all. The class is suitable for merging large amount of similar
        widgets. :
         */
        class CV_EXPORTS WWidgetMerger : public Widget3D
        {
        public:
            WWidgetMerger();

            //! Add widget to merge with optional position change
            void addWidget(const Widget3D& widget, const Affine3d &pose = Affine3d::Identity());

            //! Repacks internal structure to single widget
            void finalize();
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
        template<> CV_EXPORTS WWidgetMerger Widget::cast<WWidgetMerger>();

//! @}

    } /* namespace viz */
} /* namespace cv */

#endif
