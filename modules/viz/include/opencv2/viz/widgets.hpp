#pragma once

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
            COLOR,
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

        class CV_EXPORTS LineWidget : public Widget3D
        {
        public:
            LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
        };

        class CV_EXPORTS PlaneWidget : public Widget3D
        {
        public:
            PlaneWidget(const Vec4f& coefs, double size = 1.0, const Color &color = Color::white());
            PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size = 1.0, const Color &color = Color::white());
        private:
            struct SetSizeImpl;
        };

        class CV_EXPORTS SphereWidget : public Widget3D
        {
        public:
            SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white());
        };

        class CV_EXPORTS ArrowWidget : public Widget3D
        {
        public:
            ArrowWidget(const Point3f& pt1, const Point3f& pt2, double thickness = 0.03, const Color &color = Color::white());
        };

        class CV_EXPORTS CircleWidget : public Widget3D
        {
        public:
            CircleWidget(const Point3f& pt, double radius, double thickness = 0.01, const Color &color = Color::white());
        };

        class CV_EXPORTS CylinderWidget : public Widget3D
        {
        public:
            CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30, const Color &color = Color::white());
        };

        class CV_EXPORTS CubeWidget : public Widget3D
        {
        public:
            CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white());
        };

        class CV_EXPORTS CoordinateSystemWidget : public Widget3D
        {
        public:
            CoordinateSystemWidget(double scale = 1.0);
        };

        class CV_EXPORTS PolyLineWidget : public Widget3D
        {
        public:
            PolyLineWidget(InputArray points, const Color &color = Color::white());

        private:
            struct CopyImpl;
        };

        class CV_EXPORTS GridWidget : public Widget3D
        {
        public:
            //! Creates grid at the origin
            GridWidget(const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
            //! Creates grid based on the plane equation
            GridWidget(const Vec4f &coeffs, const Vec2i &dimensions, const Vec2d &spacing, const Color &color = Color::white());
            
        private:
            struct GridImpl;
        
        };

        class CV_EXPORTS Text3DWidget : public Widget3D
        {
        public:
            Text3DWidget(const String &text, const Point3f &position, double text_scale = 1.0, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };

        class CV_EXPORTS TextWidget : public Widget2D
        {
        public:
            TextWidget(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white());

            void setText(const String &text);
            String getText() const;
        };
        
        class CV_EXPORTS ImageOverlayWidget : public Widget2D
        {
        public:
            ImageOverlayWidget(const Mat &image, const Rect &rect);
            
            void setImage(const Mat &image);
        };
        
        class CV_EXPORTS Image3DWidget : public Widget3D
        {
        public:
            //! Creates 3D image at the origin
            Image3DWidget(const Mat &image, const Size &size);
            //! Creates 3D image at a given position, pointing in the direction of the normal, and having the up_vector orientation
            Image3DWidget(const Vec3f &position, const Vec3f &normal, const Vec3f &up_vector, const Mat &image, const Size &size);
            
            void setImage(const Mat &image);
        };
        
        class CV_EXPORTS CameraPositionWidget : public Widget3D
        {
        public:
            //! Creates camera coordinate frame (axes) at the origin
            CameraPositionWidget(double scale = 1.0);
            //! Creates frustum based on the intrinsic marix K at the origin
            CameraPositionWidget(const Matx33f &K, double scale = 1.0, const Color &color = Color::white());
            //! Creates frustum based on the field of view at the origin
            CameraPositionWidget(const Vec2f &fov, double scale = 1.0, const Color &color = Color::white());
            //! Creates frustum and display given image at the far plane
            CameraPositionWidget(const Matx33f &K, const Mat &img, double scale = 1.0, const Color &color = Color::white());
        };
        
        class CV_EXPORTS TrajectoryWidget : public Widget3D
        {
        public:
            enum {DISPLAY_FRAMES = 1, DISPLAY_PATH = 2};
            
            //! Displays trajectory of the given path either by coordinate frames or polyline
            TrajectoryWidget(const std::vector<Affine3f> &path, int display_mode = TrajectoryWidget::DISPLAY_PATH, const Color &color = Color::white(), double scale = 1.0);
            //! Displays trajectory of the given path by frustums
            TrajectoryWidget(const std::vector<Affine3f> &path, const Matx33f &K, double scale = 1.0, const Color &color = Color::white());
            //! Displays trajectory of the given path by frustums
            TrajectoryWidget(const std::vector<Affine3f> &path, const Vec2f &fov, double scale = 1.0, const Color &color = Color::white());
            
        private:
            struct ApplyPath;
        };
        
        class CV_EXPORTS SpheresTrajectoryWidget : public Widget3D
        {
        public:
            SpheresTrajectoryWidget(const std::vector<Affine3f> &path, float line_length = 0.05f, double init_sphere_radius = 0.021,
                                    double sphere_radius = 0.007, const Color &line_color = Color::white(), const Color &sphere_color = Color::white());
        };

        class CV_EXPORTS CloudWidget : public Widget3D
        {
        public:
            //! Each point in cloud is mapped to a color in colors
            CloudWidget(InputArray cloud, InputArray colors);
            //! All points in cloud have the same color
            CloudWidget(InputArray cloud, const Color &color = Color::white());

        private:
            struct CreateCloudWidget;
        };

        class CV_EXPORTS CloudCollectionWidget : public Widget3D
        {
        public:
            CloudCollectionWidget();
            
            //! Each point in cloud is mapped to a color in colors
            void addCloud(InputArray cloud, InputArray colors, const Affine3f &pose = Affine3f::Identity());
            //! All points in cloud have the same color
            void addCloud(InputArray cloud, const Color &color = Color::white(), const Affine3f &pose = Affine3f::Identity());
            
        private:
            struct CreateCloudWidget;
        };
        
        class CV_EXPORTS CloudNormalsWidget : public Widget3D
        {
        public:
            CloudNormalsWidget(InputArray cloud, InputArray normals, int level = 100, float scale = 0.02f, const Color &color = Color::white());

        private:
            struct ApplyCloudNormals;
        };
        
        class CV_EXPORTS MeshWidget : public Widget3D
        {
        public:
            MeshWidget(const Mesh3d &mesh);
            
        private:
            struct CopyImpl;
        };

        template<> CV_EXPORTS Widget2D Widget::cast<Widget2D>();
        template<> CV_EXPORTS Widget3D Widget::cast<Widget3D>();
        template<> CV_EXPORTS LineWidget Widget::cast<LineWidget>();
        template<> CV_EXPORTS PlaneWidget Widget::cast<PlaneWidget>();
        template<> CV_EXPORTS SphereWidget Widget::cast<SphereWidget>();
        template<> CV_EXPORTS CylinderWidget Widget::cast<CylinderWidget>();
        template<> CV_EXPORTS ArrowWidget Widget::cast<ArrowWidget>();
        template<> CV_EXPORTS CircleWidget Widget::cast<CircleWidget>();
        template<> CV_EXPORTS CubeWidget Widget::cast<CubeWidget>();
        template<> CV_EXPORTS CoordinateSystemWidget Widget::cast<CoordinateSystemWidget>();
        template<> CV_EXPORTS PolyLineWidget Widget::cast<PolyLineWidget>();
        template<> CV_EXPORTS GridWidget Widget::cast<GridWidget>();
        template<> CV_EXPORTS Text3DWidget Widget::cast<Text3DWidget>();
        template<> CV_EXPORTS TextWidget Widget::cast<TextWidget>();
        template<> CV_EXPORTS ImageOverlayWidget Widget::cast<ImageOverlayWidget>();
        template<> CV_EXPORTS Image3DWidget Widget::cast<Image3DWidget>();
        template<> CV_EXPORTS CameraPositionWidget Widget::cast<CameraPositionWidget>();
        template<> CV_EXPORTS TrajectoryWidget Widget::cast<TrajectoryWidget>();
        template<> CV_EXPORTS SpheresTrajectoryWidget Widget::cast<SpheresTrajectoryWidget>();
        template<> CV_EXPORTS CloudWidget Widget::cast<CloudWidget>();
        template<> CV_EXPORTS CloudCollectionWidget Widget::cast<CloudCollectionWidget>();
        template<> CV_EXPORTS CloudNormalsWidget Widget::cast<CloudNormalsWidget>();
        template<> CV_EXPORTS MeshWidget Widget::cast<MeshWidget>();

    } /* namespace viz */
} /* namespace cv */