#pragma once

#include <opencv2/viz/types.hpp>
#include <common.h>


namespace cv
{
    namespace viz
    {
        /////////////////////////////////////////////////////////////////////////////
        /// The base class for all widgets
        class CV_EXPORTS Widget
        {
        public:
            Widget();
            Widget(const Widget &other);
            Widget& operator =(const Widget &other);

            ~Widget();

            template<typename _W> _W cast();
        private:
            class Impl;
            Impl *impl_;
            friend struct WidgetAccessor;

            void create();
            void release();
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

            void setLineWidth(float line_width);
            float getLineWidth();
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
            GridWidget(Vec2i dimensions, Vec2d spacing, const Color &color = Color::white());
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
            ImageOverlayWidget(const Mat &image, const Point2i &pos);
            
            void setImage(const Mat &image);
            
        private:
            struct CopyImpl;
        };
        
        class CV_EXPORTS Image3DWidget : public Widget3D
        {
        public:
            Image3DWidget(const Mat &image);
            
            void setImage(const Mat &image);
        
        private:
            struct CopyImpl;
        };
        
        class CV_EXPORTS CameraPositionWidget : public Widget3D
        {
        public:
            CameraPositionWidget(double scale = 1.0);
            CameraPositionWidget(const Vec3f &position, const Vec3f &look_at, const Vec3f &up_vector, double scale = 1.0);
            CameraPositionWidget(const Matx33f &K, double scale = 1.0, const Color &color = Color::white());
            CameraPositionWidget(const Vec2f &fov, double scale = 1.0, const Color &color = Color::white());
        };
        
        class CV_EXPORTS TrajectoryWidget : public Widget3D
        {
        public:
            TrajectoryWidget(const std::vector<Affine3f> &path, const Color &color = Color::white(), bool show_frames = false, double scale = 1.0);
            TrajectoryWidget(const std::vector<Affine3f> &path, const Matx33f &K, double scale = 1.0, const Color &color = Color::white()); // Camera frustums
        };

        class CV_EXPORTS CloudWidget : public Widget3D
        {
        public:
            CloudWidget(InputArray cloud, InputArray colors);
            CloudWidget(InputArray cloud, const Color &color = Color::white());

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
        template<> CV_EXPORTS CloudWidget Widget::cast<CloudWidget>();
        template<> CV_EXPORTS CloudNormalsWidget Widget::cast<CloudNormalsWidget>();
        template<> CV_EXPORTS MeshWidget Widget::cast<MeshWidget>();

    } /* namespace viz */
} /* namespace cv */