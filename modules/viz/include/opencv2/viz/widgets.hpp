#pragma once

#include <opencv2/viz/types.hpp>


namespace temp_viz
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
        Widget3D(const Widget& other);
        Widget3D& operator =(const Widget &other);
        
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
        Widget2D(const Widget &other);
        Widget2D& operator=(const Widget &other);
        
        void setColor(const Color &color);
    };
    

    class CV_EXPORTS LineWidget : public Widget3D
    {
    public:
        LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
        LineWidget(const Widget &other) : Widget3D(other) {}
        LineWidget& operator=(const Widget &other);
        
        void setLineWidth(float line_width);
        float getLineWidth();
    };
    
    class CV_EXPORTS PlaneWidget : public Widget3D
    {
    public:
        PlaneWidget(const Vec4f& coefs, double size = 1.0, const Color &color = Color::white());
        PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size = 1.0, const Color &color = Color::white());
        PlaneWidget(const Widget& other) : Widget3D(other) {}
        PlaneWidget& operator=(const Widget& other);
    };
    
    class CV_EXPORTS SphereWidget : public Widget3D
    {
    public:
        SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white());
        SphereWidget(const Widget &other) : Widget3D(other) {}
        SphereWidget& operator=(const Widget &other);
    };
    
    class CV_EXPORTS ArrowWidget : public Widget3D
    {
    public:
        ArrowWidget(const Point3f& pt1, const Point3f& pt2, const Color &color = Color::white());
        ArrowWidget(const Widget &other) : Widget3D(other) {}
        ArrowWidget& operator=(const Widget &other);
    };

    class CV_EXPORTS CircleWidget : public Widget3D
    {
    public:
        CircleWidget(const Point3f& pt, double radius, double thickness = 0.01, const Color &color = Color::white());
        CircleWidget(const Widget& other) : Widget3D(other) {}
        CircleWidget& operator=(const Widget &other);
    };
    
    class CV_EXPORTS CylinderWidget : public Widget3D
    {
    public:
        CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30, const Color &color = Color::white());
        CylinderWidget(const Widget& other) : Widget3D(other) {}
        CylinderWidget& operator=(const Widget &other);
    };
    
    class CV_EXPORTS CubeWidget : public Widget3D
    {
    public:
        CubeWidget(const Point3f& pt_min, const Point3f& pt_max, bool wire_frame = true, const Color &color = Color::white());
        CubeWidget(const Widget& other) : Widget3D(other) {}
        CubeWidget& operator=(const Widget &other);
    };
    
    class CV_EXPORTS CoordinateSystemWidget : public Widget3D
    {
    public:
        CoordinateSystemWidget(double scale, const Affine3f& affine);
        CoordinateSystemWidget(const Widget &other) : Widget3D(other) {}
        CoordinateSystemWidget& operator=(const Widget &other);
    };
    
    class CV_EXPORTS TextWidget : public Widget2D
    {
    public:
        TextWidget(const String &text, const Point2i &pos, int font_size = 10, const Color &color = Color::white());
        TextWidget(const Widget& other) : Widget2D(other) {}
        TextWidget& operator=(const Widget &other);
        
        void setText(const String &text);
        String getText() const;
    };
    
    class CV_EXPORTS CloudWidget : public Widget3D
    {
    public:
        CloudWidget(InputArray _cloud, InputArray _colors);
        CloudWidget(InputArray _cloud, const Color &color = Color::white());
        CloudWidget(const Widget &other) : Widget3D(other) {}
        CloudWidget& operator=(const Widget &other);
        
    private:
        struct CreateCloudWidget;
    };
    
    class CV_EXPORTS CloudNormalsWidget : public Widget3D
    {
    public:
        CloudNormalsWidget(InputArray _cloud, InputArray _normals, int level = 100, float scale = 0.02f, const Color &color = Color::white());
        CloudNormalsWidget(const Widget &other) : Widget3D(other) {}
        CloudNormalsWidget& operator=(const Widget &other);

    private:
        struct ApplyCloudNormals;
    };
}
