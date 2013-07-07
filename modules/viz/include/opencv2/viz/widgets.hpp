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
        
        void copyTo(Widget &dst);

        void setColor(const Color &color);
        void setPose(const Affine3f &pose);
        void updatePose(const Affine3f &pose);
        Affine3f getPose() const;

    private:
        class Impl;
        Impl* impl_;
        
        void create();
        void release();
        
        friend struct WidgetAccessor;
    };


    class CV_EXPORTS LineWidget : public Widget
    {
    public:
        LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
        
        void setLineWidth(float line_width);
        float getLineWidth();
    };
    
    class CV_EXPORTS PlaneWidget : public Widget
    {
    public:
        PlaneWidget(const Vec4f& coefs, double size = 1.0, const Color &color = Color::white());
        PlaneWidget(const Vec4f& coefs, const Point3f& pt, double size = 1.0, const Color &color = Color::white());
    };
    
    class CV_EXPORTS SphereWidget : public Widget
    {
    public:
        SphereWidget(const cv::Point3f &center, float radius, int sphere_resolution = 10, const Color &color = Color::white());
    };
    
    class CV_EXPORTS ArrowWidget : public Widget
    {
    public:
        ArrowWidget(const Point3f& pt1, const Point3f& pt2, const Color &color = Color::white());
    };

    class CV_EXPORTS CircleWidget : public Widget
    {
    public:
        CircleWidget(const Point3f& pt, double radius, double thickness = 0.01, const Color &color = Color::white());
    };
    
    class CV_EXPORTS CylinderWidget : public Widget
    {
    public:
        CylinderWidget(const Point3f& pt_on_axis, const Point3f& axis_direction, double radius, int numsides = 30, const Color &color = Color::white());
    };
    
    class CV_EXPORTS CubeWidget : public Widget
    {
    public:
        CubeWidget(const Point3f& pt_min, const Point3f& pt_max, const Color &color = Color::white());
    };
    
    class CV_EXPORTS CoordinateSystemWidget : public Widget
    {
    public:
        CoordinateSystemWidget(double scale, const Affine3f& affine);
    };

}
