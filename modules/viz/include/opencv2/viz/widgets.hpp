#pragma once

#include <opencv2/viz/types.hpp>


namespace temp_viz
{
    /////////////////////////////////////////////////////////////////////////////
    /// brief The base class for all widgets
    class CV_EXPORTS Widget
    {
    public:
        Widget();
        Widget(const Widget &other);
        Widget& operator =(const Widget &other);

        void copyTo(Widget &dst);

        void setColor(const Color &color);
        void setPose(const Affine3f &pose);
        void updatePose(const Affine3f &pose);
        Affine3f getPose() const;

    private:
        class Impl;
        cv::Ptr<Impl> impl_;
        friend struct WidgetAccessor;
    };


    class CV_EXPORTS LineWidget : public Widget
    {
    public:
        LineWidget(const Point3f &pt1, const Point3f &pt2, const Color &color = Color::white());
    };


}
