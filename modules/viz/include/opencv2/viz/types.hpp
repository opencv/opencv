#pragma once

#include <vector>
#include <opencv2/core/cvdef.h>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

namespace temp_viz
{
    //qt creator hack
    typedef cv::Scalar Scalar;
    typedef cv::Mat Mat;
    typedef std::string String;

    typedef cv::Vec3d Vec3d;
    typedef cv::Vec3f Vec3f;
    typedef cv::Vec4d Vec4d;
    typedef cv::Vec4f Vec4f;
    typedef cv::Vec2d Vec2d;
    typedef cv::Vec2i Vec2i;
    typedef cv::Vec3b Vec3b;
    typedef cv::Matx33d Matx33d;
    typedef cv::Affine3f Affine3f;
    typedef cv::Affine3d Affine3d;
    typedef cv::Point2i Point2i;
    typedef cv::Point3f Point3f;
    typedef cv::Point3d Point3d;
    typedef cv::Matx44d Matx44d;
    typedef cv::Matx44f Matx44f;
    typedef cv::Size Size;
    typedef cv::Point Point;
    typedef cv::InputArray InputArray;
    using cv::Point3_;
    using cv::Vec;
    using cv::Mat_;
    using cv::DataDepth;
    using cv::DataType;

    class CV_EXPORTS Color : public Scalar
    {
    public:
        Color();
        Color(double gray);
        Color(double blue, double green, double red);

        Color(const Scalar& color);

        static Color black();
        static Color blue();
        static Color green();
        static Color cyan();

        static Color red();
        static Color magenta();
        static Color yellow();
        static Color white();

        static Color gray();
    };


    struct CV_EXPORTS Vertices
    {
        std::vector<unsigned int> vertices;
    };

    class CV_EXPORTS Mesh3d
    {
    public:
        typedef Ptr<Mesh3d> Ptr;

        Mat cloud, colors;
        std::vector<Vertices> polygons;

        static Mesh3d::Ptr mesh_load(const String& file);

    };

    /////////////////////////////////////////////////////////////////////////////
    /// Utility functions

    inline Color vtkcolor(const Color& color)
    {
        Color scaled_color = color * (1.0/255.0);
        std::swap(scaled_color[0], scaled_color[2]);
        return scaled_color;
    }

    inline Vec3d vtkpoint(const Point3f& point) { return Vec3d(point.x, point.y, point.z); }
    template<typename _Tp> inline _Tp normalized(const _Tp& v) { return v * 1/cv::norm(v); }
    
    inline bool isNan(float x)
    {
        unsigned int *u = reinterpret_cast<unsigned int *>(&x);
        return ((u[0] & 0x7f800000) == 0x7f800000) && (u[0] & 0x007fffff);
    }
    
    inline bool isNan(double x)
    {
        unsigned int *u = reinterpret_cast<unsigned int *>(&x);
        return (u[1] & 0x7ff00000) == 0x7ff00000 && (u[0] != 0 || (u[1] & 0x000fffff) != 0);
    }    

    template<typename _Tp, int cn> inline bool isNan(const Vec<_Tp, cn>& v)
    { return isNan(v.val[0]) || isNan(v.val[1]) || isNan(v.val[2]); }

    template<typename _Tp> inline bool isNan(const Point3_<_Tp>& p)
    { return isNan(p.x) || isNan(p.y) || isNan(p.z); }
}
