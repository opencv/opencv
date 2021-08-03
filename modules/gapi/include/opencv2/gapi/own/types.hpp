// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_TYPES_HPP
#define OPENCV_GAPI_TYPES_HPP

#include <algorithm>              // std::max, std::min
#include <ostream>

namespace cv
{
namespace gapi
{

/**
 * @brief This namespace contains G-API own data structures used in
 * its standalone mode build.
 */
namespace own
{

class Point
{
public:
    Point() = default;
    Point(int _x, int _y) : x(_x),  y(_y)  {};

    int x = 0;
    int y = 0;
};

class Point2f
{
public:
    Point2f() = default;
    Point2f(float _x, float _y) : x(_x),  y(_y)  {};

    float x = 0.f;
    float y = 0.f;
};

class Rect
{
public:
    Rect() = default;
    Rect(int _x, int _y, int _width, int _height) : x(_x), y(_y),   width(_width),  height(_height)  {};
#if !defined(GAPI_STANDALONE)
    Rect(const cv::Rect& other) : x(other.x), y(other.y), width(other.width), height(other.height) {};
    inline Rect& operator=(const cv::Rect& other)
    {
        x = other.x;
        y = other.x;
        width  = other.width;
        height = other.height;
        return *this;
    }
#endif // !defined(GAPI_STANDALONE)

    int x      = 0; //!< x coordinate of the top-left corner
    int y      = 0; //!< y coordinate of the top-left corner
    int width  = 0; //!< width of the rectangle
    int height = 0; //!< height of the rectangle
};

inline bool operator==(const Rect& lhs, const Rect& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height;
}

inline bool operator!=(const Rect& lhs, const Rect& rhs)
{
    return !(lhs == rhs);
}

inline Rect& operator&=(Rect& lhs, const Rect& rhs)
{
    int x1 = std::max(lhs.x, rhs.x);
    int y1 = std::max(lhs.y, rhs.y);
    lhs.width  = std::min(lhs.x + lhs.width,  rhs.x + rhs.width) -  x1;
    lhs.height = std::min(lhs.y + lhs.height, rhs.y + rhs.height) - y1;
    lhs.x = x1;
    lhs.y = y1;
    if( lhs.width <= 0 || lhs.height <= 0 )
        lhs = Rect();
    return lhs;
}

inline const Rect operator&(const Rect& lhs, const Rect& rhs)
{
    Rect result = lhs;
    return result &= rhs;
}

inline std::ostream& operator<<(std::ostream& o, const Rect& rect)
{
    return o << "[" << rect.width << " x " << rect.height << " from (" << rect.x << ", " << rect.y << ")]";
}

class Size
{
public:
    Size() = default;
    Size(int _width, int _height) : width(_width),  height(_height)  {};
#if !defined(GAPI_STANDALONE)
    Size(const cv::Size& other) : width(other.width), height(other.height) {};
    inline Size& operator=(const cv::Size& rhs)
    {
        width  = rhs.width;
        height = rhs.height;
        return *this;
    }
#endif // !defined(GAPI_STANDALONE)

    int width  = 0;
    int height = 0;
};

inline Size& operator+=(Size& lhs, const Size& rhs)
{
    lhs.width  += rhs.width;
    lhs.height += rhs.height;
    return lhs;
}

inline bool operator==(const Size& lhs, const Size& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

inline bool operator!=(const Size& lhs, const Size& rhs)
{
    return !(lhs == rhs);
}


inline std::ostream& operator<<(std::ostream& o, const Size& s)
{
    o << "[" << s.width << " x " << s.height << "]";
    return o;
}

} // namespace own
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_TYPES_HPP
