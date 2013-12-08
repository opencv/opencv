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

#ifndef __OPENCV_VIZ_TYPES_HPP__
#define __OPENCV_VIZ_TYPES_HPP__

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

namespace cv
{
    namespace viz
    {
        class Color : public Scalar
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

            static Color mlab();

            static Color navy();
            static Color olive();
            static Color maroon();
            static Color teal();
            static Color rose();
            static Color azure();
            static Color lime();
            static Color gold();
            static Color brown();
            static Color orange();
            static Color chartreuse();
            static Color orange_red();
            static Color purple();
            static Color indigo();

            static Color pink();
            static Color cherry();
            static Color bluberry();
            static Color raspberry();
            static Color silver();
            static Color violet();
            static Color apricot();
            static Color turquoise();
            static Color celestial_blue();
            static Color amethyst();
        };

        class CV_EXPORTS Mesh3d
        {
        public:

            Mat cloud, colors;
            Mat polygons;

            //! Loads mesh from a given ply file
            static cv::viz::Mesh3d loadMesh(const String& file);

        private:
            struct loadMeshImpl;
        };

        class CV_EXPORTS Camera
        {
        public:
            Camera(float fx, float fy, float cx, float cy, const Size &window_size);
            explicit Camera(const Vec2f &fov, const Size &window_size);
            explicit Camera(const cv::Matx33f &K, const Size &window_size);
            explicit Camera(const cv::Matx44f &proj, const Size &window_size);

            inline const Vec2d & getClip() const { return clip_; }
            inline void setClip(const Vec2d &clip) { clip_ = clip; }

            inline const Size & getWindowSize() const { return window_size_; }
            void setWindowSize(const Size &window_size);

            inline const Vec2f & getFov() const { return fov_; }
            inline void setFov(const Vec2f & fov) { fov_ = fov; }

            inline const Vec2f & getPrincipalPoint() const { return principal_point_; }
            inline const Vec2f & getFocalLength() const { return focal_; }

            void computeProjectionMatrix(Matx44f &proj) const;

            static Camera KinectCamera(const Size &window_size);

        private:
            void init(float fx, float fy, float cx, float cy, const Size &window_size);

            Vec2d clip_;
            Vec2f fov_;
            Size window_size_;
            Vec2f principal_point_;
            Vec2f focal_;
        };

        class CV_EXPORTS KeyboardEvent
        {
        public:
            enum { NONE = 0, ALT = 1, CTRL = 2, SHIFT = 4 };
            enum Action { KEY_UP = 0, KEY_DOWN = 1 };

            KeyboardEvent(Action action, const String& symbol, unsigned char code, int modifiers);

            Action action;
            String symbol;
            unsigned char code;
            int modifiers;
        };

        class CV_EXPORTS MouseEvent
        {
        public:
            enum Type { MouseMove = 1, MouseButtonPress, MouseButtonRelease, MouseScrollDown, MouseScrollUp, MouseDblClick } ;
            enum MouseButton { NoButton = 0, LeftButton, MiddleButton, RightButton, VScroll } ;

            MouseEvent(const Type& type, const MouseButton& button, const Point& pointer, int modifiers);

            Type type;
            MouseButton button;
            Point pointer;
            int modifiers;
        };
    } /* namespace viz */
} /* namespace cv */

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

inline cv::viz::Color::Color() : Scalar(0, 0, 0) {}
inline cv::viz::Color::Color(double _gray) : Scalar(_gray, _gray, _gray) {}
inline cv::viz::Color::Color(double _blue, double _green, double _red) : Scalar(_blue, _green, _red) {}
inline cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

inline cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0,   0); }
inline cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255,   0); }
inline cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0,   0); }
inline cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255,   0); }
inline cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
inline cv::viz::Color cv::viz::Color::yellow()  { return Color(  0, 255, 255); }
inline cv::viz::Color cv::viz::Color::magenta() { return Color(255,   0, 255); }
inline cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }
inline cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }

inline cv::viz::Color cv::viz::Color::mlab()    { return Color(235, 118, 118); }

inline cv::viz::Color cv::viz::Color::navy()       { return Color(0,     0, 128); }
inline cv::viz::Color cv::viz::Color::olive()      { return Color(0,   128, 128); }
inline cv::viz::Color cv::viz::Color::maroon()     { return Color(0,     0, 128); }
inline cv::viz::Color cv::viz::Color::teal()       { return Color(128, 128,   0); }
inline cv::viz::Color cv::viz::Color::rose()       { return Color(127,   0, 255); }
inline cv::viz::Color cv::viz::Color::azure()      { return Color(255, 127,   0); }
inline cv::viz::Color cv::viz::Color::lime()       { return Color(0,   255, 191); }
inline cv::viz::Color cv::viz::Color::gold()       { return Color(0,   215, 255); }
inline cv::viz::Color cv::viz::Color::brown()      { return Color(0,    75, 150); }
inline cv::viz::Color cv::viz::Color::orange()     { return Color(0,   165, 255); }
inline cv::viz::Color cv::viz::Color::chartreuse() { return Color(0,   255, 127); }
inline cv::viz::Color cv::viz::Color::orange_red() { return Color(0,    69, 255); }
inline cv::viz::Color cv::viz::Color::purple()     { return Color(128,   0, 128); }
inline cv::viz::Color cv::viz::Color::indigo()     { return Color(130,   0,  75); }

inline cv::viz::Color cv::viz::Color::pink()           { return Color(203, 192, 255); }
inline cv::viz::Color cv::viz::Color::cherry()         { return Color( 99,  29, 222); }
inline cv::viz::Color cv::viz::Color::bluberry()       { return Color(247, 134,  79); }
inline cv::viz::Color cv::viz::Color::raspberry()      { return Color( 92,  11, 227); }
inline cv::viz::Color cv::viz::Color::silver()         { return Color(192, 192, 192); }
inline cv::viz::Color cv::viz::Color::violet()         { return Color(226,  43, 138); }
inline cv::viz::Color cv::viz::Color::apricot()        { return Color(177, 206, 251); }
inline cv::viz::Color cv::viz::Color::turquoise()      { return Color(208, 224,  64); }
inline cv::viz::Color cv::viz::Color::celestial_blue() { return Color(208, 151,  73); }
inline cv::viz::Color cv::viz::Color::amethyst()       { return Color(204, 102, 153); }





#endif
