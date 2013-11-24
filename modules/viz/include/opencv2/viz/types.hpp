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

        class CV_EXPORTS KeyboardEvent
        {
        public:
            static const unsigned int Alt   = 1;
            static const unsigned int Ctrl  = 2;
            static const unsigned int Shift = 4;

            //! Create a keyboard event
            //! - Note that action is true if key is pressed, false if released
            KeyboardEvent(bool action, const String& key_sym, unsigned char key, bool alt, bool ctrl, bool shift);

            bool isAltPressed() const;
            bool isCtrlPressed() const;
            bool isShiftPressed() const;

            unsigned char getKeyCode() const;

            const String& getKeySym() const;
            bool keyDown() const;
            bool keyUp() const;

        protected:

            bool action_;
            unsigned int modifiers_;
            unsigned char key_code_;
            String key_sym_;
        };

        class CV_EXPORTS MouseEvent
        {
        public:
            enum Type { MouseMove = 1, MouseButtonPress, MouseButtonRelease, MouseScrollDown, MouseScrollUp, MouseDblClick } ;
            enum MouseButton { NoButton = 0, LeftButton, MiddleButton, RightButton, VScroll } ;

            MouseEvent(const Type& type, const MouseButton& button, const Point& p, bool alt, bool ctrl, bool shift);

            Type type;
            MouseButton button;
            Point pointer;
            unsigned int key_state;
        };

        class CV_EXPORTS Camera
        {
        public:
            Camera(float f_x, float f_y, float c_x, float c_y, const Size &window_size);
            Camera(const Vec2f &fov, const Size &window_size);
            Camera(const cv::Matx33f &K, const Size &window_size);
            Camera(const cv::Matx44f &proj, const Size &window_size);

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
            void init(float f_x, float f_y, float c_x, float c_y, const Size &window_size);

            Vec2d clip_;
            Vec2f fov_;
            Size window_size_;
            Vec2f principal_point_;
            Vec2f focal_;
        };
    } /* namespace viz */
} /* namespace cv */

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

inline cv::viz::Color::Color() : Scalar(0, 0, 0) {}
inline cv::viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
inline cv::viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
inline cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

inline cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0, 0); }
inline cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255, 0); }
inline cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0, 0); }
inline cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255, 0); }
inline cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
inline cv::viz::Color cv::viz::Color::yellow()  { return Color(  0, 255, 255); }
inline cv::viz::Color cv::viz::Color::magenta() { return Color(255,   0, 255); }
inline cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }
inline cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }


#endif
