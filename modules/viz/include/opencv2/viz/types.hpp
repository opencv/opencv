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

#ifndef OPENCV_VIZ_TYPES_HPP
#define OPENCV_VIZ_TYPES_HPP

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

namespace cv
{
    namespace viz
    {

//! @addtogroup viz
//! @{

        /** @brief This class represents color in BGR order.
        */
        class Color : public Scalar
        {
        public:
            Color();
            //! The three channels will have the same value equal to gray.
            Color(double gray);
            Color(double blue, double green, double red);

            Color(const Scalar& color);

            operator Vec3b() const;

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

            static Color not_set();
        };

        /** @brief This class wraps mesh attributes, and it can load a mesh from a ply file. :
        */
        class CV_EXPORTS Mesh
        {
        public:
            enum {
                LOAD_AUTO = 0,
                LOAD_PLY = 1,
                LOAD_OBJ = 2
            };

            Mat cloud; //!< point coordinates of type CV_32FC3 or CV_64FC3 with only 1 row
            Mat colors; //!< point color of type CV_8UC3 or CV_8UC4 with only 1 row
            Mat normals; //!< point normals of type CV_32FC3, CV_32FC4, CV_64FC3 or CV_64FC4 with only 1 row

            //! Raw integer list of the form: (n,id1,id2,...,idn, n,id1,id2,...,idn, ...)
            //! where n is the number of points in the polygon, and id is a zero-offset index into an associated cloud.
            Mat polygons; //!< CV_32SC1 with only 1 row

            Mat texture;
            Mat tcoords; //!< CV_32FC2 or CV_64FC2 with only 1 row

            /** @brief Loads a mesh from a ply or a obj file.

            @param file File name
            @param type File type (for now only PLY and OBJ are supported)

            **File type** can be one of the following:
            -   **LOAD_PLY**
            -   **LOAD_OBJ**
             */
            static Mesh load(const String& file, int type = LOAD_PLY);

        };

        /** @brief This class wraps intrinsic parameters of a camera.

        It provides several constructors that can extract the intrinsic parameters from field of
        view, intrinsic matrix and projection matrix. :
         */
        class CV_EXPORTS Camera
        {
        public:

            /** @brief Constructs a Camera.

            @param fx Horizontal focal length.
            @param fy Vertical focal length.
            @param cx x coordinate of the principal point.
            @param cy y coordinate of the principal point.
            @param window_size Size of the window. This together with focal length and principal
            point determines the field of view.
             */
            Camera(double fx, double fy, double cx, double cy, const Size &window_size);

            /** @overload
            @param fov Field of view (horizontal, vertical)
            @param window_size Size of the window. Principal point is at the center of the window
            by default.
            */
            Camera(const Vec2d &fov, const Size &window_size);

            /** @overload
            @param K Intrinsic matrix of the camera with the following form
            \f[
              \begin{bmatrix}
              f_x &   0 & c_x\\
                0 & f_y & c_y\\
                0 &   0 &   1\\
              \end{bmatrix}
            \f]
            @param window_size Size of the window. This together with intrinsic matrix determines
            the field of view.
            */
            Camera(const Matx33d &K, const Size &window_size);

            /** @overload
            @param proj Projection matrix of the camera with the following form
            \f[
              \begin{bmatrix}
              \frac{2n}{r-l} &        0       & \frac{r+l}{r-l}  & 0\\
                    0        & \frac{2n}{t-b} & \frac{t+b}{t-b}  & 0\\
                    0        &        0       & -\frac{f+n}{f-n} & -\frac{2fn}{f-n}\\
                    0        &        0       & -1               & 0\\
              \end{bmatrix}
            \f]

            @param window_size Size of the window. This together with projection matrix determines
            the field of view.
            */
            explicit Camera(const Matx44d &proj, const Size &window_size);

            const Vec2d & getClip() const { return clip_; }
            void setClip(const Vec2d &clip) { clip_ = clip; }

            const Size & getWindowSize() const { return window_size_; }
            void setWindowSize(const Size &window_size);

            const Vec2d& getFov() const { return fov_; }
            void setFov(const Vec2d& fov) { fov_ = fov; }

            const Vec2d& getPrincipalPoint() const { return principal_point_; }
            const Vec2d& getFocalLength() const { return focal_; }

            /** @brief Computes projection matrix using intrinsic parameters of the camera.


            @param proj Output projection matrix with the following form
            \f[
              \begin{bmatrix}
              \frac{2n}{r-l} &        0       & \frac{r+l}{r-l}  & 0\\
                    0        & \frac{2n}{t-b} & \frac{t+b}{t-b}  & 0\\
                    0        &        0       & -\frac{f+n}{f-n} & -\frac{2fn}{f-n}\\
                    0        &        0       & -1               & 0\\
              \end{bmatrix}
            \f]
             */
            void computeProjectionMatrix(Matx44d &proj) const;

            /** @brief Creates a Kinect Camera with
              - fx = fy = 525
              - cx = 320
              - cy = 240

            @param window_size Size of the window. This together with intrinsic matrix of a Kinect Camera
            determines the field of view.
             */
            static Camera KinectCamera(const Size &window_size);

        private:
            void init(double fx, double fy, double cx, double cy, const Size &window_size);

            /** The near plane and the far plane.
             *  - clip_[0]: the near plane; default value is 0.01
             *  - clip_[1]: the far plane; default value is 1000.01
             */
            Vec2d clip_;

            /**
             * Field of view.
             *  - fov_[0]: horizontal(x-axis) field of view in radians
             *  - fov_[1]: vertical(y-axis) field of view in radians
             */
            Vec2d fov_;

            /** Window size.*/
            Size window_size_;

            /**
             * Principal point.
             *  - principal_point_[0]: cx
             *  - principal_point_[1]: cy
             */
            Vec2d principal_point_;
            /**
             * Focal length.
             *  - focal_[0]: fx
             *  - focal_[1]: fy
             */
            Vec2d focal_;
        };

        /** @brief This class represents a keyboard event.
        */
        class CV_EXPORTS KeyboardEvent
        {
        public:
            enum { NONE = 0, ALT = 1, CTRL = 2, SHIFT = 4 };
            enum Action { KEY_UP = 0, KEY_DOWN = 1 };

            /** @brief Constructs a KeyboardEvent.

            @param action Signals if key is pressed or released.
            @param symbol Name of the key.
            @param code Code of the key.
            @param modifiers Signals if alt, ctrl or shift are pressed or their combination.
             */
            KeyboardEvent(Action action, const String& symbol, unsigned char code, int modifiers);

            Action action;
            String symbol;
            unsigned char code;
            int modifiers;
        };

        /** @brief This class represents a mouse event.
        */
        class CV_EXPORTS MouseEvent
        {
        public:
            enum Type { MouseMove = 1, MouseButtonPress, MouseButtonRelease, MouseScrollDown, MouseScrollUp, MouseDblClick } ;
            enum MouseButton { NoButton = 0, LeftButton, MiddleButton, RightButton, VScroll } ;

            /** @brief Constructs a MouseEvent.

            @param type Type of the event. This can be **MouseMove**, **MouseButtonPress**,
            **MouseButtonRelease**, **MouseScrollDown**, **MouseScrollUp**, **MouseDblClick**.
            @param button Mouse button. This can be **NoButton**, **LeftButton**, **MiddleButton**,
            **RightButton**, **VScroll**.
            @param pointer Position of the event.
            @param modifiers Signals if alt, ctrl or shift are pressed or their combination.
             */
            MouseEvent(const Type& type, const MouseButton& button, const Point& pointer, int modifiers);

            Type type;
            MouseButton button;
            Point pointer;
            int modifiers;
        };

//! @} viz

    } /* namespace viz */
} /* namespace cv */

//! @cond IGNORED

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

inline cv::viz::Color::Color() : Scalar(0, 0, 0) {}
inline cv::viz::Color::Color(double _gray) : Scalar(_gray, _gray, _gray) {}
inline cv::viz::Color::Color(double _blue, double _green, double _red) : Scalar(_blue, _green, _red) {}
inline cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

inline cv::viz::Color::operator cv::Vec3b() const { return cv::Vec3d(val); }

inline cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0,   0); }
inline cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255,   0); }
inline cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0,   0); }
inline cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255,   0); }
inline cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
inline cv::viz::Color cv::viz::Color::yellow()  { return Color(  0, 255, 255); }
inline cv::viz::Color cv::viz::Color::magenta() { return Color(255,   0, 255); }
inline cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }
inline cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }

inline cv::viz::Color cv::viz::Color::mlab()    { return Color(255, 128, 128); }

inline cv::viz::Color cv::viz::Color::navy()       { return Color(0,     0, 128); }
inline cv::viz::Color cv::viz::Color::olive()      { return Color(0,   128, 128); }
inline cv::viz::Color cv::viz::Color::maroon()     { return Color(0,     0, 128); }
inline cv::viz::Color cv::viz::Color::teal()       { return Color(128, 128,   0); }
inline cv::viz::Color cv::viz::Color::rose()       { return Color(128,   0, 255); }
inline cv::viz::Color cv::viz::Color::azure()      { return Color(255, 128,   0); }
inline cv::viz::Color cv::viz::Color::lime()       { return Color(0,   255, 191); }
inline cv::viz::Color cv::viz::Color::gold()       { return Color(0,   215, 255); }
inline cv::viz::Color cv::viz::Color::brown()      { return Color(42,    42, 165); }
inline cv::viz::Color cv::viz::Color::orange()     { return Color(0,   165, 255); }
inline cv::viz::Color cv::viz::Color::chartreuse() { return Color(0,   255, 128); }
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

inline cv::viz::Color cv::viz::Color::not_set()        { return Color(-1, -1, -1); }

//! @endcond

#endif
