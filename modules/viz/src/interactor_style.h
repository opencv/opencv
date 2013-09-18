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

#ifndef __OPENCV_VIZ_INTERACTOR_STYLE_H__
#define __OPENCV_VIZ_INTERACTOR_STYLE_H__

#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
    {
        class InteractorStyle : public vtkInteractorStyleTrackballCamera
        {
        public:

            enum KeyboardModifier
            {
                KB_MOD_ALT,
                KB_MOD_CTRL,
                KB_MOD_SHIFT
            };

            static InteractorStyle *New();

            InteractorStyle() {}
            virtual ~InteractorStyle() {}

            // this macro defines Superclass, the isA functionality and the safe downcast method
            vtkTypeMacro(InteractorStyle, vtkInteractorStyleTrackballCamera)

            /** \brief Initialization routine. Must be called before anything else. */
            virtual void Initialize();

            inline void setWidgetActorMap(const Ptr<WidgetActorMap>& actors) { widget_actor_map_ = actors; }
            void setRenderer(vtkSmartPointer<vtkRenderer>& ren) { renderer_ = ren; }
            void registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie = 0);
            void registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void * cookie = 0);
            void saveScreenshot(const std::string &file);

            /** \brief Change the default keyboard modified from ALT to a different special key.*/
            inline void setKeyboardModifier(const KeyboardModifier &modifier) { modifier_ = modifier; }

        protected:
            /** \brief Set to true after initialization is complete. */
            bool init_;

            /** \brief Collection of vtkRenderers stored internally. */
            vtkSmartPointer<vtkRenderer> renderer_;

            /** \brief Actor map stored internally. */
            cv::Ptr<WidgetActorMap> widget_actor_map_;

            /** \brief The current window width/height. */
            Vec2i win_size_;

            /** \brief The current window position x/y. */
            Vec2i win_pos_;

            /** \brief The maximum resizeable window width/height. */
            Vec2i max_win_size_;

            /** \brief A PNG writer for screenshot captures. */
            vtkSmartPointer<vtkPNGWriter> snapshot_writer_;

            /** \brief Internal window to image filter. Needed by \a snapshot_writer_. */
            vtkSmartPointer<vtkWindowToImageFilter> wif_;

            /** \brief Interactor style internal method. Gets called whenever a key is pressed. */
            virtual void OnChar();

            // Keyboard events
            virtual void OnKeyDown();
            virtual void OnKeyUp();

            // mouse button events
            virtual void OnMouseMove();
            virtual void OnLeftButtonDown();
            virtual void OnLeftButtonUp();
            virtual void OnMiddleButtonDown();
            virtual void OnMiddleButtonUp();
            virtual void OnRightButtonDown();
            virtual void OnRightButtonUp();
            virtual void OnMouseWheelForward();
            virtual void OnMouseWheelBackward();

            /** \brief Interactor style internal method. Gets called periodically if a timer is set. */
            virtual void OnTimer();

            void zoomIn();
            void zoomOut();

            /** \brief True if we're using red-blue colors for anaglyphic stereo, false if magenta-green. */
            bool stereo_anaglyph_mask_default_;

            KeyboardModifier modifier_;

            void (*keyboardCallback_)(const KeyboardEvent&, void*);
            void *keyboard_callback_cookie_;

            void (*mouseCallback_)(const MouseEvent&, void*);
            void *mouse_callback_cookie_;
        };
    }
}

#endif
