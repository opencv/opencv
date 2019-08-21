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

#ifndef __OPENCV_VIZ_INTERACTOR_STYLE_H__
#define __OPENCV_VIZ_INTERACTOR_STYLE_H__

#include <vtkInteractorStyle.h>

namespace cv
{
    namespace viz
    {
        class vtkVizInteractorStyle : public vtkInteractorStyle
        {
        public:
            static vtkVizInteractorStyle *New();
            vtkTypeMacro(vtkVizInteractorStyle, vtkInteractorStyle)
            void PrintSelf(ostream& os, vtkIndent indent);

            virtual void OnChar();
            virtual void OnKeyDown();
            virtual void OnKeyUp();

            virtual void OnMouseMove();
            virtual void OnLeftButtonDown();
            virtual void OnLeftButtonUp();
            virtual void OnMiddleButtonDown();
            virtual void OnMiddleButtonUp();
            virtual void OnRightButtonDown();
            virtual void OnRightButtonUp();
            virtual void OnMouseWheelForward();
            virtual void OnMouseWheelBackward();
            virtual void OnTimer();

            virtual void Rotate();
            virtual void Spin();
            virtual void Pan();
            virtual void Dolly();

            vtkSetMacro(FlyMode,bool)
            vtkGetMacro(FlyMode,bool)


            vtkSetMacro(MotionFactor, double)
            vtkGetMacro(MotionFactor, double)

            void registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie = 0);
            void registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void * cookie = 0);

            void setWidgetActorMap(const Ptr<WidgetActorMap>& actors) { widget_actor_map_ = actors; }
            void saveScreenshot(const String &file);
            void exportScene(const String &file);
            void exportScene();
            void changePointsSize(float delta);
            void setRepresentationToPoints();
            void printCameraParams();
            void toggleFullScreen();
            void resetViewerPose();
            void toggleStereo();
            void printHelp();

            // Set the basic unit step size : by default 1/250 of bounding diagonal
            vtkSetMacro(MotionStepSize,double)
            vtkGetMacro(MotionStepSize,double)

            // Set acceleration factor when shift key is applied : default 10
            vtkSetMacro(MotionAccelerationFactor,double)
            vtkGetMacro(MotionAccelerationFactor,double)

            // Set the basic angular unit for turning : default 1 degree
            vtkSetMacro(AngleStepSize,double)
            vtkGetMacro(AngleStepSize,double)

        private:
            Ptr<WidgetActorMap> widget_actor_map_;

            Vec2i win_size_;
            Vec2i win_pos_;
            Vec2i max_win_size_;

            void zoomIn();
            void zoomOut();

        protected:
            vtkVizInteractorStyle();
            ~vtkVizInteractorStyle();

            virtual void Dolly(double factor);

            void Fly();
            void FlyByMouse();
            void FlyByKey();
            void SetupMotionVars();
            void MotionAlongVector(const Vec3d& vector, double amount, vtkCamera* cam);

        private:
            vtkVizInteractorStyle(const vtkVizInteractorStyle&);
            vtkVizInteractorStyle& operator=(const vtkVizInteractorStyle&);

            //! True for red-blue colors, false for magenta-green.
            bool stereo_anaglyph_redblue_;

            void (*keyboardCallback_)(const KeyboardEvent&, void*);
            void *keyboard_callback_cookie_;

            void (*mouseCallback_)(const MouseEvent&, void*);
            void *mouse_callback_cookie_;

            bool FlyMode;
            double MotionFactor;

            int getModifiers();

            // from fly
            unsigned char KeysDown;
            double        DiagonalLength;
            double        MotionStepSize;
            double        MotionUserScale;
            double        MotionAccelerationFactor;
            double        AngleStepSize;
            double        DeltaYaw;
            double        DeltaPitch;
        };
    } // end namespace viz
} // end namespace cv

#endif
