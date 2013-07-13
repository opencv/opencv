#pragma once

#include "viz_types.h"
#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
    {
        /** \brief InteractorStyle defines an unique, custom VTK based interactory style Viz applications.
          * Besides defining the rendering style, we also create a list of custom actions
          * that are triggered on different keys being pressed:
          *
          * -        p, P   : switch to a point-based representation
          * -        w, W   : switch to a wireframe-based representation (where available)
          * -        s, S   : switch to a surface-based representation (where available)
          * -        j, J   : take a .PNG snapshot of the current window view
          * -        c, C   : display current camera/window parameters
          * -        f, F   : fly to point mode
          * -        e, E   : exit the interactor
          * -        q, Q   : stop and call VTK's TerminateApp
          * -       + / -   : increment/decrement overall point size
          * -  r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]
          * -  ALT + s, S   : turn stereo mode on/off
          * -  ALT + f, F   : switch between maximized window mode and original size
          * -
          * -  SHIFT + left click   : select a point
          *
          * \author Radu B. Rusu
          */
        class InteractorStyle : public vtkInteractorStyleTrackballCamera
        {
        public:

            enum KeyboardModifier
            {
                KB_MOD_ALT,
                KB_MOD_CTRL,
                KB_MOD_SHIFT
            };

            static InteractorStyle *New ();

            InteractorStyle () {}
            virtual ~InteractorStyle () {}

            // this macro defines Superclass, the isA functionality and the safe downcast method
            vtkTypeMacro (InteractorStyle, vtkInteractorStyleTrackballCamera)

            /** \brief Initialization routine. Must be called before anything else. */
            virtual void Initialize ();

            inline void setCloudActorMap (const Ptr<CloudActorMap>& actors) { actors_ = actors; }
            void setRenderer (vtkSmartPointer<vtkRenderer>& ren) { renderer_ = ren; }
            void registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie = 0);
            void registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void * cookie = 0);
            void saveScreenshot (const std::string &file);

            /** \brief Change the default keyboard modified from ALT to a different special key.
                  * Allowed values are:
                  * - KB_MOD_ALT
                  * - KB_MOD_CTRL
                  * - KB_MOD_SHIFT
                  * \param[in] modifier the new keyboard modifier
                  */
            inline void setKeyboardModifier (const KeyboardModifier &modifier) { modifier_ = modifier; }
        protected:
            /** \brief Set to true after initialization is complete. */
            bool init_;

            /** \brief Collection of vtkRenderers stored internally. */
            //vtkSmartPointer<vtkRendererCollection> rens_;
            vtkSmartPointer<vtkRenderer> renderer_;

            /** \brief Actor map stored internally. */
            cv::Ptr<CloudActorMap> actors_;

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
            virtual void OnChar ();

            // Keyboard events
            virtual void OnKeyDown ();
            virtual void OnKeyUp ();

            // mouse button events
            virtual void OnMouseMove ();
            virtual void OnLeftButtonDown ();
            virtual void OnLeftButtonUp ();
            virtual void OnMiddleButtonDown ();
            virtual void OnMiddleButtonUp ();
            virtual void OnRightButtonDown ();
            virtual void OnRightButtonUp ();
            virtual void OnMouseWheelForward ();
            virtual void OnMouseWheelBackward ();

            /** \brief Interactor style internal method. Gets called periodically if a timer is set. */
            virtual void OnTimer ();

            void zoomIn ();
            void zoomOut ();

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
