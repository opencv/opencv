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

#include "precomp.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::Initialize()
{
    modifier_ = cv::viz::InteractorStyle::KB_MOD_ALT;
    // Set windows size (width, height) to unknown (-1)
    win_size_ = Vec2i(-1, -1);
    win_pos_ = Vec2i(0, 0);
    max_win_size_ = Vec2i(-1, -1);

    // Create the image filter and PNG writer objects
    wif_ = vtkSmartPointer<vtkWindowToImageFilter>::New();
    snapshot_writer_ = vtkSmartPointer<vtkPNGWriter>::New();
    snapshot_writer_->SetInputConnection(wif_->GetOutputPort());

    init_ = true;
    stereo_anaglyph_mask_default_ = true;

    // Initialize the keyboard event callback as none
    keyboardCallback_ = 0;
    keyboard_callback_cookie_ = 0;

    // Initialize the mouse event callback as none
    mouseCallback_ = 0;
    mouse_callback_cookie_ = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::saveScreenshot(const String &file)
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    wif_->SetInput(Interactor->GetRenderWindow());
    wif_->Modified(); // Update the WindowToImageFilter
    snapshot_writer_->Modified();
    snapshot_writer_->SetFileName(file.c_str());
    snapshot_writer_->Write();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::zoomIn()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    // Zoom in
    StartDolly();
    double factor = 10.0 * 0.2 * .5;
    Dolly(std::pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::zoomOut()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    // Zoom out
    StartDolly();
    double factor = 10.0 * -0.2 * .5;
    Dolly(std::pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnChar()
{
    // Make sure we ignore the same events we handle in OnKeyDown to avoid calling things twice
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    if (Interactor->GetKeyCode() >= '0' && Interactor->GetKeyCode() <= '9')
        return;

    String key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != String::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != String::npos)
        zoomOut();

    int keymod = false;
    switch (modifier_)
    {
    case KB_MOD_ALT:   keymod = Interactor->GetAltKey(); break;
    case KB_MOD_CTRL:  keymod = Interactor->GetControlKey(); break;
    case KB_MOD_SHIFT: keymod = Interactor->GetShiftKey(); break;
    }

    switch (Interactor->GetKeyCode())
    {
    // All of the options below simply exit
    case 'h': case 'H':
    case 'l': case 'L':
    case 'p': case 'P':
    case 'j': case 'J':
    case 'c': case 'C':
    case 43:        // KEY_PLUS
    case 45:        // KEY_MINUS
    case 'f': case 'F':
    case 'g': case 'G':
    case 'o': case 'O':
    case 'u': case 'U':
    case 'q': case 'Q':
    {
        break;
    }
        // S and R have a special !ALT case
    case 'r': case 'R':
    case 's': case 'S':
    {
        if (!keymod)
            Superclass::OnChar();
        break;
    }
    default:
    {
        Superclass::OnChar();
        break;
    }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie)
{
    // Register the callback function and store the user data
    mouseCallback_ = callback;
    mouse_callback_cookie_ = cookie;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void *cookie)
{
    // Register the callback function and store the user data
    keyboardCallback_ = callback;
    keyboard_callback_cookie_ = cookie;
}

//////////////////////////////////////////////////////////////////////////////////////////////
bool cv::viz::InteractorStyle::getAltKey() { return Interactor->GetAltKey() != 0; }
bool cv::viz::InteractorStyle::getShiftKey() { return Interactor->GetShiftKey()!= 0; }
bool cv::viz::InteractorStyle::getControlKey() { return Interactor->GetControlKey()!= 0; }

//////////////////////////////////////////////////////////////////////////////////////////////
void
cv::viz::InteractorStyle::OnKeyDown()
{
    CV_Assert("Interactor style not initialized. Please call Initialize() before continuing" && init_);
    CV_Assert("No renderer given! Use SetRendererCollection() before continuing." && renderer_);

    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);

    if (wif_->GetInput() == NULL)
    {
        wif_->SetInput(Interactor->GetRenderWindow());
        wif_->Modified();
        snapshot_writer_->Modified();
    }

    // Save the initial windows width/height
    if (win_size_[0] == -1 || win_size_[1] == -1)
        win_size_ = Vec2i(Interactor->GetRenderWindow()->GetSize());


    // Get the status of special keys (Cltr+Alt+Shift)
    bool shift = getShiftKey();
    bool ctrl  = getControlKey();
    bool alt   = getAltKey();

    bool keymod = false;
    switch (modifier_)
    {
    case KB_MOD_ALT:   keymod = alt;   break;
    case KB_MOD_CTRL:  keymod = ctrl;  break;
    case KB_MOD_SHIFT: keymod = shift; break;
    }

    std::string key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != std::string::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != std::string::npos)
        zoomOut();

    switch (Interactor->GetKeyCode())
    {
    case 'h': case 'H':
    {
        std::cout << "| Help:\n"
                     "-------\n"
                     "          p, P   : switch to a point-based representation\n"
                     "          w, W   : switch to a wireframe-based representation (where available)\n"
                     "          s, S   : switch to a surface-based representation (where available)\n"
                     "\n"
                     "          j, J   : take a .PNG snapshot of the current window view\n"
                     "          c, C   : display current camera/window parameters\n"
                     "          f, F   : fly to point mode\n"
                     "\n"
                     "          e, E   : exit the interactor\n"
                     "          q, Q   : stop and call VTK's TerminateApp\n"
                     "\n"
                     "           +/-   : increment/decrement overall point size\n"
                     "     +/- [+ ALT] : zoom in/out \n"
                     "\n"
                     "    r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]\n"
                     "\n"
                     "    ALT + s, S   : turn stereo mode on/off\n"
                     "    ALT + f, F   : switch between maximized window mode and original size\n"
                     "\n"
                     << std::endl;
        break;
    }

        // Switch representation to points
    case 'p': case 'P':
    {
        vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
        vtkCollectionSimpleIterator ait;
        for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
            for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
            {
                vtkActor* apart = reinterpret_cast <vtkActor*>(path->GetLastNode()->GetViewProp());
                apart->GetProperty()->SetRepresentationToPoints();
            }
        break;
    }
        // Save a PNG snapshot with the current screen
    case 'j': case 'J':
    {
        unsigned int t = static_cast<unsigned int>(time(0));
        String png_file = cv::format("screenshot-%d.png", t);
        String cam_file = cv::format("screenshot-%d.cam", t);

        vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();
        Vec2d clip;
        Vec3d focal, pos, view;
        cam->GetClippingRange(clip.val);
        cam->GetFocalPoint(focal.val);
        cam->GetPosition(pos.val);
        cam->GetViewUp(view.val);
        Vec2i win_pos(Interactor->GetRenderWindow()->GetPosition());
        Vec2i win_size(Interactor->GetRenderWindow()->GetSize());
        double angle = cam->GetViewAngle() / 180.0 * CV_PI;

        String data = cv::format("%f,%f/%f,%f,%f/%f,%f,%f/%f,%f,%f/%f/%d,%d/%d,%d", clip[0],clip[1], focal[0],focal[1],focal[2],
                 pos[0],pos[1],pos[2], view[0],view[1], view[2], angle , win_size[0],win_size[1], win_pos[0], win_pos[1]);

        saveScreenshot(png_file);
        ofstream ofs_cam(cam_file.c_str());
        ofs_cam << data.c_str() << endl;
        ofs_cam.close();

        cout << "Screenshot (" << png_file.c_str() << ") and camera information (" << cam_file.c_str() << ") successfully captured." << endl;
        break;
    }
        // display current camera settings/parameters
    case 'c': case 'C':
    {
        vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();

        Vec2d clip;
        Vec3d focal, pose, view;
        cam->GetClippingRange(clip.val);
        cam->GetFocalPoint(focal.val);
        cam->GetPosition(pose.val);
        cam->GetViewUp(view.val);
        Vec2i win_pos(Interactor->GetRenderWindow()->GetPosition());
        Vec2i win_size(Interactor->GetRenderWindow()->GetSize());

        cv::print(Mat(clip, false).reshape(1, 1));
        std::cout << "/";
        cv::print(Mat(focal, false).reshape(1, 1));
        std::cout << "/";
        cv::print(Mat(pose, false).reshape(1, 1));
        std::cout << "/";
        cv::print(Mat(view, false).reshape(1, 1));
        std::cout << "/" << cam->GetViewAngle () / 180.0 * CV_PI;
        cv::print(Mat(win_size, false).reshape(1, 1));
        std::cout << "/";
        cv::print(Mat(win_pos, false).reshape(1, 1));
        std::cout << std::endl;
        break;
    }
    case '=':
    {
        zoomIn();
        break;
    }
    case 43:        // KEY_PLUS
    {
        if (alt)
            zoomIn();
        else
        {
            vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
            vtkCollectionSimpleIterator ait;
            for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
                for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
                {
                    vtkActor* apart = reinterpret_cast <vtkActor*>(path->GetLastNode()->GetViewProp());
                    float psize = apart->GetProperty()->GetPointSize();
                    if (psize < 63.0f)
                        apart->GetProperty()->SetPointSize(psize + 1.0f);
                }
        }
        break;
    }
    case 45:        // KEY_MINUS
    {
        if (alt)
            zoomOut();
        else
        {
            vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
            vtkCollectionSimpleIterator ait;
            for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
                for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
                {
                    vtkActor* apart = static_cast<vtkActor*>(path->GetLastNode()->GetViewProp());
                    float psize = apart->GetProperty()->GetPointSize();
                    if (psize > 1.0f)
                        apart->GetProperty()->SetPointSize(psize - 1.0f);
                }
        }
        break;
    }
        // Switch between maximize and original window size
    case 'f': case 'F':
    {
        if (keymod)
        {
            Vec2i screen_size(Interactor->GetRenderWindow()->GetScreenSize());
            Vec2i win_size(Interactor->GetRenderWindow()->GetSize());

            // Is window size = max?
            if (win_size == max_win_size_)
            {
                Interactor->GetRenderWindow()->SetSize(win_size_.val);
                Interactor->GetRenderWindow()->SetPosition(win_pos_.val);
                Interactor->GetRenderWindow()->Render();
                Interactor->Render();
            }
            // Set to max
            else
            {
                win_pos_ = Vec2i(Interactor->GetRenderWindow()->GetPosition());
                win_size_ = win_size;

                Interactor->GetRenderWindow()->SetSize(screen_size.val);
                Interactor->GetRenderWindow()->Render();
                Interactor->Render();
                max_win_size_ = Vec2i(Interactor->GetRenderWindow()->GetSize());
            }
        }
        else
        {
            AnimState = VTKIS_ANIM_ON;
            vtkAssemblyPath *path = NULL;
            Interactor->GetPicker()->Pick(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1], 0.0, CurrentRenderer);
            vtkAbstractPropPicker *picker;
            if ((picker = vtkAbstractPropPicker::SafeDownCast(Interactor->GetPicker())))
                path = picker->GetPath();
            if (path != NULL)
                Interactor->FlyTo(CurrentRenderer, picker->GetPickPosition());
            AnimState = VTKIS_ANIM_OFF;
        }
        break;
    }
        // 's'/'S' w/out ALT
    case 's': case 'S':
    {
        if (keymod)
        {
            int stereo_render = Interactor->GetRenderWindow()->GetStereoRender();
            if (!stereo_render)
            {
                if (stereo_anaglyph_mask_default_)
                {
                    Interactor->GetRenderWindow()->SetAnaglyphColorMask(4, 3);
                    stereo_anaglyph_mask_default_ = false;
                }
                else
                {
                    Interactor->GetRenderWindow()->SetAnaglyphColorMask(2, 5);
                    stereo_anaglyph_mask_default_ = true;
                }
            }
            Interactor->GetRenderWindow()->SetStereoRender(!stereo_render);
            Interactor->GetRenderWindow()->Render();
            Interactor->Render();
        }
        else
            Superclass::OnKeyDown();
        break;
    }

    case 'o': case 'O':
    {
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        cam->SetParallelProjection(!cam->GetParallelProjection());
        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->Render();
        break;
    }

    // Overwrite the camera reset
    case 'r': case 'R':
    {
        if (!keymod)
        {
            Superclass::OnKeyDown();
            break;
        }

        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();

        static WidgetActorMap::iterator it = widget_actor_map_->begin();
        // it might be that some actors don't have a valid transformation set -> we skip them to avoid a seg fault.
        bool found_transformation = false;

        for (size_t idx = 0; idx < widget_actor_map_->size(); ++idx, ++it)
        {
            if (it == widget_actor_map_->end())
                it = widget_actor_map_->begin();

            vtkProp3D * actor = vtkProp3D::SafeDownCast(it->second);
            if (actor && actor->GetUserMatrix())
            {
                found_transformation = true;
                break;
            }
        }

        // if a valid transformation was found, use it otherwise fall back to default view point.
        if (found_transformation)
        {
            const vtkMatrix4x4* m = vtkProp3D::SafeDownCast(it->second)->GetUserMatrix();

            cam->SetFocalPoint(m->GetElement(0, 3) - m->GetElement(0, 2),
                               m->GetElement(1, 3) - m->GetElement(1, 2),
                               m->GetElement(2, 3) - m->GetElement(2, 2));

            cam->SetViewUp  (m->GetElement(0, 1), m->GetElement(1, 1), m->GetElement(2, 1));
            cam->SetPosition(m->GetElement(0, 3), m->GetElement(1, 3), m->GetElement(2, 3));
        }
        else
        {
            cam->SetPosition(0, 0, 0);
            cam->SetFocalPoint(0, 0, 1);
            cam->SetViewUp(0, -1, 0);
        }

        // go to the next actor for the next key-press event.
        if (it != widget_actor_map_->end())
            ++it;
        else
            it = widget_actor_map_->begin();

        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Render();
        break;
    }

    case 'q': case 'Q':
    {
        Interactor->ExitCallback();
        return;
    }
    default:
    {
        Superclass::OnKeyDown();
        break;
    }
    }

    KeyboardEvent event(true, Interactor->GetKeySym(), Interactor->GetKeyCode(), getAltKey(), getControlKey(), getShiftKey());
    // Check if there is a keyboard callback registered
    if (keyboardCallback_)
      keyboardCallback_(event, keyboard_callback_cookie_);

    renderer_->Render();
    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnKeyUp()
{
    KeyboardEvent event(false, Interactor->GetKeySym(), Interactor->GetKeyCode(), getAltKey(), getControlKey(), getShiftKey());
    // Check if there is a keyboard callback registered
    if (keyboardCallback_)
      keyboardCallback_(event, keyboard_callback_cookie_);

    Superclass::OnKeyUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnMouseMove()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseMove, MouseEvent::NoButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnMouseMove();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnLeftButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::LeftButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnLeftButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnLeftButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::LeftButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnLeftButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnMiddleButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());

    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::MiddleButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnMiddleButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnMiddleButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::MiddleButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnMiddleButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnRightButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());

    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::RightButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnRightButtonDown();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnRightButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::RightButton, p, getAltKey(), getControlKey(), getShiftKey());
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    Superclass::OnRightButtonUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnMouseWheelForward()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseScrollUp, MouseEvent::VScroll, p, getAltKey(), getControlKey(), getShiftKey());
    // If a mouse callback registered, call it!
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);
    if (Interactor->GetRepeatCount() && mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);

    if (Interactor->GetAltKey())
    {
        // zoom
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        double opening_angle = cam->GetViewAngle();
        if (opening_angle > 15.0)
            opening_angle -= 1.0;

        cam->SetViewAngle(opening_angle);
        cam->Modified();
        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        CurrentRenderer->Render();
        renderer_->Render();
        Interactor->Render();
    }
    else
        Superclass::OnMouseWheelForward();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnMouseWheelBackward()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseScrollDown, MouseEvent::VScroll, p, getAltKey(), getControlKey(), getShiftKey());
    // If a mouse callback registered, call it!
    if (mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);

    if (Interactor->GetRepeatCount() && mouseCallback_)
      mouseCallback_(event, mouse_callback_cookie_);

    if (Interactor->GetAltKey())
    {
        // zoom
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        double opening_angle = cam->GetViewAngle();
        if (opening_angle < 170.0)
            opening_angle += 1.0;

        cam->SetViewAngle(opening_angle);
        cam->Modified();
        CurrentRenderer->SetActiveCamera(cam);
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        CurrentRenderer->Render();
        renderer_->Render();
        Interactor->Render();
    }
    else
        Superclass::OnMouseWheelBackward();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::InteractorStyle::OnTimer()
{
    CV_Assert("Interactor style not initialized." && init_);
    CV_Assert("Renderer has not been set." && renderer_);
    renderer_->Render();
    Interactor->Render();
}

namespace cv { namespace viz
{
    //Standard VTK macro for *New()
    vtkStandardNewMacro(InteractorStyle)
}}
