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

#include "../precomp.hpp"

namespace cv { namespace viz
{
    vtkStandardNewMacro(vtkVizInteractorStyle)
}}

//////////////////////////////////////////////////////////////////////////////////////////////

cv::viz::vtkVizInteractorStyle::vtkVizInteractorStyle()
{
    FlyMode = false;
    MotionFactor = 10.0;

    keyboardCallback_ = 0;
    keyboard_callback_cookie_ = 0;

    mouseCallback_ = 0;
    mouse_callback_cookie_ = 0;

    // Set windows size (width, height) to unknown (-1)
    win_size_ = Vec2i(-1, -1);
    win_pos_ = Vec2i(0, 0);
    max_win_size_ = Vec2i(-1, -1);

    stereo_anaglyph_redblue_ = true;

    //from fly
    KeysDown     = 0;
    UseTimers    = 1;

    DiagonalLength           = 1.0;
    MotionStepSize           = 1.0/100.0;
    MotionUserScale          = 1.0;  // +/- key adjustment
    MotionAccelerationFactor = 10.0;
    AngleStepSize            = 1.0;
}

cv::viz::vtkVizInteractorStyle::~vtkVizInteractorStyle() {}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::saveScreenshot(const String &file)
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);

    vtkSmartPointer<vtkWindowToImageFilter> wif = vtkSmartPointer<vtkWindowToImageFilter>::New();
    wif->SetInput(Interactor->GetRenderWindow());

    vtkSmartPointer<vtkPNGWriter> snapshot_writer = vtkSmartPointer<vtkPNGWriter>::New();
    snapshot_writer->SetInputConnection(wif->GetOutputPort());
    snapshot_writer->SetFileName(file.c_str());
    snapshot_writer->Write();

    cout << "Screenshot successfully captured (" << file.c_str() << ")" << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::exportScene(const String &file)
{
    vtkSmartPointer<vtkExporter> exporter;
    if (file.size() > 5 && file.substr(file.size() - 5) == ".vrml")
    {
        exporter = vtkSmartPointer<vtkVRMLExporter>::New();
        vtkVRMLExporter::SafeDownCast(exporter)->SetFileName(file.c_str());
    }
    else
    {
        exporter = vtkSmartPointer<vtkOBJExporter>::New();
        vtkOBJExporter::SafeDownCast(exporter)->SetFilePrefix(file.c_str());
    }

    exporter->SetInput(Interactor->GetRenderWindow());
    exporter->Write();

    cout << "Scene successfully exported (" << file.c_str() << ")" << endl;
}

void cv::viz::vtkVizInteractorStyle::exportScene()
{
    // Export scene as in obj or vrml format
    String format = Interactor->GetAltKey() ? "scene-%d.vrml" : "scene-%d";
    exportScene(cv::format(format.c_str(), (unsigned int)time(0)));
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::changePointsSize(float delta)
{
    vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
    vtkCollectionSimpleIterator ait;

    for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
        for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
        {
            vtkActor* apart = vtkActor::SafeDownCast(path->GetLastNode()->GetViewProp());
            float psize = apart->GetProperty()->GetPointSize() + delta;
            psize = std::max(1.f, std::min(63.f, psize));
            apart->GetProperty()->SetPointSize(psize);
        }
}

void cv::viz::vtkVizInteractorStyle::setRepresentationToPoints()
{
    vtkSmartPointer<vtkActorCollection> ac = CurrentRenderer->GetActors();
    vtkCollectionSimpleIterator ait;
    for (ac->InitTraversal(ait); vtkActor* actor = ac->GetNextActor(ait); )
        for (actor->InitPathTraversal(); vtkAssemblyPath* path = actor->GetNextPath(); )
        {
            vtkActor* apart = vtkActor::SafeDownCast(path->GetLastNode()->GetViewProp());
            apart->GetProperty()->SetRepresentationToPoints();
        }
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::printCameraParams()
{
    vtkSmartPointer<vtkCamera> cam = Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera();

    Vec2d clip(cam->GetClippingRange());
    Vec3d focal(cam->GetFocalPoint()), pos(cam->GetPosition()), view(cam->GetViewUp());
    Vec2i win_pos(Interactor->GetRenderWindow()->GetPosition());
    Vec2i win_size(Interactor->GetRenderWindow()->GetSize());
    double angle = cam->GetViewAngle () / 180.0 * CV_PI;

    String data = cv::format("clip(%f,%f) focal(%f,%f,%f) pos(%f,%f,%f) view(%f,%f,%f) angle(%f) winsz(%d,%d) winpos(%d,%d)",
                             clip[0], clip[1], focal[0], focal[1], focal[2], pos[0], pos[1], pos[2], view[0], view[1], view[2],
                             angle, win_size[0], win_size[1], win_pos[0], win_pos[1]);

    std::cout << data.c_str() << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::toggleFullScreen()
{
    Vec2i screen_size(Interactor->GetRenderWindow()->GetScreenSize());
    Vec2i win_size(Interactor->GetRenderWindow()->GetSize());

    // Is window size = max?
    if (win_size == max_win_size_)
    {
        Interactor->GetRenderWindow()->SetSize(win_size_.val);
        Interactor->GetRenderWindow()->SetPosition(win_pos_.val);
        Interactor->Render();
    }
    // Set to max
    else
    {
        win_pos_ = Vec2i(Interactor->GetRenderWindow()->GetPosition());
        win_size_ = win_size;

        Interactor->GetRenderWindow()->SetSize(screen_size.val);
        Interactor->Render();
        max_win_size_ = Vec2i(Interactor->GetRenderWindow()->GetSize());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::resetViewerPose()
{
    WidgetActorMap::iterator it = widget_actor_map_->begin();
    // it might be that some actors don't have a valid transformation set -> we skip them to avoid a seg fault.
    for (; it != widget_actor_map_->end();  ++it)
    {
        vtkProp3D * actor = vtkProp3D::SafeDownCast(it->second);
        if (actor && actor->GetUserMatrix())
            break;
    }

    vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();

    // if a valid transformation was found, use it otherwise fall back to default view point.
    if (it != widget_actor_map_->end())
    {
        vtkMatrix4x4* m = vtkProp3D::SafeDownCast(it->second)->GetUserMatrix();

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
    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::toggleStereo()
{
    vtkSmartPointer<vtkRenderWindow> window = Interactor->GetRenderWindow();
    if (!window->GetStereoRender())
    {
        static Vec2i red_blue(4, 3), magenta_green(2, 5);
        window->SetAnaglyphColorMask (stereo_anaglyph_redblue_ ? red_blue.val : magenta_green.val);
        stereo_anaglyph_redblue_ = !stereo_anaglyph_redblue_;
    }
    window->SetStereoRender(!window->GetStereoRender());
    Interactor->Render();

}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::printHelp()
{
    std::cout << "| Help:\n"
                 "-------\n"
                 "          p, P   : switch to a point-based representation\n"
                 "          w, W   : switch to a wireframe-based representation (where available)\n"
                 "          s, S   : switch to a surface-based representation (where available)\n"
                 "\n"
                 "          j, J   : take a .PNG snapshot of the current window view\n"
                 "          k, K   : export scene to Wavefront .obj format\n"
                 "    ALT + k, K   : export scene to VRML format\n"
                 "          c, C   : display current camera/window parameters\n"
                 "          F5     : enable/disable fly mode (changes control style)\n"
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
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::zoomIn()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    // Zoom in
    StartDolly();
    double factor = 10.0 * 0.2 * .5;
    Dolly(std::pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::zoomOut()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);
    // Zoom out
    StartDolly();
    double factor = 10.0 * -0.2 * .5;
    Dolly(std::pow(1.1, factor));
    EndDolly();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnChar()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);

    String key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != String::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != String::npos)
        zoomOut();

    switch (Interactor->GetKeyCode())
    {
//    // All of the options below simply exit
//    case 'l': case 'L': case 'j': case 'J': case 'c': case 'C': case 'q': case 'Q':
//    case 'f': case 'F': case 'g': case 'G': case 'o': case 'O': case 'u': case 'U':
    case 'p': case 'P':
        break;

    case '+':
        if (FlyMode)
            MotionUserScale = std::min(16.0, MotionUserScale*2.0);
        break;
    case '-':
        if (FlyMode)
            MotionUserScale = std::max(MotionUserScale * 0.5, 0.0625);
        break;

    case 'r': case 'R': case 's': case 'S':
        if (!Interactor->GetAltKey())
            Superclass::OnChar();
        break;
    default:
        Superclass::OnChar();
        break;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::registerMouseCallback(void (*callback)(const MouseEvent&, void*), void* cookie)
{
    mouseCallback_ = callback;
    mouse_callback_cookie_ = cookie;
}

void cv::viz::vtkVizInteractorStyle::registerKeyboardCallback(void (*callback)(const KeyboardEvent&, void*), void *cookie)
{
    keyboardCallback_ = callback;
    keyboard_callback_cookie_ = cookie;
}

//////////////////////////////////////////////////////////////////////////////////////////////
int cv::viz::vtkVizInteractorStyle::getModifiers()
{
    int modifiers = KeyboardEvent::NONE;

    if (Interactor->GetAltKey())
        modifiers |= KeyboardEvent::ALT;

    if (Interactor->GetControlKey())
        modifiers |= KeyboardEvent::CTRL;

    if (Interactor->GetShiftKey())
        modifiers |= KeyboardEvent::SHIFT;
    return modifiers;
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnKeyDown()
{
    FindPokedRenderer(Interactor->GetEventPosition()[0], Interactor->GetEventPosition()[1]);

    String key(Interactor->GetKeySym());
    if (key.find("XF86ZoomIn") != String::npos)
        zoomIn();
    else if (key.find("XF86ZoomOut") != String::npos)
        zoomOut();
    else if (key.find("F5") != String::npos)
    {
        FlyMode = !FlyMode;
        std::cout << (FlyMode ? "Fly mode: on" : "Fly mode: off") << std::endl;
    }

    // Save the initial windows width/height
    if (win_size_[0] == -1 || win_size_[1] == -1)
        win_size_ = Vec2i(Interactor->GetRenderWindow()->GetSize());

    switch (Interactor->GetKeyCode())
    {
    case 'a': case 'A' : KeysDown |=16; break;
    case 'z': case 'Z' : KeysDown |=32; break;
    case 'h': case 'H' : printHelp();   break;
    case 'p': case 'P' : setRepresentationToPoints(); break;
    case 'k': case 'K' : exportScene(); break;
    case 'j': case 'J' : saveScreenshot(cv::format("screenshot-%d.png", (unsigned int)time(0))); break;
    case 'c': case 'C' : printCameraParams(); break;
    case '=':           zoomIn();            break;
    case 43:        // KEY_PLUS
    {
        if (FlyMode)
            break;
        if (Interactor->GetAltKey())
            zoomIn();
        else
            changePointsSize(+1.f);
        break;
    }
    case 45:        // KEY_MINUS
    {
        if (FlyMode)
            break;
        if (Interactor->GetAltKey())
            zoomOut();
        else
           changePointsSize(-1.f);
        break;
    }
        // Switch between maximize and original window size
    case 'f': case 'F':
    {
        if (Interactor->GetAltKey())
            toggleFullScreen();
        break;
    }
        // 's'/'S' w/out ALT
    case 's': case 'S':
    {
        if (Interactor->GetAltKey())
            toggleStereo();
        break;
    }

    case 'o': case 'O':
    {
        vtkSmartPointer<vtkCamera> cam = CurrentRenderer->GetActiveCamera();
        cam->SetParallelProjection(!cam->GetParallelProjection());
        Interactor->Render();
        break;
    }

    // Overwrite the camera reset
    case 'r': case 'R':
    {
        if (Interactor->GetAltKey())
            resetViewerPose();
        break;
    }
    case 'q': case 'Q':
        Interactor->ExitCallback(); return;
    default:
        Superclass::OnKeyDown(); break;
    }

    KeyboardEvent event(KeyboardEvent::KEY_DOWN, Interactor->GetKeySym(), Interactor->GetKeyCode(), getModifiers());
    if (keyboardCallback_)
        keyboardCallback_(event, keyboard_callback_cookie_);

    if (FlyMode && (KeysDown & (32+16)) == (32+16))
    {
        if (State == VTKIS_FORWARDFLY || State == VTKIS_REVERSEFLY)
            StopState();
    }
    else if (FlyMode && (KeysDown & 32) == 32)
    {
        if (State == VTKIS_FORWARDFLY)
            StopState();

        if (State == VTKIS_NONE)
            StartState(VTKIS_REVERSEFLY);
    }
    else if (FlyMode && (KeysDown & 16) == 16)
    {
        if (State == VTKIS_REVERSEFLY)
            StopState();

        if (State == VTKIS_NONE)
            StartState(VTKIS_FORWARDFLY);
    }

    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnKeyUp()
{
    KeyboardEvent event(KeyboardEvent::KEY_UP, Interactor->GetKeySym(), Interactor->GetKeyCode(), getModifiers());
    if (keyboardCallback_)
        keyboardCallback_(event, keyboard_callback_cookie_);

    switch (Interactor->GetKeyCode())
    {
    case 'a': case 'A' : KeysDown &= ~16; break;
    case 'z': case 'Z' : KeysDown &= ~32; break;
    }

    if (State == VTKIS_FORWARDFLY && (KeysDown & 16) == 0)
        StopState();

    if (State == VTKIS_REVERSEFLY && (KeysDown & 32) == 0)
        StopState();

    Superclass::OnKeyUp();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnMouseMove()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseMove, MouseEvent::NoButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    FindPokedRenderer(p[0], p[1]);

    if (State == VTKIS_ROTATE || State == VTKIS_PAN || State == VTKIS_DOLLY || State == VTKIS_SPIN)
    {
        switch (State)
        {
        case VTKIS_ROTATE: Rotate(); break;
        case VTKIS_PAN:    Pan();    break;
        case VTKIS_DOLLY:  Dolly();  break;
        case VTKIS_SPIN:   Spin();   break;
        }

        InvokeEvent(vtkCommand::InteractionEvent, NULL);
    }

    if (State == VTKIS_FORWARDFLY || State == VTKIS_REVERSEFLY)
    {
        vtkCamera *cam = CurrentRenderer->GetActiveCamera();
        Vec2i thispos(Interactor->GetEventPosition());
        Vec2i lastpos(Interactor->GetLastEventPosition());

        // we want to steer by an amount proportional to window viewangle and size
        // compute dx and dy increments relative to last mouse click
        Vec2i size(Interactor->GetSize());
        double scalefactor = 5*cam->GetViewAngle()/size[0];

        double dx = - (thispos[0] - lastpos[0])*scalefactor*AngleStepSize;
        double dy =   (thispos[1] - lastpos[1])*scalefactor*AngleStepSize;

        // Temporary until I get smooth flight working
        DeltaPitch = dy;
        DeltaYaw = dx;

        InvokeEvent(vtkCommand::InteractionEvent, NULL);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnLeftButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::LeftButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    FindPokedRenderer(p[0], p[1]);
    if (!CurrentRenderer)
        return;

    GrabFocus(EventCallbackCommand);

    if (FlyMode)
    {
        if(State == VTKIS_REVERSEFLY)
            State = VTKIS_FORWARDFLY;
        else
        {
            SetupMotionVars();
            if (State == VTKIS_NONE)
                StartState(VTKIS_FORWARDFLY);
        }
    }
    else
    {
        if (Interactor->GetShiftKey())
        {
            if (Interactor->GetControlKey())
                StartDolly();
            else
                StartPan();
        }
        else
        {
            if (Interactor->GetControlKey())
                StartSpin();
            else
                StartRotate();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnLeftButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::LeftButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    switch (State)
    {
    case VTKIS_DOLLY:      EndDolly();  break;
    case VTKIS_PAN:        EndPan();    break;
    case VTKIS_SPIN:       EndSpin();   break;
    case VTKIS_ROTATE:     EndRotate(); break;
    case VTKIS_FORWARDFLY: StopState(); break;
    }

    if (Interactor )
        ReleaseFocus();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnMiddleButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::MiddleButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    FindPokedRenderer(p[0], p[1]);
    if (!CurrentRenderer)
        return;

    GrabFocus(EventCallbackCommand);
    StartPan();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnMiddleButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::MiddleButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    if (State == VTKIS_PAN)
    {
        EndPan();
        if (Interactor)
            ReleaseFocus();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnRightButtonDown()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent::Type type = (Interactor->GetRepeatCount() == 0) ? MouseEvent::MouseButtonPress : MouseEvent::MouseDblClick;
    MouseEvent event(type, MouseEvent::RightButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    FindPokedRenderer(p[0], p[1]);
    if (!CurrentRenderer)
        return;

    GrabFocus(EventCallbackCommand);

    if (FlyMode)
    {
        if (State == VTKIS_FORWARDFLY)
            State = VTKIS_REVERSEFLY;
        else
        {
            SetupMotionVars();
            if (State == VTKIS_NONE)
                StartState(VTKIS_REVERSEFLY);
        }

    }
    else
        StartDolly();
}


//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnRightButtonUp()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseButtonRelease, MouseEvent::RightButton, p, getModifiers());
    if (mouseCallback_)
        mouseCallback_(event, mouse_callback_cookie_);

    if(State == VTKIS_DOLLY)
    {
        EndDolly();
        if (Interactor)
            ReleaseFocus();
    }

    if (State == VTKIS_REVERSEFLY)
    {
        StopState();
        if (Interactor)
            ReleaseFocus();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnMouseWheelForward()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseScrollUp, MouseEvent::VScroll, p, getModifiers());
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
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        Interactor->Render();
    }
    else
    {
        FindPokedRenderer(p[0], p[1]);
        if (!CurrentRenderer)
            return;

        GrabFocus(EventCallbackCommand);
        StartDolly();
        Dolly(pow(1.1, MotionFactor * 0.2 * MouseWheelMotionFactor));
        EndDolly();
        ReleaseFocus();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnMouseWheelBackward()
{
    Vec2i p(Interactor->GetEventPosition());
    MouseEvent event(MouseEvent::MouseScrollDown, MouseEvent::VScroll, p, getModifiers());
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
        CurrentRenderer->ResetCameraClippingRange();
        CurrentRenderer->Modified();
        Interactor->Render();
    }
    else
    {
        FindPokedRenderer(p[0], p[1]);
        if (!CurrentRenderer)
            return;

        GrabFocus(EventCallbackCommand);
        StartDolly();
        Dolly(pow(1.1, MotionFactor * -0.2 * MouseWheelMotionFactor));
        EndDolly();
        ReleaseFocus();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::OnTimer()
{
    if  (State == VTKIS_FORWARDFLY || State == VTKIS_REVERSEFLY)
        Fly();

    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::Rotate()
{
    if (!CurrentRenderer)
        return;

    Vec2i dxy = Vec2i(Interactor->GetEventPosition()) - Vec2i(Interactor->GetLastEventPosition());
    Vec2i size(CurrentRenderer->GetRenderWindow()->GetSize());

    double delta_elevation = -20.0 / size[1];
    double delta_azimuth   = -20.0 / size[0];

    double rxf = dxy[0] * delta_azimuth * MotionFactor;
    double ryf = dxy[1] * delta_elevation * MotionFactor;

    vtkCamera *camera = CurrentRenderer->GetActiveCamera();
    camera->Azimuth(rxf);
    camera->Elevation(ryf);
    camera->OrthogonalizeViewUp();

    if (AutoAdjustCameraClippingRange)
        CurrentRenderer->ResetCameraClippingRange();

    if (Interactor->GetLightFollowCamera())
        CurrentRenderer->UpdateLightsGeometryToFollowCamera();

    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::Spin()
{
    if (!CurrentRenderer)
        return;

    vtkRenderWindowInteractor *rwi = Interactor;

    double *center = CurrentRenderer->GetCenter();

    double newAngle = vtkMath::DegreesFromRadians( atan2( rwi->GetEventPosition()[1]     - center[1], rwi->GetEventPosition()[0]     - center[0] ) );
    double oldAngle = vtkMath::DegreesFromRadians( atan2( rwi->GetLastEventPosition()[1] - center[1], rwi->GetLastEventPosition()[0] - center[0] ) );

    vtkCamera *camera = CurrentRenderer->GetActiveCamera();
    camera->Roll( newAngle - oldAngle );
    camera->OrthogonalizeViewUp();

    rwi->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////
void cv::viz::vtkVizInteractorStyle::Pan()
{
    if (!CurrentRenderer)
        return;

    vtkRenderWindowInteractor *rwi = Interactor;

    double viewFocus[4], focalDepth, viewPoint[3];
    double newPickPoint[4], oldPickPoint[4], motionVector[3];

    // Calculate the focal depth since we'll be using it a lot

    vtkCamera *camera = CurrentRenderer->GetActiveCamera();
    camera->GetFocalPoint(viewFocus);
    ComputeWorldToDisplay(viewFocus[0], viewFocus[1], viewFocus[2], viewFocus);
    focalDepth = viewFocus[2];

    ComputeDisplayToWorld(rwi->GetEventPosition()[0], rwi->GetEventPosition()[1], focalDepth, newPickPoint);

    // Has to recalc old mouse point since the viewport has moved, so can't move it outside the loop
    ComputeDisplayToWorld(rwi->GetLastEventPosition()[0], rwi->GetLastEventPosition()[1], focalDepth, oldPickPoint);

    // Camera motion is reversed
    motionVector[0] = oldPickPoint[0] - newPickPoint[0];
    motionVector[1] = oldPickPoint[1] - newPickPoint[1];
    motionVector[2] = oldPickPoint[2] - newPickPoint[2];

    camera->GetFocalPoint(viewFocus);
    camera->GetPosition(viewPoint);
    camera->SetFocalPoint(motionVector[0] + viewFocus[0], motionVector[1] + viewFocus[1], motionVector[2] + viewFocus[2]);
    camera->SetPosition(  motionVector[0] + viewPoint[0], motionVector[1] + viewPoint[1], motionVector[2] + viewPoint[2]);

    if (Interactor->GetLightFollowCamera())
        CurrentRenderer->UpdateLightsGeometryToFollowCamera();

    Interactor->Render();
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::Dolly()
{
    if (!CurrentRenderer)
        return;

    int dy = Interactor->GetEventPosition()[1] - Interactor->GetLastEventPosition()[1];
    Dolly(pow(1.1, MotionFactor * dy / CurrentRenderer->GetCenter()[1]));
}

void cv::viz::vtkVizInteractorStyle::Dolly(double factor)
{
    if (!CurrentRenderer)
        return;

    vtkCamera *camera = CurrentRenderer->GetActiveCamera();
    if (camera->GetParallelProjection())
        camera->SetParallelScale(camera->GetParallelScale() / factor);
    else
    {
        camera->Dolly(factor);
        if (AutoAdjustCameraClippingRange)
            CurrentRenderer->ResetCameraClippingRange();
    }

    if (Interactor->GetLightFollowCamera())
        CurrentRenderer->UpdateLightsGeometryToFollowCamera();

    Interactor->Render();
}
//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::Fly()
{
    if (CurrentRenderer == NULL)
        return;

    if (KeysDown)
        FlyByKey();
    else
        FlyByMouse();

    CurrentRenderer->GetActiveCamera()->OrthogonalizeViewUp();

    if (AutoAdjustCameraClippingRange)
        CurrentRenderer->ResetCameraClippingRange();

    if (Interactor->GetLightFollowCamera())
        CurrentRenderer->UpdateLightsGeometryToFollowCamera();
}

void cv::viz::vtkVizInteractorStyle::SetupMotionVars()
{
    Vec6d bounds;
    CurrentRenderer->ComputeVisiblePropBounds(bounds.val);

    if ( !vtkMath::AreBoundsInitialized(bounds.val) )
        DiagonalLength = 1.0;
    else
        DiagonalLength = norm(Vec3d(bounds[0], bounds[2], bounds[4]) - Vec3d(bounds[1], bounds[3], bounds[5]));
}

void cv::viz::vtkVizInteractorStyle::MotionAlongVector(const Vec3d& vector, double amount, vtkCamera* cam)
{
    // move camera and focus along DirectionOfProjection
    Vec3d campos = Vec3d(cam->GetPosition())   - amount * vector;
    Vec3d camfoc = Vec3d(cam->GetFocalPoint()) - amount * vector;

    cam->SetPosition(campos.val);
    cam->SetFocalPoint(camfoc.val);
}

void cv::viz::vtkVizInteractorStyle::FlyByMouse()
{
    vtkCamera* cam = CurrentRenderer->GetActiveCamera();
    double speed  = DiagonalLength * MotionStepSize * MotionUserScale;
    speed = speed * ( Interactor->GetShiftKey() ? MotionAccelerationFactor : 1.0);

    // Sidestep
    if (Interactor->GetAltKey())
    {
        if (DeltaYaw!=0.0)
        {
            vtkMatrix4x4 *vtm = cam->GetViewTransformMatrix();
            Vec3d a_vector(vtm->GetElement(0,0), vtm->GetElement(0,1), vtm->GetElement(0,2));

            MotionAlongVector(a_vector, -DeltaYaw*speed, cam);
        }
        if (DeltaPitch!=0.0)
        {
            Vec3d a_vector(cam->GetViewUp());
            MotionAlongVector(a_vector, DeltaPitch*speed, cam);
        }
    }
    else
    {
        cam->Yaw(DeltaYaw);
        cam->Pitch(DeltaPitch);
        DeltaYaw = 0;
        DeltaPitch = 0;
    }
    //
    if (!Interactor->GetControlKey())
    {
        Vec3d a_vector(cam->GetDirectionOfProjection()); // reversed (use -speed)
        switch (State)
        {
        case VTKIS_FORWARDFLY: MotionAlongVector(a_vector, -speed, cam); break;
        case VTKIS_REVERSEFLY: MotionAlongVector(a_vector, speed, cam); break;
        }
    }
}

void cv::viz::vtkVizInteractorStyle::FlyByKey()
{
    vtkCamera* cam = CurrentRenderer->GetActiveCamera();

    double speed  = DiagonalLength * MotionStepSize * MotionUserScale;
    speed = speed * ( Interactor->GetShiftKey() ? MotionAccelerationFactor : 1.0);

    // Left and right
    if (Interactor->GetAltKey())
    { // Sidestep
        vtkMatrix4x4 *vtm = cam->GetViewTransformMatrix();
        Vec3d a_vector(vtm->GetElement(0,0), vtm->GetElement(0,1), vtm->GetElement(0,2));

        if (KeysDown & 1)
            MotionAlongVector(a_vector, -speed, cam);

        if (KeysDown & 2)
            MotionAlongVector(a_vector,  speed, cam);
    }
    else
    {
        if (KeysDown & 1)
            cam->Yaw( AngleStepSize);

        if (KeysDown & 2)
            cam->Yaw(-AngleStepSize);
    }

    // Up and Down
    if (Interactor->GetControlKey())
    { // Sidestep
        Vec3d a_vector = Vec3d(cam->GetViewUp());
        if (KeysDown & 4)
            MotionAlongVector(a_vector,-speed, cam);

        if (KeysDown & 8)
            MotionAlongVector(a_vector, speed, cam);
    }
    else
    {
        if (KeysDown & 4)
            cam->Pitch(-AngleStepSize);

        if (KeysDown & 8)
            cam->Pitch( AngleStepSize);
    }

    // forward and backward
    Vec3d a_vector(cam->GetDirectionOfProjection());
    if (KeysDown & 16)
        MotionAlongVector(a_vector, speed, cam);

    if (KeysDown & 32)
        MotionAlongVector(a_vector,-speed, cam);
}

//////////////////////////////////////////////////////////////////////////////////////////////

void cv::viz::vtkVizInteractorStyle::PrintSelf(ostream& os, vtkIndent indent)
{
    Superclass::PrintSelf(os, indent);
    os << indent << "MotionFactor: " << MotionFactor << "\n";
    os << indent << "MotionStepSize: " << MotionStepSize << "\n";
    os << indent << "MotionAccelerationFactor: "<< MotionAccelerationFactor << "\n";
    os << indent << "AngleStepSize: " << AngleStepSize << "\n";
    os << indent << "MotionUserScale: "<< MotionUserScale << "\n";
}
