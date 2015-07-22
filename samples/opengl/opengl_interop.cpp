/*
// Sample demonstrating interoperability of OpenCV UMat with OpenGL texture.
// At first, the data obtained from video file or camera and placed onto
// OpenGL texture, following mapping of this OpenGL texture to OpenCV UMat
// and call cv::Blur function. The result is mapped back to OpenGL texture
// and rendered through OpenGL API.
*/
#if defined(WIN32) || defined(_WIN32)
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#elif defined(__linux__)
# include <X11/X.h>
# include <X11/Xlib.h>
#endif

#include <iostream>
#include <queue>
#include <string>

#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "winapp.hpp"

#if defined(WIN32) || defined(_WIN32)
# pragma comment(lib, "opengl32.lib")
# pragma comment(lib, "glu32.lib")
#endif

/*
// Press key   to
//       0     no processing
//       1     processing on CPU
//       2     processing on GPU
//       9     toggle texture/buffer
//       space toggle processing on/off, preserve mode
//       esc   quit
*/

class GLWinApp : public WinApp
{
public:
    GLWinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown = false;
        m_mode = 0;
        m_modeStr[0] = cv::String("Texture/No processing");
        m_modeStr[1] = cv::String("Texture/Processing on CPU");
        m_modeStr[2] = cv::String("Texture/Processing on GPU");
        m_modeStr[3] = cv::String("Buffer/No processing");
        m_modeStr[4] = cv::String("Buffer/Processing on CPU");
        m_modeStr[5] = cv::String("Buffer/Processing on GPU");
        m_disableProcessing = false;
        m_cap = cap;
    }

    ~GLWinApp() {}

    virtual void cleanup()
    {
        m_shutdown = true;
#if defined(__linux__)
        glXMakeCurrent(m_display, None, NULL);
        glXDestroyContext(m_display, m_glctx);
#endif
        WinApp::cleanup();
    }

#if defined(WIN32) || defined(_WIN32)
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_CHAR:
            if (wParam >= '0' && wParam <= '2')
            {
                set_mode((char)wParam - '0');
                return 0;
            }
            else if (wParam == '9')
            {
                toggle_buffer();
                return 0;
            }
            else if (wParam == VK_SPACE)
            {
                m_disableProcessing = !m_disableProcessing;
                return 0;
            }
            else if (wParam == VK_ESCAPE)
            {
                cleanup();
                return 0;
            }
            break;

        case WM_CLOSE:
            cleanup();
            return 0;

        case WM_DESTROY:
            ::PostQuitMessage(0);
            return 0;
        }

        return ::DefWindowProc(hWnd, message, wParam, lParam);
    }
#endif

    static float getFps()
    {
        static std::queue<int64> time_queue;

        int64 now = cv::getTickCount();
        int64 then = 0;
        time_queue.push(now);

        if (time_queue.size() >= 2)
            then = time_queue.front();

        if (time_queue.size() >= 25)
            time_queue.pop();

        return time_queue.size() * (float)cv::getTickFrequency() / (now - then);
    }

#if defined(__linux__)
    int handle_event(XEvent& e)
    {
        switch(e.type)
        {
        case ClientMessage:
            if ((Atom)e.xclient.data.l[0] == m_WM_DELETE_WINDOW)
            {
                m_end_loop = true;
                cleanup();
            }
            else
            {
                return 0;
            }
            break;
        case Expose:
            render();
            break;
        case KeyPress:
            switch(keycode_to_keysym(e.xkey.keycode))
            {
            case XK_space:
                m_disableProcessing = !m_disableProcessing;
                break;
            case XK_0:
                set_mode(0);
                break;
            case XK_1:
                set_mode(1);
                break;
            case XK_2:
                set_mode(2);
                break;
            case XK_9:
                toggle_buffer();
                break;
            case XK_Escape:
                m_end_loop = true;
                cleanup();
                break;
            }
            break;
        default:
            return 0;
        }
        return 1;
    }
#endif

    int init()
    {
#if defined(WIN32) || defined(_WIN32)
        m_hDC = GetDC(m_hWnd);

        if (setup_pixel_format() != 0)
        {
            std::cerr << "Can't setup pixel format" << std::endl;
            return -1;
        }

        m_hRC = wglCreateContext(m_hDC);
        wglMakeCurrent(m_hDC, m_hRC);
#elif defined(__linux__)
        m_glctx = glXCreateContext(m_display, m_visual_info, NULL, GL_TRUE);
        glXMakeCurrent(m_display, m_window, m_glctx);
#endif

        glEnable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);

        glViewport(0, 0, m_width, m_height);

        if (cv::ocl::haveOpenCL())
        {
            (void) cv::ogl::ocl::initializeContextFromGL();
        }

        m_oclDevName = cv::ocl::useOpenCL() ?
            cv::ocl::Context::getDefault().device(0).name() :
            (char*) "No OpenCL device";

        return 0;
    } // init()

    int get_frame(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer)
    {
        if (!m_cap.read(m_frame_bgr))
            return -1;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, CV_RGB2RGBA);

        if (use_buffer())
            buffer.copyFrom(m_frame_rgba);
        else
            texture.copyFrom(m_frame_rgba);

        return 0;
    }

    void print_info(int mode, float fps, cv::String oclDevName)
    {
#if defined(WIN32) || defined(_WIN32)
        HDC hDC = m_hDC;

        HFONT hFont = (HFONT)::GetStockObject(SYSTEM_FONT);

        HFONT hOldFont = (HFONT)::SelectObject(hDC, hFont);

        if (hOldFont)
        {
            TEXTMETRIC tm;
            ::GetTextMetrics(hDC, &tm);

            char buf[256+1];
            int  y = 0;

            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "Mode: %s", m_modeStr[mode].c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "FPS: %2.1f", fps);
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "OpenCL device: %s", oclDevName.c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            ::SelectObject(hDC, hOldFont);
        }
#elif defined(__linux__)

        char buf[256+1];
        snprintf(buf, sizeof(buf)-1, "FPS: %2.1f Mode: %s Device: %s", fps, m_modeStr[mode].c_str(), oclDevName.c_str());
        XStoreName(m_display, m_window, buf);
#endif
    }

    void idle()
    {
        render();
    }

    int render()
    {
        try
        {
            if (m_shutdown)
                return 0;

            int r;
            cv::ogl::Texture2D texture;
            cv::ogl::Buffer buffer;

            r = get_frame(texture, buffer);
            if (r != 0)
            {
                return -1;
            }

            bool do_buffer = use_buffer();
            switch (get_mode())
            {
                case 0:
                    // no processing
                    break;

                case 1:
                {
                    // process video frame on CPU
                    cv::Mat m(m_height, m_width, CV_8UC4);

                    if (do_buffer)
                        buffer.copyTo(m);
                    else
                        texture.copyTo(m);

                    if (!m_disableProcessing)
                    {
                        // blur texture image with OpenCV on CPU
                        cv::blur(m, m, cv::Size(15, 15), cv::Point(-7, -7));
                    }

                    if (do_buffer)
                        buffer.copyFrom(m);
                    else
                        texture.copyFrom(m);

                    break;
                }

                case 2:
                {
                    // process video frame on GPU
                    cv::UMat u;

                    if (do_buffer)
                        u = cv::ogl::mapGLBuffer(buffer);
                    else
                        cv::ogl::convertFromGLTexture2D(texture, u);

                    if (!m_disableProcessing)
                    {
                        // blur texture image with OpenCV on GPU with OpenCL
                        cv::blur(u, u, cv::Size(15, 15), cv::Point(-7, -7));
                    }

                    if (do_buffer)
                        cv::ogl::unmapGLBuffer(u);
                    else
                        cv::ogl::convertToGLTexture2D(u, texture);

                    break;
                }

            } // switch

            if (do_buffer) // buffer -> texture
            {
                cv::Mat m(m_height, m_width, CV_8UC4);
                buffer.copyTo(m);
                texture.copyFrom(m);
            }

#if defined(__linux__)
            XWindowAttributes window_attributes;
            XGetWindowAttributes(m_display, m_window, &window_attributes);
            glViewport(0, 0, window_attributes.width, window_attributes.height);
#endif

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glLoadIdentity();
            glEnable(GL_TEXTURE_2D);

            texture.bind();

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, 1.0f, 0.1f);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, -1.0f, 0.1f);
            glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.1f);
            glEnd();

#if defined(WIN32) || defined(_WIN32)
            SwapBuffers(m_hDC);
#elif defined(__linux__)
            glXSwapBuffers(m_display, m_window);
#endif

            print_info(m_mode, getFps(), m_oclDevName);
        }


        catch (cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return 0;
    }

protected:

#if defined(WIN32) || defined(_WIN32)
    int setup_pixel_format()
    {
        PIXELFORMATDESCRIPTOR  pfd;

        pfd.nSize           = sizeof(PIXELFORMATDESCRIPTOR);
        pfd.nVersion        = 1;
        pfd.dwFlags         = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL  | PFD_DOUBLEBUFFER;
        pfd.iPixelType      = PFD_TYPE_RGBA;
        pfd.cColorBits      = 24;
        pfd.cRedBits        = 8;
        pfd.cRedShift       = 0;
        pfd.cGreenBits      = 8;
        pfd.cGreenShift     = 0;
        pfd.cBlueBits       = 8;
        pfd.cBlueShift      = 0;
        pfd.cAlphaBits      = 8;
        pfd.cAlphaShift     = 0;
        pfd.cAccumBits      = 0;
        pfd.cAccumRedBits   = 0;
        pfd.cAccumGreenBits = 0;
        pfd.cAccumBlueBits  = 0;
        pfd.cAccumAlphaBits = 0;
        pfd.cDepthBits      = 24;
        pfd.cStencilBits    = 8;
        pfd.cAuxBuffers     = 0;
        pfd.iLayerType      = PFD_MAIN_PLANE;
        pfd.bReserved       = 0;
        pfd.dwLayerMask     = 0;
        pfd.dwVisibleMask   = 0;
        pfd.dwDamageMask    = 0;

        int pfmt = ChoosePixelFormat(m_hDC, &pfd);
        if (pfmt == 0)
            return -1;
        if (SetPixelFormat(m_hDC, pfmt, &pfd) == 0)
            return -2;
        return 0;
    }
#endif

#if defined(__linux__)
    KeySym keycode_to_keysym(unsigned keycode)
    {   // note that XKeycodeToKeysym() is considered deprecated
        int keysyms_per_keycode_return = 0;
        KeySym *keysyms = XGetKeyboardMapping(m_display, keycode, 1, &keysyms_per_keycode_return);
        KeySym keysym = keysyms[0];
        XFree(keysyms);
        return keysym;
    }
#endif

    // modes: 0,1,2 - use texture
    //        3,4,5 - use buffer
    bool use_buffer()
    {
        return bool(m_mode >= 3);
    }
    void toggle_buffer()
    {
        if (m_mode < 3)
            m_mode += 3;
        else
            m_mode -= 3;
    }
    int get_mode()
    {
        return (m_mode % 3);
    }
    void set_mode(int mode)
    {
        bool do_buffer = bool(m_mode >= 3);
        m_mode = (mode % 3);
        if (do_buffer)
            m_mode += 3;
    }

private:
    bool               m_shutdown;
    int                m_mode;
    cv::String         m_modeStr[3*2];
    int                m_disableProcessing;
#if defined(WIN32) || defined(_WIN32)
    HDC                m_hDC;
    HGLRC              m_hRC;
#elif defined(__linux__)
    GLXContext         m_glctx;
#endif
    cv::VideoCapture   m_cap;
    cv::Mat            m_frame_bgr;
    cv::Mat            m_frame_rgba;
    cv::String         m_oclDevName;
};

using namespace cv;

int main(int argc, char** argv)
{
    cv::VideoCapture cap;

    if (argc > 1)
        cap.open(argv[1]);
    else
        cap.open(0);

    int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    std::string wndname = "WGL Window";

    GLWinApp app(width, height, wndname, cap);

    try
    {
        app.create();
        return app.run();
    }
    catch (cv::Exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 10;
    }
    catch (...)
    {
        std::cerr << "FATAL ERROR: Unknown exception" << std::endl;
        return 11;
    }
}
