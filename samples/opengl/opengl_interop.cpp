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
//       1     processing on CPU
//       2     processing on GPU
//       9     toggle texture/buffer
//       space toggle processing on/off, preserve mode
//       esc   quit
*/

class GLWinApp : public WinApp
{
public:
    enum MODE
    {
        MODE_CPU = 0,
        MODE_GPU
    };

    GLWinApp(int width, int height, std::string& window_name, cv::VideoCapture& cap) :
        WinApp(width, height, window_name)
    {
        m_shutdown        = false;
        m_use_buffer      = false;
        m_demo_processing = true;
        m_mode            = MODE_CPU;
        m_modeStr[0]      = cv::String("Processing on CPU");
        m_modeStr[1]      = cv::String("Processing on GPU");
        m_cap             = cap;
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
            if (wParam == '1')
            {
                set_mode(MODE_CPU);
                return 0;
            }
            if (wParam == '2')
            {
                set_mode(MODE_GPU);
                return 0;
            }
            else if (wParam == '9')
            {
                toggle_buffer();
                return 0;
            }
            else if (wParam == VK_SPACE)
            {
                m_demo_processing = !m_demo_processing;
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
                m_demo_processing = !m_demo_processing;
                break;
            case XK_1:
                set_mode(MODE_CPU);
                break;
            case XK_2:
                set_mode(MODE_GPU);
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

    int get_frame(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        if (!m_cap.read(m_frame_bgr))
            return -1;

        cv::cvtColor(m_frame_bgr, m_frame_rgba, CV_RGB2RGBA);

        if (do_buffer)
            buffer.copyFrom(m_frame_rgba, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m_frame_rgba, true);

        return 0;
    }

    void print_info(MODE mode, float time, cv::String& oclDevName)
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
            sprintf_s(buf, sizeof(buf)-1, "Mode: %s OpenGL %s", m_modeStr[mode].c_str(), use_buffer() ? "buffer" : "texture");
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "Time, msec: %2.1f", time);
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            y += tm.tmHeight;
            buf[0] = 0;
            sprintf_s(buf, sizeof(buf)-1, "OpenCL device: %s", oclDevName.c_str());
            ::TextOut(hDC, 0, y, buf, (int)strlen(buf));

            ::SelectObject(hDC, hOldFont);
        }
#elif defined(__linux__)

        char buf[256+1];
        snprintf(buf, sizeof(buf)-1, "Time, msec: %2.1f, Mode: %s OpenGL %s, Device: %s", time, m_modeStr[mode].c_str(), use_buffer() ? "buffer" : "texture", oclDevName.c_str());
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

            texture.setAutoRelease(true);
            buffer.setAutoRelease(true);

            MODE mode = get_mode();
            bool do_buffer = use_buffer();

            r = get_frame(texture, buffer, do_buffer);
            if (r != 0)
            {
                return -1;
            }

            switch (mode)
            {
                case MODE_CPU: // process frame on CPU
                    processFrameCPU(texture, buffer, do_buffer);
                    break;

                case MODE_GPU: // process frame on GPU
                    processFrameGPU(texture, buffer, do_buffer);
                    break;
            } // switch

            if (do_buffer) // buffer -> texture
            {
                cv::Mat m(m_height, m_width, CV_8UC4);
                buffer.copyTo(m);
                texture.copyFrom(m, true);
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

            print_info(mode, m_timer.time(Timer::MSEC), m_oclDevName);
        }


        catch (cv::Exception& e)
        {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 10;
        }

        return 0;
    }

protected:

    void processFrameCPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::Mat m(m_height, m_width, CV_8UC4);

        m_timer.start();

        if (do_buffer)
            buffer.copyTo(m);
        else
            texture.copyTo(m);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on CPU
            cv::blur(m, m, cv::Size(15, 15), cv::Point(-7, -7));
        }

        if (do_buffer)
            buffer.copyFrom(m, cv::ogl::Buffer::PIXEL_UNPACK_BUFFER, true);
        else
            texture.copyFrom(m, true);

        m_timer.stop();
    }

    void processFrameGPU(cv::ogl::Texture2D& texture, cv::ogl::Buffer& buffer, bool do_buffer)
    {
        cv::UMat u;

        m_timer.start();

        if (do_buffer)
            u = cv::ogl::mapGLBuffer(buffer);
        else
            cv::ogl::convertFromGLTexture2D(texture, u);

        if (m_demo_processing)
        {
            // blur texture image with OpenCV on GPU with OpenCL
            cv::blur(u, u, cv::Size(15, 15), cv::Point(-7, -7));
        }

        if (do_buffer)
            cv::ogl::unmapGLBuffer(u);
        else
            cv::ogl::convertToGLTexture2D(u, texture);

        m_timer.stop();
    }

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

    bool use_buffer()        { return m_use_buffer; }
    void toggle_buffer()     { m_use_buffer = !m_use_buffer; }
    MODE get_mode()          { return m_mode; }
    void set_mode(MODE mode) { m_mode = mode; }

private:
    bool               m_shutdown;
    bool               m_use_buffer;
    bool               m_demo_processing;
    MODE               m_mode;
    cv::String         m_modeStr[2];
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

static void help()
{
    printf(
        "\nSample demonstrating interoperability of OpenGL and OpenCL with OpenCV.\n"
        "Hot keys: \n"
        "  SPACE - turn processing on/off\n"
        "    1   - process GL data through OpenCV on CPU\n"
        "    2   - process GL data through OpenCV on GPU (via OpenCL)\n"
        "    9   - toggle use of GL texture/GL buffer\n"
        "  ESC   - exit\n\n");
}

static const char* keys =
{
    "{c camera | true  | use camera or not}"
    "{f file   |       | movie file name  }"
    "{h help   | false | print help info  }"
};

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    bool   useCamera = parser.get<bool>("camera");
    string file      = parser.get<string>("file");
    bool   showHelp  = parser.get<bool>("help");

    if (showHelp)
    {
        help();
        return 0;
    }

    parser.printMessage();

    cv::VideoCapture cap;

    if (useCamera)
        cap.open(0);
    else
        cap.open(file.c_str());

    if (!cap.isOpened())
    {
        printf("can not open camera or video file\n");
        return -1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

#if defined(WIN32) || defined(_WIN32)
    string wndname = "WGL Window";
#elif defined(__linux__)
    string wndname = "GLX Window";
#endif

    GLWinApp app(width, height, wndname, cap);

    try
    {
        app.create();
        return app.run();
    }
    catch (cv::Exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 10;
    }
    catch (...)
    {
        cerr << "FATAL ERROR: Unknown exception" << endl;
        return 11;
    }
}
