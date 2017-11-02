#if defined(_WIN32)
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#elif defined(__linux__)
# include <X11/X.h>
# include <X11/Xlib.h>
# include <X11/Xutil.h>
#endif

#include <string>

#include <GL/gl.h>
#if defined(_WIN32)
# include <GL/glu.h>
#elif defined(__linux__)
# include <GL/glx.h>
#endif

#if defined(_WIN32)
# define WINCLASS "WinAppWnd"
#endif

#define SAFE_RELEASE(p) if (p) { p->Release(); p = NULL; }

class Timer
{
public:
    enum UNITS
    {
        USEC = 0,
        MSEC,
        SEC
    };

    Timer() : m_t0(0), m_diff(0)
    {
        m_tick_frequency = (float)cv::getTickFrequency();

        m_unit_mul[USEC] = 1000000;
        m_unit_mul[MSEC] = 1000;
        m_unit_mul[SEC]  = 1;
    }

    void clear()
    {
        m_t0 = m_diff = 0;
    }

    void start()
    {
        m_t0 = cv::getTickCount();
    }

    void stop()
    {
        m_diff = cv::getTickCount() - m_t0;
    }

    float time(UNITS u = MSEC)
    {
        float sec = m_diff / m_tick_frequency;

        return sec * m_unit_mul[u];
    }

public:
    float m_tick_frequency;
    int64 m_t0;
    int64 m_diff;
    int   m_unit_mul[3];
};

class WinApp
{
public:
    WinApp(int width, int height, std::string& window_name)
    {
        m_width       = width;
        m_height      = height;
        m_window_name = window_name;
#if defined(_WIN32)
        m_hInstance   = ::GetModuleHandle(NULL);
#endif
    }

    virtual ~WinApp()
    {
#if defined(_WIN32)
        ::UnregisterClass(WINCLASS, m_hInstance);
#endif
    }

    int create()
    {
#if defined(_WIN32)
        WNDCLASSEX wcex;

        wcex.cbSize        = sizeof(WNDCLASSEX);
        wcex.style         = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc   = &WinApp::StaticWndProc;
        wcex.cbClsExtra    = 0;
        wcex.cbWndExtra    = 0;
        wcex.hInstance     = m_hInstance;
        wcex.hIcon         = LoadIcon(0, IDI_APPLICATION);
        wcex.hCursor       = LoadCursor(0, IDC_ARROW);
        wcex.hbrBackground = 0;
        wcex.lpszMenuName  = 0L;
        wcex.lpszClassName = WINCLASS;
        wcex.hIconSm       = 0;

        ATOM wc = ::RegisterClassEx(&wcex);

        RECT rc = { 0, 0, m_width, m_height };
        ::AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, false);

        m_hWnd = ::CreateWindow(
                     (LPCTSTR)wc, m_window_name.c_str(),
                     WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                     rc.right - rc.left, rc.bottom - rc.top,
                     NULL, NULL, m_hInstance, (void*)this);

        if (!m_hWnd)
            return -1;

        ::ShowWindow(m_hWnd, SW_SHOW);
        ::UpdateWindow(m_hWnd);
        ::SetFocus(m_hWnd);
#elif defined(__linux__)
        m_display = XOpenDisplay(NULL);

        if (m_display == NULL)
        {
            return -1;
        }

        m_WM_DELETE_WINDOW = XInternAtom(m_display, "WM_DELETE_WINDOW", False);

        static GLint visual_attributes[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
        m_visual_info = glXChooseVisual(m_display, 0, visual_attributes);

        if (m_visual_info == NULL)
        {
            XCloseDisplay(m_display);
            return -2;
        }

        Window root = DefaultRootWindow(m_display);

        m_event_mask = ExposureMask | KeyPressMask;

        XSetWindowAttributes window_attributes;
        window_attributes.colormap = XCreateColormap(m_display, root, m_visual_info->visual, AllocNone);
        window_attributes.event_mask = m_event_mask;

        m_window = XCreateWindow(
            m_display, root, 0, 0, m_width, m_height, 0, m_visual_info->depth,
            InputOutput, m_visual_info->visual, CWColormap | CWEventMask, &window_attributes);

        XMapWindow(m_display, m_window);
        XSetWMProtocols(m_display, m_window, &m_WM_DELETE_WINDOW, 1);
        XStoreName(m_display, m_window, m_window_name.c_str());
#endif

        return init();
    }

    virtual void cleanup()
    {
#if defined(_WIN32)
        ::DestroyWindow(m_hWnd);
#elif defined(__linux__)
        XDestroyWindow(m_display, m_window);
        XCloseDisplay(m_display);
#endif
    }

#if defined(_WIN32)
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) = 0;
#endif

    int run()
    {
#if defined(_WIN32)
        MSG msg;

        ::ZeroMemory(&msg, sizeof(msg));

        while (msg.message != WM_QUIT)
        {
            if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
            {
                ::TranslateMessage(&msg);
                ::DispatchMessage(&msg);
            }
            else
            {
                idle();
            }
        }

        return static_cast<int>(msg.wParam);
#elif defined(__linux__)
        m_end_loop = false;

        do {
            XEvent e;

            if (!XCheckWindowEvent(m_display, m_window, m_event_mask, &e) || !handle_event(e))
            {
                idle();
            }
        } while (!m_end_loop);

        return 0;
#endif
    }

protected:

#if defined(_WIN32)
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        WinApp* pWnd;

        if (message == WM_NCCREATE)
        {
            LPCREATESTRUCT pCreateStruct = ((LPCREATESTRUCT)lParam);
            pWnd = (WinApp*)(pCreateStruct->lpCreateParams);
            ::SetWindowLongPtr(hWnd, GWLP_USERDATA, (LONG_PTR)pWnd);
        }

        pWnd = GetObjectFromWindow(hWnd);

        if (pWnd)
            return pWnd->WndProc(hWnd, message, wParam, lParam);
        else
            return ::DefWindowProc(hWnd, message, wParam, lParam);
    }

    inline static WinApp* GetObjectFromWindow(HWND hWnd)
    {
        return (WinApp*)::GetWindowLongPtr(hWnd, GWLP_USERDATA);
    }
#endif

#if defined(__linux__)
    virtual int handle_event(XEvent& e) = 0;
#endif

    virtual int init() = 0;
    virtual int render() = 0;

    virtual void idle() = 0;

#if defined(_WIN32)
    HINSTANCE    m_hInstance;
    HWND         m_hWnd;
#elif defined(__linux__)
    Display*     m_display;
    XVisualInfo* m_visual_info;
    Window       m_window;
    long         m_event_mask;
    Atom         m_WM_DELETE_WINDOW;
    bool         m_end_loop;
#endif
    int          m_width;
    int          m_height;
    std::string  m_window_name;
    Timer        m_timer;
};
