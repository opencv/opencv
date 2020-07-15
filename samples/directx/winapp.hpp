/*
// Sample demonstrating interoperability of OpenCV UMat with Direct X surface
// Base class for Windows application
*/
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>


#define WINCLASS "WinAppWnd"

class WinApp
{
public:
    WinApp(int width, int height, std::string& window_name)
    {
        m_width       = width;
        m_height      = height;
        m_window_name = window_name;
        m_hInstance   = ::GetModuleHandle(NULL);
        m_hWnd        = 0;
    }

    virtual ~WinApp() {}


    virtual int create()
    {
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
        if (!wc)
            return -1;

        RECT rc = { 0, 0, m_width, m_height };
        if(!::AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, false))
            return -1;

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

        return 0;
    } // create()


    int run()
    {
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
    } // run()


    virtual int cleanup()
    {
        ::DestroyWindow(m_hWnd);
        ::UnregisterClass(WINCLASS, m_hInstance);
        return 0;
    } // cleanup()

protected:
    // dispatch message handling to method of class
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        WinApp* pWnd;

        if (message == WM_NCCREATE)
        {
            LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
            pWnd = static_cast<WinApp*>(pCreateStruct->lpCreateParams);
            ::SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
        }

        pWnd = GetObjectFromWindow(hWnd);

        if (pWnd)
            return pWnd->WndProc(hWnd, message, wParam, lParam);
        else
            return ::DefWindowProc(hWnd, message, wParam, lParam);
    } // StaticWndProc()

    inline static WinApp* GetObjectFromWindow(HWND hWnd) { return (WinApp*)::GetWindowLongPtr(hWnd, GWLP_USERDATA); }

    // actual wnd message handling
    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) = 0;
    // idle processing
    virtual int idle() = 0;

    HINSTANCE   m_hInstance;
    HWND        m_hWnd;
    int         m_width;
    int         m_height;
    std::string m_window_name;
};
