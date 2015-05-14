#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>


#define WINCLASS "WinAppWnd"

#define SAFE_RELEASE(p) if (p) { p->Release(); p = NULL; }

class WinApp
{
public:
    WinApp(int width, int height, std::string& window_name)
    {
        m_width       = width;
        m_height      = height;
        m_window_name = window_name;
        m_hInstance   = ::GetModuleHandle(NULL);
    }

    virtual ~WinApp()
    {
        ::UnregisterClass(WINCLASS, m_hInstance);
    }

    int Create()
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

        return init();
    }

    virtual LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) = 0;

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
                render();
            }
        }

        return static_cast<int>(msg.wParam);
    }

protected:
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

    virtual int init() = 0;
    virtual int render() = 0;

    HINSTANCE   m_hInstance;
    HWND        m_hWnd;
    int         m_width;
    int         m_height;
    std::string m_window_name;
};
