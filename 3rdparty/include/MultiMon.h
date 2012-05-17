//=============================================================================
//
// multimon.h -- Stub module that fakes multiple monitor apis on Win32 OSes
//               without them.
//
// By using this header your code will get back default values from
// GetSystemMetrics() for new metrics, and the new multimonitor APIs
// will act like only one display is present on a Win32 OS without
// multimonitor APIs.
//
// Exactly one source must include this with COMPILE_MULTIMON_STUBS defined.
//
// Copyright (c) Microsoft Corporation. All rights reserved. 
//
//=============================================================================

#ifdef __cplusplus
extern "C" {            // Assume C declarations for C++
#endif // __cplusplus

//
// If we are building with Win95/NT4 headers, we need to declare
// the multimonitor-related metrics and APIs ourselves.
//
#ifndef SM_CMONITORS

#define SM_XVIRTUALSCREEN       76
#define SM_YVIRTUALSCREEN       77
#define SM_CXVIRTUALSCREEN      78
#define SM_CYVIRTUALSCREEN      79
#define SM_CMONITORS            80
#define SM_SAMEDISPLAYFORMAT    81

// HMONITOR is already declared if WINVER >= 0x0500 in windef.h
// This is for components built with an older version number.
//
#if !defined(HMONITOR_DECLARED) && (WINVER < 0x0500)
DECLARE_HANDLE(HMONITOR);
#define HMONITOR_DECLARED
#endif

#define MONITOR_DEFAULTTONULL       0x00000000
#define MONITOR_DEFAULTTOPRIMARY    0x00000001
#define MONITOR_DEFAULTTONEAREST    0x00000002

#define MONITORINFOF_PRIMARY        0x00000001

typedef struct tagMONITORINFO
{
    DWORD   cbSize;
    RECT    rcMonitor;
    RECT    rcWork;
    DWORD   dwFlags;
} MONITORINFO, *LPMONITORINFO;

#ifndef CCHDEVICENAME
#define CCHDEVICENAME 32
#endif

#ifdef __cplusplus
typedef struct tagMONITORINFOEXA : public tagMONITORINFO
{
    CHAR        szDevice[CCHDEVICENAME];
} MONITORINFOEXA, *LPMONITORINFOEXA;
typedef struct tagMONITORINFOEXW : public tagMONITORINFO
{
    WCHAR       szDevice[CCHDEVICENAME];
} MONITORINFOEXW, *LPMONITORINFOEXW;
#ifdef UNICODE
typedef MONITORINFOEXW MONITORINFOEX;
typedef LPMONITORINFOEXW LPMONITORINFOEX;
#else
typedef MONITORINFOEXA MONITORINFOEX;
typedef LPMONITORINFOEXA LPMONITORINFOEX;
#endif // UNICODE
#else // ndef __cplusplus
typedef struct tagMONITORINFOEXA
{
    MONITORINFO;
    CHAR        szDevice[CCHDEVICENAME];
} MONITORINFOEXA, *LPMONITORINFOEXA;
typedef struct tagMONITORINFOEXW
{
    MONITORINFO;
    WCHAR       szDevice[CCHDEVICENAME];
} MONITORINFOEXW, *LPMONITORINFOEXW;
#ifdef UNICODE
typedef MONITORINFOEXW MONITORINFOEX;
typedef LPMONITORINFOEXW LPMONITORINFOEX;
#else
typedef MONITORINFOEXA MONITORINFOEX;
typedef LPMONITORINFOEXA LPMONITORINFOEX;
#endif // UNICODE
#endif

typedef BOOL (CALLBACK* MONITORENUMPROC)(HMONITOR, HDC, LPRECT, LPARAM);

#ifndef DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
typedef struct _DISPLAY_DEVICEA {
    DWORD  cb;
    CHAR   DeviceName[32];
    CHAR   DeviceString[128];
    DWORD  StateFlags;
    CHAR   DeviceID[128];
    CHAR   DeviceKey[128];
} DISPLAY_DEVICEA, *PDISPLAY_DEVICEA, *LPDISPLAY_DEVICEA;
typedef struct _DISPLAY_DEVICEW {
    DWORD  cb;
    WCHAR  DeviceName[32];
    WCHAR  DeviceString[128];
    DWORD  StateFlags;
    WCHAR  DeviceID[128];
    WCHAR  DeviceKey[128];
} DISPLAY_DEVICEW, *PDISPLAY_DEVICEW, *LPDISPLAY_DEVICEW;
#ifdef UNICODE
typedef DISPLAY_DEVICEW DISPLAY_DEVICE;
typedef PDISPLAY_DEVICEW PDISPLAY_DEVICE;
typedef LPDISPLAY_DEVICEW LPDISPLAY_DEVICE;
#else
typedef DISPLAY_DEVICEA DISPLAY_DEVICE;
typedef PDISPLAY_DEVICEA PDISPLAY_DEVICE;
typedef LPDISPLAY_DEVICEA LPDISPLAY_DEVICE;
#endif // UNICODE

#define DISPLAY_DEVICE_ATTACHED_TO_DESKTOP 0x00000001
#define DISPLAY_DEVICE_MULTI_DRIVER        0x00000002
#define DISPLAY_DEVICE_PRIMARY_DEVICE      0x00000004
#define DISPLAY_DEVICE_MIRRORING_DRIVER    0x00000008
#define DISPLAY_DEVICE_VGA_COMPATIBLE      0x00000010
#endif

#endif  // SM_CMONITORS

#undef GetMonitorInfo
#undef GetSystemMetrics
#undef MonitorFromWindow
#undef MonitorFromRect
#undef MonitorFromPoint
#undef EnumDisplayMonitors
#undef EnumDisplayDevices

//
// Define COMPILE_MULTIMON_STUBS to compile the stubs;
// otherwise, you get the declarations.
//
#ifdef COMPILE_MULTIMON_STUBS

//-----------------------------------------------------------------------------
//
// Implement the API stubs.
//
//-----------------------------------------------------------------------------

#ifndef _MULTIMON_USE_SECURE_CRT
#if defined(__GOT_SECURE_LIB__) && __GOT_SECURE_LIB__ >= 200402L
#define _MULTIMON_USE_SECURE_CRT 1
#else
#define _MULTIMON_USE_SECURE_CRT 0
#endif
#endif

#ifndef MULTIMON_FNS_DEFINED

int      (WINAPI* g_pfnGetSystemMetrics)(int) = NULL;
HMONITOR (WINAPI* g_pfnMonitorFromWindow)(HWND, DWORD) = NULL;
HMONITOR (WINAPI* g_pfnMonitorFromRect)(LPCRECT, DWORD) = NULL;
HMONITOR (WINAPI* g_pfnMonitorFromPoint)(POINT, DWORD) = NULL;
BOOL     (WINAPI* g_pfnGetMonitorInfo)(HMONITOR, LPMONITORINFO) = NULL;
BOOL     (WINAPI* g_pfnEnumDisplayMonitors)(HDC, LPCRECT, MONITORENUMPROC, LPARAM) = NULL;
BOOL     (WINAPI* g_pfnEnumDisplayDevices)(PVOID, DWORD, PDISPLAY_DEVICE,DWORD) = NULL;
BOOL     g_fMultiMonInitDone = FALSE;
BOOL     g_fMultimonPlatformNT = FALSE;

#endif

BOOL IsPlatformNT()
{ 
    OSVERSIONINFOA osvi = {0};
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    GetVersionExA((OSVERSIONINFOA*)&osvi);
    return (VER_PLATFORM_WIN32_NT == osvi.dwPlatformId);    
}

BOOL InitMultipleMonitorStubs(void)
{
    HMODULE hUser32;
    if (g_fMultiMonInitDone)
    {
        return g_pfnGetMonitorInfo != NULL;
    }

    g_fMultimonPlatformNT = IsPlatformNT();
    hUser32 = GetModuleHandle(TEXT("USER32"));
    if (hUser32 &&
        (*(FARPROC*)&g_pfnGetSystemMetrics    = GetProcAddress(hUser32,"GetSystemMetrics")) != NULL &&
        (*(FARPROC*)&g_pfnMonitorFromWindow   = GetProcAddress(hUser32,"MonitorFromWindow")) != NULL &&
        (*(FARPROC*)&g_pfnMonitorFromRect     = GetProcAddress(hUser32,"MonitorFromRect")) != NULL &&
        (*(FARPROC*)&g_pfnMonitorFromPoint    = GetProcAddress(hUser32,"MonitorFromPoint")) != NULL &&
        (*(FARPROC*)&g_pfnEnumDisplayMonitors = GetProcAddress(hUser32,"EnumDisplayMonitors")) != NULL &&
#ifdef UNICODE
        (*(FARPROC*)&g_pfnEnumDisplayDevices  = GetProcAddress(hUser32,"EnumDisplayDevicesW")) != NULL &&
        (*(FARPROC*)&g_pfnGetMonitorInfo      = g_fMultimonPlatformNT ? GetProcAddress(hUser32,"GetMonitorInfoW") : 
                                                GetProcAddress(hUser32,"GetMonitorInfoA")) != NULL
#else
        (*(FARPROC*)&g_pfnGetMonitorInfo      = GetProcAddress(hUser32,"GetMonitorInfoA")) != NULL &&
        (*(FARPROC*)&g_pfnEnumDisplayDevices  = GetProcAddress(hUser32,"EnumDisplayDevicesA")) != NULL
#endif
    ) {
        g_fMultiMonInitDone = TRUE;
        return TRUE;
    }
    else
    {
        g_pfnGetSystemMetrics    = NULL;
        g_pfnMonitorFromWindow   = NULL;
        g_pfnMonitorFromRect     = NULL;
        g_pfnMonitorFromPoint    = NULL;
        g_pfnGetMonitorInfo      = NULL;
        g_pfnEnumDisplayMonitors = NULL;
        g_pfnEnumDisplayDevices  = NULL;

        g_fMultiMonInitDone = TRUE;
        return FALSE;
    }
}

//-----------------------------------------------------------------------------
//
// fake implementations of Monitor APIs that work with the primary display
// no special parameter validation is made since these run in client code
//
//-----------------------------------------------------------------------------

int WINAPI
xGetSystemMetrics(int nIndex)
{
    if (InitMultipleMonitorStubs())
        return g_pfnGetSystemMetrics(nIndex);

    switch (nIndex)
    {
    case SM_CMONITORS:
    case SM_SAMEDISPLAYFORMAT:
        return 1;

    case SM_XVIRTUALSCREEN:
    case SM_YVIRTUALSCREEN:
        return 0;

    case SM_CXVIRTUALSCREEN:
        nIndex = SM_CXSCREEN;
        break;

    case SM_CYVIRTUALSCREEN:
        nIndex = SM_CYSCREEN;
        break;
    }

    return GetSystemMetrics(nIndex);
}

#define xPRIMARY_MONITOR ((HMONITOR)0x12340042)

HMONITOR WINAPI
xMonitorFromPoint(POINT ptScreenCoords, DWORD dwFlags)
{
    if (InitMultipleMonitorStubs())
        return g_pfnMonitorFromPoint(ptScreenCoords, dwFlags);

    if ((dwFlags & (MONITOR_DEFAULTTOPRIMARY | MONITOR_DEFAULTTONEAREST)) ||
        ((ptScreenCoords.x >= 0) &&
        (ptScreenCoords.x < GetSystemMetrics(SM_CXSCREEN)) &&
        (ptScreenCoords.y >= 0) &&
        (ptScreenCoords.y < GetSystemMetrics(SM_CYSCREEN))))
    {
        return xPRIMARY_MONITOR;
    }

    return NULL;
}

HMONITOR WINAPI
xMonitorFromRect(LPCRECT lprcScreenCoords, DWORD dwFlags)
{
    if (InitMultipleMonitorStubs())
        return g_pfnMonitorFromRect(lprcScreenCoords, dwFlags);

    if ((dwFlags & (MONITOR_DEFAULTTOPRIMARY | MONITOR_DEFAULTTONEAREST)) ||
        ((lprcScreenCoords->right > 0) &&
        (lprcScreenCoords->bottom > 0) &&
        (lprcScreenCoords->left < GetSystemMetrics(SM_CXSCREEN)) &&
        (lprcScreenCoords->top < GetSystemMetrics(SM_CYSCREEN))))
    {
        return xPRIMARY_MONITOR;
    }

    return NULL;
}

HMONITOR WINAPI
xMonitorFromWindow(HWND hWnd, DWORD dwFlags)
{
    WINDOWPLACEMENT wp;

    if (InitMultipleMonitorStubs())
        return g_pfnMonitorFromWindow(hWnd, dwFlags);

    if (dwFlags & (MONITOR_DEFAULTTOPRIMARY | MONITOR_DEFAULTTONEAREST))
        return xPRIMARY_MONITOR;

    if (IsIconic(hWnd) ?
            GetWindowPlacement(hWnd, &wp) :
            GetWindowRect(hWnd, &wp.rcNormalPosition)) {

        return xMonitorFromRect(&wp.rcNormalPosition, dwFlags);
    }

    return NULL;
}

BOOL WINAPI
xGetMonitorInfo(HMONITOR hMonitor, __inout LPMONITORINFO lpMonitorInfo)
{
    RECT rcWork;

    if (InitMultipleMonitorStubs())
    {
        BOOL f = g_pfnGetMonitorInfo(hMonitor, lpMonitorInfo);
#ifdef UNICODE
        if (f && !g_fMultimonPlatformNT && (lpMonitorInfo->cbSize >= sizeof(MONITORINFOEX)))
        { 
            MultiByteToWideChar(CP_ACP, 0,
                (LPSTR)((MONITORINFOEX*)lpMonitorInfo)->szDevice, -1,
                ((MONITORINFOEX*)lpMonitorInfo)->szDevice, (sizeof(((MONITORINFOEX*)lpMonitorInfo)->szDevice)/sizeof(TCHAR)));
        }
#endif
        return f;
    }

    if ((hMonitor == xPRIMARY_MONITOR) &&
        lpMonitorInfo &&
        (lpMonitorInfo->cbSize >= sizeof(MONITORINFO)) &&
        SystemParametersInfoA(SPI_GETWORKAREA, 0, &rcWork, 0))
    {
        lpMonitorInfo->rcMonitor.left = 0;
        lpMonitorInfo->rcMonitor.top  = 0;
        lpMonitorInfo->rcMonitor.right  = GetSystemMetrics(SM_CXSCREEN);
        lpMonitorInfo->rcMonitor.bottom = GetSystemMetrics(SM_CYSCREEN);
        lpMonitorInfo->rcWork = rcWork;
        lpMonitorInfo->dwFlags = MONITORINFOF_PRIMARY;

        if (lpMonitorInfo->cbSize >= sizeof(MONITORINFOEX))
        {
#ifdef UNICODE
            MultiByteToWideChar(CP_ACP, 0, "DISPLAY", -1, ((MONITORINFOEX*)lpMonitorInfo)->szDevice, (sizeof(((MONITORINFOEX*)lpMonitorInfo)->szDevice)/sizeof(TCHAR)));
#else // UNICODE
#if _MULTIMON_USE_SECURE_CRT
            strncpy_s(((MONITORINFOEX*)lpMonitorInfo)->szDevice, (sizeof(((MONITORINFOEX*)lpMonitorInfo)->szDevice)/sizeof(TCHAR)), TEXT("DISPLAY"), (sizeof(((MONITORINFOEX*)lpMonitorInfo)->szDevice)/sizeof(TCHAR)) - 1);
#else
            lstrcpyn(((MONITORINFOEX*)lpMonitorInfo)->szDevice, TEXT("DISPLAY"), (sizeof(((MONITORINFOEX*)lpMonitorInfo)->szDevice)/sizeof(TCHAR)));
#endif // _MULTIMON_USE_SECURE_CRT
#endif // UNICODE
        }

        return TRUE;
    }

    return FALSE;
}

BOOL WINAPI
xEnumDisplayMonitors(
        HDC             hdcOptionalForPainting,
        LPCRECT         lprcEnumMonitorsThatIntersect,
        MONITORENUMPROC lpfnEnumProc,
        LPARAM          dwData)
{
    RECT rcLimit;

    if (InitMultipleMonitorStubs()) {
        return g_pfnEnumDisplayMonitors(
                hdcOptionalForPainting,
                lprcEnumMonitorsThatIntersect,
                lpfnEnumProc,
                dwData);
    }

    if (!lpfnEnumProc)
        return FALSE;

    rcLimit.left   = 0;
    rcLimit.top    = 0;
    rcLimit.right  = GetSystemMetrics(SM_CXSCREEN);
    rcLimit.bottom = GetSystemMetrics(SM_CYSCREEN);

    if (hdcOptionalForPainting)
    {
        RECT    rcClip;
        POINT   ptOrg;

        switch (GetClipBox(hdcOptionalForPainting, &rcClip))
        {
        default:
            if (!GetDCOrgEx(hdcOptionalForPainting, &ptOrg))
                return FALSE;

            OffsetRect(&rcLimit, -ptOrg.x, -ptOrg.y);
            if (IntersectRect(&rcLimit, &rcLimit, &rcClip) &&
                (!lprcEnumMonitorsThatIntersect ||
                     IntersectRect(&rcLimit, &rcLimit, lprcEnumMonitorsThatIntersect))) {

                break;
            }
            //fall thru
        case NULLREGION:
             return TRUE;
        case ERROR:
             return FALSE;
        }
    } else {
        if (    lprcEnumMonitorsThatIntersect &&
                !IntersectRect(&rcLimit, &rcLimit, lprcEnumMonitorsThatIntersect)) {

            return TRUE;
        }
    }

    return lpfnEnumProc(
            xPRIMARY_MONITOR,
            hdcOptionalForPainting,
            &rcLimit,
            dwData);
}

BOOL WINAPI
xEnumDisplayDevices(
    PVOID Unused,
    DWORD iDevNum,
    __inout PDISPLAY_DEVICE lpDisplayDevice,
    DWORD dwFlags)
{
    if (InitMultipleMonitorStubs())
        return g_pfnEnumDisplayDevices(Unused, iDevNum, lpDisplayDevice, dwFlags);

    if (Unused != NULL)
        return FALSE;

    if (iDevNum != 0)
        return FALSE;

    if (lpDisplayDevice == NULL || lpDisplayDevice->cb < sizeof(DISPLAY_DEVICE))
        return FALSE;

#ifdef UNICODE
    MultiByteToWideChar(CP_ACP, 0, "DISPLAY", -1, lpDisplayDevice->DeviceName, (sizeof(lpDisplayDevice->DeviceName)/sizeof(TCHAR)));
    MultiByteToWideChar(CP_ACP, 0, "DISPLAY", -1, lpDisplayDevice->DeviceString, (sizeof(lpDisplayDevice->DeviceString)/sizeof(TCHAR)));
#else // UNICODE
#if _MULTIMON_USE_SECURE_CRT
    strncpy_s((LPTSTR)lpDisplayDevice->DeviceName, (sizeof(lpDisplayDevice->DeviceName)/sizeof(TCHAR)), TEXT("DISPLAY"), (sizeof(lpDisplayDevice->DeviceName)/sizeof(TCHAR)) - 1);
    strncpy_s((LPTSTR)lpDisplayDevice->DeviceString, (sizeof(lpDisplayDevice->DeviceString)/sizeof(TCHAR)), TEXT("DISPLAY"), (sizeof(lpDisplayDevice->DeviceName)/sizeof(TCHAR)) - 1);
#else
    lstrcpyn((LPTSTR)lpDisplayDevice->DeviceName,   TEXT("DISPLAY"), (sizeof(lpDisplayDevice->DeviceName)/sizeof(TCHAR)));
    lstrcpyn((LPTSTR)lpDisplayDevice->DeviceString, TEXT("DISPLAY"), (sizeof(lpDisplayDevice->DeviceString)/sizeof(TCHAR)));
#endif // _MULTIMON_USE_SECURE_CRT
#endif // UNICODE

    lpDisplayDevice->StateFlags = DISPLAY_DEVICE_ATTACHED_TO_DESKTOP | DISPLAY_DEVICE_PRIMARY_DEVICE;

    return TRUE;
}

#undef xPRIMARY_MONITOR
#undef COMPILE_MULTIMON_STUBS

#else   // COMPILE_MULTIMON_STUBS

extern int  WINAPI xGetSystemMetrics(int);
extern HMONITOR WINAPI xMonitorFromWindow(HWND, DWORD);
extern HMONITOR WINAPI xMonitorFromRect(LPCRECT, DWORD);
extern HMONITOR WINAPI xMonitorFromPoint(POINT, DWORD);
extern BOOL WINAPI xGetMonitorInfo(HMONITOR, LPMONITORINFO);
extern BOOL WINAPI xEnumDisplayMonitors(HDC, LPCRECT, MONITORENUMPROC, LPARAM);
extern BOOL WINAPI xEnumDisplayDevices(PVOID, DWORD, PDISPLAY_DEVICE, DWORD);

#endif  // COMPILE_MULTIMON_STUBS

//
// build defines that replace the regular APIs with our versions
//
#define GetSystemMetrics    xGetSystemMetrics
#define MonitorFromWindow   xMonitorFromWindow
#define MonitorFromRect     xMonitorFromRect
#define MonitorFromPoint    xMonitorFromPoint
#define GetMonitorInfo      xGetMonitorInfo
#define EnumDisplayMonitors xEnumDisplayMonitors
#define EnumDisplayDevices  xEnumDisplayDevices

#ifdef __cplusplus
}
#endif  // __cplusplus


