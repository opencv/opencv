/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
//M*/

#include "precomp.hpp"

#if defined WIN32 || defined _WIN32

#if _MSC_VER >= 1200
#pragma warning( disable: 4710 )
#endif

#include <commctrl.h>
#include <winuser.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#ifdef HAVE_OPENGL
#include <memory>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/gpumat.hpp"
#include <GL\gl.h>
#endif

static const char* trackbar_text =
"                                                                                             ";

#if defined _M_X64 || defined __x86_64

#define icvGetWindowLongPtr GetWindowLongPtr
#define icvSetWindowLongPtr( hwnd, id, ptr ) SetWindowLongPtr( hwnd, id, (LONG_PTR)(ptr) )
#define icvGetClassLongPtr  GetClassLongPtr

#define CV_USERDATA GWLP_USERDATA
#define CV_WNDPROC GWLP_WNDPROC
#define CV_HCURSOR GCLP_HCURSOR
#define CV_HBRBACKGROUND GCLP_HBRBACKGROUND

#else

#define icvGetWindowLongPtr GetWindowLong
#define icvSetWindowLongPtr( hwnd, id, ptr ) SetWindowLong( hwnd, id, (size_t)ptr )
#define icvGetClassLongPtr GetClassLong

#define CV_USERDATA GWL_USERDATA
#define CV_WNDPROC GWL_WNDPROC
#define CV_HCURSOR GCL_HCURSOR
#define CV_HBRBACKGROUND GCL_HBRBACKGROUND

#endif

void FillBitmapInfo( BITMAPINFO* bmi, int width, int height, int bpp, int origin )
{
    assert( bmi && width >= 0 && height >= 0 && (bpp == 8 || bpp == 24 || bpp == 32));

    BITMAPINFOHEADER* bmih = &(bmi->bmiHeader);

    memset( bmih, 0, sizeof(*bmih));
    bmih->biSize = sizeof(BITMAPINFOHEADER);
    bmih->biWidth = width;
    bmih->biHeight = origin ? abs(height) : -abs(height);
    bmih->biPlanes = 1;
    bmih->biBitCount = (unsigned short)bpp;
    bmih->biCompression = BI_RGB;

    if( bpp == 8 )
    {
        RGBQUAD* palette = bmi->bmiColors;
        int i;
        for( i = 0; i < 256; i++ )
        {
            palette[i].rgbBlue = palette[i].rgbGreen = palette[i].rgbRed = (BYTE)i;
            palette[i].rgbReserved = 0;
        }
    }
}

struct CvWindow;

typedef struct CvTrackbar
{
    int signature;
    HWND hwnd;
    char* name;
    CvTrackbar* next;
    CvWindow* parent;
    HWND buddy;
    int* data;
    int pos;
    int maxval;
    void (*notify)(int);
    void (*notify2)(int, void*);
    void* userdata;
    int id;
}
CvTrackbar;

typedef struct CvWindow
{
    int signature;
    HWND hwnd;
    char* name;
    CvWindow* prev;
    CvWindow* next;
    HWND frame;

    HDC dc;
    HGDIOBJ image;
    int last_key;
    int flags;
	int status;//0 normal, 1 fullscreen (YV)

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    struct
    {
        HWND toolbar;
        int pos;
        int rows;
        WNDPROC toolBarProc;
        CvTrackbar* first;
    }
    toolbar;

    int width;
    int height;

    // OpenGL support

#ifdef HAVE_OPENGL
    bool useGl;
    HGLRC hGLRC;

    CvOpenGLCallback glDrawCallback;
    void* glDrawData;

    CvOpenGlCleanCallback glCleanCallback;
    void* glCleanData;

    cv::gpu::GlFuncTab* glFuncTab;
#endif
}
CvWindow;


#define HG_BUDDY_WIDTH  130

#ifndef TBIF_SIZE
    #define TBIF_SIZE  0x40
#endif

#ifndef TB_SETBUTTONINFO
    #define TB_SETBUTTONINFO (WM_USER + 66)
#endif

#ifndef TBM_GETTOOLTIPS
    #define TBM_GETTOOLTIPS  (WM_USER + 30)
#endif

static LRESULT CALLBACK HighGUIProc(  HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK WindowProc(  HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK MainWindowProc(  HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
static void icvUpdateWindowPos( CvWindow* window );

static CvWindow* hg_windows = 0;

typedef int (CV_CDECL * CvWin32WindowCallback)(HWND, UINT, WPARAM, LPARAM, int*);
static CvWin32WindowCallback hg_on_preprocess = 0, hg_on_postprocess = 0;
static HINSTANCE hg_hinstance = 0;

static const char* highGUIclassName = "HighGUI class";
static const char* mainHighGUIclassName = "Main HighGUI class";

static void icvCleanupHighgui()
{
    cvDestroyAllWindows();
    UnregisterClass(highGUIclassName, hg_hinstance);
    UnregisterClass(mainHighGUIclassName, hg_hinstance);
}

CV_IMPL int cvInitSystem( int, char** )
{
    static int wasInitialized = 0;

    // check initialization status
    if( !wasInitialized )
    {
        // Initialize the stogare
        hg_windows = 0;

        // Register the class
        WNDCLASS wndc;
        wndc.style = CS_OWNDC | CS_VREDRAW | CS_HREDRAW;
        wndc.lpfnWndProc = WindowProc;
        wndc.cbClsExtra = 0;
        wndc.cbWndExtra = 0;
        wndc.hInstance = hg_hinstance;
        wndc.lpszClassName = highGUIclassName;
        wndc.lpszMenuName = highGUIclassName;
        wndc.hIcon = LoadIcon(0, IDI_APPLICATION);
        wndc.hCursor = (HCURSOR)LoadCursor(0, (LPSTR)(size_t)IDC_CROSS );
        wndc.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);

        RegisterClass(&wndc);

        wndc.lpszClassName = mainHighGUIclassName;
        wndc.lpszMenuName = mainHighGUIclassName;
        wndc.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
        wndc.lpfnWndProc = MainWindowProc;

        RegisterClass(&wndc);
        atexit( icvCleanupHighgui );

        wasInitialized = 1;
    }

    return 0;
}

CV_IMPL int cvStartWindowThread(){
    return 0;
}

static CvWindow* icvFindWindowByName( const char* name )
{
    CvWindow* window = hg_windows;

    for( ; window != 0 && strcmp( name, window->name) != 0; window = window->next )
        ;

    return window;
}


static CvWindow* icvWindowByHWND( HWND hwnd )
{
    CvWindow* window = (CvWindow*)icvGetWindowLongPtr( hwnd, CV_USERDATA );
    return window != 0 && hg_windows != 0 &&
           window->signature == CV_WINDOW_MAGIC_VAL ? window : 0;
}


static CvTrackbar* icvTrackbarByHWND( HWND hwnd )
{
    CvTrackbar* trackbar = (CvTrackbar*)icvGetWindowLongPtr( hwnd, CV_USERDATA );
    return trackbar != 0 && trackbar->signature == CV_TRACKBAR_MAGIC_VAL &&
           trackbar->hwnd == hwnd ? trackbar : 0;
}


static const char* icvWindowPosRootKey = "Software\\OpenCV\\HighGUI\\Windows\\";

// Window positions saving/loading added by Philip Gruebele.
//<a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>
// Restores the window position from the registry saved position.
static void
icvLoadWindowPos( const char* name, CvRect& rect )
{
    HKEY hkey;
    char szKey[1024];
    strcpy( szKey, icvWindowPosRootKey );
    strcat( szKey, name );

    rect.x = rect.y = CW_USEDEFAULT;
    rect.width = rect.height = 320;

    if( RegOpenKeyEx(HKEY_CURRENT_USER,szKey,0,KEY_QUERY_VALUE,&hkey) == ERROR_SUCCESS )
    {
        // Yes we are installed.
        DWORD dwType = 0;
        DWORD dwSize = sizeof(int);

        RegQueryValueEx(hkey, "Left", NULL, &dwType, (BYTE*)&rect.x, &dwSize);
        RegQueryValueEx(hkey, "Top", NULL, &dwType, (BYTE*)&rect.y, &dwSize);
        RegQueryValueEx(hkey, "Width", NULL, &dwType, (BYTE*)&rect.width, &dwSize);
        RegQueryValueEx(hkey, "Height", NULL, &dwType, (BYTE*)&rect.height, &dwSize);

        if( rect.x != (int)CW_USEDEFAULT && (rect.x < -200 || rect.x > 3000) )
            rect.x = 100;
        if( rect.y != (int)CW_USEDEFAULT && (rect.y < -200 || rect.y > 3000) )
            rect.y = 100;

        if( rect.width != (int)CW_USEDEFAULT && (rect.width < 0 || rect.width > 3000) )
            rect.width = 100;
        if( rect.height != (int)CW_USEDEFAULT && (rect.height < 0 || rect.height > 3000) )
            rect.height = 100;

        RegCloseKey(hkey);
    }
}


// Window positions saving/loading added by Philip Gruebele.
//<a href="mailto:pgruebele@cox.net">pgruebele@cox.net</a>
// philipg.  Saves the window position in the registry
static void
icvSaveWindowPos( const char* name, CvRect rect )
{
    static const DWORD MAX_RECORD_COUNT = 100;
    HKEY hkey;
    char szKey[1024];
    char rootKey[1024];
    strcpy( szKey, icvWindowPosRootKey );
    strcat( szKey, name );
    
    if( RegOpenKeyEx( HKEY_CURRENT_USER,szKey,0,KEY_READ,&hkey) != ERROR_SUCCESS )
    {
        HKEY hroot;
        DWORD count = 0;
        FILETIME oldestTime = { UINT_MAX, UINT_MAX };
        char oldestKey[1024];
        char currentKey[1024];

        strcpy( rootKey, icvWindowPosRootKey );
        rootKey[strlen(rootKey)-1] = '\0';
        if( RegCreateKeyEx(HKEY_CURRENT_USER, rootKey, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_READ+KEY_WRITE, 0, &hroot, NULL) != ERROR_SUCCESS )
            //RegOpenKeyEx( HKEY_CURRENT_USER,rootKey,0,KEY_READ,&hroot) != ERROR_SUCCESS )
            return;

        for(;;)
        {
            DWORD csize = sizeof(currentKey);
            FILETIME accesstime = { 0, 0 };
            LONG code = RegEnumKeyEx( hroot, count, currentKey, &csize, NULL, NULL, NULL, &accesstime );
            if( code != ERROR_SUCCESS && code != ERROR_MORE_DATA )
                break;
            count++;
            if( oldestTime.dwHighDateTime > accesstime.dwHighDateTime ||
                (oldestTime.dwHighDateTime == accesstime.dwHighDateTime &&
                oldestTime.dwLowDateTime > accesstime.dwLowDateTime) )
            {
                oldestTime = accesstime;
                strcpy( oldestKey, currentKey );
            }
        }

        if( count >= MAX_RECORD_COUNT )
            RegDeleteKey( hroot, oldestKey );
        RegCloseKey( hroot );

        if( RegCreateKeyEx(HKEY_CURRENT_USER,szKey,0,NULL,REG_OPTION_NON_VOLATILE, KEY_WRITE, 0, &hkey, NULL) != ERROR_SUCCESS )
            return;
    }
    else
    {
        RegCloseKey( hkey );
        if( RegOpenKeyEx( HKEY_CURRENT_USER,szKey,0,KEY_WRITE,&hkey) != ERROR_SUCCESS )
            return;
    }
    
    RegSetValueEx(hkey, "Left", 0, REG_DWORD, (BYTE*)&rect.x, sizeof(rect.x));
    RegSetValueEx(hkey, "Top", 0, REG_DWORD, (BYTE*)&rect.y, sizeof(rect.y));
    RegSetValueEx(hkey, "Width", 0, REG_DWORD, (BYTE*)&rect.width, sizeof(rect.width));
    RegSetValueEx(hkey, "Height", 0, REG_DWORD, (BYTE*)&rect.height, sizeof(rect.height));
    RegCloseKey(hkey);
}

double cvGetModeWindow_W32(const char* name)//YV
{
	double result = -1;
	
	CV_FUNCNAME( "cvGetModeWindow_W32" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here
        
    result = window->status;
        
    __END__;
    return result;   
}

#ifdef MONITOR_DEFAULTTONEAREST
void cvSetModeWindow_W32( const char* name, double prop_value)//Yannick Verdie
{
	CV_FUNCNAME( "cvSetModeWindow_W32" );

	__BEGIN__;

	CvWindow* window;

	if(!name)
		CV_ERROR( CV_StsNullPtr, "NULL name string" );

	window = icvFindWindowByName( name );
	if( !window )
		CV_ERROR( CV_StsNullPtr, "NULL window" );

	if(window->flags & CV_WINDOW_AUTOSIZE)//if the flag CV_WINDOW_AUTOSIZE is set
		EXIT;

	{
		DWORD dwStyle = (DWORD)GetWindowLongPtr(window->frame, GWL_STYLE);
		CvRect position;

		if (window->status==CV_WINDOW_FULLSCREEN && prop_value==CV_WINDOW_NORMAL)
		{
			icvLoadWindowPos(window->name,position );
			SetWindowLongPtr(window->frame, GWL_STYLE, dwStyle | WS_CAPTION | WS_THICKFRAME);

			SetWindowPos(window->frame, HWND_TOP, position.x, position.y , position.width,position.height, SWP_NOZORDER | SWP_FRAMECHANGED);
			window->status=CV_WINDOW_NORMAL;

			EXIT;
		}

		if (window->status==CV_WINDOW_NORMAL && prop_value==CV_WINDOW_FULLSCREEN)
		{
			//save dimension
			RECT rect;
			GetWindowRect(window->frame, &rect);
			CvRect RectCV = cvRect(rect.left, rect.top,rect.right - rect.left, rect.bottom - rect.top);
			icvSaveWindowPos(window->name,RectCV );

			//Look at coordinate for fullscreen
			HMONITOR hMonitor;
			MONITORINFO mi;
			hMonitor = MonitorFromRect(&rect, MONITOR_DEFAULTTONEAREST);

			mi.cbSize = sizeof(mi);
			GetMonitorInfo(hMonitor, &mi);

			//fullscreen
			position.x=mi.rcMonitor.left;position.y=mi.rcMonitor.top;
			position.width=mi.rcMonitor.right - mi.rcMonitor.left;position.height=mi.rcMonitor.bottom - mi.rcMonitor.top;
			SetWindowLongPtr(window->frame, GWL_STYLE, dwStyle & ~WS_CAPTION & ~WS_THICKFRAME);

			SetWindowPos(window->frame, HWND_TOP, position.x, position.y , position.width,position.height, SWP_NOZORDER | SWP_FRAMECHANGED);
			window->status=CV_WINDOW_FULLSCREEN;

			EXIT;
		}
	}

	__END__;
}
#else
void cvSetModeWindow_W32( const char*, double)
{
}
#endif

double cvGetPropWindowAutoSize_W32(const char* name)
{
    double result = -1;

    CV_FUNCNAME( "cvSetCloseCallback" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here

    result = window->flags & CV_WINDOW_AUTOSIZE;

    __END__;

    return result;
}

double cvGetRatioWindow_W32(const char* name)
{
	double result = -1;
	
	CV_FUNCNAME( "cvGetRatioWindow_W32" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here
        
    result = static_cast<double>(window->width) / window->height;
        
    __END__;

    return result;   
}

double cvGetOpenGlProp_W32(const char* name)
{
	double result = -1;

#ifdef HAVE_OPENGL	
	CV_FUNCNAME( "cvGetOpenGlProp_W32" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here
        
    result = window->useGl;
        
    __END__;
#endif

    return result;   
}


// OpenGL support

#ifdef HAVE_OPENGL

#ifndef APIENTRY
    #define APIENTRY
#endif

#ifndef APIENTRYP
    #define APIENTRYP APIENTRY *
#endif

#ifndef GL_VERSION_1_5
    /* GL types for handling large vertex buffer objects */
    typedef ptrdiff_t GLintptr;
    typedef ptrdiff_t GLsizeiptr;
#endif

namespace
{
    typedef void (APIENTRYP PFNGLGENBUFFERSPROC   ) (GLsizei n, GLuint *buffers);
    typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);

    typedef void (APIENTRYP PFNGLBUFFERDATAPROC   ) (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);
    typedef void (APIENTRYP PFNGLBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, const GLvoid* data);

    typedef void (APIENTRYP PFNGLBINDBUFFERPROC   ) (GLenum target, GLuint buffer);

    typedef GLvoid* (APIENTRYP PFNGLMAPBUFFERPROC) (GLenum target, GLenum access);
    typedef GLboolean (APIENTRYP PFNGLUNMAPBUFFERPROC) (GLenum target);

    class GlFuncTab_W32 : public cv::gpu::GlFuncTab
    {
    public:
        PFNGLGENBUFFERSPROC    glGenBuffersExt;
        PFNGLDELETEBUFFERSPROC glDeleteBuffersExt;

        PFNGLBUFFERDATAPROC    glBufferDataExt;
        PFNGLBUFFERSUBDATAPROC glBufferSubDataExt;

        PFNGLBINDBUFFERPROC    glBindBufferExt;

        PFNGLMAPBUFFERPROC     glMapBufferExt;
        PFNGLUNMAPBUFFERPROC   glUnmapBufferExt;

        bool initialized;

        GlFuncTab_W32()
        {
            glGenBuffersExt    = 0;
            glDeleteBuffersExt = 0;

            glBufferDataExt    = 0;
            glBufferSubDataExt = 0;

            glBindBufferExt    = 0;

            glMapBufferExt     = 0;
            glUnmapBufferExt   = 0;

            initialized = false;
        }

        void genBuffers(int n, unsigned int* buffers) const
        {
            CV_FUNCNAME( "genBuffers" );

            __BEGIN__;

            if (!glGenBuffersExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glGenBuffersExt(n, buffers);
            CV_CheckGlError();

            __END__;
        }

        void deleteBuffers(int n, const unsigned int* buffers) const
        {
            CV_FUNCNAME( "deleteBuffers" );

            __BEGIN__;

            if (!glDeleteBuffersExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glDeleteBuffersExt(n, buffers);
            CV_CheckGlError();

            __END__;
        }

        void bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const
        {
            CV_FUNCNAME( "bufferData" );

            __BEGIN__;

            if (!glBufferDataExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glBufferDataExt(target, size, data, usage);
            CV_CheckGlError();

            __END__;
        }

        void bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const
        {
            CV_FUNCNAME( "bufferSubData" );

            __BEGIN__;

            if (!glBufferSubDataExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glBufferSubDataExt(target, offset, size, data);
            CV_CheckGlError();

            __END__;
        }

        void bindBuffer(unsigned int target, unsigned int buffer) const
        {
            CV_FUNCNAME( "bindBuffer" );

            __BEGIN__;

            if (!glBindBufferExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glBindBufferExt(target, buffer);
            CV_CheckGlError();

            __END__;
        }

        void* mapBuffer(unsigned int target, unsigned int access) const
        {
            CV_FUNCNAME( "mapBuffer" );

            void* res = 0;

            __BEGIN__;

            if (!glMapBufferExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            res = glMapBufferExt(target, access);
            CV_CheckGlError();

            __END__;

            return res;
        }

        void unmapBuffer(unsigned int target) const
        {
            CV_FUNCNAME( "unmapBuffer" );

            __BEGIN__;

            if (!glUnmapBufferExt)
                CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

            glUnmapBufferExt(target);
            CV_CheckGlError();

            __END__;
        }

        bool isGlContextInitialized() const
        {
            return initialized;
        }
    };

    void initGl(CvWindow* window)
    {
        std::auto_ptr<GlFuncTab_W32> glFuncTab(new GlFuncTab_W32);

        // Load extensions
        PROC func;

        func = wglGetProcAddress("glGenBuffers");
        glFuncTab->glGenBuffersExt = (PFNGLGENBUFFERSPROC)func;

        func = wglGetProcAddress("glDeleteBuffers");
        glFuncTab->glDeleteBuffersExt = (PFNGLDELETEBUFFERSPROC)func;

        func = wglGetProcAddress("glBufferData");
        glFuncTab->glBufferDataExt = (PFNGLBUFFERDATAPROC)func;

        func = wglGetProcAddress("glBufferSubData");
        glFuncTab->glBufferSubDataExt = (PFNGLBUFFERSUBDATAPROC)func;

        func = wglGetProcAddress("glBindBuffer");
        glFuncTab->glBindBufferExt = (PFNGLBINDBUFFERPROC)func;

        func = wglGetProcAddress("glMapBuffer");
        glFuncTab->glMapBufferExt = (PFNGLMAPBUFFERPROC)func;

        func = wglGetProcAddress("glUnmapBuffer");
        glFuncTab->glUnmapBufferExt = (PFNGLUNMAPBUFFERPROC)func;

        glFuncTab->initialized = true;

        window->glFuncTab = glFuncTab.release();

        cv::gpu::setGlFuncTab(window->glFuncTab);
    }

    void createGlContext(HWND hWnd, HDC& hGLDC, HGLRC& hGLRC, bool& useGl)
    {
        CV_FUNCNAME( "createGlContext" );

        __BEGIN__;

        useGl = false;

        int PixelFormat;

        static PIXELFORMATDESCRIPTOR pfd =
        {
            sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
            1,                             // Version Number
            PFD_DRAW_TO_WINDOW |           // Format Must Support Window
            PFD_SUPPORT_OPENGL |           // Format Must Support OpenGL
            PFD_DOUBLEBUFFER,              // Must Support Double Buffering
            PFD_TYPE_RGBA,                 // Request An RGBA Format
            32,                            // Select Our Color Depth
            0, 0, 0, 0, 0, 0,              // Color Bits Ignored
            0,                             // No Alpha Buffer
            0,                             // Shift Bit Ignored
            0,                             // No Accumulation Buffer
            0, 0, 0, 0,                    // Accumulation Bits Ignored
            32,                            // 32 Bit Z-Buffer (Depth Buffer)  
            0,                             // No Stencil Buffer
            0,                             // No Auxiliary Buffer
            PFD_MAIN_PLANE,                // Main Drawing Layer
            0,                             // Reserved
            0, 0, 0	                       // Layer Masks Ignored
        };

        hGLDC = GetDC(hWnd);
        if (!hGLDC)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Device Context" );

        PixelFormat = ChoosePixelFormat(hGLDC, &pfd);
        if (!PixelFormat)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Find A Suitable PixelFormat" );

        if (!SetPixelFormat(hGLDC, PixelFormat, &pfd))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Set The PixelFormat" );

        hGLRC = wglCreateContext(hGLDC);
        if (!hGLRC)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Rendering Context" );

        if (!wglMakeCurrent(hGLDC, hGLRC))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        useGl = true;

        __END__;
    }

    void releaseGlContext(CvWindow* window)
    {
        CV_FUNCNAME( "releaseGlContext" );

        __BEGIN__;

        delete window->glFuncTab;

        if (window->hGLRC)
        {
            wglDeleteContext(window->hGLRC);
            window->hGLRC = NULL;
        }

        if (window->dc)	
        {
            ReleaseDC(window->hwnd, window->dc);
            window->dc = NULL;
        }

        window->useGl = false;

        __END__;
    }

    void drawGl(CvWindow* window)
    {
        CV_FUNCNAME( "drawGl" );

        __BEGIN__;

        if (!wglMakeCurrent(window->dc, window->hGLRC))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (window->glDrawCallback)
            window->glDrawCallback(window->glDrawData);

        CV_CheckGlError();

        if (!SwapBuffers(window->dc))
            CV_ERROR( CV_OpenGlApiCallError, "Can't swap OpenGL buffers" );

        __END__;
    }

    void resizeGl(CvWindow* window)
    {
        CV_FUNCNAME( "resizeGl" );

        __BEGIN__;

        if (!wglMakeCurrent(window->dc, window->hGLRC))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        glViewport(0, 0, window->width, window->height);

        __END__;
    }
}

#endif // HAVE_OPENGL


CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    int result = 0;
    CV_FUNCNAME( "cvNamedWindow" );

    __BEGIN__;

    HWND hWnd, mainhWnd;
    CvWindow* window;
    DWORD defStyle = WS_VISIBLE | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU;
    int len;
    CvRect rect;

    cvInitSystem(0,0);

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    // Check the name in the storage
    window = icvFindWindowByName( name );
    if (window != 0)
    {
        result = 1;

        #ifdef HAVE_OPENGL
            if (window->useGl && !(flags & CV_WINDOW_OPENGL))
            {
                wglMakeCurrent(window->dc, window->hGLRC);

                if (window->glCleanCallback)
                {
                    window->glCleanCallback(window->glCleanData);
                    window->glCleanCallback = 0;
                    window->glCleanData = 0;
                }

                releaseGlContext(window);

                window->dc = CreateCompatibleDC(0);
                window->hGLRC = 0;
                window->useGl = false;
            }
            else if (!window->useGl && (flags & CV_WINDOW_OPENGL))
            {
                if (window->dc && window->image)
                    DeleteObject(SelectObject(window->dc, window->image));

                if (window->dc)
                    DeleteDC(window->dc);

                bool useGl = false;
                HDC hGLDC = 0;
                HGLRC hGLRC = 0;

                createGlContext(window->hwnd, hGLDC, hGLRC, useGl);

                if (!useGl)
                {
                    window->dc = CreateCompatibleDC(0);
                    window->hGLRC = 0;
                    window->useGl = false;

                    result = 0;
                }
                else
                {
                    window->dc = hGLDC;
                    window->hGLRC = hGLRC;
                    window->useGl = true;
                    initGl(window);
                }
            }
        #endif // HAVE_OPENGL

        EXIT;
    }

    if( !(flags & CV_WINDOW_AUTOSIZE))//YV add border in order to resize the window
       defStyle |= WS_SIZEBOX;

    icvLoadWindowPos( name, rect );

    mainhWnd = CreateWindow( "Main HighGUI class", name, defStyle | WS_OVERLAPPED,
                             rect.x, rect.y, rect.width, rect.height, 0, 0, hg_hinstance, 0 );
    if( !mainhWnd )
        CV_ERROR( CV_StsError, "Frame window can not be created" );

    ShowWindow(mainhWnd, SW_SHOW);

	//YV- remove one border by changing the style
    hWnd = CreateWindow("HighGUI class", "", (defStyle & ~WS_SIZEBOX) | WS_CHILD, CW_USEDEFAULT, 0, rect.width, rect.height, mainhWnd, 0, hg_hinstance, 0);
    if( !hWnd )
        CV_ERROR( CV_StsError, "Frame window can not be created" );

#ifndef HAVE_OPENGL
    if (flags & CV_WINDOW_OPENGL)
        CV_ERROR( CV_OpenGlNotSupported, "Library was built without OpenGL support" );
#else
    bool useGl = false;
    HDC hGLDC = 0;
    HGLRC hGLRC = 0;

    if (flags & CV_WINDOW_OPENGL)
        createGlContext(hWnd, hGLDC, hGLRC, useGl);
#endif

    ShowWindow(hWnd, SW_SHOW);

    len = (int)strlen(name);
    CV_CALL( window = (CvWindow*)cvAlloc(sizeof(CvWindow) + len + 1));

    window->signature = CV_WINDOW_MAGIC_VAL;
    window->hwnd = hWnd;
    window->frame = mainhWnd;
    window->name = (char*)(window + 1);
    memcpy( window->name, name, len + 1 );
    window->flags = flags;
    window->image = 0;

#ifndef HAVE_OPENGL
    window->dc = CreateCompatibleDC(0);
#else
    window->glFuncTab = 0;
    if (!useGl)
    {
        window->dc = CreateCompatibleDC(0);
        window->hGLRC = 0;
        window->useGl = false;
    }
    else
    {
        window->dc = hGLDC;
        window->hGLRC = hGLRC;
        window->useGl = true;
        initGl(window);
    }

    window->glDrawCallback = 0;
    window->glDrawData = 0;

    window->glCleanCallback = 0;
    window->glCleanData = 0;
#endif

    window->last_key = 0;
    window->status = CV_WINDOW_NORMAL;//YV

    window->on_mouse = 0;
    window->on_mouse_param = 0;

    memset( &window->toolbar, 0, sizeof(window->toolbar));

    window->next = hg_windows;
    window->prev = 0;
    if( hg_windows )
        hg_windows->prev = window;
    hg_windows = window;
    icvSetWindowLongPtr( hWnd, CV_USERDATA, window );
    icvSetWindowLongPtr( mainhWnd, CV_USERDATA, window );

    // Recalculate window position
    icvUpdateWindowPos( window );

    result = 1;
    __END__;

    return result;
}

#ifdef HAVE_OPENGL

CV_IMPL void cvSetOpenGlContext(const char* name)
{
    CV_FUNCNAME( "cvSetOpenGlContext" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    if (!window->useGl)
        CV_ERROR( CV_OpenGlNotSupported, "Window doesn't support OpenGL" );

    if (!wglMakeCurrent(window->dc, window->hGLRC))
        CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

    cv::gpu::setGlFuncTab(window->glFuncTab);

    __END__;
}

CV_IMPL void cvUpdateWindow(const char* name)
{
    CV_FUNCNAME( "cvUpdateWindow" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT;

    InvalidateRect(window->hwnd, 0, 0);

    __END__;
}

CV_IMPL void cvCreateOpenGLCallback(const char* name, CvOpenGLCallback callback, void* userdata, double, double, double)
{
    CV_FUNCNAME( "cvCreateOpenGLCallback" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        EXIT;

    if (!window->useGl)
        CV_ERROR( CV_OpenGlNotSupported, "Window was created without OpenGL context" );

    window->glDrawCallback = callback;
    window->glDrawData = userdata;

    __END__;
}

void icvSetOpenGlCleanCallback(const char* name, CvOpenGlCleanCallback callback, void* userdata)
{
    CV_FUNCNAME( "icvSetOpenGlCleanCallback" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR(CV_StsNullPtr, "NULL name string");

    window = icvFindWindowByName(name);
    if (!window)
        EXIT;

    if (window->glCleanCallback)
        window->glCleanCallback(window->glCleanData);

    window->glCleanCallback = callback;
    window->glCleanData = userdata;

    __END__;
}

#endif // HAVE_OPENGL

static void icvRemoveWindow( CvWindow* window )
{
    CvTrackbar* trackbar = NULL;
    RECT wrect={0,0,0,0};

#ifdef HAVE_OPENGL
    if (window->useGl)
    {
        wglMakeCurrent(window->dc, window->hGLRC);

        if (window->glCleanCallback)
        {
            window->glCleanCallback(window->glCleanData);
            window->glCleanCallback = 0;
            window->glCleanData = 0;
        }

        releaseGlContext(window);
    }
#endif

    if( window->frame )
        GetWindowRect( window->frame, &wrect );
    if( window->name )
        icvSaveWindowPos( window->name, cvRect(wrect.left, wrect.top,
            wrect.right-wrect.left, wrect.bottom-wrect.top) );

    if( window->hwnd )
        icvSetWindowLongPtr( window->hwnd, CV_USERDATA, 0 );
    if( window->frame )
        icvSetWindowLongPtr( window->frame, CV_USERDATA, 0 );

    if( window->toolbar.toolbar )
        icvSetWindowLongPtr(window->toolbar.toolbar, CV_USERDATA, 0);

    if( window->prev )
        window->prev->next = window->next;
    else
        hg_windows = window->next;

    if( window->next )
        window->next->prev = window->prev;

    window->prev = window->next = 0;

    if( window->dc && window->image )
        DeleteObject(SelectObject(window->dc,window->image));

    if( window->dc )
        DeleteDC(window->dc);

    for( trackbar = window->toolbar.first; trackbar != 0; )
    {
        CvTrackbar* next = trackbar->next;
        if( trackbar->hwnd )
        {
            icvSetWindowLongPtr( trackbar->hwnd, CV_USERDATA, 0 );
            cvFree( &trackbar );
        }
        trackbar = next;
    }

    cvFree( &window );
}


CV_IMPL void cvDestroyWindow( const char* name )
{
    CV_FUNCNAME( "cvDestroyWindow" );

    __BEGIN__;

    CvWindow* window;
    HWND mainhWnd;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        EXIT;

    mainhWnd = window->frame;

    SendMessage(window->hwnd, WM_CLOSE, 0, 0);
    SendMessage( mainhWnd, WM_CLOSE, 0, 0);
    // Do NOT call _remove_window -- CvWindow list will be updated automatically ...

    __END__;
}


static void icvScreenToClient( HWND hwnd, RECT* rect )
{
    POINT p;
    p.x = rect->left;
    p.y = rect->top;
    ScreenToClient(hwnd, &p);
    OffsetRect( rect, p.x - rect->left, p.y - rect->top );
}


/* Calculatess the window coordinates relative to the upper left corner of the mainhWnd window */
static RECT icvCalcWindowRect( CvWindow* window )
{
    const int gutter = 1;
    RECT crect, trect, rect;

    assert(window);

    GetClientRect(window->frame, &crect);
    if(window->toolbar.toolbar)
    {
        GetWindowRect(window->toolbar.toolbar, &trect);
        icvScreenToClient(window->frame, &trect);
        SubtractRect( &rect, &crect, &trect);
    }
    else
        rect = crect;

    rect.top += gutter;
    rect.left += gutter;
    rect.bottom -= gutter;
    rect.right -= gutter;

    return rect;
}

// returns TRUE if there is a problem such as ERROR_IO_PENDING.
static bool icvGetBitmapData( CvWindow* window, SIZE* size, int* channels, void** data )
{
    BITMAP bmp;
    GdiFlush();
    HGDIOBJ h = GetCurrentObject( window->dc, OBJ_BITMAP );
    if( size )
        size->cx = size->cy = 0;
    if( data )
        *data = 0;

    if (h == NULL)
        return true;
    if (GetObject(h, sizeof(bmp), &bmp) == 0)
        return true;

    if( size )
    {
        size->cx = abs(bmp.bmWidth);
        size->cy = abs(bmp.bmHeight);
    }

    if( channels )
        *channels = bmp.bmBitsPixel/8;

    if( data )
        *data = bmp.bmBits;

    return false;
}


static void icvUpdateWindowPos( CvWindow* window )
{
    RECT rect;
    assert(window);

    if( (window->flags & CV_WINDOW_AUTOSIZE) && window->image )
    {
        int i;
        SIZE size = {0,0};
        icvGetBitmapData( window, &size, 0, 0 );

        // Repeat two times because after the first resizing of the mainhWnd window
        // toolbar may resize too
        for(i = 0; i < (window->toolbar.toolbar ? 2 : 1); i++)
        {
            RECT rmw, rw = icvCalcWindowRect(window );
            MoveWindow(window->hwnd, rw.left, rw.top,
                rw.right - rw.left + 1, rw.bottom - rw.top + 1, FALSE);
            GetClientRect(window->hwnd, &rw);
            GetWindowRect(window->frame, &rmw);
            // Resize the mainhWnd window in order to make the bitmap fit into the child window
            MoveWindow(window->frame, rmw.left, rmw.top,
                rmw.right - rmw.left + size.cx - rw.right + rw.left,
                rmw.bottom  - rmw.top + size.cy - rw.bottom + rw.top, TRUE );
        }
    }

    rect = icvCalcWindowRect(window);
    MoveWindow(window->hwnd, rect.left, rect.top,
               rect.right - rect.left + 1,
               rect.bottom - rect.top + 1, TRUE );
}

CV_IMPL void
cvShowImage( const char* name, const CvArr* arr )
{
    CV_FUNCNAME( "cvShowImage" );

    __BEGIN__;

    CvWindow* window;
    SIZE size = { 0, 0 };
    int channels = 0;
    void* dst_ptr = 0;
    const int channels0 = 3;
    int origin = 0;
    CvMat stub, dst, *image;
    bool changed_size = false; // philipg

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
	if(!window)
	{
        #ifndef HAVE_OPENGL
		    cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
        #else
		    cvNamedWindow(name, CV_WINDOW_AUTOSIZE | CV_WINDOW_OPENGL);
        #endif

		window = icvFindWindowByName(name);
	}

    if( !window || !arr )
        EXIT; // keep silence here.

    if( CV_IS_IMAGE_HDR( arr ))
        origin = ((IplImage*)arr)->origin;

    CV_CALL( image = cvGetMat( arr, &stub ));

#ifdef HAVE_OPENGL
    if (window->useGl)
    {
        cv::Mat im(image);
        cv::imshow(name, im);
        return;
    }
#endif

    if (window->image)
        // if there is something wrong with these system calls, we cannot display image...
        if (icvGetBitmapData( window, &size, &channels, &dst_ptr ))
            return;

    if( size.cx != image->width || size.cy != image->height || channels != channels0 )
    {
        changed_size = true;

        uchar buffer[sizeof(BITMAPINFO) + 255*sizeof(RGBQUAD)];
        BITMAPINFO* binfo = (BITMAPINFO*)buffer;

        DeleteObject( SelectObject( window->dc, window->image ));
        window->image = 0;

        size.cx = image->width;
        size.cy = image->height;
        channels = channels0;

        FillBitmapInfo( binfo, size.cx, size.cy, channels*8, 1 );

        window->image = SelectObject( window->dc, CreateDIBSection(window->dc, binfo,
                                      DIB_RGB_COLORS, &dst_ptr, 0, 0));
    }

    cvInitMatHeader( &dst, size.cy, size.cx, CV_8UC3,
                     dst_ptr, (size.cx * channels + 3) & -4 );
    cvConvertImage( image, &dst, origin == 0 ? CV_CVTIMG_FLIP : 0 );

    // ony resize window if needed
    if (changed_size)
        icvUpdateWindowPos(window);
    InvalidateRect(window->hwnd, 0, 0);
    // philipg: this is not needed and just slows things down
//    UpdateWindow(window->hwnd);

    __END__;
}


CV_IMPL void cvResizeWindow(const char* name, int width, int height )
{
    CV_FUNCNAME( "cvResizeWindow" );

    __BEGIN__;

    int i;
    CvWindow* window;
    RECT rmw, rw, rect;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    // Repeat two times because after the first resizing of the mainhWnd window
    // toolbar may resize too
    for(i = 0; i < (window->toolbar.toolbar ? 2 : 1); i++)
    {
        rw = icvCalcWindowRect(window);
        MoveWindow(window->hwnd, rw.left, rw.top,
            rw.right - rw.left + 1, rw.bottom - rw.top + 1, FALSE);
        GetClientRect(window->hwnd, &rw);
        GetWindowRect(window->frame, &rmw);
        // Resize the mainhWnd window in order to make the bitmap fit into the child window
        MoveWindow(window->frame, rmw.left, rmw.top,
            rmw.right - rmw.left + width - rw.right + rw.left,
            rmw.bottom  - rmw.top + height - rw.bottom + rw.top, TRUE);
    }

    rect = icvCalcWindowRect(window);
    MoveWindow(window->hwnd, rect.left, rect.top,
        rect.right - rect.left + 1, rect.bottom - rect.top + 1, TRUE);

    __END__;
}


CV_IMPL void cvMoveWindow( const char* name, int x, int y )
{
    CV_FUNCNAME( "cvMoveWindow" );

    __BEGIN__;

    CvWindow* window;
    RECT rect;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    GetWindowRect( window->frame, &rect );
    MoveWindow( window->frame, x, y, rect.right - rect.left, rect.bottom - rect.top, TRUE);

    __END__;
}


static LRESULT CALLBACK
MainWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    CvWindow* window = icvWindowByHWND( hwnd );
    if( !window )
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    switch(uMsg)
    {
    case WM_DESTROY:

        icvRemoveWindow(window);
        // Do nothing!!!
        //PostQuitMessage(0);
        break;

    case WM_GETMINMAXINFO:
        if( !(window->flags & CV_WINDOW_AUTOSIZE) )
        {
            MINMAXINFO* minmax = (MINMAXINFO*)lParam;
            RECT rect;
            LRESULT retval = DefWindowProc(hwnd, uMsg, wParam, lParam);

            minmax->ptMinTrackSize.y = 100;
            minmax->ptMinTrackSize.x = 100;

            if( window->toolbar.first )
            {
                GetWindowRect( window->toolbar.first->hwnd, &rect );
                minmax->ptMinTrackSize.y += window->toolbar.rows*(rect.bottom - rect.top);
                minmax->ptMinTrackSize.x = MAX(rect.right - rect.left + HG_BUDDY_WIDTH, HG_BUDDY_WIDTH*2);
            }
            return retval;
        }
        break;

    case WM_WINDOWPOSCHANGED:
        {
            WINDOWPOS* pos = (WINDOWPOS*)lParam;

            // Update the toolbar position/size
            if(window->toolbar.toolbar)
            {
                RECT rect;
                GetWindowRect(window->toolbar.toolbar, &rect);
                MoveWindow(window->toolbar.toolbar, 0, 0, pos->cx, rect.bottom - rect.top, TRUE);
            }

            if(!(window->flags & CV_WINDOW_AUTOSIZE))
                icvUpdateWindowPos(window);

            break;
        }

    case WM_ACTIVATE:
        if(LOWORD(wParam) == WA_ACTIVE || LOWORD(wParam) == WA_CLICKACTIVE)
            SetFocus(window->hwnd);
        break;

    case WM_ERASEBKGND:
        {
            RECT cr, tr, wrc;
            HRGN rgn, rgn1, rgn2;
            int ret;
            HDC hdc = (HDC)wParam;
            GetWindowRect(window->hwnd, &cr);
            icvScreenToClient(window->frame, &cr);
            if(window->toolbar.toolbar)
            {
                GetWindowRect(window->toolbar.toolbar, &tr);
                icvScreenToClient(window->frame, &tr);
            }
            else
                tr.left = tr.top = tr.right = tr.bottom = 0;

            GetClientRect(window->frame, &wrc);

            rgn = CreateRectRgn(0, 0, wrc.right, wrc.bottom);
            rgn1 = CreateRectRgn(cr.left, cr.top, cr.right, cr.bottom);
            rgn2 = CreateRectRgn(tr.left, tr.top, tr.right, tr.bottom);
            ret = CombineRgn(rgn, rgn, rgn1, RGN_DIFF);
            ret = CombineRgn(rgn, rgn, rgn2, RGN_DIFF);

            if(ret != NULLREGION && ret != ERROR)
                FillRgn(hdc, rgn, (HBRUSH)icvGetClassLongPtr(hwnd, CV_HBRBACKGROUND));

            DeleteObject(rgn);
            DeleteObject(rgn1);
            DeleteObject(rgn2);
        }
        return 1;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


static LRESULT CALLBACK HighGUIProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    CvWindow* window = icvWindowByHWND(hwnd);
    if( !window )
        // This window is not mentioned in HighGUI storage
        // Actually, this should be error except for the case of calls to CreateWindow
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    // Process the message
    switch(uMsg)
    {
    case WM_WINDOWPOSCHANGING:
        {
            LPWINDOWPOS pos = (LPWINDOWPOS)lParam;
            RECT rect = icvCalcWindowRect(window);
            pos->x = rect.left;
            pos->y = rect.top;
            pos->cx = rect.right - rect.left + 1;
            pos->cy = rect.bottom - rect.top + 1;
        }
        break;

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDBLCLK:
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    case WM_MOUSEMOVE:
        if( window->on_mouse )
        {
            POINT pt;
            RECT rect;
            SIZE size = {0,0};

            int flags = (wParam & MK_LBUTTON ? CV_EVENT_FLAG_LBUTTON : 0)|
                        (wParam & MK_RBUTTON ? CV_EVENT_FLAG_RBUTTON : 0)|
                        (wParam & MK_MBUTTON ? CV_EVENT_FLAG_MBUTTON : 0)|
                        (wParam & MK_CONTROL ? CV_EVENT_FLAG_CTRLKEY : 0)|
                        (wParam & MK_SHIFT ? CV_EVENT_FLAG_SHIFTKEY : 0)|
                        (GetKeyState(VK_MENU) < 0 ? CV_EVENT_FLAG_ALTKEY : 0);
            int event = uMsg == WM_LBUTTONDOWN ? CV_EVENT_LBUTTONDOWN :
                        uMsg == WM_RBUTTONDOWN ? CV_EVENT_RBUTTONDOWN :
                        uMsg == WM_MBUTTONDOWN ? CV_EVENT_MBUTTONDOWN :
                        uMsg == WM_LBUTTONUP ? CV_EVENT_LBUTTONUP :
                        uMsg == WM_RBUTTONUP ? CV_EVENT_RBUTTONUP :
                        uMsg == WM_MBUTTONUP ? CV_EVENT_MBUTTONUP :
                        uMsg == WM_LBUTTONDBLCLK ? CV_EVENT_LBUTTONDBLCLK :
                        uMsg == WM_RBUTTONDBLCLK ? CV_EVENT_RBUTTONDBLCLK :
                        uMsg == WM_MBUTTONDBLCLK ? CV_EVENT_MBUTTONDBLCLK :
                                                   CV_EVENT_MOUSEMOVE;
            if( uMsg == WM_LBUTTONDOWN || uMsg == WM_RBUTTONDOWN || uMsg == WM_MBUTTONDOWN )
                SetCapture( hwnd );
            if( uMsg == WM_LBUTTONUP || uMsg == WM_RBUTTONUP || uMsg == WM_MBUTTONUP )
                ReleaseCapture();

            pt.x = LOWORD( lParam );
            pt.y = HIWORD( lParam );

            GetClientRect( window->hwnd, &rect );
            icvGetBitmapData( window, &size, 0, 0 );

            window->on_mouse( event, pt.x*size.cx/MAX(rect.right - rect.left,1),
                                     pt.y*size.cy/MAX(rect.bottom - rect.top,1), flags,
                                     window->on_mouse_param );
        }
        break;

    case WM_PAINT:
        if(window->image != 0)
        {
            int nchannels = 3;
            SIZE size = {0,0};
            PAINTSTRUCT paint;
            HDC hdc;
            RGBQUAD table[256];

            // Determine the bitmap's dimensions
            icvGetBitmapData( window, &size, &nchannels, 0 );

            hdc = BeginPaint(hwnd, &paint);
            SetStretchBltMode(hdc, COLORONCOLOR);

            if( nchannels == 1 )
            {
                int i;
                for(i = 0; i < 256; i++)
                {
                    table[i].rgbBlue = (unsigned char)i;
                    table[i].rgbGreen = (unsigned char)i;
                    table[i].rgbRed = (unsigned char)i;
                }
                SetDIBColorTable(window->dc, 0, 255, table);
            }

            if(window->flags & CV_WINDOW_AUTOSIZE)
            {
                BitBlt( hdc, 0, 0, size.cx, size.cy, window->dc, 0, 0, SRCCOPY );
            }
            else
            {
                RECT rect;
                GetClientRect(window->hwnd, &rect);
                StretchBlt( hdc, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                            window->dc, 0, 0, size.cx, size.cy, SRCCOPY );
            }
            //DeleteDC(hdc);
            EndPaint(hwnd, &paint);
        }
#ifdef HAVE_OPENGL
        else if(window->useGl) 
        {
            drawGl(window);            
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
#endif
        else
        {
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
        return 0;

    case WM_ERASEBKGND:
        if(window->image)
            return 0;
        break;

    case WM_DESTROY:

        icvRemoveWindow(window);
        // Do nothing!!!
        //PostQuitMessage(0);
        break;

    case WM_SETCURSOR:
        SetCursor((HCURSOR)icvGetClassLongPtr(hwnd, CV_HCURSOR));
        return 0;

    case WM_KEYDOWN:
        window->last_key = (int)wParam;
        return 0;

    case WM_SIZE:
        window->width = LOWORD(lParam);
        window->height = HIWORD(lParam);

#ifdef HAVE_OPENGL
        if (window->useGl)
            resizeGl(window);
#endif
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}


static LRESULT CALLBACK WindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    LRESULT ret;

    if( hg_on_preprocess )
    {
        int was_processed = 0;
        int ret = hg_on_preprocess(hwnd, uMsg, wParam, lParam, &was_processed);
        if( was_processed )
            return ret;
    }
    ret = HighGUIProc(hwnd, uMsg, wParam, lParam);

    if(hg_on_postprocess)
    {
        int was_processed = 0;
        int ret = hg_on_postprocess(hwnd, uMsg, wParam, lParam, &was_processed);
        if( was_processed )
            return ret;
    }

    return ret;
}


static void icvUpdateTrackbar( CvTrackbar* trackbar, int pos )
{
    const int max_name_len = 10;
    const char* suffix = "";
    char pos_text[32];
    int name_len;

    if( trackbar->data )
        *trackbar->data = pos;

    if( trackbar->pos != pos )
    {
        trackbar->pos = pos;
        if( trackbar->notify2 )
            trackbar->notify2(pos, trackbar->userdata);
        if( trackbar->notify )
            trackbar->notify(pos);

        name_len = (int)strlen(trackbar->name);

        if( name_len > max_name_len )
        {
            int start_len = max_name_len*2/3;
            int end_len = max_name_len - start_len - 2;
            memcpy( pos_text, trackbar->name, start_len );
            memcpy( pos_text + start_len, "...", 3 );
            memcpy( pos_text + start_len + 3, trackbar->name + name_len - end_len, end_len + 1 );
        }
        else
        {
            memcpy( pos_text, trackbar->name, name_len + 1);
        }

        sprintf( pos_text + strlen(pos_text), "%s: %d\n", suffix, pos );
        SetWindowText( trackbar->buddy, pos_text );
    }
}


static LRESULT CALLBACK HGToolbarProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    CvWindow* window = icvWindowByHWND( hwnd );
    if(!window)
        return DefWindowProc(hwnd, uMsg, wParam, lParam);

    // Control messages processing
    switch(uMsg)
    {
    // Slider processing
    case WM_HSCROLL:
        {
            HWND slider = (HWND)lParam;
            int pos = (int)SendMessage(slider, TBM_GETPOS, 0, 0);
            CvTrackbar* trackbar = icvTrackbarByHWND( slider );

            if( trackbar )
            {
                if( trackbar->pos != pos )
                    icvUpdateTrackbar( trackbar, pos );
            }

            SetFocus( window->hwnd );
            return 0;
        }

    case WM_NCCALCSIZE:
        {
            LRESULT ret = CallWindowProc(window->toolbar.toolBarProc, hwnd, uMsg, wParam, lParam);
            int rows = (int)SendMessage(hwnd, TB_GETROWS, 0, 0);

            if(window->toolbar.rows != rows)
            {
                SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);
                CvTrackbar* trackbar = window->toolbar.first;

                for( ; trackbar != 0; trackbar = trackbar->next )
                {
                    RECT rect;
                    SendMessage(window->toolbar.toolbar, TB_GETITEMRECT,
                               (WPARAM)trackbar->id, (LPARAM)&rect);
                    MoveWindow(trackbar->hwnd, rect.left + HG_BUDDY_WIDTH, rect.top,
                               rect.right - rect.left - HG_BUDDY_WIDTH,
                               rect.bottom - rect.top, FALSE);
                    MoveWindow(trackbar->buddy, rect.left, rect.top,
                               HG_BUDDY_WIDTH, rect.bottom - rect.top, FALSE);
                }
                window->toolbar.rows = rows;
            }
            return ret;
        }
    }

    return CallWindowProc(window->toolbar.toolBarProc, hwnd, uMsg, wParam, lParam);
}


CV_IMPL void
cvDestroyAllWindows(void)
{
    CvWindow* window = hg_windows;

    while( window )
    {
        HWND mainhWnd = window->frame;
        HWND hwnd = window->hwnd;
        window = window->next;

        SendMessage( hwnd, WM_CLOSE, 0, 0 );
        SendMessage( mainhWnd, WM_CLOSE, 0, 0 );
    }
}


CV_IMPL int
cvWaitKey( int delay )
{
    int time0 = GetTickCount();

    for(;;)
    {
        CvWindow* window;
        MSG message;
        int is_processed = 0;

        if( (delay > 0 && abs((int)(GetTickCount() - time0)) >= delay) || hg_windows == 0 )
            return -1;

        if( delay <= 0 )
            GetMessage(&message, 0, 0, 0);
        else if( PeekMessage(&message, 0, 0, 0, PM_REMOVE) == FALSE )
        {
            Sleep(1);
            continue;
        }

        for( window = hg_windows; window != 0 && is_processed == 0; window = window->next )
        {
            if( window->hwnd == message.hwnd || window->frame == message.hwnd )
            {
                is_processed = 1;
                switch(message.message)
                {
                case WM_DESTROY:
                case WM_CHAR:
                    DispatchMessage(&message);
                    return (int)message.wParam;

                case WM_SYSKEYDOWN:
                    if( message.wParam == VK_F10 )
                    {
                        is_processed = 1;
                        return (int)(message.wParam << 16);
                    }
                    break;

                case WM_KEYDOWN:
                    TranslateMessage(&message);
                    if( (message.wParam >= VK_F1 && message.wParam <= VK_F24) ||
                        message.wParam == VK_HOME || message.wParam == VK_END ||
                        message.wParam == VK_UP || message.wParam == VK_DOWN ||
                        message.wParam == VK_LEFT || message.wParam == VK_RIGHT ||
                        message.wParam == VK_INSERT || message.wParam == VK_DELETE ||
                        message.wParam == VK_PRIOR || message.wParam == VK_NEXT )
                    {
                        DispatchMessage(&message);
                        is_processed = 1;
                        return (int)(message.wParam << 16);
                    }
                default:
                    DispatchMessage(&message);
                    is_processed = 1;
                    break;
                }
            }
        }

        if( !is_processed )
        {
            TranslateMessage(&message);
            DispatchMessage(&message);
        }
    }
}


static CvTrackbar*
icvFindTrackbarByName( const CvWindow* window, const char* name )
{
    CvTrackbar* trackbar = window->toolbar.first;

    for( ; trackbar != 0 && strcmp( trackbar->name, name ) != 0; trackbar = trackbar->next )
        ;

    return trackbar;
}


typedef struct
{
    UINT cbSize;
    DWORD dwMask;
    int idCommand;
    int iImage;
    BYTE fsState;
    BYTE fsStyle;
    WORD cx;
    DWORD lParam;
    LPSTR pszText;
    int cchText;
}
ButtonInfo;


static int
icvCreateTrackbar( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback on_notify,
                   CvTrackbarCallback2 on_notify2, void* userdata )
{
    int result = 0;

    CV_FUNCNAME( "icvCreateTrackbar" );

    __BEGIN__;

    char slider_name[32];
    CvWindow* window = 0;
    CvTrackbar* trackbar = 0;
    int pos = 0;

    if( !window_name || !trackbar_name )
        CV_ERROR( CV_StsNullPtr, "NULL window or trackbar name" );

    if( count <= 0 )
        CV_ERROR( CV_StsOutOfRange, "Bad trackbar maximal value" );

    window = icvFindWindowByName(window_name);
    if( !window )
        EXIT;

    trackbar = icvFindTrackbarByName(window,trackbar_name);
    if( !trackbar )
    {
        TBBUTTON tbs;
        ButtonInfo tbis;
        RECT rect;
        int bcount;
        int len = (int)strlen( trackbar_name );

        // create toolbar if it is not created yet
        if( !window->toolbar.toolbar )
        {
            const int default_height = 30;

            window->toolbar.toolbar = CreateToolbarEx(
                    window->frame, WS_CHILD | CCS_TOP | TBSTYLE_WRAPABLE,
                    1, 0, 0, 0, 0, 0, 16, 20, 16, 16, sizeof(TBBUTTON));
            GetClientRect(window->frame, &rect);
            MoveWindow( window->toolbar.toolbar, 0, 0,
                        rect.right - rect.left, default_height, TRUE);
            SendMessage(window->toolbar.toolbar, TB_AUTOSIZE, 0, 0);
            ShowWindow(window->toolbar.toolbar, SW_SHOW);

            window->toolbar.first = 0;
            window->toolbar.pos = 0;
            window->toolbar.rows = 0;
            window->toolbar.toolBarProc =
                (WNDPROC)icvGetWindowLongPtr(window->toolbar.toolbar, CV_WNDPROC);

            icvUpdateWindowPos(window);

            // Subclassing from toolbar
            icvSetWindowLongPtr(window->toolbar.toolbar, CV_WNDPROC, HGToolbarProc);
            icvSetWindowLongPtr(window->toolbar.toolbar, CV_USERDATA, window);
        }

        /* Retrieve current buttons count */
        bcount = (int)SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);

        if(bcount > 1)
        {
            /* If this is not the first button then we need to
            separate it from the previous one */
            tbs.iBitmap = 0;
            tbs.idCommand = bcount; // Set button id to it's number
            tbs.iString = 0;
            tbs.fsStyle = TBSTYLE_SEP;
            tbs.fsState = TBSTATE_ENABLED;
            SendMessage(window->toolbar.toolbar, TB_ADDBUTTONS, 1, (LPARAM)&tbs);

            // Retrieve current buttons count
            bcount = (int)SendMessage(window->toolbar.toolbar, TB_BUTTONCOUNT, 0, 0);
        }

        /* Add a button which we're going to cover with the slider */
        tbs.iBitmap = 0;
        tbs.idCommand = bcount; // Set button id to it's number
        tbs.fsState = TBSTATE_ENABLED;
#if 0/*!defined WIN64 && !defined EM64T*/
        tbs.fsStyle = 0;
        tbs.iString = 0;
#else

#ifndef TBSTYLE_AUTOSIZE
#define TBSTYLE_AUTOSIZE        0x0010
#endif

#ifndef TBSTYLE_GROUP
#define TBSTYLE_GROUP           0x0004
#endif
        //tbs.fsStyle = TBSTYLE_AUTOSIZE;
        tbs.fsStyle = TBSTYLE_GROUP;
        tbs.iString = (INT_PTR)trackbar_text;
#endif
        SendMessage(window->toolbar.toolbar, TB_ADDBUTTONS, 1, (LPARAM)&tbs);

        /* Adjust button size to the slider */
        tbis.cbSize = sizeof(tbis);
        tbis.dwMask = TBIF_SIZE;

        GetClientRect(window->hwnd, &rect);
        tbis.cx = (unsigned short)(rect.right - rect.left);

        SendMessage(window->toolbar.toolbar, TB_SETBUTTONINFO,
            (WPARAM)tbs.idCommand, (LPARAM)&tbis);

        /* Get button position */
        SendMessage(window->toolbar.toolbar, TB_GETITEMRECT,
            (WPARAM)tbs.idCommand, (LPARAM)&rect);

        /* Create a slider */
        trackbar = (CvTrackbar*)cvAlloc( sizeof(CvTrackbar) + len + 1 );
        trackbar->signature = CV_TRACKBAR_MAGIC_VAL;
        trackbar->notify = 0;
        trackbar->notify2 = 0;
        trackbar->parent = window;
        trackbar->pos = 0;
        trackbar->data = 0;
        trackbar->id = bcount;
        trackbar->next = window->toolbar.first;
        trackbar->name = (char*)(trackbar + 1);
        memcpy( trackbar->name, trackbar_name, len + 1 );
        window->toolbar.first = trackbar;

        sprintf(slider_name, "Trackbar%p", val);
        trackbar->hwnd = CreateWindowEx(0, TRACKBAR_CLASS, slider_name,
                            WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS |
                            TBS_FIXEDLENGTH | TBS_HORZ | TBS_BOTTOM,
                            rect.left + HG_BUDDY_WIDTH, rect.top,
                            rect.right - rect.left - HG_BUDDY_WIDTH,
                            rect.bottom - rect.top, window->toolbar.toolbar,
                            (HMENU)(size_t)bcount, hg_hinstance, 0);

        sprintf(slider_name,"Buddy%p", val);
        trackbar->buddy = CreateWindowEx(0, "STATIC", slider_name,
                            WS_CHILD | SS_RIGHT,
                            rect.left, rect.top,
                            HG_BUDDY_WIDTH, rect.bottom - rect.top,
                            window->toolbar.toolbar, 0, hg_hinstance, 0);

        icvSetWindowLongPtr( trackbar->hwnd, CV_USERDATA, trackbar );

        /* Minimize the number of rows */
        SendMessage( window->toolbar.toolbar, TB_SETROWS,
                     MAKEWPARAM(1, FALSE), (LPARAM)&rect );
    }
    else
    {
        trackbar->data = 0;
        trackbar->notify = 0;
        trackbar->notify2 = 0;
    }

    trackbar->maxval = count;

    /* Adjust slider parameters */
    SendMessage(trackbar->hwnd, TBM_SETRANGE, (WPARAM)TRUE, (LPARAM)MAKELONG(0, count));
    SendMessage(trackbar->hwnd, TBM_SETTICFREQ, (WPARAM)1, (LPARAM)0 );
    if( val )
        pos = *val;

    SendMessage(trackbar->hwnd, TBM_SETPOS, (WPARAM)TRUE, (LPARAM)pos );
    SendMessage(window->toolbar.toolbar, TB_AUTOSIZE, 0, 0);

    trackbar->pos = -1;
    icvUpdateTrackbar( trackbar, pos );
    ShowWindow( trackbar->buddy, SW_SHOW );
    ShowWindow( trackbar->hwnd, SW_SHOW );

    trackbar->notify = on_notify;
    trackbar->notify2 = on_notify2;
    trackbar->userdata = userdata;
    trackbar->data = val;

    /* Resize the window to reflect the toolbar resizing*/
    icvUpdateWindowPos(window);

    result = 1;

    __END__;

    return result;
}

CV_IMPL int
cvCreateTrackbar( const char* trackbar_name, const char* window_name,
                  int* val, int count, CvTrackbarCallback on_notify )
{
    return icvCreateTrackbar( trackbar_name, window_name, val, count,
        on_notify, 0, 0 );
}

CV_IMPL int
cvCreateTrackbar2( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback2 on_notify2,
                   void* userdata )
{
    return icvCreateTrackbar( trackbar_name, window_name, val, count,
        0, on_notify2, userdata );
}

CV_IMPL void
cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse, void* param )
{
    CV_FUNCNAME( "cvSetMouseCallback" );

    __BEGIN__;

    CvWindow* window = 0;

    if( !window_name )
        CV_ERROR( CV_StsNullPtr, "NULL window name" );

    window = icvFindWindowByName(window_name);
    if( !window )
        EXIT;

    window->on_mouse = on_mouse;
    window->on_mouse_param = param;

    __END__;
}


CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
    int pos = -1;

    CV_FUNCNAME( "cvGetTrackbarPos" );

    __BEGIN__;

    CvWindow* window;
    CvTrackbar* trackbar = 0;

    if( trackbar_name == 0 || window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        trackbar = icvFindTrackbarByName( window, trackbar_name );

    if( trackbar )
        pos = trackbar->pos;

    __END__;

    return pos;
}


CV_IMPL void cvSetTrackbarPos( const char* trackbar_name, const char* window_name, int pos )
{
    CV_FUNCNAME( "cvSetTrackbarPos" );

    __BEGIN__;

    CvWindow* window;
    CvTrackbar* trackbar = 0;

    if( trackbar_name == 0 || window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        trackbar = icvFindTrackbarByName( window, trackbar_name );

    if( trackbar )
    {
        if( pos < 0 )
            pos = 0;

        if( pos > trackbar->maxval )
            pos = trackbar->maxval;

        SendMessage( trackbar->hwnd, TBM_SETPOS, (WPARAM)TRUE, (LPARAM)pos );
        icvUpdateTrackbar( trackbar, pos );
    }

    __END__;
}


CV_IMPL void* cvGetWindowHandle( const char* window_name )
{
    void* hwnd = 0;

    CV_FUNCNAME( "cvGetWindowHandle" );

    __BEGIN__;

    CvWindow* window;

    if( window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        hwnd = (void*)window->hwnd;

    __END__;

    return hwnd;
}


CV_IMPL const char* cvGetWindowName( void* window_handle )
{
    const char* window_name = "";

    CV_FUNCNAME( "cvGetWindowName" );

    __BEGIN__;

    CvWindow* window;

    if( window_handle == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    window = icvWindowByHWND( (HWND)window_handle );
    if( window )
        window_name = window->name;

    __END__;

    return window_name;
}


CV_IMPL void
cvSetPreprocessFuncWin32_(const void* callback)
{
    hg_on_preprocess = (CvWin32WindowCallback)callback;
}

CV_IMPL void
cvSetPostprocessFuncWin32_(const void* callback)
{
    hg_on_postprocess = (CvWin32WindowCallback)callback;
}

#endif //WIN32
