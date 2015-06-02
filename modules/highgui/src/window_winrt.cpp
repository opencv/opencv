#include "precomp.hpp"

#if defined WINRT && !defined WINRT_8_0

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

struct CvWindow;

typedef struct CvTrackbar
{
    int signature;
    void* hwnd; // TODO: use proper handle type
    char* name;
    CvTrackbar* next;
    CvWindow* parent;
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
    void* hwnd; // TODO: use proper handle type
    char* name;
    CvWindow* prev;
    CvWindow* next;

    HGDIOBJ image;
    int flags;

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    struct
    {
        void* toolbar; // TODO: use proper handle type
        int pos;
        int rows;
        CvTrackbar* first;
    }
    toolbar;

    int width;
    int height;
}
CvWindow;

static CvWindow* hg_windows = 0;

// typedef int (CV_CDECL * CvWin32WindowCallback)(HWND, UINT, WPARAM, LPARAM, int*);

static CvWindow* icvFindWindowByName(const char* name) {
    CvWindow* window = hg_windows;

    for (; window != 0 && strcmp(name, window->name) != 0; window = window->next)
        ;

    return window;
}

static CvTrackbar*
icvFindTrackbarByName(const CvWindow* window, const char* name) {
    CvTrackbar* trackbar = window->toolbar.first;

    for (; trackbar != 0 && strcmp(trackbar->name, name) != 0; trackbar = trackbar->next)
        ;

    return trackbar;
}

CV_IMPL int cvInitSystem( int, char** )
{
    static int wasInitialized = 0;

    if (!wasInitialized)
    {
        hg_windows = 0;
    }

    return 0;
}

CV_IMPL int cvStartWindowThread(){
    return 0;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    int result = 0;
    CV_FUNCNAME( "cvNamedWindow" );

    __BEGIN__;
    __END__;

    return result;
}

CV_IMPL void cvDestroyWindow( const char* name )
{
    CV_FUNCNAME( "cvDestroyWindow" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName(name);
    if( !window )
        EXIT;

    __END__;
}

CV_IMPL void cvShowImage( const char* name, const CvArr* arr )
{
    CV_FUNCNAME( "cvShowImage" );

    __BEGIN__;

    CvWindow* window;
    SIZE size = { 0, 0 };
    int channels = 0;
    void* dst_ptr = 0;
    const int channels_def = 3;
    int origin = 0;
    CvMat stub, dst, *image;
    bool changed_size = false;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
    {
        cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
        window = icvFindWindowByName(name);
    }

    if( !window || !arr )
        EXIT;

    if( CV_IS_IMAGE_HDR( arr ))
        origin = ((IplImage*)arr)->origin;

    CV_CALL( image = cvGetMat( arr, &stub ));

#ifdef HAVE_OPENGL
    if (window->useGl)
    {
        cv::imshow(name, cv::cvarrToMat(image));
        return;
    }
#endif

    if (window->image)
    {
        //TODO: validate image
    }

    if (size.cx != image->width || size.cy != image->height || channels != channels_def)
    {
        changed_size = true;

        //TODO: handle image resize
    }

    cvInitMatHeader( &dst, size.cy, size.cx, CV_8UC3,
                     dst_ptr, (size.cx * channels + 3) & -4 );
    cvConvertImage( image, &dst, origin == 0 ? CV_CVTIMG_FLIP : 0 );

    if (changed_size)
        //TODO: handle consequent image resize

    __END__;
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height )
{
    CV_FUNCNAME( "cvResizeWindow" );

    __BEGIN__;

    CvWindow* window;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    // TODO: implement appropriate logic here

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

    // TODO: implement appropriate logic here

    __END__;
}



CV_IMPL void cvDestroyAllWindows(void)
{
    // TODO: implement appropriate logic here
}

CV_IMPL int cvWaitKey( int delay )
{
    // see https://msdn.microsoft.com/en-us/library/windows/desktop/ms724411(v=vs.85).aspx
    int time0 = GetTickCount64();

    for(;;)
    {
        CvWindow* window;

        if ((delay > 0 && abs((int)(GetTickCount64() - time0)) >= delay) || hg_windows == 0)
            return -1;

        if (delay <= 0)
        {
            // TODO: implement appropriate logic here
        }

        for( window = hg_windows; window != 0; window = window->next )
        {
        }
    }
}



CV_IMPL int
cvCreateTrackbar( const char* trackbar_name, const char* window_name,
                  int* val, int count, CvTrackbarCallback on_notify )
{
    // TODO: implement appropriate logic here
    return 0;
}

CV_IMPL int
cvCreateTrackbar2( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback2 on_notify2,
                   void* userdata )
{
    // TODO: implement appropriate logic here
    return 0;
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

    // TODO: implement appropriate logic here

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

        //TODO: update trackbar
    }

    __END__;
}


CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    CV_FUNCNAME( "cvSetTrackbarMax" );

    __BEGIN__;

    if (maxval >= 0)
    {
        CvWindow* window = 0;
        CvTrackbar* trackbar = 0;
        if (trackbar_name == 0 || window_name == 0)
        {
            CV_ERROR(CV_StsNullPtr, "NULL trackbar or window name");
        }

        window = icvFindWindowByName(window_name);
        if (window)
        {
            trackbar = icvFindTrackbarByName(window, trackbar_name);
            if (trackbar)
            {
                // The position will be min(pos, maxval).
                trackbar->maxval = maxval;

                //TODO: update trackbar
            }
        }
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

    CvWindow* window = 0;

    if( window_handle == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    // window = TODO: find window by handle
    if( window )
        window_name = window->name;

    __END__;

    return 0;
}

#endif //defined WINRT && !defined WINRT_8_0