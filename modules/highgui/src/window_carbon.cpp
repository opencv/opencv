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

#include <Carbon/Carbon.h>
#include <Quicktime/Quicktime.h>//YV

#include <unistd.h>
#include <cstdio>
#include <cmath>

//#define MS_TO_TICKS(a) a*3/50

#define LABELWIDTH 64
#define INTERWIDGETSPACE 16
#define WIDGETHEIGHT 32
#define NO_KEY -1

struct CvWindow;

typedef struct CvTrackbar
{
    int signature;

    ControlRef trackbar;
    ControlRef label;

    char* name;
    CvTrackbar* next;
    CvWindow* parent;
    int* data;
    int pos;
    int maxval;
    int labelSize;//Yannick Verdie
    CvTrackbarCallback notify;
    CvTrackbarCallback2 notify2;
    void* userdata;
}
CvTrackbar;


typedef struct CvWindow
{
    int signature;

    char* name;
    CvWindow* prev;
    CvWindow* next;

    WindowRef window;
    WindowRef oldwindow;//YV
    CGImageRef imageRef;
    int imageWidth;//FD
    int imageHeight;//FD

    CvMat* image;
    CvMat* dst_image;
    int converted;
    int last_key;
    int flags;
    int status;//YV
    Ptr restoreState;//YV

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    struct {
        int pos;
        int rows;
        CvTrackbar* first;
    }
    toolbar;
    int trackbarheight;
}
CvWindow;

static CvWindow* hg_windows = 0;

#define Assert(exp)                                             \
if( !(exp) )                                                    \
{                                                               \
    printf("Assertion: %s  %s: %d\n", #exp, __FILE__, __LINE__);\
    assert(exp);                                                \
}

static int wasInitialized = 0;
static char lastKey = NO_KEY;
OSStatus keyHandler(EventHandlerCallRef hcr, EventRef theEvent, void* inUserData);
static pascal OSStatus windowEventHandler(EventHandlerCallRef nextHandler, EventRef theEvent, void *inUserData);

static const EventTypeSpec applicationKeyboardEvents[] =
{
    { kEventClassKeyboard, kEventRawKeyDown },
};

CV_IMPL int cvInitSystem( int argc, char** argv )
{
    OSErr err = noErr;
    if( !wasInitialized )
    {

        hg_windows = 0;
        err = InstallApplicationEventHandler(NewEventHandlerUPP( keyHandler),GetEventTypeCount(applicationKeyboardEvents),applicationKeyboardEvents,NULL,NULL);
        if (err != noErr)
        {
             fprintf(stderr,"InstallApplicationEventHandler was not ok\n");
        }
        wasInitialized = 1;
    }
    setlocale(LC_NUMERIC,"C");

    return 0;
}

// TODO: implement missing functionality
CV_IMPL int cvStartWindowThread()
{
    return 0;
}

static int icvCountTrackbarInWindow( const CvWindow* window)
{
    CvTrackbar* trackbar = window->toolbar.first;
    int count = 0;
    while (trackbar != 0) {
        count++;
        trackbar = trackbar->next;
    }
    return count;
}

static CvTrackbar* icvTrackbarByHandle( void * handle )
{
    CvWindow* window = hg_windows;
    CvTrackbar* trackbar = NULL;
    while( window != 0 && window->window != handle) {
        trackbar = window->toolbar.first;
        while (trackbar != 0 && trackbar->trackbar != handle)
            trackbar = trackbar->next;
        if (trackbar != 0 && trackbar->trackbar == handle)
            break;
        window = window->next;
    }
    return trackbar;
}

static CvWindow* icvWindowByHandle( void * handle )
{
    CvWindow* window = hg_windows;

    while( window != 0 && window->window != handle)
        window = window->next;

    return window;
}

CV_IMPL CvWindow * icvFindWindowByName( const char* name)
{
    CvWindow* window = hg_windows;
    while( window != 0 && strcmp(name, window->name) != 0 )
        window = window->next;

    return window;
}

static CvTrackbar* icvFindTrackbarByName( const CvWindow* window, const char* name )
{
    CvTrackbar* trackbar = window->toolbar.first;

    while (trackbar != 0 && strcmp( trackbar->name, name ) != 0)
        trackbar = trackbar->next;

    return trackbar;
}

//FD
/* draw image to frame */
static void icvDrawImage( CvWindow* window )
{
    Assert( window != 0 );
    if( window->imageRef == 0 ) return;

    CGContextRef myContext;
    CGRect rect;
    Rect portrect;
    int width = window->imageWidth;
    int height = window->imageHeight;

        GetWindowPortBounds(window->window, &portrect);

    if(!( window->flags & CV_WINDOW_AUTOSIZE) ) //YV
    {
        CGPoint origin = {0,0};
        CGSize size = {portrect.right-portrect.left, portrect.bottom-portrect.top-window->trackbarheight};
        rect.origin = origin; rect.size = size;
    }
    else
    {
        CGPoint origin = {0, portrect.bottom - height - window->trackbarheight};
        CGSize size = {width, height};
        rect.origin = origin; rect.size = size;
    }

    /* To be sybnchronous we are using this, better would be to susbcribe to the draw event and process whenever requested, we might save SOME CPU cycles*/
    SetPortWindowPort (window->window);
    QDBeginCGContext (GetWindowPort (window->window), &myContext);
    CGContextSetInterpolationQuality (myContext, kCGInterpolationLow);
    CGContextDrawImage(myContext,rect,window->imageRef);
    CGContextFlush(myContext);// 4
    QDEndCGContext (GetWindowPort(window->window), &myContext);// 5
}

//FD
/* update imageRef */
static void icvPutImage( CvWindow* window )
{
    Assert( window != 0 );
    if( window->image == 0 ) return;

    CGColorSpaceRef colorspace = NULL;
    CGDataProviderRef provider = NULL;
    int width = window->imageWidth = window->image->cols;
    int height = window->imageHeight = window->image->rows;

    colorspace = CGColorSpaceCreateDeviceRGB();

    int size = 8;
    int nbChannels = 3;

    provider = CGDataProviderCreateWithData(NULL, window->image->data.ptr, width * height , NULL );

    if (window->imageRef != NULL){
        CGImageRelease(window->imageRef);
        window->imageRef = NULL;
    }

    window->imageRef = CGImageCreate( width, height, size , size*nbChannels , window->image->step, colorspace,  kCGImageAlphaNone , provider, NULL, true, kCGRenderingIntentDefault );
    icvDrawImage( window );

    /* release the provider's memory */
    CGDataProviderRelease( provider );
}

static void icvUpdateWindowSize( const CvWindow* window )
{
    int width = 0, height = 240;
    Rect globalBounds;

    GetWindowBounds(window->window, kWindowContentRgn, &globalBounds);

    int minWidth = 320;

    if( window->image ) {
        width = MAX(MAX(window->image->width, width), minWidth);
        height = window->image->height;
    } else
        width = minWidth;

    height += window->trackbarheight;

    //height +=WIDGETHEIGHT; /* 32 pixels are spearating tracbars from the video display */

    globalBounds.right = globalBounds.left + width;
    globalBounds.bottom = globalBounds.top + height;
    SetWindowBounds(window->window, kWindowContentRgn, &globalBounds);
}

static void icvDeleteWindow( CvWindow* window )
{
    CvTrackbar* trackbar;

    if( window->prev )
        window->prev->next = window->next;
    else
        hg_windows = window->next;

    if( window->next )
        window->next->prev = window->prev;

    window->prev = window->next = 0;

    cvReleaseMat( &window->image );
    cvReleaseMat( &window->dst_image );

    for( trackbar = window->toolbar.first; trackbar != 0; )
    {
        CvTrackbar* next = trackbar->next;
        cvFree( (void**)&trackbar );
        trackbar = next;
    }

    if (window->imageRef != NULL)
        CGImageRelease(window->imageRef);

    DisposeWindow (window->window);//YV

    cvFree( (void**)&window );
}


CV_IMPL void cvDestroyWindow( const char* name)
{
    CV_FUNCNAME( "cvDestroyWindow" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        EXIT;

    icvDeleteWindow( window );

    __END__;
}


CV_IMPL void cvDestroyAllWindows( void )
{
    while( hg_windows )
    {
        CvWindow* window = hg_windows;
        icvDeleteWindow( window );
    }
}


CV_IMPL void cvShowImage( const char* name, const CvArr* arr)
{
    CV_FUNCNAME( "cvShowImage" );

    __BEGIN__;

    CvWindow* window;
    int origin = 0;
    int resize = 0;
    CvMat stub, *image;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
    {
        cvNamedWindow(name, 1);
        window = icvFindWindowByName(name);
    }

    if( !window || !arr )
        EXIT; // keep silence here.

    if( CV_IS_IMAGE_HDR( arr ))
        origin = ((IplImage*)arr)->origin;

    CV_CALL( image = cvGetMat( arr, &stub ));

    /*
     if( !window->image )
     cvResizeWindow( name, image->cols, image->rows );
     */

    if( window->image &&
        !CV_ARE_SIZES_EQ(window->image, image) ) {
        if ( ! (window->flags & CV_WINDOW_AUTOSIZE) )//FD
            resize = 1;
        cvReleaseMat( &window->image );
    }

    if( !window->image ) {
        resize = 1;//FD
        window->image = cvCreateMat( image->rows, image->cols, CV_8UC3 );
    }

    cvConvertImage( image, window->image, (origin != 0 ? CV_CVTIMG_FLIP : 0) + CV_CVTIMG_SWAP_RB );
    icvPutImage( window );
    if ( resize )//FD
        icvUpdateWindowSize( window );

    __END__;
}

CV_IMPL void cvResizeWindow( const char* name, int width, int height)
{
    CV_FUNCNAME( "cvResizeWindow" );

    __BEGIN__;

    CvWindow* window;
    //CvTrackbar* trackbar;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    SizeWindow(window->window, width, height, true);

    __END__;
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y)
{
    CV_FUNCNAME( "cvMoveWindow" );

    __BEGIN__;

    CvWindow* window;
    //CvTrackbar* trackbar;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    MoveWindow(window->window, x, y, true);

    __END__;
}

void TrackbarActionProcPtr (ControlRef theControl, ControlPartCode partCode)
{
    CvTrackbar * trackbar = icvTrackbarByHandle (theControl);

    if (trackbar == NULL)
    {
        fprintf(stderr,"Error getting trackbar\n");
        return;
    }
    else
    {
        int pos = GetControl32BitValue (theControl);
        if ( trackbar->data )
            *trackbar->data = pos;
        if ( trackbar->notify )
            trackbar->notify(pos);
        else if ( trackbar->notify2 )
            trackbar->notify2(pos, trackbar->userdata);

        //--------YV---------------------------
        CFStringEncoding encoding = kCFStringEncodingASCII;
        CFAllocatorRef alloc_default = kCFAllocatorDefault;  // = NULL;

        char valueinchar[20];
        sprintf(valueinchar, " (%d)",  *trackbar->data);

        // create an empty CFMutableString
        CFIndex maxLength = 256;
        CFMutableStringRef cfstring = CFStringCreateMutable(alloc_default,maxLength);

        // append some c strings into it.
        CFStringAppendCString(cfstring,trackbar->name,encoding);
        CFStringAppendCString(cfstring,valueinchar,encoding);

        SetControlData(trackbar->label, kControlEntireControl,kControlStaticTextCFStringTag, sizeof(cfstring), &cfstring);
        DrawControls(trackbar->parent->window);
        //-----------------------------------------
    }
}


static int icvCreateTrackbar (const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback on_notify,
                              CvTrackbarCallback2 on_notify2,
                              void* userdata)
{
    int result = 0;

    CV_FUNCNAME( "icvCreateTrackbar" );
    __BEGIN__;

    /*char slider_name[32];*/
    CvWindow* window = 0;
    CvTrackbar* trackbar = 0;
    Rect  stboundsRect;
    ControlRef outControl;
    ControlRef stoutControl;
    Rect bounds;

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
        int len = strlen(trackbar_name);
        trackbar = (CvTrackbar*)cvAlloc(sizeof(CvTrackbar) + len + 1);
        memset( trackbar, 0, sizeof(*trackbar));
        trackbar->signature = CV_TRACKBAR_MAGIC_VAL;
        trackbar->name = (char*)(trackbar+1);
        memcpy( trackbar->name, trackbar_name, len + 1 );
        trackbar->parent = window;
        trackbar->next = window->toolbar.first;
        window->toolbar.first = trackbar;

        if( val )
        {
            int value = *val;
            if( value < 0 )
                value = 0;
            if( value > count )
                value = count;
            trackbar->pos = value;
            trackbar->data = val;
        }

        trackbar->maxval = count;

        //----------- YV ----------------------
        //get nb of digits
        int nbDigit = 0;
        while((count/=10)>10){
            nbDigit++;
        }

        //pad size maxvalue in pixel
        Point	qdSize;
        char valueinchar[strlen(trackbar_name)+1 +1 +1+nbDigit+1];//length+\n +space +(+nbDigit+)
        sprintf(valueinchar, "%s (%d)",trackbar_name, trackbar->maxval);
        SInt16	baseline;
        CFStringRef text = CFStringCreateWithCString(NULL,valueinchar,kCFStringEncodingASCII);
        GetThemeTextDimensions( text, kThemeCurrentPortFont, kThemeStateActive, false, &qdSize, &baseline );
        trackbar->labelSize = qdSize.h;
        //--------------------------------------

        int c = icvCountTrackbarInWindow(window);

        GetWindowBounds(window->window,kWindowContentRgn,&bounds);

        stboundsRect.top = (INTERWIDGETSPACE +WIDGETHEIGHT)* (c-1)+INTERWIDGETSPACE;
        stboundsRect.left = INTERWIDGETSPACE;
        stboundsRect.bottom = stboundsRect.top + WIDGETHEIGHT;
        stboundsRect.right = stboundsRect.left+LABELWIDTH;

        //fprintf(stdout,"create trackabar bounds (%d %d %d %d)\n",stboundsRect.top,stboundsRect.left,stboundsRect.bottom,stboundsRect.right);
     //----------- YV ----------------------
     sprintf(valueinchar, "%s (%d)",trackbar_name, trackbar->pos);
        CreateStaticTextControl (window->window,&stboundsRect,CFStringCreateWithCString(NULL,valueinchar,kCFStringEncodingASCII),NULL,&stoutControl);
        //--------------------------------------

        stboundsRect.top = (INTERWIDGETSPACE +WIDGETHEIGHT)* (c-1)+INTERWIDGETSPACE;
        stboundsRect.left = INTERWIDGETSPACE*2+LABELWIDTH;
        stboundsRect.bottom = stboundsRect.top + WIDGETHEIGHT;
        stboundsRect.right =  bounds.right-INTERWIDGETSPACE;

        CreateSliderControl (window->window,&stboundsRect, trackbar->pos,0,trackbar->maxval,kControlSliderLiveFeedback,0,true,NewControlActionUPP(TrackbarActionProcPtr),&outControl);

        bounds.bottom += INTERWIDGETSPACE + WIDGETHEIGHT;
        SetControlVisibility (outControl,true,true);
        SetControlVisibility (stoutControl,true,true);

        trackbar->trackbar = outControl;
        trackbar->label = stoutControl;
        if (c == 1)
            window->trackbarheight = INTERWIDGETSPACE*2 + WIDGETHEIGHT;
        else
            window->trackbarheight += INTERWIDGETSPACE + WIDGETHEIGHT;
        icvUpdateWindowSize( window );
    }

    trackbar->notify = on_notify;
    trackbar->notify2 = on_notify2;
    trackbar->userdata = userdata;

    result = 1;

    __END__;
    return result;
}


CV_IMPL int cvCreateTrackbar (const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback on_notify)
{
    return icvCreateTrackbar(trackbar_name, window_name, val, count, on_notify, 0, 0);
}


CV_IMPL int cvCreateTrackbar2(const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback2 on_notify2,
                              void* userdata)
{
    return icvCreateTrackbar(trackbar_name, window_name, val,
                             count, 0, on_notify2, userdata);
}


CV_IMPL void
cvSetMouseCallback( const char* name, CvMouseCallback function, void* info)
{
    CvWindow* window = icvFindWindowByName( name );
    if (window != NULL)
    {
        window->on_mouse = function;
        window->on_mouse_param = info;
    }
    else
    {
        fprintf(stdout,"Error with cvSetMouseCallback. Window not found : %s\n",name);
    }
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

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
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

    // Set new value and redraw the trackbar
    SetControlValue( trackbar->trackbar, pos );
    Draw1Control( trackbar->trackbar );
    }

    __END__;
    return ;
}

CV_IMPL void* cvGetWindowHandle( const char* name )
{
    WindowRef result = 0;

    __BEGIN__;

    CvWindow* window;
    window = icvFindWindowByName( name );
    if (window != NULL)
        result = window->window;
    else
        result = NULL;

    __END__;

    return result;
}


CV_IMPL const char* cvGetWindowName( void* window_handle )
{
    const char* window_name = "";

    CV_FUNCNAME( "cvGetWindowName" );

    __BEGIN__;

    CvWindow* window;

    if( window_handle == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window" );
    window = icvWindowByHandle(window_handle );
    if( window )
        window_name = window->name;

    __END__;

    return window_name;
}

double cvGetModeWindow_CARBON(const char* name)//YV
{
    double result = -1;

    CV_FUNCNAME( "cvGetModeWindow_QT" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    result = window->status;

    __END__;
    return result;
}

void cvSetModeWindow_CARBON( const char* name, double prop_value)//Yannick Verdie
{
    OSStatus err = noErr;


    CV_FUNCNAME( "cvSetModeWindow_QT" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    if(window->flags & CV_WINDOW_AUTOSIZE)//if the flag CV_WINDOW_AUTOSIZE is set
        EXIT;

    if (window->status==CV_WINDOW_FULLSCREEN && prop_value==CV_WINDOW_NORMAL)
    {
        err = EndFullScreen(window->restoreState,0);
        if (err != noErr)
            fprintf(stdout,"Error EndFullScreen\n");
        window->window = window->oldwindow;
        ShowWindow( window->window );

        window->status=CV_WINDOW_NORMAL;
        EXIT;
    }

    if (window->status==CV_WINDOW_NORMAL && prop_value==CV_WINDOW_FULLSCREEN)
    {
        GDHandle device;
        err = GetWindowGreatestAreaDevice(window->window, kWindowTitleBarRgn, &device, NULL);
        if (err != noErr)
            fprintf(stdout,"Error GetWindowGreatestAreaDevice\n");

        HideWindow(window->window);
        window->oldwindow = window->window;
        err = BeginFullScreen(&(window->restoreState), device, 0, 0, &window->window, 0, fullScreenAllowEvents | fullScreenDontSwitchMonitorResolution);
        if (err != noErr)
            fprintf(stdout,"Error BeginFullScreen\n");

        window->status=CV_WINDOW_FULLSCREEN;
        EXIT;
    }

    __END__;
}

void cv::setWindowTitle(const String& winname, const String& title)
{
    CvWindow* window = icvFindWindowByName(winname.c_str());

    if (!window)
    {
        namedWindow(winname);
        window = icvFindWindowByName(winname.c_str());
    }

    if (!window)
        CV_Error(Error::StsNullPtr, "NULL window");

    if (noErr != SetWindowTitleWithCFString(window->window, CFStringCreateWithCString(NULL, title.c_str(), kCFStringEncodingASCII)))
        CV_Error_(Error::StsError, ("Failed to set \"%s\" window title to \"%s\"", winname.c_str(), title.c_str()));
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    int result = 0;
    CV_FUNCNAME( "cvNamedWindow" );
    if (!wasInitialized)
        cvInitSystem(0, NULL);

    // to be able to display a window, we need to be a 'faceful' application
    // http://lists.apple.com/archives/carbon-dev/2005/Jun/msg01414.html
    static bool switched_to_faceful = false;
    if (! switched_to_faceful)
    {
        ProcessSerialNumber psn = { 0, kCurrentProcess };
        OSStatus ret = TransformProcessType (&psn, kProcessTransformToForegroundApplication );

        if (ret == noErr)
        {
            SetFrontProcess( &psn );
            switched_to_faceful = true;
        }
        else
        {
            fprintf(stderr, "Failed to tranform process type: %d\n", (int) ret);
            fflush (stderr);
        }
    }

    __BEGIN__;

    WindowRef       outWindow = NULL;
    OSStatus              err = noErr;
    Rect        contentBounds = {100,100,320,440};

    CvWindow* window;
    UInt wAttributes = 0;

    int len;

    const EventTypeSpec genericWindowEventHandler[] = {
        { kEventClassMouse, kEventMouseMoved},
        { kEventClassMouse, kEventMouseDragged},
        { kEventClassMouse, kEventMouseUp},
        { kEventClassMouse, kEventMouseDown},
        { kEventClassWindow, kEventWindowClose },
        { kEventClassWindow, kEventWindowBoundsChanged }//FD
    };

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    if( icvFindWindowByName( name ) != 0 ){
        result = 1;
        EXIT;
    }
    len = strlen(name);
    CV_CALL( window = (CvWindow*)cvAlloc(sizeof(CvWindow) + len + 1));
    memset( window, 0, sizeof(*window));
    window->name = (char*)(window + 1);
    memcpy( window->name, name, len + 1 );
    window->flags = flags;
    window->status = CV_WINDOW_NORMAL;//YV
    window->signature = CV_WINDOW_MAGIC_VAL;
    window->image = 0;
    window->last_key = 0;
    window->on_mouse = 0;
    window->on_mouse_param = 0;

    window->next = hg_windows;
    window->prev = 0;
    if( hg_windows )
        hg_windows->prev = window;
    hg_windows = window;
    wAttributes =  kWindowStandardDocumentAttributes | kWindowStandardHandlerAttribute | kWindowLiveResizeAttribute;


    if (window->flags & CV_WINDOW_AUTOSIZE)//Yannick verdie, remove the handler at the bottom-right position of the window in AUTORESIZE mode
    {
    wAttributes = 0;
    wAttributes = kWindowCloseBoxAttribute | kWindowFullZoomAttribute | kWindowCollapseBoxAttribute | kWindowStandardHandlerAttribute  |  kWindowLiveResizeAttribute;
    }

    err = CreateNewWindow ( kDocumentWindowClass,wAttributes,&contentBounds,&outWindow);
    if (err != noErr)
        fprintf(stderr,"Error while creating the window\n");

    SetWindowTitleWithCFString(outWindow,CFStringCreateWithCString(NULL,name,kCFStringEncodingASCII));
    if (err != noErr)
        fprintf(stdout,"Error SetWindowTitleWithCFString\n");

    window->window = outWindow;
    window->oldwindow = 0;//YV

    err = InstallWindowEventHandler(outWindow, NewEventHandlerUPP(windowEventHandler), GetEventTypeCount(genericWindowEventHandler), genericWindowEventHandler, outWindow, NULL);

    ShowWindow( outWindow );
    result = 1;

    __END__;
    return result;
}

static pascal OSStatus windowEventHandler(EventHandlerCallRef nextHandler, EventRef theEvent, void *inUserData)
{
    CvWindow* window = NULL;
    UInt32 eventKind, eventClass;
    OSErr err = noErr;
    int event = 0;
    UInt32 count = 0;
    HIPoint point = {0,0};
    EventMouseButton eventMouseButton = 0;//FD
    UInt32 modifiers;//FD

    WindowRef theWindow = (WindowRef)inUserData;
    if (theWindow == NULL)
        return eventNotHandledErr;
    window = icvWindowByHandle(theWindow);
    if ( window == NULL)
        return eventNotHandledErr;

    eventKind = GetEventKind(theEvent);
    eventClass = GetEventClass(theEvent);

    switch (eventClass) {
    case kEventClassMouse : {
        switch (eventKind){
        case kEventMouseUp :
        case kEventMouseDown :
        case kEventMouseMoved :
        case kEventMouseDragged : {
            err = CallNextEventHandler(nextHandler, theEvent);
            if (err != eventNotHandledErr)
                return err;
            err = GetEventParameter(theEvent, kEventParamMouseButton, typeMouseButton, NULL, sizeof(eventMouseButton), NULL, &eventMouseButton);
            err = GetEventParameter(theEvent, kEventParamKeyModifiers, typeUInt32, NULL, sizeof(modifiers), NULL, &modifiers);
            err = GetEventParameter(theEvent,kEventParamClickCount,typeUInt32,NULL,sizeof(UInt32),NULL,&count);
            if (err == noErr){
                if (count >1) event += 6;
            } else {
                event = CV_EVENT_MOUSEMOVE;
            }

            if (eventKind == kEventMouseUp)
                event +=4;
            if (eventKind == kEventMouseDown)
                event +=1;

            unsigned int flags = 0;

            err = GetEventParameter(theEvent, kEventParamWindowMouseLocation, typeHIPoint, NULL, sizeof(point), NULL, &point);
            if (eventKind != kEventMouseMoved){
                switch(eventMouseButton){
                    case kEventMouseButtonPrimary:
                        if (modifiers & controlKey){
                            flags += CV_EVENT_FLAG_RBUTTON;
                            event += 1;
                        } else {
                            flags += CV_EVENT_FLAG_LBUTTON;
                        }
                        break;
                    case kEventMouseButtonSecondary:
                        flags += CV_EVENT_FLAG_RBUTTON;
                        event += 1;
                        break;
                    case kEventMouseButtonTertiary:
                        flags += CV_EVENT_FLAG_MBUTTON;
                        event += 2;
                        break;
                }
            }

            if (modifiers&controlKey) flags += CV_EVENT_FLAG_CTRLKEY;
            if (modifiers&shiftKey)   flags += CV_EVENT_FLAG_SHIFTKEY;
            if (modifiers& cmdKey )   flags += CV_EVENT_FLAG_ALTKEY;

            if (window->on_mouse != NULL){
                int lx,ly;
                Rect structure, content;
                GetWindowBounds(theWindow, kWindowStructureRgn, &structure);
                GetWindowBounds(theWindow, kWindowContentRgn, &content);
                lx = (int)point.x - content.left + structure.left;
                ly = (int)point.y - window->trackbarheight  - content.top + structure.top;
                if (window->flags & CV_WINDOW_AUTOSIZE) {//FD
                                                         //printf("was %d,%d\n", lx, ly);
                    /* scale the mouse coordinates */
                    lx = lx * window->imageWidth / (content.right - content.left);
                    ly = ly * window->imageHeight / (content.bottom - content.top - window->trackbarheight);
                }

                if (lx>0 && ly >0){
                    window->on_mouse (event, lx, ly, flags, window->on_mouse_param);
                    return noErr;
                }
            }
        }
        default : return eventNotHandledErr;
        }
    }
    case kEventClassWindow : {//FD
        switch (eventKind){
        case kEventWindowBoundsChanged :
        {
            /* resize the trackbars */
            CvTrackbar *t;
            Rect bounds;
            GetWindowBounds(window->window,kWindowContentRgn,&bounds);
            for ( t = window->toolbar.first; t != 0; t = t->next )
                SizeControl(t->trackbar,bounds.right - bounds.left - INTERWIDGETSPACE*3 - LABELWIDTH , WIDGETHEIGHT);
        }
            /* redraw the image */
            icvDrawImage(window);
            break;
        default :
            return eventNotHandledErr;
        }
    }
    default:
        return eventNotHandledErr;
    }

    return eventNotHandledErr;
}

OSStatus keyHandler(EventHandlerCallRef hcr, EventRef theEvent, void* inUserData)
{
    UInt32 eventKind;
    UInt32 eventClass;
    OSErr  err        = noErr;

    eventKind  = GetEventKind     (theEvent);
    eventClass = GetEventClass    (theEvent);
    err        = GetEventParameter(theEvent, kEventParamKeyMacCharCodes, typeChar, NULL, sizeof(lastKey), NULL, &lastKey);
    if (err != noErr)
        lastKey = NO_KEY;

    return noErr;
}

CV_IMPL int cvWaitKey (int maxWait)
{
    EventRecord theEvent;

    // wait at least for one event (to allow mouse, etc. processing), exit if maxWait milliseconds passed (nullEvent)
    UInt32 start = TickCount();
    int iters=0;
    do
    {
        // remaining time until maxWait is over
        UInt32 wait = EventTimeToTicks (maxWait / 1000.0) - (TickCount() - start);
        if ((int)wait <= 0)
        {
            if( maxWait > 0 && iters > 0 )
                break;
            wait = 1;
        }
        iters++;
        WaitNextEvent (everyEvent, &theEvent, maxWait > 0 ? wait : kDurationForever, NULL);
    }
    while (lastKey == NO_KEY  &&  theEvent.what != nullEvent);

    int key = lastKey;
    lastKey = NO_KEY;
    return key;
}

/* End of file. */
