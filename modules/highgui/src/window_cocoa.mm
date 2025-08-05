/* The file is the modified version of window_cocoa.mm from opencv-cocoa project by Andre Cohen */

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                         License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010, Willow Garage Inc., all rights reserved.
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
#include "opencv2/imgproc.hpp"

#import <TargetConditionals.h>

#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
/*** begin IPhone OS Stubs ***/
// When highgui functions are referred to on iPhone OS, they will fail silently.
CV_IMPL int cvInitSystem( int argc, char** argv) { return 0;}
CV_IMPL int cvStartWindowThread(){ return 0; }
CV_IMPL void cvDestroyWindow( const char* name) {}
CV_IMPL void cvDestroyAllWindows( void ) {}
CV_IMPL void cvShowImage( const char* name, const CvArr* arr) {}
CV_IMPL void cvResizeWindow( const char* name, int width, int height) {}
CV_IMPL void cvMoveWindow( const char* name, int x, int y){}
CV_IMPL int cvCreateTrackbar (const char* trackbar_name,const char* window_name,
                              int* val, int count, CvTrackbarCallback on_notify) {return  0;}
CV_IMPL int cvCreateTrackbar2(const char* trackbar_name,const char* window_name,
                              int* val, int count, CvTrackbarCallback2 on_notify2, void* userdata) {return 0;}
CV_IMPL void cvSetMouseCallback( const char* name, CvMouseCallback function, void* info) {}
CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name ) {return 0;}
CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos) {}
CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval) {}
CV_IMPL void cvSetTrackbarMin(const char* trackbar_name, const char* window_name, int minval) {}
CV_IMPL void* cvGetWindowHandle( const char* name ) {return NULL;}
CV_IMPL const char* cvGetWindowName( void* window_handle ) {return NULL;}
CV_IMPL int cvNamedWindow( const char* name, int flags ) {return 0; }
CV_IMPL int cvWaitKey (int maxWait) {return 0;}
//*** end IphoneOS Stubs ***/
#else

#import <Cocoa/Cocoa.h>

#include <iostream>

const int MIN_SLIDER_WIDTH=200;

static NSApplication *application = nil;
static NSMutableDictionary *windows = nil;
static bool wasInitialized = false;

@interface CVView : NSView
@property(retain) NSView *imageView;
@property(retain) NSImage *image;
@property int sliderHeight;
- (void)setImageData:(CvArr *)arr;
@end

@interface CVSlider : NSView {
    NSSlider *slider;
    NSTextField *name;
    NSString *initialName;
    int *value;
    void *userData;
    CvTrackbarCallback callback;
    CvTrackbarCallback2 callback2;
}
@property(retain) NSSlider *slider;
@property(retain) NSTextField *name;
@property(retain) NSString *initialName;
@property(assign) int *value;
@property(assign) void *userData;
@property(assign) CvTrackbarCallback callback;
@property(assign) CvTrackbarCallback2 callback2;
@end

@interface CVWindow : NSWindow {
    NSMutableDictionary *sliders;
    NSMutableArray *slidersKeys;
    CvMouseCallback mouseCallback;
    void *mouseParam;
    BOOL autosize;
    BOOL firstContent;
    int status;
    int x0, y0;
}
@property(assign) CvMouseCallback mouseCallback;
@property(assign) void *mouseParam;
@property(assign) BOOL autosize;
@property(assign) BOOL firstContent;
@property(assign) int x0;
@property(assign) int y0;
@property(retain) NSMutableDictionary *sliders;
@property(retain) NSMutableArray *slidersKeys;
@property(readwrite) int status;
- (CVView *)contentView;
- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags;
- (void)cvMouseEvent:(NSEvent *)event;
- (void)createSliderWithName:(const char *)name maxValue:(int)max value:(int *)value callback:(CvTrackbarCallback)callback;
@end

CV_IMPL int cvInitSystem( int , char** )
{
    wasInitialized = true;

    application = [NSApplication sharedApplication];
    windows = [[NSMutableDictionary alloc] init];

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6

#ifndef NSAppKitVersionNumber10_5
#define NSAppKitVersionNumber10_5 949
#endif
    if( floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_5 )
        [application setActivationPolicy:NSApplicationActivationPolicyRegular];
#endif
    setlocale(LC_NUMERIC,"C");

    return 0;
}

static CVWindow *cvGetWindow(const char *name) {
    CVWindow *retval = nil;
    @autoreleasepool{
        NSString *cvname = [NSString stringWithFormat:@"%s", name];
        retval = (CVWindow*) [windows valueForKey:cvname];
        if (retval != nil) {
            [retval retain];
        }
    }
    return [retval autorelease];
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

CV_IMPL void cvDestroyWindow( const char* name)
{
    @autoreleasepool {
        CVWindow *window = cvGetWindow(name);
        if(window) {
            if ([window styleMask] & NSFullScreenWindowMask) {
                [window toggleFullScreen:nil];
            }
            [window close];
            [windows removeObjectForKey:[NSString stringWithFormat:@"%s", name]];
        }
    }
}


CV_IMPL void cvDestroyAllWindows( void )
{
    @autoreleasepool {
        NSDictionary* list = [NSDictionary dictionaryWithDictionary:windows];
        for(NSString *key in list) {
            cvDestroyWindow([key cStringUsingEncoding:NSASCIIStringEncoding]);
        }
    }
}


CV_IMPL void cvShowImage( const char* name, const CvArr* arr)
{
    @autoreleasepool{
        CVWindow *window = cvGetWindow(name);
        if(!window)
        {
            cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
            window = cvGetWindow(name);
        }

        if(window)
        {
            bool empty = [[window contentView] image] == nil;
            NSRect vrectOld = [[window contentView] frame];
            NSSize oldImageSize = [[[window contentView] image] size];
            [[window contentView] setImageData:(CvArr *)arr];
            if([window autosize] || [window firstContent] || empty)
            {
                NSSize imageSize = [[[window contentView] image] size];
                // Only adjust the image size if the new image is a different size from the previous
                if (oldImageSize.height != imageSize.height || oldImageSize.width != imageSize.width)
                {
                    //Set new view size considering sliders (reserve height and min width)
                    NSSize scaledImageSize = imageSize;
                    if ([[window contentView] respondsToSelector:@selector(convertSizeFromBacking:)])
                    {
                        // Only resize for retina displays if the image is bigger than the screen
                        NSSize screenSize = NSScreen.mainScreen.visibleFrame.size;
                        CGFloat titleBarHeight = window.frame.size.height - [window contentRectForFrameRect:window.frame].size.height;
                        screenSize.height -= titleBarHeight;
                        if (imageSize.width > screenSize.width || imageSize.height > screenSize.height)
                        {
                            CGFloat fx = screenSize.width/std::max(imageSize.width, (CGFloat)1.f);
                            CGFloat fy = screenSize.height/std::max(imageSize.height, (CGFloat)1.f);
                            CGFloat min_f = std::min(fx, fy);
                            scaledImageSize = [[window contentView] convertSizeFromBacking:imageSize];
                            scaledImageSize.width = std::min(scaledImageSize.width, min_f*imageSize.width);
                            scaledImageSize.height = std::min(scaledImageSize.height, min_f*imageSize.height);
                        }
                    }
                    NSSize contentSize = vrectOld.size;
                    contentSize.height = scaledImageSize.height + [window contentView].sliderHeight;
                    contentSize.width = std::max<int>(scaledImageSize.width, MIN_SLIDER_WIDTH);
                    [window setContentSize:contentSize]; //adjust sliders to fit new window size
                    if([window firstContent])
                    {
                        int x = [window x0];
                        int y = [window y0];
                        if(x >= 0 && y >= 0)
                        {
                            y = [[window screen] visibleFrame].size.height - y;
                            [window setFrameTopLeftPoint:NSMakePoint(x, y)];
                        }
                    }
                }
            }
            [window setFirstContent:NO];
        }
    }
}

CV_IMPL void cvResizeWindow( const char* name, int width, int height)
{
    @autoreleasepool{
        CVWindow *window = cvGetWindow(name);
        if(window && ![window autosize]) {
            height += [window contentView].sliderHeight;
            NSSize size = { (CGFloat)width, (CGFloat)height };
            [window setContentSize:size];
        }
    }
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y)
{
    CV_FUNCNAME("cvMoveWindow");

    __BEGIN__;

    CVWindow *window = nil;

    if(name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL window name" );
    @autoreleasepool{
        window = cvGetWindow(name);
        if(window) {
            if([window firstContent]) {
                [window setX0:x];
                [window setY0:y];
            }
            else {
                y = [[window screen] visibleFrame].size.height - y;
                [window setFrameTopLeftPoint:NSMakePoint(x, y)];
            }
        }
    }
    __END__;
}

CV_IMPL int cvCreateTrackbar (const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback on_notify)
{
    CV_FUNCNAME("cvCreateTrackbar");

    int result = 0;
    CVWindow *window = nil;

    __BEGIN__;
    if(window_name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL window name" );
    @autoreleasepool {
        window = cvGetWindow(window_name);
        if(window) {
            [window createSliderWithName:trackbar_name
                                maxValue:count
                                value:val
                                callback:on_notify];
            result = 1;
        }
    }
    __END__;
    return result;
}


CV_IMPL int cvCreateTrackbar2(const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback2 on_notify2,
                              void* userdata)
{
    int res = cvCreateTrackbar(trackbar_name, window_name, val, count, NULL);
    @autoreleasepool{
        if(res) {
            CVWindow *window = cvGetWindow(window_name);
            if (window && [window respondsToSelector:@selector(sliders)]) {
                CVSlider *slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
                [slider setCallback2:on_notify2];
                [slider setUserData:userdata];
            }
        }
    }
    return res;
}


CV_IMPL void
cvSetMouseCallback( const char* name, CvMouseCallback function, void* info)
{
    CV_FUNCNAME("cvSetMouseCallback");

    CVWindow *window = nil;
    __BEGIN__;
    if(name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL window name" );
    @autoreleasepool{
        window = cvGetWindow(name);
        if(window) {
            [window setMouseCallback:function];
            [window setMouseParam:info];
        }
    }
    __END__;
}

 CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
    CV_FUNCNAME("cvGetTrackbarPos");

    CVWindow *window = nil;
    int pos = -1;

    __BEGIN__;
    if(trackbar_name == NULL || window_name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    @autoreleasepool{
        window = cvGetWindow(window_name);
        if(window && [window respondsToSelector:@selector(sliders)]) {
            CVSlider *slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
            if(slider) {
                pos = [[slider slider] intValue];
            }
        }
    }
    __END__;
    return pos;
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
    CV_FUNCNAME("cvSetTrackbarPos");

    CVWindow *window = nil;
    CVSlider *slider = nil;

    __BEGIN__;
    if(trackbar_name == NULL || window_name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    if(pos < 0)
        CV_ERROR( CV_StsOutOfRange, "Bad trackbar maximal value" );

    @autoreleasepool{
        window = cvGetWindow(window_name);
        if(window && [window respondsToSelector:@selector(sliders)]) {
            slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
            if(slider) {
                [[slider slider] setIntValue:pos];
                if([slider respondsToSelector:@selector(handleSlider)]) {
                    [slider performSelector:@selector(handleSlider)];
                }
            }
        }
    }
    __END__;
}

CV_IMPL void cvSetTrackbarMax(const char* trackbar_name, const char* window_name, int maxval)
{
    CV_FUNCNAME("cvSetTrackbarMax");

    CVWindow *window = nil;
    CVSlider *slider = nil;

    __BEGIN__;
    if(trackbar_name == NULL || window_name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    @autoreleasepool{
        window = cvGetWindow(window_name);
        if(window && [window respondsToSelector:@selector(sliders)]) {
            slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
            if(slider) {
                if(maxval >= 0) {
                    int minval = [[slider slider] minValue];
                    maxval = (minval>maxval)?minval:maxval;
                    [[slider slider] setMaxValue:maxval];
                }
            }
        }
    }
    __END__;
}

CV_IMPL void cvSetTrackbarMin(const char* trackbar_name, const char* window_name, int minval)
{
    CV_FUNCNAME("cvSetTrackbarMin");

    CVWindow *window = nil;
    CVSlider *slider = nil;

    __BEGIN__;
    if(trackbar_name == NULL || window_name == NULL)
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    @autoreleasepool{
        window = cvGetWindow(window_name);
        if(window && [window respondsToSelector:@selector(sliders)]) {
            slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
            if(slider) {
                if(minval >= 0) {
                    int maxval = [[slider slider] maxValue];
                    minval = (minval<maxval)?minval:maxval;
                    [[slider slider] setMinValue:minval];
                }
            }
        }
    }
    __END__;
}

CV_IMPL void* cvGetWindowHandle( const char* name )
{
    return cvGetWindow(name);
}


CV_IMPL const char* cvGetWindowName( void* window_handle )
{
    @autoreleasepool{
        for(NSString *key in windows) {
            if([windows valueForKey:key] == window_handle) {
                return [key UTF8String];
            }
        }
    }
    return 0;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    if( !wasInitialized )
        cvInitSystem(0, 0);

    @autoreleasepool{
        CVWindow *window = cvGetWindow(name);
        if( window )
        {
            [window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];
            return 0;
        }

        NSScreen* mainDisplay = [NSScreen mainScreen];
        NSString *windowName = [NSString stringWithFormat:@"%s", name];
        NSUInteger showResize = NSResizableWindowMask;
        NSUInteger styleMask = NSTitledWindowMask|NSMiniaturizableWindowMask|showResize;
        CGFloat windowWidth = [NSWindow minFrameWidthWithTitle:windowName styleMask:styleMask];
        NSRect initContentRect = NSMakeRect(0, 0, windowWidth, 0);
        if (mainDisplay) {
            NSRect dispFrame = [mainDisplay visibleFrame];
            initContentRect.origin.y = dispFrame.size.height-20;
        }

        window = [[CVWindow alloc] initWithContentRect:initContentRect
                                            styleMask:NSTitledWindowMask|NSMiniaturizableWindowMask|showResize
                                            backing:NSBackingStoreBuffered
                                                defer:YES
                                                screen:mainDisplay];

        [window setFrameTopLeftPoint:initContentRect.origin];

        [window setFirstContent:YES];
        [window setX0:-1];
        [window setY0:-1];

        [window setContentView:[[CVView alloc] init]];

        [NSApp activateIgnoringOtherApps:YES];

        [window setHasShadow:YES];
        [window setAcceptsMouseMovedEvents:YES];
        [window useOptimizedDrawing:YES];
        [window setTitle:windowName];
        [window makeKeyAndOrderFront:nil];

        [window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];

        [windows setValue:window forKey:windowName];
        return [windows count]-1;
    }
}

CV_IMPL int cvWaitKey (int maxWait)
{
    int returnCode = -1;
    @autoreleasepool{
        double start = [[NSDate date] timeIntervalSince1970];

        while(true) {
            if(([[NSDate date] timeIntervalSince1970] - start) * 1000 >= maxWait && maxWait>0)
                break;
            NSEvent *event =
            [application
            nextEventMatchingMask:NSAnyEventMask
            untilDate://[NSDate dateWithTimeIntervalSinceNow: 1./100]
            [NSDate distantPast]
            inMode:NSDefaultRunLoopMode
            dequeue:YES];

            if([event type] == NSKeyDown && [[event characters] length]) {
                returnCode = [[event characters] characterAtIndex:0];
                break;
            }

            [application sendEvent:event];
            [application updateWindows];

            [NSThread sleepForTimeInterval:1/100.];
        }
        return returnCode;
    }
}

CvRect cvGetWindowRect_COCOA( const char* name )
{
    CvRect result = cvRect(-1, -1, -1, -1);
    CVWindow *window = nil;

    CV_FUNCNAME( "cvGetWindowRect_COCOA" );

    __BEGIN__;
    if( name == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL name string" );
    }

    window = cvGetWindow( name );
    if ( window == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL window" );
    } else {
        @autoreleasepool {
            NSRect rect = [window frame];
#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_6
            NSPoint pt = [window convertRectToScreen:rect].origin;
#else
            NSPoint pt = [window convertBaseToScreen:rect.origin];
#endif
            NSSize sz = [[[window contentView] image] size];
            result = cvRect(pt.x, pt.y, sz.width, sz.height);
        }
    }
    __END__;
    return result;
}

double cvGetModeWindow_COCOA( const char* name )
{
    double result = -1;
    CVWindow *window = nil;

    CV_FUNCNAME( "cvGetModeWindow_COCOA" );

    __BEGIN__;
    if( name == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL name string" );
    }

    window = cvGetWindow( name );
    if ( window == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL window" );
    }

    result = window.status;
    __END__;
    return result;
}

void cvSetModeWindow_COCOA( const char* name, double prop_value )
{
    CVWindow *window = nil;

#if MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
    NSDictionary *fullscreenOptions = nil;
#endif

    CV_FUNCNAME( "cvSetModeWindow_COCOA" );

    __BEGIN__;
    if( name == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL name string" );
    }

    window = cvGetWindow(name);
    if ( window == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL window" );
    }

    if ( [window autosize] )
    {
        return;
    }

    @autoreleasepool{
#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_6
        if ( ([window styleMask] & NSFullScreenWindowMask) && prop_value==CV_WINDOW_NORMAL )
        {
            [window toggleFullScreen:nil];

            window.status=CV_WINDOW_NORMAL;
        }
        else if( !([window styleMask] & NSFullScreenWindowMask) && prop_value==CV_WINDOW_FULLSCREEN )
        {
            [window setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];

            NSScreen* screen = [window screen];

            NSRect frame = [screen frame];
            [window setFrame:frame display:YES];

            [window setContentSize:frame.size];

            [window toggleFullScreen:nil];

            [window setFrameTopLeftPoint: frame.origin];

            window.status=CV_WINDOW_FULLSCREEN;
        }
#else
        fullscreenOptions = [NSDictionary dictionaryWithObject:[NSNumber numberWithBool:YES] forKey:NSFullScreenModeSetting];
        if ( [[window contentView] isInFullScreenMode] && prop_value==CV_WINDOW_NORMAL )
        {
            [[window contentView] exitFullScreenModeWithOptions:fullscreenOptions];
            window.status=CV_WINDOW_NORMAL;
        }
        else if( ![[window contentView] isInFullScreenMode] && prop_value==CV_WINDOW_FULLSCREEN )
        {
            [[window contentView] enterFullScreenMode:[NSScreen mainScreen] withOptions:fullscreenOptions];
            window.status=CV_WINDOW_FULLSCREEN;
        }
#endif
    }

    __END__;
}

double cvGetPropVisible_COCOA(const char* name)
{
    double    result = -1;
    CVWindow* window = nil;

    CV_FUNCNAME("cvGetPropVisible_COCOA");

    __BEGIN__;
    if (name == NULL)
    {
        CV_ERROR(CV_StsNullPtr, "NULL name string");
    }

    window = cvGetWindow(name);
    if (window == NULL)
    {
        CV_ERROR(CV_StsNullPtr, "NULL window");
    }

    result = window.isVisible ? 1 : 0;

    __END__;
    return result;
}

double cvGetPropTopmost_COCOA(const char* name)
{
    double    result = -1;
    CVWindow* window = nil;

    CV_FUNCNAME("cvGetPropTopmost_COCOA");

    __BEGIN__;
    if (name == NULL)
    {
        CV_ERROR(CV_StsNullPtr, "NULL name string");
    }

    window = cvGetWindow(name);
    if (window == NULL)
    {
        CV_ERROR(CV_StsNullPtr, "NULL window");
    }

    result = (window.level == NSStatusWindowLevel) ? 1 : 0;

    __END__;
    return result;
}

void cvSetPropTopmost_COCOA( const char* name, const bool topmost )
{
    CVWindow *window = nil;
    CV_FUNCNAME( "cvSetPropTopmost_COCOA" );

    __BEGIN__;
    if( name == NULL )
    {
        CV_ERROR( CV_StsNullPtr, "NULL name string" );
    }
    @autoreleasepool{
        window = cvGetWindow(name);
        if ( window == NULL )
        {
            CV_ERROR( CV_StsNullPtr, "NULL window" );
        }

        if (([window styleMask] & NSFullScreenWindowMask))
        {
            EXIT;
        }

        if (topmost)
        {
            [window makeKeyAndOrderFront:window.self];
            [window setLevel:CGWindowLevelForKey(kCGMaximumWindowLevelKey)];
        }
        else
        {
            [window makeKeyAndOrderFront:nil];
        }
    }
    __END__;
}

void setWindowTitle_COCOA(const cv::String& winname, const cv::String& title)
{

    @autoreleasepool{
        CVWindow *window = cvGetWindow(winname.c_str());

        if (window == NULL)
        {
            cv::namedWindow(winname);
            window = cvGetWindow(winname.c_str());
        }

        if (window == NULL)
            CV_Error(cv::Error::StsNullPtr, "NULL window");

        NSString *windowTitle = [NSString stringWithFormat:@"%s", title.c_str()];
        [window setTitle:windowTitle];
    }

}

static NSSize constrainAspectRatio(NSSize base, NSSize constraint) {
    CGFloat heightDiff = (base.height / constraint.height);
    CGFloat widthDiff = (base.width / constraint.width);
    if (heightDiff == 0) heightDiff = widthDiff;
    if (widthDiff == heightDiff) {
        return base;
    }
    else if (widthDiff > heightDiff) {
        NSSize out = { constraint.width / constraint.height * base.height, base.height };
        return out;
    }
    else {
        NSSize out = { base.width, constraint.height / constraint.width * base.width };
        return out;
    }
}

@implementation CVWindow

@synthesize mouseCallback;
@synthesize mouseParam;
@synthesize autosize;
@synthesize firstContent;
@synthesize x0;
@synthesize y0;
@synthesize sliders;
@synthesize slidersKeys;
@synthesize status;

- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags {
    (void)event;
    NSPoint mp = [NSEvent mouseLocation];
    mp = [self convertScreenToBase: mp];
    CVView *contentView = [self contentView];
    NSSize viewSize = contentView.frame.size;
    if (contentView.imageView) {
        viewSize = contentView.imageView.frame.size;
    }
    else {
        viewSize.height -= contentView.sliderHeight;
    }
    mp.y = viewSize.height - mp.y;

    NSSize imageSize = contentView.image.size;
    mp.y *= (imageSize.height / std::max(viewSize.height, 1.));
    mp.x *= (imageSize.width / std::max(viewSize.width, 1.));

    if( [event type] == NSEventTypeScrollWheel ) {
        if( event.hasPreciseScrollingDeltas ) {
            mp.x = int(event.scrollingDeltaX);
            mp.y = int(event.scrollingDeltaY);
        } else {
            mp.x = int(event.scrollingDeltaX / 0.100006);
            mp.y = int(event.scrollingDeltaY / 0.100006);
        }
        if( mp.x && !mp.y && cv::EVENT_MOUSEWHEEL == type ) {
            type = cv::EVENT_MOUSEHWHEEL;
        }
        mouseCallback(type, mp.x, mp.y, flags, mouseParam);
    } else if( mp.x >= 0 && mp.y >= 0 && mp.x < imageSize.width && mp.y < imageSize.height ) {
        mouseCallback(type, mp.x, mp.y, flags, mouseParam);
    }
}

- (void)cvMouseEvent:(NSEvent *)event {
    if(!mouseCallback)
        return;

    int flags = 0;
    if([event modifierFlags] & NSShiftKeyMask)		flags |= cv::EVENT_FLAG_SHIFTKEY;
    if([event modifierFlags] & NSControlKeyMask)	flags |= cv::EVENT_FLAG_CTRLKEY;
    if([event modifierFlags] & NSAlternateKeyMask)	flags |= cv::EVENT_FLAG_ALTKEY;

    //modified code using ternary operator:
    if ([event type] == NSLeftMouseDown) {
    [self cvSendMouseEvent:event
                      type:([event modifierFlags] & NSControlKeyMask) ? cv::EVENT_RBUTTONDOWN : cv::EVENT_LBUTTONDOWN
                     flags:flags | (([event modifierFlags] & NSControlKeyMask) ? cv::EVENT_FLAG_RBUTTON : cv::EVENT_FLAG_LBUTTON)];
    }

    if ([event type] == NSLeftMouseUp) {
        [self cvSendMouseEvent:event
                        type:([event modifierFlags] & NSControlKeyMask) ? cv::EVENT_RBUTTONUP : cv::EVENT_LBUTTONUP
                        flags:flags | (([event modifierFlags] & NSControlKeyMask) ? cv::EVENT_FLAG_RBUTTON : cv::EVENT_FLAG_LBUTTON)];
    }

    if([event type] == NSRightMouseDown){[self cvSendMouseEvent:event type:cv::EVENT_RBUTTONDOWN flags:flags | cv::EVENT_FLAG_RBUTTON];}
    if([event type] == NSRightMouseUp)	{[self cvSendMouseEvent:event type:cv::EVENT_RBUTTONUP   flags:flags | cv::EVENT_FLAG_RBUTTON];}
    if([event type] == NSOtherMouseDown){[self cvSendMouseEvent:event type:cv::EVENT_MBUTTONDOWN flags:flags];}
    if([event type] == NSOtherMouseUp)	{[self cvSendMouseEvent:event type:cv::EVENT_MBUTTONUP   flags:flags];}
    if([event type] == NSMouseMoved)	{[self cvSendMouseEvent:event type:cv::EVENT_MOUSEMOVE   flags:flags];}
    if([event type] == NSLeftMouseDragged) {[self cvSendMouseEvent:event type:cv::EVENT_MOUSEMOVE   flags:flags | cv::EVENT_FLAG_LBUTTON];}
    if([event type] == NSRightMouseDragged)	{[self cvSendMouseEvent:event type:cv::EVENT_MOUSEMOVE   flags:flags | cv::EVENT_FLAG_RBUTTON];}
    if([event type] == NSOtherMouseDragged)	{[self cvSendMouseEvent:event type:cv::EVENT_MOUSEMOVE   flags:flags | cv::EVENT_FLAG_MBUTTON];}
    if([event type] == NSEventTypeScrollWheel) {[self cvSendMouseEvent:event type:cv::EVENT_MOUSEWHEEL   flags:flags ];}
}

-(void)scrollWheel:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)keyDown:(NSEvent *)theEvent {
    [super keyDown:theEvent];
}
- (void)rightMouseDragged:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)rightMouseUp:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)rightMouseDown:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)mouseMoved:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)otherMouseDragged:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)otherMouseUp:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)otherMouseDown:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)mouseDragged:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)mouseUp:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}
- (void)mouseDown:(NSEvent *)theEvent {
    [self cvMouseEvent:theEvent];
}

- (void)createSliderWithName:(const char *)name maxValue:(int)max value:(int *)value callback:(CvTrackbarCallback)callback {
    if(sliders == nil)
        sliders = [[NSMutableDictionary alloc] init];

    if(slidersKeys == nil)
        slidersKeys = [[NSMutableArray alloc] init];

    NSString *cvname = [NSString stringWithFormat:@"%s", name];

    // Avoid overwriting slider
    if([sliders valueForKey:cvname]!=nil)
        return;

    // Create slider
    CVSlider *slider = [[CVSlider alloc] init];
    [[slider name] setStringValue:cvname];
    slider.initialName = [NSString stringWithFormat:@"%s", name];
    [[slider slider] setMaxValue:max];
    [[slider slider] setMinValue:0];
    if(value)
    {
        [[slider slider] setIntValue:*value];
        [slider setValue:value];
        NSString *temp = [slider initialName];
        NSString *text = [NSString stringWithFormat:@"%@ %d", temp, *value];
        [[slider name] setStringValue: text];
    }
    if(callback)
        [slider setCallback:callback];

    // Save slider
    [sliders setValue:slider forKey:cvname];
    [slidersKeys addObject:cvname];
    [[self contentView] addSubview:slider];

    //update contentView size to contain sliders
    NSSize viewSize=[[self contentView] frame].size,
           sliderSize=[slider frame].size;
    viewSize.height += sliderSize.height;
    viewSize.width = std::max<int>(viewSize.width, MIN_SLIDER_WIDTH);

    // Update slider sizes
    [self contentView].sliderHeight += sliderSize.height;

    if ([[self contentView] image] && ![[self contentView] imageView]) {
        [[self contentView] setNeedsDisplay:YES];
    }

    //update window size to contain sliders
    [self setContentSize: viewSize];
}

- (CVView *)contentView {
    return (CVView*)[super contentView];
}

@end

@implementation CVView

@synthesize image;

- (id)init {
    [super init];
    return self;
}

- (void)setImageData:(CvArr *)arr {
    cv::Mat arrMat = cv::cvarrToMat(arr);
    @autoreleasepool{
        NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:NULL
                    pixelsWide:arrMat.cols
                    pixelsHigh:arrMat.rows
                    bitsPerSample:8
                    samplesPerPixel:3
                    hasAlpha:NO
                    isPlanar:NO
                    colorSpaceName:NSDeviceRGBColorSpace
                    bitmapFormat: kCGImageAlphaNone
                    bytesPerRow:((arrMat.cols * 3 + 3) & -4)
                    bitsPerPixel:24];

        if (bitmap) {
            cv::Mat dst(arrMat.rows, arrMat.cols, CV_8UC3, [bitmap bitmapData], [bitmap bytesPerRow]);
            convertToShow(arrMat, dst);
        }
        else {
            // It's not guaranteed to like the bitsPerPixel:24, but this is a lot slower so we'd rather not do it
            bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:NULL
                pixelsWide:arrMat.cols
                pixelsHigh:arrMat.rows
                bitsPerSample:8
                samplesPerPixel:3
                hasAlpha:NO
                isPlanar:NO
                colorSpaceName:NSDeviceRGBColorSpace
                bytesPerRow:(arrMat.cols * 4)
                bitsPerPixel:32];
            cv::Mat dst(arrMat.rows, arrMat.cols, CV_8UC4, [bitmap bitmapData], [bitmap bytesPerRow]);
            convertToShow(arrMat, dst);
        }

        if( image ) {
            [image release];
        }

        image = [[NSImage alloc] init];
        [image addRepresentation:bitmap];
        [bitmap release];

        // This isn't supported on older versions of macOS
        // The performance issues this solves are mainly on newer versions of macOS, so that's fine
        if( floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_5 ) {
            if (![self imageView]) {
                [self setImageView:[[NSView alloc] init]];
                [[self imageView] setWantsLayer:true];
                [self addSubview:[self imageView]];
            }

            [[[self imageView] layer] setContents:image];

            NSRect imageViewFrame = [self frame];
            imageViewFrame.size.height -= [self sliderHeight];
            NSRect constrainedFrame = { imageViewFrame.origin, constrainAspectRatio(imageViewFrame.size, [image size]) };
            [[self imageView] setFrame:constrainedFrame];
        }
        else {
            NSRect redisplayRect = [self frame];
            redisplayRect.size.height -= [self sliderHeight];
            [self setNeedsDisplayInRect:redisplayRect];
        }
    }
}

- (void)setFrameSize:(NSSize)size {
    [super setFrameSize:size];

    int height = size.height;

    @autoreleasepool{
        CVWindow *cvwindow = (CVWindow *)[self window];
        if ([cvwindow respondsToSelector:@selector(sliders)]) {
            for(NSString *key in [cvwindow slidersKeys]) {
                CVSlider *slider = [[cvwindow sliders] valueForKey:key];
                NSRect r = [slider frame];
                r.origin.y = height - r.size.height;
                r.size.width = [[cvwindow contentView] frame].size.width;

                CGRect sliderRect = slider.slider.frame;
                CGFloat targetWidth = r.size.width - (sliderRect.origin.x + 10);
                sliderRect.size.width = targetWidth < 0 ? 0 : targetWidth;
                slider.slider.frame = sliderRect;

                [slider setFrame:r];
                height -= r.size.height;
            }
        }
        NSRect frame = self.frame;
        if (frame.size.height < self.sliderHeight) {
            frame.size.height = self.sliderHeight;
            self.frame = frame;
        }
        if ([self imageView]) {
            NSRect imageViewFrame = frame;
            imageViewFrame.size.height -= [self sliderHeight];
            NSRect constrainedFrame = { imageViewFrame.origin, constrainAspectRatio(imageViewFrame.size, [image size]) };
            [[self imageView] setFrame:constrainedFrame];
        }
    }
}

- (void)drawRect:(NSRect)rect {
    [super drawRect:rect];
    // If imageView exists, all drawing will be done by it and nothing needs to happen here
    if ([self image] && ![self imageView]) {
        @autoreleasepool{
            if(image != nil) {
                [image drawInRect: [self frame]
                        fromRect: NSZeroRect
                        operation: NSCompositeSourceOver
                        fraction: 1.0];
            }
        }
    }
}

@end

@implementation CVSlider

@synthesize slider;
@synthesize name;
@synthesize initialName;
@synthesize value;
@synthesize userData;
@synthesize callback;
@synthesize callback2;

- (id)init {
    [super init];

    callback = NULL;
    value = NULL;
    userData = NULL;

    [self setFrame:NSMakeRect(0,0,200,30)];

    name = [[NSTextField alloc] initWithFrame:NSMakeRect(10, 0,110, 25)];
    [name setEditable:NO];
    [name setSelectable:NO];
    [name setBezeled:NO];
    [name setBordered:NO];
    [name setDrawsBackground:NO];
    [[name cell] setLineBreakMode:NSLineBreakByTruncatingTail];
    [self addSubview:name];

    slider = [[NSSlider alloc] initWithFrame:NSMakeRect(120, 0, 70, 25)];
    [slider setAutoresizingMask:NSViewWidthSizable];
    [slider setMinValue:0];
    [slider setMaxValue:100];
    [slider setContinuous:YES];
    [slider setTarget:self];
    [slider setAction:@selector(handleSliderNotification:)];
    [self addSubview:slider];

    [self setAutoresizingMask:NSViewWidthSizable];

    return self;
}

- (void)handleSliderNotification:(NSNotification *)notification {
    (void)notification;
    [self handleSlider];
}

- (void)handleSlider {
    int pos = [slider intValue];
    NSString *temp = [self initialName];
    NSString *text = [NSString stringWithFormat:@"%@ %d", temp, pos];
    [name setStringValue: text];
    if(value)
        *value = pos;
    if(callback)
        callback(pos);
    if(callback2)
        callback2(pos, userData);
}

@end

#endif

/* End of file. */
