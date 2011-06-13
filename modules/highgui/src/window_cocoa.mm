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
CV_IMPL void* cvGetWindowHandle( const char* name ) {return NULL;} 
CV_IMPL const char* cvGetWindowName( void* window_handle ) {return NULL;}
CV_IMPL int cvNamedWindow( const char* name, int flags ) {return 0; }
CV_IMPL int cvWaitKey (int maxWait) {return 0;}
//*** end IphoneOS Stubs ***/
#else

#include "precomp.hpp"
#import <Cocoa/Cocoa.h>

#include <iostream>
using namespace std; 

const int TOP_BORDER  = 7;

static NSApplication *application = nil;
static NSAutoreleasePool *pool = nil;
static NSMutableDictionary *windows = nil;
static bool wasInitialized = false;

@interface CVView : NSView {
	NSImage *image;
}
@property(retain) NSImage *image;
- (void)setImageData:(CvArr *)arr;
@end

@interface CVSlider : NSView {
	NSSlider *slider;
	NSTextField *name;
	int *value;
	void *userData;
	CvTrackbarCallback callback;
	CvTrackbarCallback2 callback2;
}
@property(retain) NSSlider *slider;
@property(retain) NSTextField *name;
@property(assign) int *value;
@property(assign) void *userData;
@property(assign) CvTrackbarCallback callback;
@property(assign) CvTrackbarCallback2 callback2;
@end

@interface CVWindow : NSWindow {
	NSMutableDictionary *sliders;
	CvMouseCallback mouseCallback;
	void *mouseParam;
	BOOL autosize;
	BOOL firstContent; 
}
@property(assign) CvMouseCallback mouseCallback;
@property(assign) void *mouseParam;
@property(assign) BOOL autosize;
@property(assign) BOOL firstContent; 
@property(retain) NSMutableDictionary *sliders;
- (CVView *)contentView;
- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags;
- (void)cvMouseEvent:(NSEvent *)event;
- (void)createSliderWithName:(const char *)name maxValue:(int)max value:(int *)value callback:(CvTrackbarCallback)callback;
@end

/*static void icvCocoaCleanup(void)
{
	//cout << "icvCocoaCleanup" << endl; 
    if( application )
    {
		cvDestroyAllWindows(); 
        //[application terminate:nil];
        application = 0;
        [pool release];
    }
}*/

CV_IMPL int cvInitSystem( int argc, char** argv) 
{
	//cout << "cvInitSystem" << endl; 
	wasInitialized = true;
	
	pool = [[NSAutoreleasePool alloc] init];
	application = [NSApplication sharedApplication];
	windows = [[NSMutableDictionary alloc] init];

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
    
#ifndef NSAppKitVersionNumber10_5
#define NSAppKitVersionNumber10_5 949
#endif
    if( floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_5 )
        [application setActivationPolicy:0/*NSApplicationActivationPolicyRegular*/];
#endif
    //[application finishLaunching];
    //atexit(icvCocoaCleanup);
	
    return 0;
}

CVWindow *cvGetWindow(const char *name) {
	//cout << "cvGetWindow" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	NSString *cvname = [NSString stringWithFormat:@"%s", name]; 
	CVWindow* retval = (CVWindow*) [windows valueForKey:cvname] ; 
	//cout << "retain count: " << [retval retainCount] << endl; 
	//retval = [retval retain]; 
	//cout << "retain count: " << [retval retainCount] << endl; 
	[localpool drain]; 
	//cout << "retain count: " << [retval retainCount] << endl; 
	return retval; 
}

CV_IMPL int cvStartWindowThread()
{
	//cout << "cvStartWindowThread" << endl; 
    return 0;
}

CV_IMPL void cvDestroyWindow( const char* name)
{
	
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	//cout << "cvDestroyWindow" << endl; 
	CVWindow *window = cvGetWindow(name);
	if(window) {
		[window performClose:nil];
		[windows removeObjectForKey:[NSString stringWithFormat:@"%s", name]];
	}
	[localpool drain]; 
}


CV_IMPL void cvDestroyAllWindows( void )
{
	//cout << "cvDestroyAllWindows" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	for(NSString *key in windows) {
		[[windows valueForKey:key] performClose:nil];
	}
	[windows removeAllObjects];
	[localpool drain]; 
}


CV_IMPL void cvShowImage( const char* name, const CvArr* arr)
{
	//cout << "cvShowImage" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	CVWindow *window = cvGetWindow(name);
    if(!window)
    {
        cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
        window = cvGetWindow(name);
    }
    
	if(window)
    {
        bool empty = [[window contentView] image] == nil;
        NSRect rect = [window frame];
        NSRect vrectOld = [[window contentView] frame];
        
		[[window contentView] setImageData:(CvArr *)arr];
		if([window autosize] || [window firstContent] || empty)
        {
			NSRect vrectNew = vrectOld;
            vrectNew.size = [[[window contentView] image] size];
            rect.size.width += vrectNew.size.width - vrectOld.size.width;
            rect.size.height += vrectNew.size.height - vrectOld.size.height;
			rect.origin.y -= vrectNew.size.height - vrectOld.size.height;
			
            [window setFrame:rect display:YES];
		}
        else
			[window display];
		[window setFirstContent:NO]; 
	}
	[localpool drain]; 
}

CV_IMPL void cvResizeWindow( const char* name, int width, int height)
{

	//cout << "cvResizeWindow" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	CVWindow *window = cvGetWindow(name);
	if(window) {
		NSRect frame = [window frame];
		frame.size.width = width;
		frame.size.height = height;
		[window setFrame:frame display:YES];
	}
	[localpool drain]; 
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y)
{
	
	CV_FUNCNAME("cvMoveWindow");
	__BEGIN__;
	
	NSAutoreleasePool* localpool1 = [[NSAutoreleasePool alloc] init];
	CVWindow *window = nil;
	
	if(name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL window name" );
	//cout << "cvMoveWindow"<< endl; 
 	window = cvGetWindow(name);
	if(window) {
		y = [[window screen] frame].size.height - y;
		[window setFrameTopLeftPoint:NSMakePoint(x, y)];
	}
	[localpool1 drain]; 
	
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
	NSAutoreleasePool* localpool2 = nil;
	
	__BEGIN__;
	if (localpool2 != nil) [localpool2 drain]; 
	localpool2 = [[NSAutoreleasePool alloc] init];
	
	if(window_name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL window name" );
		
	//cout << "cvCreateTrackbar" << endl ; 
	window = cvGetWindow(window_name);
	if(window) {
		[window createSliderWithName:trackbar_name
							maxValue:count
							   value:val 
							callback:on_notify];
		result = 1;
	}
	[localpool2 drain]; 
	__END__;
	return result;
}


CV_IMPL int cvCreateTrackbar2(const char* trackbar_name,
                              const char* window_name,
                              int* val, int count,
                              CvTrackbarCallback2 on_notify2,
                              void* userdata)
{
	//cout <<"cvCreateTrackbar2" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	int res = cvCreateTrackbar(trackbar_name, window_name, val, count, NULL);
	if(res) {
		CVSlider *slider = [[cvGetWindow(window_name) sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		[slider setCallback2:on_notify2];
		[slider setUserData:userdata];
	}
	[localpool drain]; 
	return res;
}


CV_IMPL void
cvSetMouseCallback( const char* name, CvMouseCallback function, void* info)
{
	CV_FUNCNAME("cvSetMouseCallback");
	
	CVWindow *window = nil;
	NSAutoreleasePool* localpool3 = nil; 
	__BEGIN__;
	//cout << "cvSetMouseCallback" << endl; 
	
	if (localpool3 != nil) [localpool3 drain]; 
	localpool3 = [[NSAutoreleasePool alloc] init];
	
	if(name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL window name" );
		
	window = cvGetWindow(name);
	if(window) {
		[window setMouseCallback:function];
		[window setMouseParam:info];
	}
	[localpool3 drain]; 
	
	__END__;
}

 CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
	CV_FUNCNAME("cvGetTrackbarPos");
	
	CVWindow *window = nil;
	int pos = -1;
	NSAutoreleasePool* localpool4 = nil; 
	__BEGIN__;
	
	//cout << "cvGetTrackbarPos" << endl; 
	if(trackbar_name == NULL || window_name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );
	
	if (localpool4 != nil) [localpool4 drain]; 
	localpool4 = [[NSAutoreleasePool alloc] init];
	
	window = cvGetWindow(window_name);
	if(window) {
		CVSlider *slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		if(slider) {
			pos = [[slider slider] intValue];
		}
	}
	[localpool4 drain]; 
	__END__;
	return pos;
}

CV_IMPL void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)
{
	CV_FUNCNAME("cvSetTrackbarPos");
	
	CVWindow *window = nil;
	CVSlider *slider = nil;
	NSAutoreleasePool* localpool5 = nil; 
	
	__BEGIN__;
	//cout << "cvSetTrackbarPos" << endl; 
	if(trackbar_name == NULL || window_name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );
	
	if(pos <= 0)
        CV_ERROR( CV_StsOutOfRange, "Bad trackbar maximal value" );
	
	if (localpool5 != nil) [localpool5 drain]; 
	localpool5 = [[NSAutoreleasePool alloc] init];
	
 	window = cvGetWindow(window_name);
	if(window) {
		slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		if(slider) {
			[[slider slider] setIntValue:pos];
		}
	}
	[localpool5 drain]; 
	
	__END__;
}

CV_IMPL void* cvGetWindowHandle( const char* name )
{
	//cout << "cvGetWindowHandle" << endl; 
	return cvGetWindow(name);
}


CV_IMPL const char* cvGetWindowName( void* window_handle )
{	
	//cout << "cvGetWindowName" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	for(NSString *key in windows) {
		if([windows valueForKey:key] == window_handle) {
			[localpool drain]; 
			return [key UTF8String];
		}
	}
	[localpool drain]; 
	return 0;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    if( !wasInitialized )
        cvInitSystem(0, 0);
    
	//cout << "cvNamedWindow" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	CVWindow *window = cvGetWindow(name);
    if( window )
    {
        [window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];
		[localpool drain]; 
        return 0;
    }
	
	NSScreen* mainDisplay = [NSScreen mainScreen]; 
	
	NSString *windowName = [NSString stringWithFormat:@"%s", name];
	NSUInteger showResize = (flags == CV_WINDOW_AUTOSIZE) ? 0: NSResizableWindowMask ;
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
	
	[window setContentView:[[CVView alloc] init]];
	
	[window setHasShadow:YES];
	[window setAcceptsMouseMovedEvents:YES];
	[window useOptimizedDrawing:YES];
	[window setTitle:windowName];
	[window makeKeyAndOrderFront:nil];

	[window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];	
	
	[windows setValue:window forKey:windowName];
		
	[localpool drain]; 
	return [windows count]-1;
}

CV_IMPL int cvWaitKey (int maxWait)
{
	//cout << "cvWaitKey" << endl; 
    int returnCode = -1;
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];
	double start = [[NSDate date] timeIntervalSince1970];

	while(true) {
		if(([[NSDate date] timeIntervalSince1970] - start) * 1000 >= maxWait && maxWait>0)
			break;
		
		//event = [application currentEvent];
        [localpool drain];
        localpool = [[NSAutoreleasePool alloc] init];
        
        NSEvent *event =
        [application
         nextEventMatchingMask:NSAnyEventMask
         untilDate://[NSDate dateWithTimeIntervalSinceNow: 1./100]
         [NSDate distantPast]
         inMode:NSDefaultRunLoopMode
         dequeue:YES];
        
		if([event type] == NSKeyDown) {
			returnCode = [[event characters] characterAtIndex:0];
            break;
		}
        
        [application sendEvent:event];
        [application updateWindows];
        
		[NSThread sleepForTimeInterval:1/100.];
	}
    [localpool drain];

	return returnCode;
}

@implementation CVWindow 

@synthesize mouseCallback;
@synthesize mouseParam;
@synthesize autosize;
@synthesize firstContent; 
@synthesize sliders;

- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags {
	//cout << "cvSendMouseEvent" << endl; 
	NSPoint mp = [NSEvent mouseLocation];
	NSRect visible = [[self contentView] frame];
    mp = [self convertScreenToBase: mp];
    double viewHeight = [self contentView].frame.size.height;
    double viewWidth = [self contentView].frame.size.width;
    CVWindow *window = (CVWindow *)[[self contentView] window];
    for(NSString *key in [window sliders]) {
        NSSlider *slider = [[window sliders] valueForKey:key];
        viewHeight = std::min(viewHeight, (double)([slider frame].origin.y));
    }
    viewHeight -= TOP_BORDER;
    mp.y = viewHeight - mp.y;
    
    NSImage* image = ((CVView*)[self contentView]).image;
    NSSize imageSize = [image size];
    mp.x = mp.x * imageSize.width / std::max(viewWidth, 1.);
    mp.y = mp.y * imageSize.height / std::max(viewHeight, 1.);
    
    if( mp.x >= 0 && mp.y >= 0 && mp.x < imageSize.width && mp.y < imageSize.height )
        mouseCallback(type, mp.x, mp.y, flags, mouseParam);
}

- (void)cvMouseEvent:(NSEvent *)event {
	//cout << "cvMouseEvent" << endl; 
	if(!mouseCallback) 
		return;
		
	int flags = 0;
	if([event modifierFlags] & NSShiftKeyMask)		flags |= CV_EVENT_FLAG_SHIFTKEY;
	if([event modifierFlags] & NSControlKeyMask)	flags |= CV_EVENT_FLAG_CTRLKEY;
	if([event modifierFlags] & NSAlternateKeyMask)	flags |= CV_EVENT_FLAG_ALTKEY;
		
	if([event type] == NSLeftMouseDown)	{[self cvSendMouseEvent:event type:CV_EVENT_LBUTTONDOWN flags:flags | CV_EVENT_FLAG_LBUTTON];}
	if([event type] == NSLeftMouseUp)	{[self cvSendMouseEvent:event type:CV_EVENT_LBUTTONUP   flags:flags | CV_EVENT_FLAG_LBUTTON];}
	if([event type] == NSRightMouseDown){[self cvSendMouseEvent:event type:CV_EVENT_RBUTTONDOWN flags:flags | CV_EVENT_FLAG_RBUTTON];}
	if([event type] == NSRightMouseUp)	{[self cvSendMouseEvent:event type:CV_EVENT_RBUTTONUP   flags:flags | CV_EVENT_FLAG_RBUTTON];}
	if([event type] == NSOtherMouseDown){[self cvSendMouseEvent:event type:CV_EVENT_MBUTTONDOWN flags:flags];}
	if([event type] == NSOtherMouseUp)	{[self cvSendMouseEvent:event type:CV_EVENT_MBUTTONUP   flags:flags];}
	if([event type] == NSMouseMoved)	{[self cvSendMouseEvent:event type:CV_EVENT_MOUSEMOVE   flags:flags];}
	if([event type] == NSLeftMouseDragged) {[self cvSendMouseEvent:event type:CV_EVENT_MOUSEMOVE   flags:flags | CV_EVENT_FLAG_LBUTTON];}
	if([event type] == NSRightMouseDragged)	{[self cvSendMouseEvent:event type:CV_EVENT_MOUSEMOVE   flags:flags | CV_EVENT_FLAG_RBUTTON];}
	if([event type] == NSOtherMouseDragged)	{[self cvSendMouseEvent:event type:CV_EVENT_MOUSEMOVE   flags:flags | CV_EVENT_FLAG_MBUTTON];}
}
- (void)keyDown:(NSEvent *)theEvent {
	//cout << "keyDown" << endl; 
	[super keyDown:theEvent];
}
- (void)rightMouseDragged:(NSEvent *)theEvent {
	//cout << "rightMouseDragged" << endl ;
	[self cvMouseEvent:theEvent];
}
- (void)rightMouseUp:(NSEvent *)theEvent {
	//cout << "rightMouseUp" << endl; 
	[self cvMouseEvent:theEvent];
}
- (void)rightMouseDown:(NSEvent *)theEvent {
	// Does not seem to work?
	//cout << "rightMouseDown" << endl; 
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
	//cout << "createSliderWithName" << endl; 
	if(sliders == nil)
		sliders = [[NSMutableDictionary alloc] init];
	
	NSString *cvname = [NSString stringWithFormat:@"%s", name];
	
	// Avoid overwriting slider
	if([sliders valueForKey:cvname]!=nil)
		return;
	
	// Create slider
	CVSlider *slider = [[CVSlider alloc] init];
	[[slider name] setStringValue:cvname];
	[[slider slider] setMaxValue:max];
	[[slider slider] setMinValue:0]; 
	[[slider slider] setNumberOfTickMarks:(max+1)]; 
	[[slider slider] setAllowsTickMarkValuesOnly:YES]; 
	if(value)
    {
		[[slider slider] setIntValue:*value];
        [slider setValue:value];
    }
	if(callback)
		[slider setCallback:callback];
	
	// Save slider
	[sliders setValue:slider forKey:cvname];
	[[self contentView] addSubview:slider];
	
	// Update slider sizes
	[[self contentView] setFrameSize:[[self contentView] frame].size];
	[[self contentView] setNeedsDisplay:YES];
	
	
	int height = 0;
	for(NSString *key in sliders) {
		height += [[sliders valueForKey:key] frame].size.height;
	}
	[self setContentMinSize:NSMakeSize(0, height)]; 
}

- (CVView *)contentView {
	return (CVView*)[super contentView];
}

@end

@implementation CVView 

@synthesize image;

- (id)init {
	//cout << "CVView init" << endl; 
	[super init];
	image = [[NSImage alloc] init];
	return self;
}

- (void)setImageData:(CvArr *)arr {
	//cout << "setImageData" << endl; 
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	CvMat *arrMat, *cvimage, stub;
	 
	arrMat = cvGetMat(arr, &stub);
	
	cvimage = cvCreateMat(arrMat->rows, arrMat->cols, CV_8UC3);
	cvConvertImage(arrMat, cvimage, CV_CVTIMG_SWAP_RB);
	
	/*CGColorSpaceRef colorspace = NULL;
    CGDataProviderRef provider = NULL;
	int width = cvimage->width;
    int height = cvimage->height;

    colorspace = CGColorSpaceCreateDeviceRGB();

    int size = 8;
    int nbChannels = 3;
      
    provider = CGDataProviderCreateWithData(NULL, cvimage->data.ptr, width * height , NULL );
      
	CGImageRef imageRef = CGImageCreate(width, height, size , size*nbChannels , cvimage->step, colorspace,  kCGImageAlphaNone , provider, NULL, true, kCGRenderingIntentDefault);
	
	NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithCGImage:imageRef]; 
	if(image) {
		[image release];
	}*/
	
	NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:NULL
				pixelsWide:cvimage->width
				pixelsHigh:cvimage->height
				bitsPerSample:8
				samplesPerPixel:3
				hasAlpha:NO
				isPlanar:NO
				colorSpaceName:NSDeviceRGBColorSpace
				bytesPerRow:(cvimage->width * 4)
				bitsPerPixel:32];
	
	int	pixelCount = cvimage->width * cvimage->height;
	unsigned char *src = cvimage->data.ptr;
	unsigned char *dst = [bitmap bitmapData];
	
	for( int i = 0; i < pixelCount; i++ )
	{
		dst[i * 4 + 0] = src[i * 3 + 0];
		dst[i * 4 + 1] = src[i * 3 + 1];
		dst[i * 4 + 2] = src[i * 3 + 2];
	}
	
    if( image )
        [image release]; 
    
	image = [[NSImage alloc] init];
	[image addRepresentation:bitmap];
	[bitmap release];
	
	/*CGColorSpaceRelease(colorspace); 
    CGDataProviderRelease(provider);
	CGImageRelease(imageRef);*/
	cvReleaseMat(&cvimage);
	[localpool drain]; 
	
	[self setNeedsDisplay:YES];
	 
}

- (void)setFrameSize:(NSSize)size {
	//cout << "setFrameSize" << endl; 
	[super setFrameSize:size];
	
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	int height = size.height;
	
	CVWindow *cvwindow = (CVWindow *)[self window];
	for(NSString *key in [cvwindow sliders]) {
		NSSlider *slider = [[cvwindow sliders] valueForKey:key];
		NSRect r = [slider frame];
		r.origin.y = height - r.size.height;
		[slider setFrame:r];
		height -= r.size.height;
	}
	[localpool drain]; 
}

- (void)drawRect:(NSRect)rect {
	//cout << "drawRect" << endl; 
	[super drawRect:rect];
	
	NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
	CVWindow *cvwindow = (CVWindow *)[self window];
	int height = 0;
	for(NSString *key in [cvwindow sliders]) {
		height += [[[cvwindow sliders] valueForKey:key] frame].size.height;
	}
	

	NSRect imageRect = {{0,0}, {self.frame.size.width, self.frame.size.height-height-6}};
	
	if(image != nil) {
		[image drawInRect: imageRect
		                      fromRect: NSZeroRect
		                     operation: NSCompositeSourceOver
		                      fraction: 1.0];
	}
	[localpool release]; 

}

@end

@implementation CVSlider 

@synthesize slider;
@synthesize name;
@synthesize value;
@synthesize userData;
@synthesize callback;
@synthesize callback2;

- (id)init {
	[super init];

	callback = NULL;
	value = NULL;
	userData = NULL;

	[self setFrame:NSMakeRect(0,0,200,25)];

	name = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0,120, 20)];
	[name setEditable:NO];
    [name setSelectable:NO];
    [name setBezeled:NO];
    [name setBordered:NO];
	[name setDrawsBackground:NO];
	[[name cell] setLineBreakMode:NSLineBreakByTruncatingTail];
	[self addSubview:name];
	
	slider = [[NSSlider alloc] initWithFrame:NSMakeRect(120, 0, 76, 20)];
	[slider setAutoresizingMask:NSViewWidthSizable];
	[slider setMinValue:0];
	[slider setMaxValue:100];
	[slider setContinuous:YES];
	[slider setTarget:self];
	[slider setAction:@selector(sliderChanged:)];
	[self addSubview:slider];
	
	[self setAutoresizingMask:NSViewWidthSizable];
	
	[self setFrame:NSMakeRect(12, 0, 182, 30)];
	
	return self;
}

- (void)sliderChanged:(NSNotification *)notification {
    int pos = [slider intValue];
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
