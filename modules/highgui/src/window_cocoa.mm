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

#import <Cocoa/Cocoa.h>
#include "precomp.hpp"

const int TOP_BORDER  = 7;

static NSApplication *application = nil;
static NSAutoreleasePool *pool = nil;
static NSMutableDictionary *windows = nil;
static bool wasInitialized = false;

@interface CVView : NSView {
	NSImage *image;
}
@property(assign) NSImage *image;
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
@property(assign) NSSlider *slider;
@property(assign) NSTextField *name;
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
}
@property(assign) CvMouseCallback mouseCallback;
@property(assign) void *mouseParam;
@property(assign) BOOL autosize;
@property(assign) NSMutableDictionary *sliders;
- (CVView *)contentView;
- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags;
- (void)cvMouseEvent:(NSEvent *)event;
- (void)createSliderWithName:(const char *)name maxValue:(int)max value:(int *)value callback:(CvTrackbarCallback)callback;
@end

static void icvCocoaCleanup(void)
{
    if( application )
    {
        [application terminate:nil];
        application = 0;
        [pool release];
    }
}

CV_IMPL int cvInitSystem( int argc, char** argv) 
{
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
    [application finishLaunching];
    atexit(icvCocoaCleanup);
	
    return 0;
}

CVWindow *cvGetWindow(const char *name) {
	NSString *cvname = [NSString stringWithFormat:@"%s", name];
	return (CVWindow *)[windows valueForKey:cvname];
}

CV_IMPL int cvStartWindowThread()
{
    return 0;
}

CV_IMPL void cvDestroyWindow( const char* name)
{
	CVWindow *window = cvGetWindow(name);
	if(window) {
		[window performClose:nil];
		[windows removeObjectForKey:[NSString stringWithFormat:@"%s", name]];
	}
}


CV_IMPL void cvDestroyAllWindows( void )
{
	for(NSString *key in windows) {
		[[windows valueForKey:key] performClose:nil];
	}
	[windows removeAllObjects];
}


CV_IMPL void cvShowImage( const char* name, const CvArr* arr)
{
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
		if(/*[window autosize] ||*/ empty)
        {
			NSRect vrectNew = vrectOld;
            vrectNew.size = [[[window contentView] image] size];
            rect.size.width += vrectNew.size.width - vrectOld.size.width;
            rect.size.height += vrectNew.size.height - vrectOld.size.height;
            [window setFrame:rect display:YES];
		}
        else
			[window display];
	}
}

CV_IMPL void cvResizeWindow( const char* name, int width, int height)
{
	CVWindow *window = cvGetWindow(name);
	if(window) {
		NSRect frame = [window frame];
		frame.size.width = width;
		frame.size.height = height;
		[window setFrame:frame display:YES];
	}
}

CV_IMPL void cvMoveWindow( const char* name, int x, int y)
{
	CV_FUNCNAME("cvMoveWindow");
	__BEGIN__;
	
	CVWindow *window = nil;
	
	if(name == NULL)
		CV_ERROR( CV_StsNullPtr, "NULL window name" );
	
 	window = cvGetWindow(name);
	if(window) {
		y = [[window screen] frame].size.height - y;
		[window setFrameTopLeftPoint:NSMakePoint(x, y)];
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
		
	window = cvGetWindow(window_name);
	if(window) {
		[window createSliderWithName:trackbar_name
							maxValue:count
							   value:val 
							callback:on_notify];
		result = 1;
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
	if(res) {
		CVSlider *slider = [[cvGetWindow(window_name) sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		[slider setCallback2:on_notify2];
		[slider setUserData:userdata];
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
		
	window = cvGetWindow(name);
	if(window) {
		[window setMouseCallback:function];
		[window setMouseParam:info];
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
	
	window = cvGetWindow(window_name);
	if(window) {
		CVSlider *slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		if(slider) {
			pos = [[slider slider] intValue];
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
	
	if(pos <= 0)
        CV_ERROR( CV_StsOutOfRange, "Bad trackbar maximal value" );
	
 	window = cvGetWindow(window_name);
	if(window) {
		slider = [[window sliders] valueForKey:[NSString stringWithFormat:@"%s", trackbar_name]];
		if(slider) {
			[[slider slider] setIntValue:pos];
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
	for(NSString *key in windows) {
		if([windows valueForKey:key] == window_handle)
			return [key UTF8String];
	}
	return 0;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    if( !wasInitialized )
        cvInitSystem(0, 0);
    
	CVWindow *window = cvGetWindow(name);
    if( window )
    {
        [window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];
        return 0;
    }
    
    window = [[CVWindow alloc] initWithContentRect:NSMakeRect(0,0,200,200)
        styleMask:NSTitledWindowMask|NSClosableWindowMask|NSMiniaturizableWindowMask|
        (!(flags & CV_WND_PROP_AUTOSIZE) ? NSResizableWindowMask : 0)
        backing:NSBackingStoreBuffered  
	    defer:NO];
	
	NSString *windowName = [NSString stringWithFormat:@"%s", name];
	
	[window setContentView:[[CVView alloc] init]];
	
	[window setHasShadow:YES];
	[window setAcceptsMouseMovedEvents:YES];
	[window useOptimizedDrawing:YES];
	[window setTitle:windowName];
	[window makeKeyAndOrderFront:nil];

	[window setAutosize:(flags == CV_WINDOW_AUTOSIZE)];	
	
	[windows setValue:window forKey:windowName];
	
	return [windows count]-1;
}

CV_IMPL int cvWaitKey (int maxWait)
{
    int returnCode = -1;
	double start = [[NSDate date] timeIntervalSince1970];
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	while(true) {
		if(([[NSDate date] timeIntervalSince1970] - start) * 1000 >= maxWait && maxWait>0)
			break;
		
		//event = [application currentEvent];
        [pool release];
        pool = [[NSAutoreleasePool alloc] init];
        
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
    [pool release];

	return returnCode;
}

@implementation CVWindow 

@synthesize mouseCallback;
@synthesize mouseParam;
@synthesize autosize;
@synthesize sliders;

- (void)cvSendMouseEvent:(NSEvent *)event type:(int)type flags:(int)flags {
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
	[super keyDown:theEvent];
}
- (void)rightMouseDragged:(NSEvent *)theEvent {
	[self cvMouseEvent:theEvent];
}
- (void)rightMouseUp:(NSEvent *)theEvent {
	[self cvMouseEvent:theEvent];
}
- (void)rightMouseDown:(NSEvent *)theEvent {
	// Does not seem to work?
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
	
	NSString *cvname = [NSString stringWithFormat:@"%s", name];
	
	// Avoid overwriting slider
	if([sliders valueForKey:cvname]!=nil)
		return;
	
	// Create slider
	CVSlider *slider = [[CVSlider alloc] init];
	[[slider name] setStringValue:cvname];
	[[slider slider] setMaxValue:max];
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
}

- (CVView *)contentView {
	return (CVView*)[super contentView];
}

@end

@implementation CVView 

@synthesize image;

- (id)init {
	[super init];
	image = nil;
	return self;
}

- (void)setImageData:(CvArr *)arr {
	CvMat *arrMat, *cvimage, stub;
	 
	arrMat = cvGetMat(arr, &stub);
	
	cvimage = cvCreateMat(arrMat->rows, arrMat->cols, CV_8UC3);
	cvConvertImage(arrMat, cvimage, CV_CVTIMG_SWAP_RB);
	
	CGColorSpaceRef colorspace = NULL;
    CGDataProviderRef provider = NULL;
	int width = cvimage->width;
    int height = cvimage->height;

    colorspace = CGColorSpaceCreateDeviceRGB();

    int size = 8;
    int nbChannels = 3;
      
    provider = CGDataProviderCreateWithData(NULL, cvimage->data.ptr, width * height , NULL );
      
	CGImageRef imageRef = CGImageCreate(width, height, size , size*nbChannels , cvimage->step, colorspace,  kCGImageAlphaNone , provider, NULL, true, kCGRenderingIntentDefault);
	
	NSBitmapImageRep *bitmap = [[[NSBitmapImageRep alloc] initWithCGImage:imageRef] autorelease];
	if(image) {
		[image release];
	}
	image = [[NSImage alloc] init];
	[image addRepresentation:bitmap];
	
    CGDataProviderRelease(provider);
	cvReleaseMat(&cvimage);
	
	[self setNeedsDisplay:YES];
}

- (void)setFrameSize:(NSSize)size {
	[super setFrameSize:size];
	int height = size.height;
	
	CVWindow *window = (CVWindow *)[self window];
	for(NSString *key in [window sliders]) {
		NSSlider *slider = [[window sliders] valueForKey:key];
		NSRect r = [slider frame];
		r.origin.y = height - r.size.height;
		[slider setFrame:r];
		height -= r.size.height;
	}
}

- (void)drawRect:(NSRect)rect {
	CVWindow *window = (CVWindow *)[self window];
	int height = 0;
	for(NSString *key in [window sliders]) {
		height += [[[window sliders] valueForKey:key] frame].size.height;
	}
	
	[super drawRect:rect];

	NSRect imageRect = {{0,0}, {self.frame.size.width, self.frame.size.height-height-6}};
	
	if(image != nil) {
		[image drawInRect: imageRect
		                      fromRect: NSZeroRect
		                     operation: NSCompositeSourceOver
		                      fraction: 1.0];
	}

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

/* End of file. */
