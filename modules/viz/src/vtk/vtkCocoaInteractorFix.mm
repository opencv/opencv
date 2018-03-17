/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
// Authors:
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//  This workaround code was taken from PCL library(www.pointclouds.org)
//
//M*/

#import <Cocoa/Cocoa.h>
#include <vtkCocoaRenderWindow.h>
#include <vtkCocoaRenderWindowInteractor.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>

namespace cv { namespace viz {
    vtkSmartPointer<vtkRenderWindowInteractor> vtkCocoaRenderWindowInteractorNew();
}} // namespace

#if ((VTK_MAJOR_VERSION < 6) || ((VTK_MAJOR_VERSION == 6) && (VTK_MINOR_VERSION < 2)))


//----------------------------------------------------------------------------
@interface vtkCocoaServerFix : NSObject
{
    vtkCocoaRenderWindow* renWin;
}

+ (id)cocoaServerWithRenderWindow:(vtkCocoaRenderWindow*)inRenderWindow;

- (void)start;
- (void)stop;
- (void)breakEventLoop;

@end

//----------------------------------------------------------------------------
@implementation vtkCocoaServerFix

//----------------------------------------------------------------------------
- (id)initWithRenderWindow:(vtkCocoaRenderWindow *)inRenderWindow
{
    self = [super init];
    if (self)
        renWin = inRenderWindow;
    return self;
}

//----------------------------------------------------------------------------
+ (id)cocoaServerWithRenderWindow:(vtkCocoaRenderWindow *)inRenderWindow
{
    vtkCocoaServerFix *server = [[[vtkCocoaServerFix alloc] initWithRenderWindow:inRenderWindow] autorelease];
    return server;
}

//----------------------------------------------------------------------------
- (void)start
{
    // Retrieve the NSWindow.
    NSWindow *win = nil;
    if (renWin)
    {
        win = reinterpret_cast<NSWindow*> (renWin->GetRootWindow ());

        // We don't want to be informed of every window closing, so check for nil.
        if (win != nil)
        {
            // Register for the windowWillClose notification in order to stop the run loop if the window closes.
            NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
            [nc addObserver:self selector:@selector(windowWillClose:) name:NSWindowWillCloseNotification object:win];
        }
    }
    // Start the NSApplication's run loop
    NSApplication* application = [NSApplication sharedApplication];
    [application run];
}

//----------------------------------------------------------------------------
- (void)stop
{
    [self breakEventLoop];
}

//----------------------------------------------------------------------------
- (void)breakEventLoop
{
    NSApplication* application = [NSApplication sharedApplication];
    [application stop:application];

    NSEvent *event = [NSEvent otherEventWithType:NSApplicationDefined
            location:NSMakePoint(0.0,0.0)
            modifierFlags:0
            timestamp:0
            windowNumber:-1
            context:nil
            subtype:0
            data1:0
            data2:0];
    [application postEvent:event atStart:YES];
}

//----------------------------------------------------------------------------
- (void)windowWillClose:(NSNotification*)aNotification
{
    (void)aNotification;

    NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
    [nc removeObserver:self name:NSWindowWillCloseNotification object:nil];

    if (renWin)
    {
        int windowCreated = renWin->GetWindowCreated ();
        if (windowCreated)
        {
            [self breakEventLoop];

            // The NSWindow is closing, so prevent anyone from accidentally using it
            renWin->SetRootWindow(NULL);
        }
    }
}

@end

//----------------------------------------------------------------------------

namespace cv { namespace viz
{
    class vtkCocoaRenderWindowInteractorFix : public vtkCocoaRenderWindowInteractor
    {
    public:
        static vtkCocoaRenderWindowInteractorFix *New ();
        vtkTypeMacro (vtkCocoaRenderWindowInteractorFix, vtkCocoaRenderWindowInteractor)

        virtual void Start ();
        virtual void TerminateApp ();

    protected:
        vtkCocoaRenderWindowInteractorFix () {}
        ~vtkCocoaRenderWindowInteractorFix () {}

    private:
        vtkCocoaRenderWindowInteractorFix (const vtkCocoaRenderWindowInteractorFix&);  // Not implemented.
        void operator = (const vtkCocoaRenderWindowInteractorFix&);  // Not implemented.
    };

    vtkStandardNewMacro (vtkCocoaRenderWindowInteractorFix)
}}

void cv::viz::vtkCocoaRenderWindowInteractorFix::Start ()
{
    vtkCocoaRenderWindow* renWin = vtkCocoaRenderWindow::SafeDownCast(this->GetRenderWindow ());
    if (renWin != NULL)
    {
        vtkCocoaServerFix *server = reinterpret_cast<vtkCocoaServerFix*> (this->GetCocoaServer ());
        if (!this->GetCocoaServer ())
        {
            server = [vtkCocoaServerFix cocoaServerWithRenderWindow:renWin];
            this->SetCocoaServer (reinterpret_cast<void*> (server));
        }

        [server start];
    }
}

void cv::viz::vtkCocoaRenderWindowInteractorFix::TerminateApp ()
{
    vtkCocoaRenderWindow *renWin = vtkCocoaRenderWindow::SafeDownCast (this->RenderWindow);
    if (renWin)
    {
        vtkCocoaServerFix *server = reinterpret_cast<vtkCocoaServerFix*> (this->GetCocoaServer ());
        [server stop];
    }
}

vtkSmartPointer<vtkRenderWindowInteractor> cv::viz::vtkCocoaRenderWindowInteractorNew()
{
    return vtkSmartPointer<vtkCocoaRenderWindowInteractorFix>::New();
}


#else

vtkSmartPointer<vtkRenderWindowInteractor> cv::viz::vtkCocoaRenderWindowInteractorNew()
{
    return vtkSmartPointer<vtkCocoaRenderWindowInteractor>::New();
}

#endif
