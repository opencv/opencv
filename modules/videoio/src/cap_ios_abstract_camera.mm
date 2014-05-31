/*
 *  cap_ios_abstract_camera.mm
 *  For iOS video I/O
 *  by Eduard Feicho on 29/07/12
 *  by Alexander Shishkov on 17/07/13
 *  Copyright 2012. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#import "opencv2/videoio/cap_ios.h"
#include "precomp.hpp"

#pragma mark - Private Interface

@interface CvAbstractCamera ()

@property (nonatomic, retain) AVCaptureVideoPreviewLayer* captureVideoPreviewLayer;

- (void)deviceOrientationDidChange:(NSNotification*)notification;
- (void)startCaptureSession;

- (void)setDesiredCameraPosition:(AVCaptureDevicePosition)desiredPosition;

- (void)updateSize;

@end


#pragma mark - Implementation


@implementation CvAbstractCamera



#pragma mark Public

@synthesize imageWidth;
@synthesize imageHeight;


@synthesize defaultFPS;
@synthesize defaultAVCaptureDevicePosition;
@synthesize defaultAVCaptureVideoOrientation;
@synthesize defaultAVCaptureSessionPreset;



@synthesize captureSession;
@synthesize captureVideoPreviewLayer;
@synthesize videoCaptureConnection;
@synthesize running;
@synthesize captureSessionLoaded;
@synthesize useAVCaptureVideoPreviewLayer;

@synthesize parentView;

#pragma mark - Constructors

- (id)init;
{
    self = [super init];
    if (self) {
        // react to device orientation notifications
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(deviceOrientationDidChange:)
                                                     name:UIDeviceOrientationDidChangeNotification
                                                   object:nil];
        [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
        currentDeviceOrientation = [[UIDevice currentDevice] orientation];


        // check if camera available
        cameraAvailable = [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera];
        NSLog(@"camera available: %@", (cameraAvailable == YES ? @"YES" : @"NO") );

        running = NO;

        // set camera default configuration
        self.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
        self.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
        self.defaultFPS = 15;
        self.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;

        self.parentView = nil;
        self.useAVCaptureVideoPreviewLayer = NO;
    }
    return self;
}



- (id)initWithParentView:(UIView*)parent;
{
    self = [super init];
    if (self) {
        // react to device orientation notifications
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(deviceOrientationDidChange:)
                                                     name:UIDeviceOrientationDidChangeNotification
                                                   object:nil];
        [[UIDevice currentDevice] beginGeneratingDeviceOrientationNotifications];
        currentDeviceOrientation = [[UIDevice currentDevice] orientation];


        // check if camera available
        cameraAvailable = [UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera];
        NSLog(@"camera available: %@", (cameraAvailable == YES ? @"YES" : @"NO") );

        running = NO;

        // set camera default configuration
        self.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
        self.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
        self.defaultFPS = 15;
        self.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;

        self.parentView = parent;
        self.useAVCaptureVideoPreviewLayer = YES;
    }
    return self;
}



- (void)dealloc;
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    [[UIDevice currentDevice] endGeneratingDeviceOrientationNotifications];
    [super dealloc];
}


#pragma mark - Public interface


- (void)start;
{
    if (![NSThread isMainThread]) {
        NSLog(@"[Camera] Warning: Call start only from main thread");
        [self performSelectorOnMainThread:@selector(start) withObject:nil waitUntilDone:NO];
        return;
    }

    if (running == YES) {
        return;
    }
    running = YES;

    // TOOD update image size data before actually starting (needed for recording)
    [self updateSize];

    if (cameraAvailable) {
        [self startCaptureSession];
    }
}


- (void)pause;
{
    running = NO;
    [self.captureSession stopRunning];
}



- (void)stop;
{
    running = NO;

    // Release any retained subviews of the main view.
    // e.g. self.myOutlet = nil;
    for (AVCaptureInput *input in self.captureSession.inputs) {
        [self.captureSession removeInput:input];
    }

    for (AVCaptureOutput *output in self.captureSession.outputs) {
        [self.captureSession removeOutput:output];
    }

    [self.captureSession stopRunning];
    self.captureSession = nil;
    self.captureVideoPreviewLayer = nil;
    self.videoCaptureConnection = nil;
    captureSessionLoaded = NO;
}



// use front/back camera
- (void)switchCameras;
{
    BOOL was_running = self.running;
    if (was_running) {
        [self stop];
    }
    if (self.defaultAVCaptureDevicePosition == AVCaptureDevicePositionFront) {
        self.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    } else {
        self.defaultAVCaptureDevicePosition  = AVCaptureDevicePositionFront;
    }
    if (was_running) {
        [self start];
    }
}



#pragma mark - Device Orientation Changes


- (void)deviceOrientationDidChange:(NSNotification*)notification
{
    (void)notification;
    UIDeviceOrientation orientation = [UIDevice currentDevice].orientation;

    switch (orientation)
    {
        case UIDeviceOrientationPortrait:
        case UIDeviceOrientationPortraitUpsideDown:
        case UIDeviceOrientationLandscapeLeft:
        case UIDeviceOrientationLandscapeRight:
            currentDeviceOrientation = orientation;
            break;

        case UIDeviceOrientationFaceUp:
        case UIDeviceOrientationFaceDown:
        default:
            break;
    }
    NSLog(@"deviceOrientationDidChange: %d", (int)orientation);

    [self updateOrientation];
}



#pragma mark - Private Interface

- (void)createCaptureSession;
{
    // set a av capture session preset
    self.captureSession = [[AVCaptureSession alloc] init];
    if ([self.captureSession canSetSessionPreset:self.defaultAVCaptureSessionPreset]) {
        [self.captureSession setSessionPreset:self.defaultAVCaptureSessionPreset];
    } else if ([self.captureSession canSetSessionPreset:AVCaptureSessionPresetLow]) {
        [self.captureSession setSessionPreset:AVCaptureSessionPresetLow];
    } else {
        NSLog(@"[Camera] Error: could not set session preset");
    }
}

- (void)createCaptureDevice;
{
    // setup the device
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    [self setDesiredCameraPosition:self.defaultAVCaptureDevicePosition];
    NSLog(@"[Camera] device connected? %@", device.connected ? @"YES" : @"NO");
    NSLog(@"[Camera] device position %@", (device.position == AVCaptureDevicePositionBack) ? @"back" : @"front");
}


- (void)createVideoPreviewLayer;
{
    self.captureVideoPreviewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.captureSession];

    if ([self.captureVideoPreviewLayer respondsToSelector:@selector(connection)])
    {
        if ([self.captureVideoPreviewLayer.connection isVideoOrientationSupported])
        {
            [self.captureVideoPreviewLayer.connection setVideoOrientation:self.defaultAVCaptureVideoOrientation];
        }
    }
    else
    {
        // Deprecated in 6.0; here for backward compatibility
        if ([self.captureVideoPreviewLayer isOrientationSupported])
        {
            [self.captureVideoPreviewLayer setOrientation:self.defaultAVCaptureVideoOrientation];
        }
    }

    if (parentView != nil) {
        self.captureVideoPreviewLayer.frame = self.parentView.bounds;
        self.captureVideoPreviewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
        [self.parentView.layer addSublayer:self.captureVideoPreviewLayer];
    }
    NSLog(@"[Camera] created AVCaptureVideoPreviewLayer");
}

- (void)setDesiredCameraPosition:(AVCaptureDevicePosition)desiredPosition;
{
    for (AVCaptureDevice *device in [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]) {
        if ([device position] == desiredPosition) {
            [self.captureSession beginConfiguration];

            NSError* error = nil;
            AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
            if (!input) {
                NSLog(@"error creating input %@", [error localizedDescription]);
            }

            // support for autofocus
            if ([device isFocusModeSupported:AVCaptureFocusModeContinuousAutoFocus]) {
                error = nil;
                if ([device lockForConfiguration:&error]) {
                    device.focusMode = AVCaptureFocusModeContinuousAutoFocus;
                    [device unlockForConfiguration];
                } else {
                    NSLog(@"unable to lock device for autofocos configuration %@", [error localizedDescription]);
                }
            }
            [self.captureSession addInput:input];

            for (AVCaptureInput *oldInput in self.captureSession.inputs) {
                [self.captureSession removeInput:oldInput];
            }
            [self.captureSession addInput:input];
            [self.captureSession commitConfiguration];

            break;
        }
    }
}



- (void)startCaptureSession
{
    if (!cameraAvailable) {
        return;
    }

    if (self.captureSessionLoaded == NO) {
        [self createCaptureSession];
        [self createCaptureDevice];
        [self createCaptureOutput];

        // setup preview layer
        if (self.useAVCaptureVideoPreviewLayer) {
            [self createVideoPreviewLayer];
        } else {
            [self createCustomVideoPreview];
        }

        captureSessionLoaded = YES;
    }

    [self.captureSession startRunning];
}


- (void)createCaptureOutput;
{
    [NSException raise:NSInternalInconsistencyException
                format:@"You must override %@ in a subclass", NSStringFromSelector(_cmd)];
}

- (void)createCustomVideoPreview;
{
    [NSException raise:NSInternalInconsistencyException
                format:@"You must override %@ in a subclass", NSStringFromSelector(_cmd)];
}

- (void)updateOrientation;
{
    // nothing to do here
}


- (void)updateSize;
{
    if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPresetPhoto]) {
        //TODO: find the correct resolution
        self.imageWidth = 640;
        self.imageHeight = 480;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPresetHigh]) {
        //TODO: find the correct resolution
        self.imageWidth = 640;
        self.imageHeight = 480;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPresetMedium]) {
        //TODO: find the correct resolution
        self.imageWidth = 640;
        self.imageHeight = 480;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPresetLow]) {
        //TODO: find the correct resolution
        self.imageWidth = 640;
        self.imageHeight = 480;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPreset352x288]) {
        self.imageWidth = 352;
        self.imageHeight = 288;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPreset640x480]) {
        self.imageWidth = 640;
        self.imageHeight = 480;
    } else if ([self.defaultAVCaptureSessionPreset isEqualToString:AVCaptureSessionPreset1280x720]) {
        self.imageWidth = 1280;
        self.imageHeight = 720;
    } else {
        self.imageWidth = 640;
        self.imageHeight = 480;
    }
}

- (void)lockFocus;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isFocusModeSupported:AVCaptureFocusModeLocked]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.focusMode = AVCaptureFocusModeLocked;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for locked focus configuration %@", [error localizedDescription]);
        }
    }
}

- (void) unlockFocus;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isFocusModeSupported:AVCaptureFocusModeContinuousAutoFocus]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.focusMode = AVCaptureFocusModeContinuousAutoFocus;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for autofocus configuration %@", [error localizedDescription]);
        }
    }
}

- (void)lockExposure;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isExposureModeSupported:AVCaptureExposureModeLocked]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.exposureMode = AVCaptureExposureModeLocked;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for locked exposure configuration %@", [error localizedDescription]);
        }
    }
}

- (void) unlockExposure;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.exposureMode = AVCaptureExposureModeContinuousAutoExposure;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for autoexposure configuration %@", [error localizedDescription]);
        }
    }
}

- (void)lockBalance;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeLocked]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.whiteBalanceMode = AVCaptureWhiteBalanceModeLocked;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for locked white balance configuration %@", [error localizedDescription]);
        }
    }
}

- (void) unlockBalance;
{
    AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    if ([device isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance]) {
        NSError *error = nil;
        if ([device lockForConfiguration:&error]) {
            device.whiteBalanceMode = AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance;
            [device unlockForConfiguration];
        } else {
            NSLog(@"unable to lock device for auto white balance configuration %@", [error localizedDescription]);
        }
    }
}

@end
