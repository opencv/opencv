/*
 *  cap_ios_photo_camera.mm
 *  For iOS video I/O
 *  by Eduard Feicho on 29/07/12
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


#import "opencv2/highgui/cap_ios.h"
#include "precomp.hpp"

#pragma mark - Private Interface


@interface CvPhotoCamera ()

@property (nonatomic, retain) AVCaptureStillImageOutput* stillImageOutput;

@end



#pragma mark - Implementation


@implementation CvPhotoCamera



#pragma mark Public

@synthesize stillImageOutput;
@synthesize delegate;


#pragma mark - Public interface


- (void)takePicture
{
    if (cameraAvailable == NO) {
        return;
    }
    cameraAvailable = NO;


    [self.stillImageOutput captureStillImageAsynchronouslyFromConnection:self.videoCaptureConnection
                                                       completionHandler:
     ^(CMSampleBufferRef imageSampleBuffer, NSError *error)
     {
         if (error == nil && imageSampleBuffer != NULL)
         {
             // TODO check
             //			 NSNumber* imageOrientation = [UIImage cgImageOrientationForUIDeviceOrientation:currentDeviceOrientation];
             //			 CMSetAttachment(imageSampleBuffer, kCGImagePropertyOrientation, imageOrientation, 1);

             NSData *jpegData = [AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageSampleBuffer];

             dispatch_async(dispatch_get_main_queue(), ^{
                 [self.captureSession stopRunning];

                 // Make sure we create objects on the main thread in the main context
                 UIImage* newImage = [UIImage imageWithData:jpegData];

                 //UIImageOrientation orientation = [newImage imageOrientation];

                 // TODO: only apply rotation, don't scale, since we can set this directly in the camera
                 /*
                  switch (orientation) {
                  case UIImageOrientationUp:
                  case UIImageOrientationDown:
                  newImage = [newImage imageWithAppliedRotationAndMaxSize:CGSizeMake(640.0, 480.0)];
                  break;
                  case UIImageOrientationLeft:
                  case UIImageOrientationRight:
                  newImage = [newImage imageWithMaxSize:CGSizeMake(640.0, 480.0)];
                  default:
                  break;
                  }
                  */

                 // We have captured the image, we can allow the user to take another picture
                 cameraAvailable = YES;

                 NSLog(@"CvPhotoCamera captured image");
                 if (self.delegate) {
                     [self.delegate photoCamera:self capturedImage:newImage];
                 }

                 [self.captureSession startRunning];
             });
         }
     }];


}

- (void)stop;
{
    [super stop];
    self.stillImageOutput = nil;
}


#pragma mark - Private Interface


- (void)createStillImageOutput;
{
    // setup still image output with jpeg codec
    self.stillImageOutput = [[AVCaptureStillImageOutput alloc] init];
    NSDictionary *outputSettings = [NSDictionary dictionaryWithObjectsAndKeys:AVVideoCodecJPEG, AVVideoCodecKey, nil];
    [self.stillImageOutput setOutputSettings:outputSettings];
    [self.captureSession addOutput:self.stillImageOutput];

    for (AVCaptureConnection *connection in self.stillImageOutput.connections) {
        for (AVCaptureInputPort *port in [connection inputPorts]) {
            if ([port.mediaType isEqual:AVMediaTypeVideo]) {
                self.videoCaptureConnection = connection;
                break;
            }
        }
        if (self.videoCaptureConnection) {
            break;
        }
    }
    NSLog(@"[Camera] still image output created");
}


- (void)createCaptureOutput;
{
    [self createStillImageOutput];
}

- (void)createCustomVideoPreview;
{
    //do nothing, always use AVCaptureVideoPreviewLayer
}


@end
