//
//  CvPhotoCamera2.mm
//
//  Created by Giles Payne on 2020/04/01.
//

#import "CvCamera2.h"

#pragma mark - Private Interface

@interface CvPhotoCamera2 ()
{
    id<CvPhotoCameraDelegate2> _delegate;
}

@property (nonatomic, strong) AVCaptureStillImageOutput* stillImageOutput;

@end


#pragma mark - Implementation

@implementation CvPhotoCamera2


#pragma mark Public

- (void)setDelegate:(id<CvPhotoCameraDelegate2>)newDelegate {
    _delegate = newDelegate;
}

- (id<CvPhotoCameraDelegate2>)delegate {
    return _delegate;
}

#pragma mark - Public interface

- (void)takePicture
{
    if (self.cameraAvailable == NO) {
        return;
    }
    self.cameraAvailable = NO;

    [self.stillImageOutput captureStillImageAsynchronouslyFromConnection:self.videoCaptureConnection
                                                       completionHandler:
     ^(CMSampleBufferRef imageSampleBuffer, NSError *error)
     {
         if (error == nil && imageSampleBuffer != NULL)
         {
             // TODO check
             //             NSNumber* imageOrientation = [UIImage cgImageOrientationForUIDeviceOrientation:currentDeviceOrientation];
             //             CMSetAttachment(imageSampleBuffer, kCGImagePropertyOrientation, imageOrientation, 1);

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
                 self.cameraAvailable = YES;

                 NSLog(@"CvPhotoCamera2 captured image");
                 [self.delegate photoCamera:self capturedImage:newImage];

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
