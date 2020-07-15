/*
 *  cap_ios_video_camera.mm
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
#import <UIKit/UIKit.h>


static CGFloat DegreesToRadians(CGFloat degrees) {return degrees * M_PI / 180;}

#pragma mark - Private Interface




@interface CvVideoCamera () {
    int recordingCountDown;
}

- (void)createVideoDataOutput;
- (void)createVideoFileOutput;


@property (nonatomic, strong) CALayer *customPreviewLayer;
@property (nonatomic, strong) AVCaptureVideoDataOutput *videoDataOutput;

@end



#pragma mark - Implementation



@implementation CvVideoCamera
{
    id<CvVideoCameraDelegate> _delegate;
}



@synthesize grayscaleMode;

@synthesize customPreviewLayer;
@synthesize videoDataOutput;

@synthesize recordVideo;
@synthesize rotateVideo;
//@synthesize videoFileOutput;
@synthesize recordAssetWriterInput;
@synthesize recordPixelBufferAdaptor;
@synthesize recordAssetWriter;

- (void)setDelegate:(id<CvVideoCameraDelegate>)newDelegate {
    _delegate = newDelegate;
}

- (id<CvVideoCameraDelegate>)delegate {
    return _delegate;
}

#pragma mark - Constructors

- (id)initWithParentView:(UIView*)parent;
{
    self = [super initWithParentView:parent];
    if (self) {
        self.useAVCaptureVideoPreviewLayer = NO;
        self.recordVideo = NO;
        self.rotateVideo = NO;
    }
    return self;
}



#pragma mark - Public interface


- (void)start;
{
    if (self.running == YES) {
        return;
    }

    recordingCountDown = 10;
    [super start];

    if (self.recordVideo == YES) {
        NSError* error = nil;
        if ([[NSFileManager defaultManager] fileExistsAtPath:[self videoFileString]]) {
            [[NSFileManager defaultManager] removeItemAtPath:[self videoFileString] error:&error];
        }
        if (error == nil) {
            NSLog(@"[Camera] Delete file %@", [self videoFileString]);
        }
    }
}



- (void)stop;
{
    if (self.running == YES) {
        [super stop];

        [videoDataOutput release];
        if (videoDataOutputQueue) {
            dispatch_release(videoDataOutputQueue);
        }

        if (self.recordVideo == YES) {
            if (self.recordAssetWriter) {
                if (self.recordAssetWriter.status == AVAssetWriterStatusWriting) {
                    [self.recordAssetWriter finishWriting];
                    NSLog(@"[Camera] recording stopped");
                } else {
                    NSLog(@"[Camera] Recording Error: asset writer status is not writing");
                }
                [recordAssetWriter release];
            }

            [recordAssetWriterInput release];
            [recordPixelBufferAdaptor release];
        }

        if (self.customPreviewLayer) {
            [self.customPreviewLayer removeFromSuperlayer];
            self.customPreviewLayer = nil;
        }
    }
}

// TODO fix
- (void)adjustLayoutToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation;
{

    NSLog(@"layout preview layer");
    if (self.parentView != nil) {

        CALayer* layer = self.customPreviewLayer;
        CGRect bounds = self.customPreviewLayer.bounds;
        int rotation_angle = 0;
        bool flip_bounds = false;

        switch (interfaceOrientation) {
            case UIInterfaceOrientationPortrait:
                NSLog(@"to Portrait");
                rotation_angle = 270;
                break;
            case UIInterfaceOrientationPortraitUpsideDown:
                rotation_angle = 90;
                NSLog(@"to UpsideDown");
                break;
            case UIInterfaceOrientationLandscapeLeft:
                rotation_angle = 0;
                NSLog(@"to LandscapeLeft");
                break;
            case UIInterfaceOrientationLandscapeRight:
                rotation_angle = 180;
                NSLog(@"to LandscapeRight");
                break;
            default:
                break; // leave the layer in its last known orientation
        }

        switch (self.defaultAVCaptureVideoOrientation) {
            case AVCaptureVideoOrientationLandscapeRight:
                rotation_angle += 180;
                break;
            case AVCaptureVideoOrientationPortraitUpsideDown:
                rotation_angle += 270;
                break;
            case AVCaptureVideoOrientationPortrait:
                rotation_angle += 90;
            case AVCaptureVideoOrientationLandscapeLeft:
                break;
            default:
                break;
        }
        rotation_angle = rotation_angle % 360;

        if (rotation_angle == 90 || rotation_angle == 270) {
            flip_bounds = true;
        }

        if (flip_bounds) {
            NSLog(@"flip bounds");
            bounds = CGRectMake(0, 0, bounds.size.height, bounds.size.width);
        }

        layer.position = CGPointMake(self.parentView.frame.size.width/2., self.parentView.frame.size.height/2.);
        self.customPreviewLayer.bounds = CGRectMake(0, 0, self.parentView.frame.size.width, self.parentView.frame.size.height);

        layer.affineTransform = CGAffineTransformMakeRotation( DegreesToRadians(rotation_angle) );
        layer.bounds = bounds;
    }

}

// TODO fix
- (void)layoutPreviewLayer;
{
    NSLog(@"layout preview layer");
    if (self.parentView != nil) {

        CALayer* layer = self.customPreviewLayer;
        CGRect bounds = self.customPreviewLayer.bounds;
        int rotation_angle = 0;
        bool flip_bounds = false;

        switch (currentDeviceOrientation) {
            case UIDeviceOrientationPortrait:
                rotation_angle = 270;
                break;
            case UIDeviceOrientationPortraitUpsideDown:
                rotation_angle = 90;
                break;
            case UIDeviceOrientationLandscapeLeft:
                NSLog(@"left");
                rotation_angle = 180;
                break;
            case UIDeviceOrientationLandscapeRight:
                NSLog(@"right");
                rotation_angle = 0;
                break;
            case UIDeviceOrientationFaceUp:
            case UIDeviceOrientationFaceDown:
            default:
                break; // leave the layer in its last known orientation
        }

        switch (self.defaultAVCaptureVideoOrientation) {
            case AVCaptureVideoOrientationLandscapeRight:
                rotation_angle += 180;
                break;
            case AVCaptureVideoOrientationPortraitUpsideDown:
                rotation_angle += 270;
                break;
            case AVCaptureVideoOrientationPortrait:
                rotation_angle += 90;
            case AVCaptureVideoOrientationLandscapeLeft:
                break;
            default:
                break;
        }
        rotation_angle = rotation_angle % 360;

        if (rotation_angle == 90 || rotation_angle == 270) {
            flip_bounds = true;
        }

        if (flip_bounds) {
            NSLog(@"flip bounds");
            bounds = CGRectMake(0, 0, bounds.size.height, bounds.size.width);
        }

        layer.position = CGPointMake(self.parentView.frame.size.width/2., self.parentView.frame.size.height/2.);
        layer.affineTransform = CGAffineTransformMakeRotation( DegreesToRadians(rotation_angle) );
        layer.bounds = bounds;
    }

}

#pragma mark - Private Interface

- (void)createVideoDataOutput;
{
    // Make a video data output
    self.videoDataOutput = [AVCaptureVideoDataOutput new];

    // In grayscale mode we want YUV (YpCbCr 4:2:0) so we can directly access the graylevel intensity values (Y component)
    // In color mode we, BGRA format is used
    OSType format = self.grayscaleMode ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_32BGRA;

    self.videoDataOutput.videoSettings  = [NSDictionary dictionaryWithObject:[NSNumber numberWithUnsignedInt:format]
                                                                      forKey:(id)kCVPixelBufferPixelFormatTypeKey];

    // discard if the data output queue is blocked (as we process the still image)
    [self.videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];

    if ( [self.captureSession canAddOutput:self.videoDataOutput] ) {
        [self.captureSession addOutput:self.videoDataOutput];
    }
    [[self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];


    // set default FPS
    AVCaptureDeviceInput *currentInput = [self.captureSession.inputs objectAtIndex:0];
    AVCaptureDevice *device = currentInput.device;

    NSError *error = nil;
    [device lockForConfiguration:&error];

    float maxRate = ((AVFrameRateRange*) [device.activeFormat.videoSupportedFrameRateRanges objectAtIndex:0]).maxFrameRate;
    if (maxRate > self.defaultFPS - 1 && error == nil) {
        [device setActiveVideoMinFrameDuration:CMTimeMake(1, self.defaultFPS)];
        [device setActiveVideoMaxFrameDuration:CMTimeMake(1, self.defaultFPS)];
        NSLog(@"[Camera] FPS set to %d", self.defaultFPS);
    } else {
        NSLog(@"[Camera] unable to set defaultFPS at %d FPS, max is %f FPS", self.defaultFPS, maxRate);
    }

    if (error != nil) {
        NSLog(@"[Camera] unable to set defaultFPS: %@", error);
    }

    [device unlockForConfiguration];

    // set video mirroring for front camera (more intuitive)
    if ([self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].supportsVideoMirroring) {
        if (self.defaultAVCaptureDevicePosition == AVCaptureDevicePositionFront) {
            [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoMirrored = YES;
        } else {
            [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoMirrored = NO;
        }
    }

    // set default video orientation
    if ([self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].supportsVideoOrientation) {
        [self.videoDataOutput connectionWithMediaType:AVMediaTypeVideo].videoOrientation = self.defaultAVCaptureVideoOrientation;
    }


    // create a custom preview layer
    self.customPreviewLayer = [CALayer layer];
    self.customPreviewLayer.bounds = CGRectMake(0, 0, self.parentView.frame.size.width, self.parentView.frame.size.height);
    self.customPreviewLayer.position = CGPointMake(self.parentView.frame.size.width/2., self.parentView.frame.size.height/2.);
    [self updateOrientation];

    // create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured
    // a serial dispatch queue must be used to guarantee that video frames will be delivered in order
    // see the header doc for setSampleBufferDelegate:queue: for more information
    videoDataOutputQueue = dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
    [self.videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];


    NSLog(@"[Camera] created AVCaptureVideoDataOutput");
}



- (void)createVideoFileOutput;
{
    /* Video File Output in H.264, via AVAsserWriter */
    NSLog(@"Create Video with dimensions %dx%d", self.imageWidth, self.imageHeight);

    NSDictionary *outputSettings
     = [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithInt:self.imageWidth], AVVideoWidthKey,
                                                  [NSNumber numberWithInt:self.imageHeight], AVVideoHeightKey,
                                                  AVVideoCodecH264, AVVideoCodecKey,
                                                  nil
     ];


    self.recordAssetWriterInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo outputSettings:outputSettings];


    int pixelBufferFormat = (self.grayscaleMode == YES) ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_32BGRA;

    self.recordPixelBufferAdaptor =
               [[AVAssetWriterInputPixelBufferAdaptor alloc]
                    initWithAssetWriterInput:self.recordAssetWriterInput
                    sourcePixelBufferAttributes:[NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithInt:pixelBufferFormat], kCVPixelBufferPixelFormatTypeKey, nil]];

    NSError* error = nil;
    NSLog(@"Create AVAssetWriter with url: %@", [self videoFileURL]);
    self.recordAssetWriter = [AVAssetWriter assetWriterWithURL:[self videoFileURL]
                                                      fileType:AVFileTypeMPEG4
                                                         error:&error];
    if (error != nil) {
        NSLog(@"[Camera] Unable to create AVAssetWriter: %@", error);
    }

    [self.recordAssetWriter addInput:self.recordAssetWriterInput];
    self.recordAssetWriterInput.expectsMediaDataInRealTime = YES;

    NSLog(@"[Camera] created AVAssetWriter");
}


- (void)createCaptureOutput;
{
    [self createVideoDataOutput];
    if (self.recordVideo == YES) {
        [self createVideoFileOutput];
    }
}

- (void)createCustomVideoPreview;
{
    [self.parentView.layer addSublayer:self.customPreviewLayer];
}

- (CVPixelBufferRef) pixelBufferFromCGImage: (CGImageRef) image
{

    CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGBitmapContextCompatibilityKey,
                             nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width,
                                          frameSize.height,  kCVPixelFormatType_32ARGB, (CFDictionaryRef) CFBridgingRetain(options),
                                          &pxbuffer);
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);

    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);


    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameSize.width,
                                                 frameSize.height, 8, 4*frameSize.width, rgbColorSpace,
                                                 kCGImageAlphaPremultipliedFirst);

    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image),
                                           CGImageGetHeight(image)), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);

    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);

    return pxbuffer;
}

#pragma mark - Protocol AVCaptureVideoDataOutputSampleBufferDelegate


- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    (void)captureOutput;
    (void)connection;
    auto strongDelegate = self.delegate;
    if (strongDelegate) {

        // convert from Core Media to Core Video
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CVPixelBufferLockBaseAddress(imageBuffer, 0);

        void* bufferAddress;
        size_t width;
        size_t height;
        size_t bytesPerRow;

        CGColorSpaceRef colorSpace;
        CGContextRef context;

        int format_opencv;

        OSType format = CVPixelBufferGetPixelFormatType(imageBuffer);
        if (format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {

            format_opencv = CV_8UC1;

            bufferAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
            width = CVPixelBufferGetWidthOfPlane(imageBuffer, 0);
            height = CVPixelBufferGetHeightOfPlane(imageBuffer, 0);
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, 0);

        } else { // expect kCVPixelFormatType_32BGRA

            format_opencv = CV_8UC4;

            bufferAddress = CVPixelBufferGetBaseAddress(imageBuffer);
            width = CVPixelBufferGetWidth(imageBuffer);
            height = CVPixelBufferGetHeight(imageBuffer);
            bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);

        }

        // delegate image processing to the delegate
        cv::Mat image((int)height, (int)width, format_opencv, bufferAddress, bytesPerRow);

        CGImage* dstImage;

        if ([strongDelegate respondsToSelector:@selector(processImage:)]) {
            [strongDelegate processImage:image];
        }

        // check if matrix data pointer or dimensions were changed by the delegate
        bool iOSimage = false;
        if (height == (size_t)image.rows && width == (size_t)image.cols && format_opencv == image.type() && bufferAddress == image.data && bytesPerRow == image.step) {
            iOSimage = true;
        }


        // (create color space, create graphics context, render buffer)
        CGBitmapInfo bitmapInfo;

        // basically we decide if it's a grayscale, rgb or rgba image
        if (image.channels() == 1) {
            colorSpace = CGColorSpaceCreateDeviceGray();
            bitmapInfo = kCGImageAlphaNone;
        } else if (image.channels() == 3) {
            colorSpace = CGColorSpaceCreateDeviceRGB();
            bitmapInfo = kCGImageAlphaNone;
            if (iOSimage) {
                bitmapInfo |= kCGBitmapByteOrder32Little;
            } else {
                bitmapInfo |= kCGBitmapByteOrder32Big;
            }
        } else {
            colorSpace = CGColorSpaceCreateDeviceRGB();
            bitmapInfo = kCGImageAlphaPremultipliedFirst;
            if (iOSimage) {
                bitmapInfo |= kCGBitmapByteOrder32Little;
            } else {
                bitmapInfo |= kCGBitmapByteOrder32Big;
            }
        }

        if (iOSimage) {
            context = CGBitmapContextCreate(bufferAddress, width, height, 8, bytesPerRow, colorSpace, bitmapInfo);
            dstImage = CGBitmapContextCreateImage(context);
            CGContextRelease(context);
        } else {

            NSData *data = [NSData dataWithBytes:image.data length:image.elemSize()*image.total()];
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

            // Creating CGImage from cv::Mat
            dstImage = CGImageCreate(image.cols,                                 // width
                                     image.rows,                                 // height
                                     8,                                          // bits per component
                                     8 * image.elemSize(),                       // bits per pixel
                                     image.step,                                 // bytesPerRow
                                     colorSpace,                                 // colorspace
                                     bitmapInfo,                                 // bitmap info
                                     provider,                                   // CGDataProviderRef
                                     NULL,                                       // decode
                                     false,                                      // should interpolate
                                     kCGRenderingIntentDefault                   // intent
                                     );

            CGDataProviderRelease(provider);
        }


        // render buffer
        dispatch_sync(dispatch_get_main_queue(), ^{
            self.customPreviewLayer.contents = (__bridge id)dstImage;
        });


        recordingCountDown--;
        if (self.recordVideo == YES && recordingCountDown < 0) {
            lastSampleTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
//			CMTimeShow(lastSampleTime);
            if (self.recordAssetWriter.status != AVAssetWriterStatusWriting) {
                [self.recordAssetWriter startWriting];
                [self.recordAssetWriter startSessionAtSourceTime:lastSampleTime];
                if (self.recordAssetWriter.status != AVAssetWriterStatusWriting) {
                    NSLog(@"[Camera] Recording Error: asset writer status is not writing: %@", self.recordAssetWriter.error);
                    return;
                } else {
                    NSLog(@"[Camera] Video recording started");
                }
            }

            if (self.recordAssetWriterInput.readyForMoreMediaData) {
                CVImageBufferRef pixelBuffer = [self pixelBufferFromCGImage:dstImage];
                if (! [self.recordPixelBufferAdaptor appendPixelBuffer:pixelBuffer
                                                  withPresentationTime:lastSampleTime] ) {
                    NSLog(@"Video Writing Error");
                }
                if (pixelBuffer != nullptr)
                    CVPixelBufferRelease(pixelBuffer);
            }

        }


        // cleanup
        CGImageRelease(dstImage);

        CGColorSpaceRelease(colorSpace);

        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    }
}


- (void)updateOrientation;
{
    if (self.rotateVideo == YES)
    {
        NSLog(@"rotate..");
        self.customPreviewLayer.bounds = CGRectMake(0, 0, self.parentView.frame.size.width, self.parentView.frame.size.height);
        [self layoutPreviewLayer];
    }
}


- (void)saveVideo;
{
    if (self.recordVideo == NO) {
        return;
    }

    UISaveVideoAtPathToSavedPhotosAlbum([self videoFileString], nil, nil, NULL);
}


- (NSURL *)videoFileURL;
{
    NSString *outputPath = [[NSString alloc] initWithFormat:@"%@%@", NSTemporaryDirectory(), @"output.mov"];
    NSURL *outputURL = [NSURL fileURLWithPath:outputPath];
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:outputPath]) {
        NSLog(@"file exists");
    }
    return outputURL;
}



- (NSString *)videoFileString;
{
    NSString *outputPath = [[NSString alloc] initWithFormat:@"%@%@", NSTemporaryDirectory(), @"output.mov"];
    return outputPath;
}

@end
