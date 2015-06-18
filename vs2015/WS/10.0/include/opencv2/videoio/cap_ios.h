/*  For iOS video I/O
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

#import <UIKit/UIKit.h>
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>
#include "opencv2/core.hpp"

//! @addtogroup videoio_ios
//! @{

/////////////////////////////////////// CvAbstractCamera /////////////////////////////////////

@class CvAbstractCamera;

@interface CvAbstractCamera : NSObject
{
    AVCaptureSession* captureSession;
    AVCaptureConnection* videoCaptureConnection;
    AVCaptureVideoPreviewLayer *captureVideoPreviewLayer;

    UIDeviceOrientation currentDeviceOrientation;

    BOOL cameraAvailable;
    BOOL captureSessionLoaded;
    BOOL running;
    BOOL useAVCaptureVideoPreviewLayer;

    AVCaptureDevicePosition defaultAVCaptureDevicePosition;
    AVCaptureVideoOrientation defaultAVCaptureVideoOrientation;
    NSString *const defaultAVCaptureSessionPreset;

    int defaultFPS;

    UIView* parentView;

    int imageWidth;
    int imageHeight;
}

@property (nonatomic, retain) AVCaptureSession* captureSession;
@property (nonatomic, retain) AVCaptureConnection* videoCaptureConnection;

@property (nonatomic, readonly) BOOL running;
@property (nonatomic, readonly) BOOL captureSessionLoaded;

@property (nonatomic, assign) int defaultFPS;
@property (nonatomic, assign) AVCaptureDevicePosition defaultAVCaptureDevicePosition;
@property (nonatomic, assign) AVCaptureVideoOrientation defaultAVCaptureVideoOrientation;
@property (nonatomic, assign) BOOL useAVCaptureVideoPreviewLayer;
@property (nonatomic, strong) NSString *const defaultAVCaptureSessionPreset;

@property (nonatomic, assign) int imageWidth;
@property (nonatomic, assign) int imageHeight;

@property (nonatomic, retain) UIView* parentView;

- (void)start;
- (void)stop;
- (void)switchCameras;

- (id)initWithParentView:(UIView*)parent;

- (void)createCaptureOutput;
- (void)createVideoPreviewLayer;
- (void)updateOrientation;

- (void)lockFocus;
- (void)unlockFocus;
- (void)lockExposure;
- (void)unlockExposure;
- (void)lockBalance;
- (void)unlockBalance;

@end

///////////////////////////////// CvVideoCamera ///////////////////////////////////////////

@class CvVideoCamera;

@protocol CvVideoCameraDelegate <NSObject>

#ifdef __cplusplus
// delegate method for processing image frames
- (void)processImage:(cv::Mat&)image;
#endif

@end

@interface CvVideoCamera : CvAbstractCamera<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    AVCaptureVideoDataOutput *videoDataOutput;

    dispatch_queue_t videoDataOutputQueue;
    CALayer *customPreviewLayer;

    BOOL grayscaleMode;

    BOOL recordVideo;
    BOOL rotateVideo;
    AVAssetWriterInput* recordAssetWriterInput;
    AVAssetWriterInputPixelBufferAdaptor* recordPixelBufferAdaptor;
    AVAssetWriter* recordAssetWriter;

    CMTime lastSampleTime;

}

@property (nonatomic, assign) id<CvVideoCameraDelegate> delegate;
@property (nonatomic, assign) BOOL grayscaleMode;

@property (nonatomic, assign) BOOL recordVideo;
@property (nonatomic, assign) BOOL rotateVideo;
@property (nonatomic, retain) AVAssetWriterInput* recordAssetWriterInput;
@property (nonatomic, retain) AVAssetWriterInputPixelBufferAdaptor* recordPixelBufferAdaptor;
@property (nonatomic, retain) AVAssetWriter* recordAssetWriter;

- (void)adjustLayoutToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation;
- (void)layoutPreviewLayer;
- (void)saveVideo;
- (NSURL *)videoFileURL;


@end

///////////////////////////////// CvPhotoCamera ///////////////////////////////////////////

@class CvPhotoCamera;

@protocol CvPhotoCameraDelegate <NSObject>

- (void)photoCamera:(CvPhotoCamera*)photoCamera capturedImage:(UIImage *)image;
- (void)photoCameraCancel:(CvPhotoCamera*)photoCamera;

@end

@interface CvPhotoCamera : CvAbstractCamera
{
    AVCaptureStillImageOutput *stillImageOutput;
}

@property (nonatomic, assign) id<CvPhotoCameraDelegate> delegate;

- (void)takePicture;

@end

//! @} videoio_ios
