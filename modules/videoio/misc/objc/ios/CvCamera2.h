//
//  CvCamera2.h
//
//  Created by Giles Payne on 2020/03/11.
//

#import <UIKit/UIKit.h>
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>
#import "CVObjcUtil.h"

@class Mat;

@class CvAbstractCamera2;

CV_EXPORTS @interface CvAbstractCamera2 : NSObject

@property UIDeviceOrientation currentDeviceOrientation;
@property BOOL cameraAvailable;
@property (nonatomic, strong) AVCaptureSession* captureSession;
@property (nonatomic, strong) AVCaptureConnection* videoCaptureConnection;

@property (nonatomic, readonly) BOOL running;
@property (nonatomic, readonly) BOOL captureSessionLoaded;

@property (nonatomic, assign) int defaultFPS;
@property (nonatomic, readonly) AVCaptureVideoPreviewLayer *captureVideoPreviewLayer;
@property (nonatomic, assign) AVCaptureDevicePosition defaultAVCaptureDevicePosition;
@property (nonatomic, assign) AVCaptureVideoOrientation defaultAVCaptureVideoOrientation;
@property (nonatomic, assign) BOOL useAVCaptureVideoPreviewLayer;
@property (nonatomic, strong) NSString *const defaultAVCaptureSessionPreset;
@property (nonatomic, assign) int imageWidth;
@property (nonatomic, assign) int imageHeight;
@property (nonatomic, strong) UIView* parentView;

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
@class CvVideoCamera2;

@protocol CvVideoCameraDelegate2 <NSObject>
- (void)processImage:(Mat*)image;
@end

CV_EXPORTS @interface CvVideoCamera2 : CvAbstractCamera2<AVCaptureVideoDataOutputSampleBufferDelegate>
@property (nonatomic, weak) id<CvVideoCameraDelegate2> delegate;
@property (nonatomic, assign) BOOL grayscaleMode;
@property (nonatomic, assign) BOOL recordVideo;
@property (nonatomic, assign) BOOL rotateVideo;
@property (nonatomic, strong) AVAssetWriterInput* recordAssetWriterInput;
@property (nonatomic, strong) AVAssetWriterInputPixelBufferAdaptor* recordPixelBufferAdaptor;
@property (nonatomic, strong) AVAssetWriter* recordAssetWriter;
- (void)adjustLayoutToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation;
- (void)layoutPreviewLayer;
- (void)saveVideo;
- (NSURL *)videoFileURL;
- (NSString *)videoFileString;
@end

///////////////////////////////// CvPhotoCamera ///////////////////////////////////////////
@class CvPhotoCamera2;

@protocol CvPhotoCameraDelegate2 <NSObject>
- (void)photoCamera:(CvPhotoCamera2*)photoCamera capturedImage:(UIImage*)image;
- (void)photoCameraCancel:(CvPhotoCamera2*)photoCamera;
@end

CV_EXPORTS @interface CvPhotoCamera2 : CvAbstractCamera2<AVCapturePhotoCaptureDelegate>
@property (nonatomic, weak) id<CvPhotoCameraDelegate2> delegate;
- (void)takePicture;
@end
