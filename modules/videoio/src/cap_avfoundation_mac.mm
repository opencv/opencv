/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistribution's of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistribution's in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * The name of the copyright holders may not be used to endorse or promote products
// derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the contributor be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*////////////////////////////////////////////////////////////////////////////////////////


#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#import <AVFoundation/AVFoundation.h>

/********************** Declaration of class headers ************************/

/*****************************************************************************
 *
 * CaptureDelegate Declaration.
 *
 * CaptureDelegate is notified on a separate thread by the OS whenever there
 *   is a new frame. When "updateImage" is called from the main thread, it
 *   copies this new frame into an IplImage, but only if this frame has not
 *   been copied before. When "getOutput" is called from the main thread,
 *   it gives the last copied IplImage.
 *
 *****************************************************************************/


@interface CaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    NSCondition *mHasNewFrame;
    CVPixelBufferRef mGrabbedPixels;
    CVImageBufferRef mCurrentImageBuffer;
    IplImage *mDeviceImage;
    uint8_t  *mOutImagedata;
    IplImage *mOutImage;
    size_t    currSize;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;

- (BOOL)grabImageUntilDate: (NSDate *)limit;
- (int)updateImage;
- (IplImage*)getOutput;

@end

/*****************************************************************************
 *
 * CvCaptureCAM Declaration.
 *
 * CvCaptureCAM is the instantiation of a capture source for cameras.
 *
 *****************************************************************************/

class CvCaptureCAM : public CvCapture {
public:
    CvCaptureCAM(int cameraNum = -1) ;
    ~CvCaptureCAM();
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual double getProperty(int property_id) const;
    virtual bool setProperty(int property_id, double value);
    virtual int didStart();


private:
    AVCaptureSession            *mCaptureSession;
    AVCaptureDeviceInput        *mCaptureDeviceInput;
    AVCaptureVideoDataOutput    *mCaptureVideoDataOutput;
    AVCaptureDevice             *mCaptureDevice;
    CaptureDelegate             *mCapture;

    int startCaptureDevice(int cameraNum);
    void stopCaptureDevice();

    void setWidthHeight();
    bool grabFrame(double timeOut);

    int camNum;
    int width;
    int height;
    int settingWidth;
    int settingHeight;
    OSType mInputPixelFormat;

    int started;
};


/*****************************************************************************
 *
 * CvCaptureFile Declaration.
 *
 * CvCaptureFile is the instantiation of a capture source for video files.
 *
 *****************************************************************************/

class CvCaptureFile : public CvCapture {
public:
    CvCaptureFile(const char* filename) ;
    ~CvCaptureFile();
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
    virtual double getProperty(int property_id) const;
    virtual bool setProperty(int property_id, double value);
    virtual int didStart();


private:
    AVAsset                  *mAsset;
    AVAssetTrack             *mAssetTrack;
    AVAssetReader            *mAssetReader;
    AVAssetReaderTrackOutput *mTrackOutput;

    CMSampleBufferRef mCurrentSampleBuffer;
    CVImageBufferRef  mGrabbedPixels;
    IplImage *mDeviceImage;
    uint8_t  *mOutImagedata;
    IplImage *mOutImage;
    size_t    currSize;

    bool setupReadingAt(CMTime position);
    IplImage* retrieveFramePixelBuffer();

    CMTime mFrameTimestamp;
    size_t mFrameNum;
    OSType mInputPixelFormat;

    int started;
};


/*****************************************************************************
 *
 * CvCaptureFile Declaration.
 *
 * CvCaptureFile is the instantiation of a capture source for video files.
 *
 *****************************************************************************/

class CvVideoWriter_AVFoundation_Mac : public CvVideoWriter{
    public:
        CvVideoWriter_AVFoundation_Mac(const char* filename, int fourcc,
                double fps, CvSize frame_size,
                int is_color=1);
        ~CvVideoWriter_AVFoundation_Mac();
        bool writeFrame(const IplImage* image);
    private:
        IplImage* argbimage;

        AVAssetWriter *mMovieWriter;
        AVAssetWriterInput* mMovieWriterInput;
        AVAssetWriterInputPixelBufferAdaptor* mMovieWriterAdaptor;

        NSString* path;
        NSString* codec;
        NSString* fileType;
        double mMovieFPS;
        CvSize movieSize;
        int movieColor;
        unsigned long mFrameNum;
};

/****************** Implementation of interface functions ********************/


CvCapture* cvCreateFileCapture_AVFoundation_Mac(const char* filename) {
    CvCaptureFile *retval = new CvCaptureFile(filename);

    if(retval->didStart())
        return retval;
    delete retval;
    return NULL;
}

CvCapture* cvCreateCameraCapture_AVFoundation_Mac(int index ) {
    CvCapture* retval = new CvCaptureCAM(index);
    if (!((CvCaptureCAM *)retval)->didStart())
        cvReleaseCapture(&retval);
    return retval;
}

CvVideoWriter* cvCreateVideoWriter_AVFoundation_Mac(const char* filename, int fourcc,
                                     double fps, CvSize frame_size,
                                     int is_color) {
    return new CvVideoWriter_AVFoundation_Mac(filename, fourcc, fps, frame_size,is_color);
}

/********************** Implementation of Classes ****************************/

/*****************************************************************************
 *
 * CvCaptureCAM Implementation.
 *
 * CvCaptureCAM is the instantiation of a capture source for cameras.
 *
 *****************************************************************************/

CvCaptureCAM::CvCaptureCAM(int cameraNum) {
    mCaptureSession = nil;
    mCaptureDeviceInput = nil;
    mCaptureVideoDataOutput = nil;
    mCaptureDevice = nil;
    mCapture = nil;

    width = 0;
    height = 0;
    settingWidth = 0;
    settingHeight = 0;

    camNum = cameraNum;

    if ( ! startCaptureDevice(camNum) ) {
        fprintf(stderr, "OpenCV: camera failed to properly initialize!\n");
        started = 0;
    } else {
        started = 1;
    }
}

CvCaptureCAM::~CvCaptureCAM() {
    stopCaptureDevice();
}

int CvCaptureCAM::didStart() {
    return started;
}


bool CvCaptureCAM::grabFrame() {
    return grabFrame(1);
}

bool CvCaptureCAM::grabFrame(double timeOut) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    bool isGrabbed = false;
    NSDate *limit = [NSDate dateWithTimeIntervalSinceNow: timeOut];
    if ( [mCapture grabImageUntilDate: limit] ) {
        [mCapture updateImage];
        isGrabbed = true;
    }

    [localpool drain];
    return isGrabbed;
}

IplImage* CvCaptureCAM::retrieveFrame(int) {
    return [mCapture getOutput];
}

void CvCaptureCAM::stopCaptureDevice() {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    [mCaptureSession stopRunning];

    [mCaptureSession release];
    [mCaptureDeviceInput release];
    [mCaptureDevice release];

    [mCaptureVideoDataOutput release];
    [mCapture release];

    [localpool drain];
}

int CvCaptureCAM::startCaptureDevice(int cameraNum) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    // get capture device
    NSArray *devices = [AVCaptureDevice devicesWithMediaType: AVMediaTypeVideo];

    if ( devices.count == 0 ) {
        fprintf(stderr, "OpenCV: AVFoundation didn't find any attached Video Input Devices!\n");
        [localpool drain];
        return 0;
    }

    if ( cameraNum < 0 || devices.count <= NSUInteger(cameraNum) ) {
        fprintf(stderr, "OpenCV: out device of bound (0-%ld): %d\n", devices.count-1, cameraNum);
        [localpool drain];
        return 0;
    }

    mCaptureDevice = devices[cameraNum];

    if ( ! mCaptureDevice ) {
        fprintf(stderr, "OpenCV: device %d not able to use.\n", cameraNum);
        [localpool drain];
        return 0;
    }

    // get input device
    NSError *error = nil;
    mCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice: mCaptureDevice
                                                                 error: &error];
    if ( error ) {
        fprintf(stderr, "OpenCV: error in [AVCaptureDeviceInput initWithDevice:error:]\n");
        NSLog(@"OpenCV: %@", error.localizedDescription);
        [localpool drain];
        return 0;
    }

    // create output
    mCapture = [[CaptureDelegate alloc] init];
    mCaptureVideoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    dispatch_queue_t queue = dispatch_queue_create("cameraQueue", DISPATCH_QUEUE_SERIAL);
    [mCaptureVideoDataOutput setSampleBufferDelegate: mCapture queue: queue];
    dispatch_release(queue);

    OSType pixelFormat = kCVPixelFormatType_32BGRA;
    //OSType pixelFormat = kCVPixelFormatType_422YpCbCr8;
    NSDictionary *pixelBufferOptions;
    if (width > 0 && height > 0) {
        pixelBufferOptions =
            @{
                (id)kCVPixelBufferWidthKey:  @(1.0*width),
                (id)kCVPixelBufferHeightKey: @(1.0*height),
                (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
            };
    } else {
        pixelBufferOptions =
            @{
                (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
            };
    }
    mCaptureVideoDataOutput.videoSettings = pixelBufferOptions;
    mCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames = YES;

    // create session
    mCaptureSession = [[AVCaptureSession alloc] init];
    mCaptureSession.sessionPreset = AVCaptureSessionPresetMedium;
    [mCaptureSession addInput: mCaptureDeviceInput];
    [mCaptureSession addOutput: mCaptureVideoDataOutput];

    [mCaptureSession startRunning];

    // flush old position image
    grabFrame(1);

    [localpool drain];
    return 1;
}

void CvCaptureCAM::setWidthHeight() {
    NSMutableDictionary *pixelBufferOptions = [mCaptureVideoDataOutput.videoSettings mutableCopy];

    while ( true ) {
        // auto matching
        pixelBufferOptions[(id)kCVPixelBufferWidthKey]  = @(1.0*width);
        pixelBufferOptions[(id)kCVPixelBufferHeightKey] = @(1.0*height);
        mCaptureVideoDataOutput.videoSettings = pixelBufferOptions;

        // compare matched size and my options
        CMFormatDescriptionRef format = mCaptureDevice.activeFormat.formatDescription;
        CMVideoDimensions deviceSize = CMVideoFormatDescriptionGetDimensions(format);
        if ( deviceSize.width == width && deviceSize.height == height ) {
            break;
        }

        // fit my options to matched size
        width = deviceSize.width;
        height = deviceSize.height;
    } 

    // flush old size image
    grabFrame(1);

    [pixelBufferOptions release];
}


double CvCaptureCAM::getProperty(int property_id) const{
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    CMFormatDescriptionRef format = mCaptureDevice.activeFormat.formatDescription;
    CMVideoDimensions s1 = CMVideoFormatDescriptionGetDimensions(format);
    double retval = 0;

    switch (property_id) {
        case CV_CAP_PROP_FRAME_WIDTH:
            retval = s1.width;
            break;
        case CV_CAP_PROP_FRAME_HEIGHT:
            retval = s1.height;
            break;
        case CV_CAP_PROP_FPS:
            {
                CMTime frameDuration = mCaptureDevice.activeVideoMaxFrameDuration;
                retval = frameDuration.timescale / double(frameDuration.value);
            }
            break;
        case CV_CAP_PROP_FORMAT:
            retval = CV_8UC3;
            break;
        default:
            break;
    }

    [localpool drain];
    return retval;
}

bool CvCaptureCAM::setProperty(int property_id, double value) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    bool isSucceeded = false;

    switch (property_id) {
        case CV_CAP_PROP_FRAME_WIDTH:
            width = value;
            settingWidth = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth = 0;
                settingHeight = 0;
            }
            isSucceeded = true;
            break;
        case CV_CAP_PROP_FRAME_HEIGHT:
            height = value;
            settingHeight = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth = 0;
                settingHeight = 0;
            }
            isSucceeded = true;
            break;
        case CV_CAP_PROP_FPS:
            if ( [mCaptureDevice lockForConfiguration: NULL] ) {
                NSArray * ranges = mCaptureDevice.activeFormat.videoSupportedFrameRateRanges;
                AVFrameRateRange *matchedRange = ranges[0];
                double minDiff = fabs(matchedRange.maxFrameRate - value);
                for ( AVFrameRateRange *range in ranges ) {
                    double diff = fabs(range.maxFrameRate - value);
                    if ( diff < minDiff ) {
                        minDiff = diff;
                        matchedRange = range;
                    }
                }
                mCaptureDevice.activeVideoMinFrameDuration = matchedRange.minFrameDuration;
                mCaptureDevice.activeVideoMaxFrameDuration = matchedRange.minFrameDuration;
                isSucceeded = true;
                [mCaptureDevice unlockForConfiguration];
            }
            break;
        default:
            break;
    }

    [localpool drain];
    return isSucceeded;
}


/*****************************************************************************
 *
 * CaptureDelegate Implementation.
 *
 * CaptureDelegate is notified on a separate thread by the OS whenever there
 *   is a new frame. When "updateImage" is called from the main thread, it
 *   copies this new frame into an IplImage, but only if this frame has not
 *   been copied before. When "getOutput" is called from the main thread,
 *   it gives the last copied IplImage.
 *
 *****************************************************************************/


@implementation CaptureDelegate

- (id)init {
    [super init];
    mHasNewFrame = [[NSCondition alloc] init];
    mCurrentImageBuffer = NULL;
    mGrabbedPixels = NULL;
    mDeviceImage = NULL;
    mOutImagedata = NULL;
    mOutImage = NULL;
    currSize = 0;
    return self;
}

-(void)dealloc {
    free(mOutImagedata);
    cvReleaseImage(&mOutImage);
    cvReleaseImage(&mDeviceImage);
    CVBufferRelease(mCurrentImageBuffer);
    CVBufferRelease(mGrabbedPixels);
    [mHasNewFrame release];
    [super dealloc];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    (void)captureOutput;
    (void)sampleBuffer;
    (void)connection;

    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVBufferRetain(imageBuffer);

    [mHasNewFrame lock];

    CVBufferRelease(mCurrentImageBuffer);
    mCurrentImageBuffer = imageBuffer;
    [mHasNewFrame signal];

    [mHasNewFrame unlock];

}

-(IplImage*) getOutput {
    return mOutImage;
}

-(BOOL) grabImageUntilDate: (NSDate *)limit {
    BOOL isGrabbed = NO;
    [mHasNewFrame lock];

    if ( mGrabbedPixels ) {
        CVBufferRelease(mGrabbedPixels);
    }
    if ( [mHasNewFrame waitUntilDate: limit] ) {
        isGrabbed = YES;
        mGrabbedPixels = CVBufferRetain(mCurrentImageBuffer);
    }

    [mHasNewFrame unlock];
    return isGrabbed;
}

-(int) updateImage {
    if ( ! mGrabbedPixels ) {
        return 0;
    }

    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    void *baseaddress = CVPixelBufferGetBaseAddress(mGrabbedPixels);

    size_t width = CVPixelBufferGetWidth(mGrabbedPixels);
    size_t height = CVPixelBufferGetHeight(mGrabbedPixels);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);

    if ( rowBytes == 0 ) {
        fprintf(stderr, "OpenCV: error: rowBytes == 0\n");
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    if ( currSize != width*3*height ) {
        currSize = width*3*height;
        free(mOutImagedata);
        mOutImagedata = reinterpret_cast<uint8_t*>(malloc(currSize));
    }

    if (mOutImage == NULL) {
        mOutImage = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 3);
    }
    mOutImage->width = int(width);
    mOutImage->height = int(height);
    mOutImage->nChannels = 3;
    mOutImage->depth = IPL_DEPTH_8U;
    mOutImage->widthStep = int(width*3);
    mOutImage->imageData = reinterpret_cast<char *>(mOutImagedata);
    mOutImage->imageSize = int(currSize);

    if ( pixelFormat == kCVPixelFormatType_32BGRA ) {
        if (mDeviceImage == NULL) {
            mDeviceImage = cvCreateImageHeader(cvSize(int(width),int(height)), IPL_DEPTH_8U, 4);
        }
        mDeviceImage->width = int(width);
        mDeviceImage->height = int(height);
        mDeviceImage->nChannels = 4;
        mDeviceImage->depth = IPL_DEPTH_8U;
        mDeviceImage->widthStep = int(rowBytes);
        mDeviceImage->imageData = reinterpret_cast<char *>(baseaddress);
        mDeviceImage->imageSize = int(rowBytes*height);

        cvCvtColor(mDeviceImage, mOutImage, CV_BGRA2BGR);
    } else if ( pixelFormat == kCVPixelFormatType_422YpCbCr8 ) {
        if ( currSize != width*3*height ) {
            currSize = width*3*height;
            free(mOutImagedata);
            mOutImagedata = reinterpret_cast<uint8_t*>(malloc(currSize));
        }

        if (mDeviceImage == NULL) {
            mDeviceImage = cvCreateImageHeader(cvSize(int(width),int(height)), IPL_DEPTH_8U, 2);
        }
        mDeviceImage->width = int(width);
        mDeviceImage->height = int(height);
        mDeviceImage->nChannels = 2;
        mDeviceImage->depth = IPL_DEPTH_8U;
        mDeviceImage->widthStep = int(rowBytes);
        mDeviceImage->imageData = reinterpret_cast<char *>(baseaddress);
        mDeviceImage->imageSize = int(rowBytes*height);

        cvCvtColor(mDeviceImage, mOutImage, CV_YUV2BGR_UYVY);
    } else {
        fprintf(stderr, "OpenCV: unknown pixel format 0x%08X\n", pixelFormat);
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    mGrabbedPixels = NULL;

    return 1;
}

@end


/*****************************************************************************
 *
 * CvCaptureFile Implementation.
 *
 * CvCaptureFile is the instantiation of a capture source for video files.
 *
 *****************************************************************************/

CvCaptureFile::CvCaptureFile(const char* filename) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    mAsset = nil;
    mAssetTrack = nil;
    mAssetReader = nil;
    mTrackOutput = nil;
    mDeviceImage = NULL;
    mOutImage = NULL;
    mOutImagedata = NULL;
    currSize = 0;
    mCurrentSampleBuffer = NULL;
    mGrabbedPixels = NULL;
    mFrameTimestamp = kCMTimeZero;
    mFrameNum = 0;

    started = 0;

    mAsset = [[AVAsset assetWithURL:[NSURL fileURLWithPath: @(filename)]] retain];

    if ( mAsset == nil ) {
        fprintf(stderr, "OpenCV: Couldn't read movie file \"%s\"\n", filename);
        [localpool drain];
        started = 0;
        return;
    }

    mAssetTrack = [[mAsset tracksWithMediaType: AVMediaTypeVideo][0] retain];

    if ( ! setupReadingAt(kCMTimeZero) ) {
        fprintf(stderr, "OpenCV: Couldn't read movie file \"%s\"\n", filename);
        [localpool drain];
        started = 0;
        return;
    }

    started = 1;
    [localpool drain];
}

CvCaptureFile::~CvCaptureFile() {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    free(mOutImagedata);
    cvReleaseImage(&mOutImage);
    cvReleaseImage(&mDeviceImage);
    [mAssetReader release];
    [mTrackOutput release];
    [mAssetTrack release];
    [mAsset release];
    CVBufferRelease(mGrabbedPixels);
    if ( mCurrentSampleBuffer ) {
        CFRelease(mCurrentSampleBuffer);
    }

    [localpool drain];
}

bool CvCaptureFile::setupReadingAt(CMTime position) {
    if (mAssetReader) {
        if (mAssetReader.status == AVAssetReaderStatusReading) {
            [mAssetReader cancelReading];
        }
        [mAssetReader release];
        mAssetReader = nil;
    }
    if (mTrackOutput) {
        [mTrackOutput release];
        mTrackOutput = nil;
    }

    OSType pixelFormat = kCVPixelFormatType_32BGRA;
    //OSType pixelFormat = kCVPixelFormatType_422YpCbCr8;
    NSDictionary *settings =
        @{
            (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
        };
    mTrackOutput = [[AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack: mAssetTrack
                                                               outputSettings: settings] retain];

    NSError *error = nil;
    mAssetReader = [[AVAssetReader assetReaderWithAsset: mAsset
                                                  error: &error] retain];
    if ( error ) {
        fprintf(stderr, "OpenCV: error in [AVAssetReader assetReaderWithAsset:error:]\n");
        NSLog(@"OpenCV: %@", error.localizedDescription);
        return false;
    }

    mAssetReader.timeRange = CMTimeRangeMake(position, kCMTimePositiveInfinity);
    mFrameTimestamp = position;
    mFrameNum = round((mFrameTimestamp.value * mAssetTrack.nominalFrameRate) / double(mFrameTimestamp.timescale));
    [mAssetReader addOutput: mTrackOutput];
    [mAssetReader startReading];

    return true; 
}

int CvCaptureFile::didStart() {
    return started;
}

bool CvCaptureFile::grabFrame() {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    CVBufferRelease(mGrabbedPixels);
    if ( mCurrentSampleBuffer ) {
        CFRelease(mCurrentSampleBuffer);
    }
    mCurrentSampleBuffer = [mTrackOutput copyNextSampleBuffer];
    mGrabbedPixels = CMSampleBufferGetImageBuffer(mCurrentSampleBuffer);
    CVBufferRetain(mGrabbedPixels);
    mFrameTimestamp = CMSampleBufferGetOutputPresentationTimeStamp(mCurrentSampleBuffer);
    mFrameNum++;

    bool isReading = (mAssetReader.status == AVAssetReaderStatusReading);
    [localpool drain];
    return isReading;
}


IplImage* CvCaptureFile::retrieveFramePixelBuffer() {
    if ( ! mGrabbedPixels ) {
        return 0;
    }

    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    void *baseaddress = CVPixelBufferGetBaseAddress(mGrabbedPixels);

    size_t width = CVPixelBufferGetWidth(mGrabbedPixels);
    size_t height = CVPixelBufferGetHeight(mGrabbedPixels);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);

    if ( rowBytes == 0 ) {
        fprintf(stderr, "OpenCV: error: rowBytes == 0\n");
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    if ( currSize != width*3*height ) {
        currSize = width*3*height;
        free(mOutImagedata);
        mOutImagedata = reinterpret_cast<uint8_t*>(malloc(currSize));
    }

    if (mOutImage == NULL) {
        mOutImage = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 3);
    }
    mOutImage->width = int(width);
    mOutImage->height = int(height);
    mOutImage->nChannels = 3;
    mOutImage->depth = IPL_DEPTH_8U;
    mOutImage->widthStep = int(width*3);
    mOutImage->imageData = reinterpret_cast<char *>(mOutImagedata);
    mOutImage->imageSize = int(currSize);

    if ( pixelFormat == kCVPixelFormatType_32BGRA ) {
        if (mDeviceImage == NULL) {
            mDeviceImage = cvCreateImageHeader(cvSize(int(width),int(height)), IPL_DEPTH_8U, 4);
        }
        mDeviceImage->width = int(width);
        mDeviceImage->height = int(height);
        mDeviceImage->nChannels = 4;
        mDeviceImage->depth = IPL_DEPTH_8U;
        mDeviceImage->widthStep = int(rowBytes);
        mDeviceImage->imageData = reinterpret_cast<char *>(baseaddress);
        mDeviceImage->imageSize = int(rowBytes*height);

        cvCvtColor(mDeviceImage, mOutImage, CV_BGRA2BGR);
    } else if ( pixelFormat == kCVPixelFormatType_422YpCbCr8 ) {
        if ( currSize != width*3*height ) {
            currSize = width*3*height;
            free(mOutImagedata);
            mOutImagedata = reinterpret_cast<uint8_t*>(malloc(currSize));
        }

        if (mDeviceImage == NULL) {
            mDeviceImage = cvCreateImageHeader(cvSize(int(width),int(height)), IPL_DEPTH_8U, 2);
        }
        mDeviceImage->width = int(width);
        mDeviceImage->height = int(height);
        mDeviceImage->nChannels = 2;
        mDeviceImage->depth = IPL_DEPTH_8U;
        mDeviceImage->widthStep = int(rowBytes);
        mDeviceImage->imageData = reinterpret_cast<char *>(baseaddress);
        mDeviceImage->imageSize = int(rowBytes*height);

        cvCvtColor(mDeviceImage, mOutImage, CV_YUV2BGR_UYVY);
    } else {
        fprintf(stderr, "OpenCV: unknown pixel format 0x%08X\n", pixelFormat);
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);

    [localpool drain];

    return mOutImage;
}


IplImage* CvCaptureFile::retrieveFrame(int) {
    return retrieveFramePixelBuffer();
}

double CvCaptureFile::getProperty(int property_id) const{
    if (mAsset == nil) return 0;

    CMTime t;

    switch (property_id) {
        case CV_CAP_PROP_POS_MSEC:
            return mFrameTimestamp.value * 1000.0 / mFrameTimestamp.timescale;
        case CV_CAP_PROP_POS_FRAMES:
            return  mFrameNum;
        case CV_CAP_PROP_POS_AVI_RATIO:
            t = [mAsset duration];
            return (mFrameTimestamp.value * t.timescale) / double(mFrameTimestamp.timescale * t.value);
        case CV_CAP_PROP_FRAME_WIDTH:
            return mAssetTrack.naturalSize.width;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return mAssetTrack.naturalSize.height;
        case CV_CAP_PROP_FPS:
            return mAssetTrack.nominalFrameRate;
        case CV_CAP_PROP_FRAME_COUNT:
            t = [mAsset duration];
            return round((t.value * mAssetTrack.nominalFrameRate) / double(t.timescale));
        case CV_CAP_PROP_FORMAT:
            return CV_8UC3;
        default:
            break;
    }

    return 0;
}

bool CvCaptureFile::setProperty(int property_id, double value) {
    if (mAsset == nil) return false;

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    bool retval = false;
    CMTime t;

    switch (property_id) {
        case CV_CAP_PROP_POS_MSEC:
            t = mAsset.duration;
            t.value = value * t.timescale / 1000;
            setupReadingAt(t);
            retval = true;
            break;
        case CV_CAP_PROP_POS_FRAMES:
            setupReadingAt(CMTimeMake(value, mAssetTrack.nominalFrameRate));
            retval = true;
            break;
        case CV_CAP_PROP_POS_AVI_RATIO:
            t = mAsset.duration;
            t.value = round(t.value * value);
            setupReadingAt(t);
            retval = true;
            break;
        default:
            break;
    }

    [localpool drain];
    return retval;
}


/*****************************************************************************
 *
 * CvVideoWriter Implementation.
 *
 * CvVideoWriter is the instantiation of a video output class
 *
 *****************************************************************************/


CvVideoWriter_AVFoundation_Mac::CvVideoWriter_AVFoundation_Mac(const char* filename, int fourcc,
        double fps, CvSize frame_size,
        int is_color) {

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];


    mFrameNum = 0;
    mMovieFPS = fps;
    movieSize = frame_size;
    movieColor = is_color;
    argbimage = cvCreateImage(movieSize, IPL_DEPTH_8U, 4);
    path = [[[NSString stringWithCString:filename encoding:NSASCIIStringEncoding] stringByExpandingTildeInPath] retain];


    /*
         AVFileTypeQuickTimeMovie
         UTI for the QuickTime movie file format.
         The value of this UTI is com.apple.quicktime-movie. Files are identified with the .mov and .qt extensions.

         AVFileTypeMPEG4
         UTI for the MPEG-4 file format.
         The value of this UTI is public.mpeg-4. Files are identified with the .mp4 extension.

         AVFileTypeAppleM4V
         UTI for the iTunes video file format.
         The value of this UTI is com.apple.mpeg-4-video. Files are identified with the .m4v extension.

         AVFileType3GPP
         UTI for the 3GPP file format.
         The value of this UTI is public.3gpp. Files are identified with the .3gp, .3gpp, and .sdv extensions.
     */

    NSString *fileExt =[[[path pathExtension] lowercaseString] copy];
    if ([fileExt isEqualToString:@"mov"] || [fileExt isEqualToString:@"qt"]){
        fileType = [AVFileTypeQuickTimeMovie copy];
    }else if ([fileExt isEqualToString:@"mp4"]){
        fileType = [AVFileTypeMPEG4 copy];
    }else if ([fileExt isEqualToString:@"m4v"]){
        fileType = [AVFileTypeAppleM4V copy];
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
    }else if ([fileExt isEqualToString:@"3gp"] || [fileExt isEqualToString:@"3gpp"] || [fileExt isEqualToString:@"sdv"]  ){
        fileType = [AVFileType3GPP copy];
#endif
    } else{
        fileType = [AVFileTypeMPEG4 copy];  //default mp4
    }
    [fileExt release];

    char cc[5];
    cc[0] = fourcc & 255;
    cc[1] = (fourcc >> 8) & 255;
    cc[2] = (fourcc >> 16) & 255;
    cc[3] = (fourcc >> 24) & 255;
    cc[4] = 0;
    int cc2 = CV_FOURCC(cc[0], cc[1], cc[2], cc[3]);
    if (cc2!=fourcc) {
        fprintf(stderr, "OpenCV: Didn't properly encode FourCC. Expected 0x%08X but got 0x%08X.\n", fourcc, cc2);
        //exception;
    }

    // Two codec supported AVVideoCodecH264 AVVideoCodecJPEG
    // On iPhone 3G H264 is not supported.
    if (fourcc == CV_FOURCC('J','P','E','G') || fourcc == CV_FOURCC('j','p','e','g') ||
            fourcc == CV_FOURCC('M','J','P','G') || fourcc == CV_FOURCC('m','j','p','g') ){
        codec = [AVVideoCodecJPEG copy]; // Use JPEG codec if specified, otherwise H264
    }else if(fourcc == CV_FOURCC('H','2','6','4') || fourcc == CV_FOURCC('a','v','c','1')){
            codec = [AVVideoCodecH264 copy];
    }else{
        codec = [AVVideoCodecH264 copy]; // default canonical H264.

    }

    //NSLog(@"Path: %@", path);

    NSError *error = nil;


    // Make sure the file does not already exist. Necessary to overwirte??
    /*
    NSFileManager *fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:path]){
        [fileManager removeItemAtPath:path error:&error];
    }
    */

    // Wire the writer:
    // Supported file types:
    //      AVFileTypeQuickTimeMovie AVFileTypeMPEG4 AVFileTypeAppleM4V AVFileType3GPP

    mMovieWriter = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:path]
        fileType:fileType
        error:&error];
    //NSParameterAssert(mMovieWriter);

    NSDictionary *videoSettings = [NSDictionary dictionaryWithObjectsAndKeys:
        codec, AVVideoCodecKey,
        [NSNumber numberWithInt:movieSize.width], AVVideoWidthKey,
        [NSNumber numberWithInt:movieSize.height], AVVideoHeightKey,
        nil];

    mMovieWriterInput = [[AVAssetWriterInput
        assetWriterInputWithMediaType:AVMediaTypeVideo
        outputSettings:videoSettings] retain];

    //NSParameterAssert(mMovieWriterInput);
    //NSParameterAssert([mMovieWriter canAddInput:mMovieWriterInput]);

    [mMovieWriter addInput:mMovieWriterInput];

    mMovieWriterAdaptor = [[AVAssetWriterInputPixelBufferAdaptor alloc] initWithAssetWriterInput:mMovieWriterInput sourcePixelBufferAttributes:nil];


    //Start a session:
    [mMovieWriter startWriting];
    [mMovieWriter startSessionAtSourceTime:kCMTimeZero];


    if(mMovieWriter.status == AVAssetWriterStatusFailed){
        NSLog(@"%@", [mMovieWriter.error localizedDescription]);
        // TODO: error handling, cleanup. Throw execption?
        // return;
    }

    [localpool drain];
}


CvVideoWriter_AVFoundation_Mac::~CvVideoWriter_AVFoundation_Mac() {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    [mMovieWriterInput markAsFinished];
    [mMovieWriter finishWriting];
    [mMovieWriter release];
    [mMovieWriterInput release];
    [mMovieWriterAdaptor release];
    [path release];
    [codec release];
    [fileType release];
    cvReleaseImage(&argbimage);

    [localpool drain];

}

bool CvVideoWriter_AVFoundation_Mac::writeFrame(const IplImage* iplimage) {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    // writer status check
    if (![mMovieWriterInput isReadyForMoreMediaData] || mMovieWriter.status !=  AVAssetWriterStatusWriting ) {
        NSLog(@"[mMovieWriterInput isReadyForMoreMediaData] Not ready for media data or ...");
        NSLog(@"mMovieWriter.status: %d. Error: %@", (int)mMovieWriter.status, [mMovieWriter.error localizedDescription]);
        [localpool drain];
        return false;
    }

    BOOL success = FALSE;

    if (iplimage->height!=movieSize.height || iplimage->width!=movieSize.width){
        fprintf(stderr, "OpenCV: Frame size does not match video size.\n");
        [localpool drain];
        return false;
    }

    if (movieColor) {
        //assert(iplimage->nChannels == 3);
        cvCvtColor(iplimage, argbimage, CV_BGR2BGRA);
    }else{
        //assert(iplimage->nChannels == 1);
        cvCvtColor(iplimage, argbimage, CV_GRAY2BGRA);
    }
    //IplImage -> CGImage conversion
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSData *nsData = [NSData dataWithBytes:argbimage->imageData length:argbimage->imageSize];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)nsData);
    CGImageRef cgImage = CGImageCreate(argbimage->width, argbimage->height,
            argbimage->depth, argbimage->depth * argbimage->nChannels, argbimage->widthStep,
            colorSpace, kCGImageAlphaLast|kCGBitmapByteOrderDefault,
            provider, NULL, false, kCGRenderingIntentDefault);

    //CGImage -> CVPixelBufferRef coversion
    CVPixelBufferRef pixelBuffer = NULL;
    CFDataRef cfData = CGDataProviderCopyData(CGImageGetDataProvider(cgImage));
    int status = CVPixelBufferCreateWithBytes(NULL,
            movieSize.width,
            movieSize.height,
            kCVPixelFormatType_32BGRA,
            (void*)CFDataGetBytePtr(cfData),
            CGImageGetBytesPerRow(cgImage),
            NULL,
            0,
            NULL,
            &pixelBuffer);
    if(status == kCVReturnSuccess){
        success = [mMovieWriterAdaptor appendPixelBuffer:pixelBuffer
            withPresentationTime:CMTimeMake(mFrameNum, mMovieFPS)];
    }

    //cleanup
    CFRelease(cfData);
    CVPixelBufferRelease(pixelBuffer);
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    [localpool drain];

    if (success) {
        mFrameNum ++;
        //NSLog(@"Frame #%d", mFrameNum);
        return true;
    }else{
        NSLog(@"Frame appendPixelBuffer failed.");
        return false;
    }

}
