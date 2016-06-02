/*
 *  cap_avfoundation.mm
 *  For iOS video I/O
 *  by Xiaochao Yang on 06/15/11 modified from
 *  cap_qtkit.mm for Nicholas Butko for Mac OS version.
 *  Copyright 2011. All rights reserved.
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

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#import <AVFoundation/AVFoundation.h>
#import <Foundation/NSException.h>


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

#define DISABLE_AUTO_RESTART 999

@interface CaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    int newFrame;
    CVImageBufferRef  mCurrentImageBuffer;
    char* imagedata;
    IplImage* image;
    char* bgr_imagedata;
    IplImage* bgr_image;
    IplImage* bgr_image_r90;
    size_t currSize;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
fromConnection:(AVCaptureConnection *)connection;


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
        virtual IplImage* queryFrame();
        virtual double getProperty(int property_id) const;
        virtual bool setProperty(int property_id, double value);
        virtual int didStart();

    private:
        AVCaptureSession            *mCaptureSession;
        AVCaptureDeviceInput        *mCaptureDeviceInput;
        AVCaptureVideoDataOutput    *mCaptureDecompressedVideoOutput;
        AVCaptureDevice 						*mCaptureDevice;
        CaptureDelegate							*capture;

        int startCaptureDevice(int cameraNum);
        void stopCaptureDevice();

        void setWidthHeight();
        bool grabFrame(double timeOut);

        int camNum;
        int width;
        int height;
        int settingWidth;
        int settingHeight;
        int started;
        int disableAutoRestart;
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
        virtual IplImage* queryFrame();
        virtual double getProperty(int property_id) const;
        virtual bool setProperty(int property_id, double value);
        virtual int didStart();

    private:

        AVAssetReader *mMovieReader;
        char* imagedata;
        IplImage* image;
        char* bgr_imagedata;
        IplImage* bgr_image;
        size_t currSize;

        IplImage* retrieveFramePixelBuffer();
        double getFPS();

        int movieWidth;
        int movieHeight;
        double movieFPS;
        double currentFPS;
        double movieDuration;
        int changedPos;

        int started;
};


/*****************************************************************************
 *
 * CvCaptureFile Declaration.
 *
 * CvCaptureFile is the instantiation of a capture source for video files.
 *
 *****************************************************************************/

class CvVideoWriter_AVFoundation : public CvVideoWriter{
    public:
        CvVideoWriter_AVFoundation(const char* filename, int fourcc,
                double fps, CvSize frame_size,
                int is_color=1);
        ~CvVideoWriter_AVFoundation();
        bool writeFrame(const IplImage* image);
    private:
        IplImage* argbimage;

        AVAssetWriter *mMovieWriter;
        AVAssetWriterInput* mMovieWriterInput;
        AVAssetWriterInputPixelBufferAdaptor* mMovieWriterAdaptor;

        NSString* path;
        NSString* codec;
        NSString* fileType;
        double movieFPS;
        CvSize movieSize;
        int movieColor;
        unsigned long frameCount;
};


/****************** Implementation of interface functions ********************/


CvCapture* cvCreateFileCapture_AVFoundation(const char* filename) {
    CvCaptureFile *retval = new CvCaptureFile(filename);

    if(retval->didStart())
        return retval;
    delete retval;
    return NULL;
}

CvCapture* cvCreateCameraCapture_AVFoundation(int index ) {

    CvCapture* retval = new CvCaptureCAM(index);
    if (!((CvCaptureCAM *)retval)->didStart())
        cvReleaseCapture(&retval);
    return retval;

}

CvVideoWriter* cvCreateVideoWriter_AVFoundation(const char* filename, int fourcc,
        double fps, CvSize frame_size,
        int is_color) {
    return new CvVideoWriter_AVFoundation(filename, fourcc, fps, frame_size,is_color);
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
    mCaptureDecompressedVideoOutput = nil;
    capture = nil;

    width = 0;
    height = 0;
    settingWidth = 0;
    settingHeight = 0;
    disableAutoRestart = 0;

    camNum = cameraNum;

    if (!startCaptureDevice(camNum)) {
        std::cout << "Warning, camera failed to properly initialize!" << std::endl;
        started = 0;
    } else {
        started = 1;
    }

}

CvCaptureCAM::~CvCaptureCAM() {
    stopCaptureDevice();
    //cout << "Cleaned up camera." << endl;
}

int CvCaptureCAM::didStart() {
    return started;
}


bool CvCaptureCAM::grabFrame() {
    return grabFrame(5);
}

bool CvCaptureCAM::grabFrame(double timeOut) {

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
    double sleepTime = 0.005;
    double total = 0;

    NSDate *loopUntil = [NSDate dateWithTimeIntervalSinceNow:sleepTime];
    while (![capture updateImage] && (total += sleepTime)<=timeOut &&
            [[NSRunLoop currentRunLoop] runMode: NSDefaultRunLoopMode
            beforeDate:loopUntil])
        loopUntil = [NSDate dateWithTimeIntervalSinceNow:sleepTime];

    [localpool drain];

    return total <= timeOut;
}

IplImage* CvCaptureCAM::retrieveFrame(int) {
    return [capture getOutput];
}

IplImage* CvCaptureCAM::queryFrame() {
    while (!grabFrame()) {
        std::cout << "WARNING: Couldn't grab new frame from camera!!!" << std::endl;
        /*
             cout << "Attempting to restart camera; set capture property DISABLE_AUTO_RESTART to disable." << endl;
             stopCaptureDevice();
             startCaptureDevice(camNum);
         */
    }
    return retrieveFrame(0);
}

void CvCaptureCAM::stopCaptureDevice() {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    [mCaptureSession stopRunning];

    [mCaptureSession release];
    [mCaptureDeviceInput release];

    [mCaptureDecompressedVideoOutput release];
    [capture release];
    [localpool drain];

}

int CvCaptureCAM::startCaptureDevice(int cameraNum) {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    capture = [[CaptureDelegate alloc] init];

    AVCaptureDevice *device;
    NSArray* devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    if ([devices count] == 0) {
        std::cout << "AV Foundation didn't find any attached Video Input Devices!" << std::endl;
        [localpool drain];
        return 0;
    }

    if (cameraNum >= 0) {
        camNum = cameraNum % [devices count];
        if (camNum != cameraNum) {
            std::cout << "Warning: Max Camera Num is " << [devices count]-1 << "; Using camera " << camNum << std::endl;
        }
        device = [devices objectAtIndex:camNum];
    } else {
        device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo]  ;
    }
    mCaptureDevice = device;
    //int success;
    NSError* error;

    if (device) {

        mCaptureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:device error:&error] ;
        mCaptureSession = [[AVCaptureSession alloc] init] ;

        /*
             success = [mCaptureSession addInput:mCaptureDeviceInput];

             if (!success) {
             cout << "AV Foundation failed to start capture session with opened Capture Device" << endl;
             [localpool drain];
             return 0;
             }
         */

        mCaptureDecompressedVideoOutput = [[AVCaptureVideoDataOutput alloc] init];

        dispatch_queue_t queue = dispatch_queue_create("cameraQueue", NULL);
        [mCaptureDecompressedVideoOutput setSampleBufferDelegate:capture queue:queue];
        dispatch_release(queue);


        NSDictionary *pixelBufferOptions ;
        if (width > 0 && height > 0) {
            pixelBufferOptions = [NSDictionary dictionaryWithObjectsAndKeys:
                [NSNumber numberWithDouble:1.0*width], (id)kCVPixelBufferWidthKey,
                [NSNumber numberWithDouble:1.0*height], (id)kCVPixelBufferHeightKey,
                [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA],
                (id)kCVPixelBufferPixelFormatTypeKey,
                nil];
        } else {
            pixelBufferOptions = [NSDictionary dictionaryWithObjectsAndKeys:
                [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA],
                (id)kCVPixelBufferPixelFormatTypeKey,
                nil];
        }

        //TODO: add new interface for setting fps and capturing resolution.
        [mCaptureDecompressedVideoOutput setVideoSettings:pixelBufferOptions];
        mCaptureDecompressedVideoOutput.alwaysDiscardsLateVideoFrames = YES;

#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        mCaptureDecompressedVideoOutput.minFrameDuration = CMTimeMake(1, 30);
#endif

        //Slow. 1280*720 for iPhone4, iPod back camera. 640*480 for front camera
        //mCaptureSession.sessionPreset = AVCaptureSessionPresetHigh; // fps ~= 5 slow for OpenCV

        mCaptureSession.sessionPreset = AVCaptureSessionPresetMedium; //480*360
        if (width == 0 ) width = 480;
        if (height == 0 ) height = 360;

        [mCaptureSession addInput:mCaptureDeviceInput];
        [mCaptureSession addOutput:mCaptureDecompressedVideoOutput];

        /*
        // Does not work! This is the preferred way (hardware acceleration) to change pixel buffer orientation.
        // I'm now using cvtranspose and cvflip instead, which takes cpu cycles.
        AVCaptureConnection *connection = [[mCaptureDecompressedVideoOutput connections] objectAtIndex:0];
        if([connection isVideoOrientationSupported]) {
            //NSLog(@"Setting pixel buffer orientation");
            connection.videoOrientation = AVCaptureVideoOrientationPortrait;
        }
        */

        [mCaptureSession startRunning];

        grabFrame(60);
        [localpool drain];
        return 1;
    }

    [localpool drain];
    return 0;
}

void CvCaptureCAM::setWidthHeight() {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
    NSDictionary* pixelBufferOptions = [NSDictionary dictionaryWithObjectsAndKeys:
        [NSNumber numberWithDouble:1.0*width], (id)kCVPixelBufferWidthKey,
        [NSNumber numberWithDouble:1.0*height], (id)kCVPixelBufferHeightKey,
        [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA],
        (id)kCVPixelBufferPixelFormatTypeKey,
        nil];

    [mCaptureDecompressedVideoOutput setVideoSettings:pixelBufferOptions];
    grabFrame(60);
    [localpool drain];
}

//added macros into headers in videoio_c.h
/*
#define CV_CAP_PROP_IOS_DEVICE_FOCUS 9001
#define CV_CAP_PROP_IOS_DEVICE_EXPOSURE 9002
#define CV_CAP_PROP_IOS_DEVICE_FLASH 9003
#define CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE 9004
#define CV_CAP_PROP_IOS_DEVICE_TORCH 9005
*/


/*
// All available settings are taken from iOS API

enum {
   AVCaptureFlashModeOff    = 0,
   AVCaptureFlashModeOn     = 1,
   AVCaptureFlashModeAuto   = 2
};
typedef NSInteger AVCaptureFlashMode;

enum {
   AVCaptureTorchModeOff    = 0,
   AVCaptureTorchModeOn     = 1,
   AVCaptureTorchModeAuto   = 2
};
typedef NSInteger AVCaptureTorchMode;

enum {
   AVCaptureFocusModeLocked                = 0,
   AVCaptureFocusModeAutoFocus             = 1,
   AVCaptureFocusModeContinuousAutoFocus   = 2,
};
typedef NSInteger AVCaptureFocusMode;

enum {
   AVCaptureExposureModeLocked                    = 0,
   AVCaptureExposureModeAutoExpose                = 1,
   AVCaptureExposureModeContinuousAutoExposure    = 2,
};
typedef NSInteger AVCaptureExposureMode;

enum {
   AVCaptureWhiteBalanceModeLocked             = 0,
   AVCaptureWhiteBalanceModeAutoWhiteBalance   = 1,
   AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance = 2,
};
typedef NSInteger AVCaptureWhiteBalanceMode;
*/

double CvCaptureCAM::getProperty(int property_id) const{
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    /*
         NSArray* connections = [mCaptureDeviceInput	connections];
         QTFormatDescription* format = [[connections objectAtIndex:0] formatDescription];
         NSSize s1 = [[format attributeForKey:QTFormatDescriptionVideoCleanApertureDisplaySizeAttribute] sizeValue];
     */

    NSArray* ports = mCaptureDeviceInput.ports;
    CMFormatDescriptionRef format = [[ports objectAtIndex:0] formatDescription];
    CGSize s1 = CMVideoFormatDescriptionGetPresentationDimensions(format, YES, YES);

    int w=(int)s1.width, h=(int)s1.height;

    [localpool drain];

    switch (property_id) {
        case CV_CAP_PROP_FRAME_WIDTH:
            return w;
        case CV_CAP_PROP_FRAME_HEIGHT:
            return h;

        case CV_CAP_PROP_IOS_DEVICE_FOCUS:
            return mCaptureDevice.focusMode;
        case CV_CAP_PROP_IOS_DEVICE_EXPOSURE:
            return mCaptureDevice.exposureMode;
        case CV_CAP_PROP_IOS_DEVICE_FLASH:
            return mCaptureDevice.flashMode;
        case CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE:
            return mCaptureDevice.whiteBalanceMode;
        case CV_CAP_PROP_IOS_DEVICE_TORCH:
            return mCaptureDevice.torchMode;

        default:
            return 0;
    }


}

bool CvCaptureCAM::setProperty(int property_id, double value) {
    switch (property_id) {
        case CV_CAP_PROP_FRAME_WIDTH:
            width = value;
            settingWidth = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth =0;
                settingHeight = 0;
            }
            return true;

        case CV_CAP_PROP_FRAME_HEIGHT:
            height = value;
            settingHeight = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth =0;
                settingHeight = 0;
            }
            return true;

        case CV_CAP_PROP_IOS_DEVICE_FOCUS:
            if ([mCaptureDevice isFocusModeSupported:(AVCaptureFocusMode)value]){
                NSError* error = nil;
                [mCaptureDevice lockForConfiguration:&error];
                if (error) return false;
                [mCaptureDevice setFocusMode:(AVCaptureFocusMode)value];
                [mCaptureDevice unlockForConfiguration];
                //NSLog(@"Focus set");
                return true;
            }else {
                return false;
            }

        case CV_CAP_PROP_IOS_DEVICE_EXPOSURE:
            if ([mCaptureDevice isExposureModeSupported:(AVCaptureExposureMode)value]){
                NSError* error = nil;
                [mCaptureDevice lockForConfiguration:&error];
                if (error) return false;
                [mCaptureDevice setExposureMode:(AVCaptureExposureMode)value];
                [mCaptureDevice unlockForConfiguration];
                //NSLog(@"Exposure set");
                return true;
            }else {
                return false;
            }

        case CV_CAP_PROP_IOS_DEVICE_FLASH:
            if ( [mCaptureDevice hasFlash] && [mCaptureDevice isFlashModeSupported:(AVCaptureFlashMode)value]){
                NSError* error = nil;
                [mCaptureDevice lockForConfiguration:&error];
                if (error) return false;
                [mCaptureDevice setFlashMode:(AVCaptureFlashMode)value];
                [mCaptureDevice unlockForConfiguration];
                //NSLog(@"Flash mode set");
                return true;
            }else {
                return false;
            }

        case CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE:
            if ([mCaptureDevice isWhiteBalanceModeSupported:(AVCaptureWhiteBalanceMode)value]){
                NSError* error = nil;
                [mCaptureDevice lockForConfiguration:&error];
                if (error) return false;
                [mCaptureDevice setWhiteBalanceMode:(AVCaptureWhiteBalanceMode)value];
                [mCaptureDevice unlockForConfiguration];
                //NSLog(@"White balance set");
                return true;
            }else {
                return false;
            }

        case CV_CAP_PROP_IOS_DEVICE_TORCH:
            if ([mCaptureDevice hasFlash] && [mCaptureDevice isTorchModeSupported:(AVCaptureTorchMode)value]){
                NSError* error = nil;
                [mCaptureDevice lockForConfiguration:&error];
                if (error) return false;
                [mCaptureDevice setTorchMode:(AVCaptureTorchMode)value];
                [mCaptureDevice unlockForConfiguration];
                //NSLog(@"Torch mode set");
                return true;
            }else {
                return false;
            }

        case DISABLE_AUTO_RESTART:
            disableAutoRestart = value;
            return 1;
        default:
            return false;
    }
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
    newFrame = 0;
    imagedata = NULL;
    bgr_imagedata = NULL;
    currSize = 0;
    image = NULL;
    bgr_image = NULL;
    bgr_image_r90 = NULL;
    return self;
}


-(void)dealloc {
    if (imagedata != NULL) free(imagedata);
    if (bgr_imagedata != NULL) free(bgr_imagedata);
    cvReleaseImage(&image);
    cvReleaseImage(&bgr_image);
    cvReleaseImage(&bgr_image_r90);
    [super dealloc];
}



- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
fromConnection:(AVCaptureConnection *)connection{

    // Failed
    // connection.videoOrientation = AVCaptureVideoOrientationPortrait;
    (void)captureOutput;
    (void)connection;

    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

    CVBufferRetain(imageBuffer);
    CVImageBufferRef imageBufferToRelease  = mCurrentImageBuffer;

    @synchronized (self) {

        mCurrentImageBuffer = imageBuffer;
        newFrame = 1;
    }

    CVBufferRelease(imageBufferToRelease);

}


-(IplImage*) getOutput {
    //return bgr_image;
    return bgr_image_r90;
}

-(int) updateImage {
    if (newFrame==0) return 0;
    CVPixelBufferRef pixels;

    @synchronized (self){
        pixels = CVBufferRetain(mCurrentImageBuffer);
        newFrame = 0;
    }

    CVPixelBufferLockBaseAddress(pixels, 0);
    uint32_t* baseaddress = (uint32_t*)CVPixelBufferGetBaseAddress(pixels);

    size_t width = CVPixelBufferGetWidth(pixels);
    size_t height = CVPixelBufferGetHeight(pixels);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(pixels);

    if (rowBytes != 0) {

        if (currSize != rowBytes*height*sizeof(char)) {
            currSize = rowBytes*height*sizeof(char);
            if (imagedata != NULL) free(imagedata);
            if (bgr_imagedata != NULL) free(bgr_imagedata);
            imagedata = (char*)malloc(currSize);
            bgr_imagedata = (char*)malloc(currSize);
        }

        memcpy(imagedata, baseaddress, currSize);

        if (image == NULL) {
            image = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 4);
        }
        image->width = (int)width;
        image->height = (int)height;
        image->nChannels = 4;
        image->depth = IPL_DEPTH_8U;
        image->widthStep = (int)rowBytes;
        image->imageData = imagedata;
        image->imageSize = (int)currSize;

        if (bgr_image == NULL) {
            bgr_image = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 3);
        }
        bgr_image->width = (int)width;
        bgr_image->height = (int)height;
        bgr_image->nChannels = 3;
        bgr_image->depth = IPL_DEPTH_8U;
        bgr_image->widthStep = (int)rowBytes;
        bgr_image->imageData = bgr_imagedata;
        bgr_image->imageSize = (int)currSize;

        cvCvtColor(image, bgr_image, CV_BGRA2BGR);

        // image taken from the buffer is incorrected rotated. I'm using cvTranspose + cvFlip.
        // There should be an option in iOS API to rotate the buffer output orientation.
        // iOS provides hardware accelerated rotation through AVCaptureConnection class
        // I can't get it work.
        if (bgr_image_r90 == NULL){
            bgr_image_r90 = cvCreateImage(cvSize((int)height, (int)width), IPL_DEPTH_8U, 3);
        }
        cvTranspose(bgr_image, bgr_image_r90);
        cvFlip(bgr_image_r90, NULL, 1);

    }

    CVPixelBufferUnlockBaseAddress(pixels, 0);
    CVBufferRelease(pixels);

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

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    mMovieReader = nil;
    image = NULL;
    bgr_image = NULL;
    imagedata = NULL;
    bgr_imagedata = NULL;
    currSize = 0;

    movieWidth = 0;
    movieHeight = 0;
    movieFPS = 0;
    currentFPS = 0;
    movieDuration = 0;
    changedPos = 0;

    started = 0;

    AVURLAsset *asset = [AVURLAsset URLAssetWithURL:
        [NSURL fileURLWithPath: [NSString stringWithUTF8String:filename]]
        options:nil];

    AVAssetTrack* videoTrack = nil;
    NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
    if ([tracks count] == 1)
    {
        videoTrack = [tracks objectAtIndex:0];

        movieWidth = videoTrack.naturalSize.width;
        movieHeight = videoTrack.naturalSize.height;
        movieFPS = videoTrack.nominalFrameRate;

        currentFPS = movieFPS; //Debugging !! should be getFPS();
        //Debugging. need to be checked

        // In ms
        movieDuration = videoTrack.timeRange.duration.value/videoTrack.timeRange.duration.timescale * 1000;

        started = 1;
        NSError* error = nil;
        mMovieReader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        if (error)
            NSLog(@"%@", [error localizedDescription]);

        NSDictionary* videoSettings =
            [NSDictionary dictionaryWithObject:[NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA]
            forKey:(NSString*)kCVPixelBufferPixelFormatTypeKey];

        [mMovieReader addOutput:[AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:videoTrack
            outputSettings:videoSettings]];
        [mMovieReader startReading];
    }

    /*
    // Asynchronously open the video in another thread. Always fail.
    [asset loadValuesAsynchronouslyForKeys:[NSArray arrayWithObject:@"tracks"] completionHandler:
    ^{
    // The completion block goes here.
    dispatch_async(dispatch_get_main_queue(),
    ^{
    AVAssetTrack* ::videoTrack = nil;
    NSArray* ::tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
    if ([tracks count] == 1)
    {
    videoTrack = [tracks objectAtIndex:0];

    movieWidth = videoTrack.naturalSize.width;
    movieHeight = videoTrack.naturalSize.height;
    movieFPS = videoTrack.nominalFrameRate;
    currentFPS = movieFPS; //Debugging !! should be getFPS();
    //Debugging. need to be checked
    movieDuration = videoTrack.timeRange.duration.value/videoTrack.timeRange.duration.timescale * 1000;
    started = 1;

    NSError* ::error = nil;
    // mMovieReader is a member variable
    mMovieReader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
    if (error)
    NSLog(@"%@", [error localizedDescription]);

    NSDictionary* ::videoSettings =
    [NSDictionary dictionaryWithObject:[NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA]
forKey:(NSString*)kCVPixelBufferPixelFormatTypeKey];

[mMovieReader addOutput:[AVAssetReaderTrackOutput
assetReaderTrackOutputWithTrack:videoTrack
outputSettings:videoSettings]];
[mMovieReader startReading];
}
});

}];
     */

[localpool drain];
}

CvCaptureFile::~CvCaptureFile() {

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
    if (imagedata != NULL) free(imagedata);
    if (bgr_imagedata != NULL) free(bgr_imagedata);
    cvReleaseImage(&image);
    cvReleaseImage(&bgr_image);
    [mMovieReader release];
    [localpool drain];
}

int CvCaptureFile::didStart() {
    return started;
}

bool CvCaptureFile::grabFrame() {

    //everything is done in queryFrame;
    currentFPS = movieFPS;
    return 1;


    /*
            double t1 = getProperty(CV_CAP_PROP_POS_MSEC);
            [mCaptureSession stepForward];
            double t2 = getProperty(CV_CAP_PROP_POS_MSEC);
            if (t2>t1 && !changedPos) {
            currentFPS = 1000.0/(t2-t1);
            } else {
            currentFPS = movieFPS;
            }
            changedPos = 0;

     */

}


IplImage* CvCaptureFile::retrieveFramePixelBuffer() {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    if (mMovieReader.status != AVAssetReaderStatusReading){

        return NULL;
    }


    AVAssetReaderOutput * output = [mMovieReader.outputs objectAtIndex:0];
    CMSampleBufferRef sampleBuffer = [output copyNextSampleBuffer];
    if (!sampleBuffer) {
        [localpool drain];
        return NULL;
    }
    CVPixelBufferRef frame = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferRef pixels = CVBufferRetain(frame);

    CVPixelBufferLockBaseAddress(pixels, 0);

    uint32_t* baseaddress = (uint32_t*)CVPixelBufferGetBaseAddress(pixels);
    size_t width = CVPixelBufferGetWidth(pixels);
    size_t height = CVPixelBufferGetHeight(pixels);
    size_t rowBytes = CVPixelBufferGetBytesPerRow(pixels);

    if (rowBytes != 0) {

        if (currSize != rowBytes*height*sizeof(char)) {
            currSize = rowBytes*height*sizeof(char);
            if (imagedata != NULL) free(imagedata);
            if (bgr_imagedata != NULL) free(bgr_imagedata);
            imagedata = (char*)malloc(currSize);
            bgr_imagedata = (char*)malloc(currSize);
        }

        memcpy(imagedata, baseaddress, currSize);

        if (image == NULL) {
            image = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 4);
        }

        image->width = (int)width;
        image->height = (int)height;
        image->nChannels = 4;
        image->depth = IPL_DEPTH_8U;
        image->widthStep = (int)rowBytes;
        image->imageData = imagedata;
        image->imageSize = (int)currSize;


        if (bgr_image == NULL) {
            bgr_image = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, 3);
        }

        bgr_image->width = (int)width;
        bgr_image->height = (int)height;
        bgr_image->nChannels = 3;
        bgr_image->depth = IPL_DEPTH_8U;
        bgr_image->widthStep = (int)rowBytes;
        bgr_image->imageData = bgr_imagedata;
        bgr_image->imageSize = (int)currSize;

        cvCvtColor(image, bgr_image,CV_BGRA2BGR);

    }

    CVPixelBufferUnlockBaseAddress(pixels, 0);
    CVBufferRelease(pixels);
    CMSampleBufferInvalidate(sampleBuffer);
    CFRelease(sampleBuffer);

    [localpool drain];
    return bgr_image;
}


IplImage* CvCaptureFile::retrieveFrame(int) {
    return retrieveFramePixelBuffer();
}

IplImage* CvCaptureFile::queryFrame() {
    grabFrame();
    return retrieveFrame(0);
}

double CvCaptureFile::getFPS() {

    /*
         if (mCaptureSession == nil) return 0;
         NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];
         double now = getProperty(CV_CAP_PROP_POS_MSEC);
         double retval = 0;
         if (now == 0) {
         [mCaptureSession stepForward];
         double t2 =  getProperty(CV_CAP_PROP_POS_MSEC);
         [mCaptureSession stepBackward];
         retval = 1000.0 / (t2-now);
         } else {
         [mCaptureSession stepBackward];
         double t2 = getProperty(CV_CAP_PROP_POS_MSEC);
         [mCaptureSession stepForward];
         retval = 1000.0 / (now-t2);
         }
         [localpool drain];
         return retval;
     */
    return 30.0; //TODO: Debugging
}

double CvCaptureFile::getProperty(int /*property_id*/) const{

    /*
         if (mCaptureSession == nil) return 0;

         NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

         double retval;
         QTTime t;

         switch (property_id) {
         case CV_CAP_PROP_POS_MSEC:
         [[mCaptureSession attributeForKey:QTMovieCurrentTimeAttribute] getValue:&t];
         retval = t.timeValue * 1000.0 / t.timeScale;
         break;
         case CV_CAP_PROP_POS_FRAMES:
         retval = movieFPS * getProperty(CV_CAP_PROP_POS_MSEC) / 1000;
         break;
         case CV_CAP_PROP_POS_AVI_RATIO:
         retval = (getProperty(CV_CAP_PROP_POS_MSEC)) / (movieDuration );
         break;
         case CV_CAP_PROP_FRAME_WIDTH:
         retval = movieWidth;
         break;
         case CV_CAP_PROP_FRAME_HEIGHT:
         retval = movieHeight;
         break;
         case CV_CAP_PROP_FPS:
         retval = currentFPS;
         break;
         case CV_CAP_PROP_FOURCC:
         default:
         retval = 0;
         }

         [localpool drain];
         return retval;
     */
    return 1.0; //Debugging
}

bool CvCaptureFile::setProperty(int /*property_id*/, double /*value*/) {

    /*
         if (mCaptureSession == nil) return false;

         NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

         bool retval = false;
         QTTime t;

         double ms;

         switch (property_id) {
         case CV_CAP_PROP_POS_MSEC:
         [[mCaptureSession attributeForKey:QTMovieCurrentTimeAttribute] getValue:&t];
         t.timeValue = value * t.timeScale / 1000;
         [mCaptureSession setCurrentTime:t];
         changedPos = 1;
         retval = true;
         break;
         case CV_CAP_PROP_POS_FRAMES:
         ms = (value*1000.0 -5)/ currentFPS;
         retval = setProperty(CV_CAP_PROP_POS_MSEC, ms);
         break;
         case CV_CAP_PROP_POS_AVI_RATIO:
         ms = value * movieDuration;
         retval = setProperty(CV_CAP_PROP_POS_MSEC, ms);
         break;
         case CV_CAP_PROP_FRAME_WIDTH:
    //retval = movieWidth;
    break;
    case CV_CAP_PROP_FRAME_HEIGHT:
    //retval = movieHeight;
    break;
    case CV_CAP_PROP_FPS:
    //etval = currentFPS;
    break;
    case CV_CAP_PROP_FOURCC:
    default:
    retval = false;
    }

    [localpool drain];

    return retval;
     */
    return true;
}


/*****************************************************************************
 *
 * CvVideoWriter Implementation.
 *
 * CvVideoWriter is the instantiation of a video output class
 *
 *****************************************************************************/


CvVideoWriter_AVFoundation::CvVideoWriter_AVFoundation(const char* filename, int fourcc,
        double fps, CvSize frame_size,
        int is_color) {

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];


    frameCount = 0;
    movieFPS = fps;
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
        std::cout << "WARNING: Didn't properly encode FourCC. Expected " << fourcc
            << " but got " << cc2 << "." << std::endl;
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


CvVideoWriter_AVFoundation::~CvVideoWriter_AVFoundation() {
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

bool CvVideoWriter_AVFoundation::writeFrame(const IplImage* iplimage) {
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
        std::cout<<"Frame size does not match video size."<<std::endl;
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
            withPresentationTime:CMTimeMake(frameCount, movieFPS)];
    }

    //cleanup
    CFRelease(cfData);
    CVPixelBufferRelease(pixelBuffer);
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    [localpool drain];

    if (success) {
        frameCount ++;
        //NSLog(@"Frame #%d", frameCount);
        return true;
    }else{
        NSLog(@"Frame appendPixelBuffer failed.");
        return false;
    }

}
