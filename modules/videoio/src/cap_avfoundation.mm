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

 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "cap_interface.hpp"
#include <iostream>
#include <Availability.h>
#import <AVFoundation/AVFoundation.h>
#import <Foundation/NSException.h>

#define CV_CAP_MODE_BGR CV_FOURCC_MACRO('B','G','R','3')
#define CV_CAP_MODE_RGB CV_FOURCC_MACRO('R','G','B','3')
#define CV_CAP_MODE_GRAY CV_FOURCC_MACRO('G','R','E','Y')
#define CV_CAP_MODE_YUYV CV_FOURCC_MACRO('Y', 'U', 'Y', 'V')

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
        bool grabFrame() CV_OVERRIDE;
        IplImage* retrieveFrame(int) CV_OVERRIDE;
        double getProperty(int property_id) const CV_OVERRIDE;
        bool setProperty(int property_id, double value) CV_OVERRIDE;
        int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }

        virtual IplImage* queryFrame();
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
    bool grabFrame() CV_OVERRIDE;
    IplImage* retrieveFrame(int) CV_OVERRIDE;
    double getProperty(int property_id) const CV_OVERRIDE;
    bool setProperty(int property_id, double value) CV_OVERRIDE;
    int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }

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
    uint32_t  mMode;
    int       mFormat;

    bool setupReadingAt(CMTime position);
    IplImage* retrieveFramePixelBuffer();

    CMTime mFrameTimestamp;
    size_t mFrameNum;

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
        bool writeFrame(const IplImage* image) CV_OVERRIDE;
        int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }
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


cv::Ptr<cv::IVideoCapture> cv::create_AVFoundation_capture_file(const std::string &filename)
{
    CvCaptureFile *retval = new CvCaptureFile(filename.c_str());
    if(retval->didStart())
        return makePtr<LegacyCapture>(retval);
    delete retval;
    return NULL;

}

cv::Ptr<cv::IVideoCapture> cv::create_AVFoundation_capture_cam(int index)
{
    CvCaptureCAM* retval = new CvCaptureCAM(index);
    if (retval->didStart())
        return cv::makePtr<cv::LegacyCapture>(retval);
    delete retval;
    return 0;
}

cv::Ptr<cv::IVideoWriter> cv::create_AVFoundation_writer(const std::string& filename, int fourcc,
                                                         double fps, const cv::Size &frameSize,
                                                         const cv::VideoWriterParameters& params)
{
    CvSize sz = { frameSize.width, frameSize.height };
    const bool isColor = params.get(VIDEOWRITER_PROP_IS_COLOR, true);
    CvVideoWriter_AVFoundation* wrt = new CvVideoWriter_AVFoundation(filename.c_str(), fourcc, fps, sz, isColor);
    return cv::makePtr<cv::LegacyWriter>(wrt);
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
    NSArray* devices = [[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]
            arrayByAddingObjectsFromArray:[AVCaptureDevice devicesWithMediaType:AVMediaTypeMuxed]];
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

#if (TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR) && (!defined(TARGET_OS_MACCATALYST) || !TARGET_OS_MACCATALYST)
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
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    mAsset = nil;
    mAssetTrack = nil;
    mAssetReader = nil;
    mTrackOutput = nil;
    mDeviceImage = NULL;
    mOutImage = NULL;
    mOutImagedata = NULL;
    currSize = 0;
    mMode = CV_CAP_MODE_BGR;
    mFormat = CV_8UC3;
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

    NSArray *tracks = [mAsset tracksWithMediaType:AVMediaTypeVideo];
    if ([tracks count] == 0) {
        fprintf(stderr, "OpenCV: Couldn't read video stream from file \"%s\"\n", filename);
        [localpool drain];
        started = 0;
        return;
    }

    mAssetTrack = [tracks[0] retain];

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

    // Capture in a pixel format that can be converted efficiently to the output mode.
    OSType pixelFormat;
    if (mMode == CV_CAP_MODE_BGR || mMode == CV_CAP_MODE_RGB) {
        pixelFormat = kCVPixelFormatType_32BGRA;
        mFormat = CV_8UC3;
    } else if (mMode == CV_CAP_MODE_GRAY) {
        pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
        mFormat = CV_8UC1;
    } else if (mMode == CV_CAP_MODE_YUYV) {
        pixelFormat = kCVPixelFormatType_422YpCbCr8;
        mFormat = CV_8UC2;
    } else {
        fprintf(stderr, "VIDEOIO ERROR: AVF Mac: Unsupported mode: %d\n", mMode);
        return false;
    }

    NSDictionary *settings =
    @{
      (id)kCVPixelBufferPixelFormatTypeKey: @(pixelFormat)
      };
    mTrackOutput = [[AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack: mAssetTrack
                                                               outputSettings: settings] retain];

    if ( !mTrackOutput ) {
        fprintf(stderr, "OpenCV: error in [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:outputSettings:]\n");
        return false;
    }

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
    return [mAssetReader startReading];
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
    void *baseaddress;
    size_t width, height, rowBytes;

    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);

    if (CVPixelBufferIsPlanar(mGrabbedPixels)) {
        baseaddress = CVPixelBufferGetBaseAddressOfPlane(mGrabbedPixels, 0);
        width = CVPixelBufferGetWidthOfPlane(mGrabbedPixels, 0);
        height = CVPixelBufferGetHeightOfPlane(mGrabbedPixels, 0);
        rowBytes = CVPixelBufferGetBytesPerRowOfPlane(mGrabbedPixels, 0);
    } else {
        baseaddress = CVPixelBufferGetBaseAddress(mGrabbedPixels);
        width = CVPixelBufferGetWidth(mGrabbedPixels);
        height = CVPixelBufferGetHeight(mGrabbedPixels);
        rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    }

    if ( rowBytes == 0 ) {
        fprintf(stderr, "OpenCV: error: rowBytes == 0\n");
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    int outChannels;
    if (mMode == CV_CAP_MODE_BGR || mMode == CV_CAP_MODE_RGB) {
        outChannels = 3;
    } else if (mMode == CV_CAP_MODE_GRAY) {
        outChannels = 1;
    } else if (mMode == CV_CAP_MODE_YUYV) {
        outChannels = 2;
    } else {
        fprintf(stderr, "VIDEOIO ERROR: AVF Mac: Unsupported mode: %d\n", mMode);
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    if ( currSize != width*outChannels*height ) {
        currSize = width*outChannels*height;
        free(mOutImagedata);
        mOutImagedata = reinterpret_cast<uint8_t*>(malloc(currSize));
    }

    if (mOutImage == NULL) {
        mOutImage = cvCreateImageHeader(cvSize((int)width,(int)height), IPL_DEPTH_8U, outChannels);
    }
    mOutImage->width = int(width);
    mOutImage->height = int(height);
    mOutImage->nChannels = outChannels;
    mOutImage->depth = IPL_DEPTH_8U;
    mOutImage->widthStep = int(width*outChannels);
    mOutImage->imageData = reinterpret_cast<char *>(mOutImagedata);
    mOutImage->imageSize = int(currSize);

    int deviceChannels;
    int cvtCode;

    if ( pixelFormat == kCVPixelFormatType_32BGRA ) {
        deviceChannels = 4;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = CV_BGRA2BGR;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = CV_BGRA2RGB;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = CV_BGRA2GRAY;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return 0;
        }
    } else if ( pixelFormat == kCVPixelFormatType_24RGB ) {
        deviceChannels = 3;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = CV_RGB2BGR;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = 0;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = CV_RGB2GRAY;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return 0;
        }
    } else if ( pixelFormat == kCVPixelFormatType_422YpCbCr8 ) {    // 422 (2vuy, UYVY)
        deviceChannels = 2;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = CV_YUV2BGR_UYVY;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = CV_YUV2RGB_UYVY;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = CV_YUV2GRAY_UYVY;
        } else if (mMode == CV_CAP_MODE_YUYV) {
            cvtCode = -1;    // Copy
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return 0;
        }
    } else if ( pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||   // 420v
               pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ) {   // 420f
        height = height * 3 / 2;
        deviceChannels = 1;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = CV_YUV2BGR_YV12;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = CV_YUV2RGB_YV12;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = CV_YUV2GRAY_420;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return 0;
        }
    } else {
        char pfBuf[] = { (char)pixelFormat, (char)(pixelFormat >> 8),
                         (char)(pixelFormat >> 16), (char)(pixelFormat >> 24), '\0' };
        fprintf(stderr, "OpenCV: unsupported pixel format '%s'\n", pfBuf);
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return 0;
    }

    if (mDeviceImage == NULL) {
        mDeviceImage = cvCreateImageHeader(cvSize(int(width),int(height)), IPL_DEPTH_8U, deviceChannels);
    }
    mDeviceImage->width = int(width);
    mDeviceImage->height = int(height);
    mDeviceImage->nChannels = deviceChannels;
    mDeviceImage->depth = IPL_DEPTH_8U;
    mDeviceImage->widthStep = int(rowBytes);
    mDeviceImage->imageData = reinterpret_cast<char *>(baseaddress);
    mDeviceImage->imageSize = int(rowBytes*height);

    if (cvtCode == -1) {
        cv::cvarrToMat(mDeviceImage).copyTo(cv::cvarrToMat(mOutImage));
    } else {
        cvCvtColor(mDeviceImage, mOutImage, cvtCode);
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
            return mAssetTrack.nominalFrameRate > 0 ? mFrameNum : 0;
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
            return mFormat;
        case CV_CAP_PROP_FOURCC:
            return mMode;
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
            retval = setupReadingAt(t);
            break;
        case CV_CAP_PROP_POS_FRAMES:
            retval = mAssetTrack.nominalFrameRate > 0 ? setupReadingAt(CMTimeMake(value, mAssetTrack.nominalFrameRate)) : false;
            break;
        case CV_CAP_PROP_POS_AVI_RATIO:
            t = mAsset.duration;
            t.value = round(t.value * value);
            retval = setupReadingAt(t);
            break;
        case CV_CAP_PROP_FOURCC:
            uint32_t mode;
            mode = cvRound(value);
            if (mMode == mode) {
                retval = true;
            } else {
                switch (mode) {
                    case CV_CAP_MODE_BGR:
                    case CV_CAP_MODE_RGB:
                    case CV_CAP_MODE_GRAY:
                    case CV_CAP_MODE_YUYV:
                        mMode = mode;
                        retval = setupReadingAt(mFrameTimestamp);
                        break;
                    default:
                        fprintf(stderr, "VIDEOIO ERROR: AVF iOS: Unsupported mode: %d\n", mode);
                        retval=false;
                        break;
                }
            }
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

    // Three codec supported AVVideoCodecH264 AVVideoCodecJPEG AVVideoCodecTypeHEVC
    // On iPhone 3G H264 is not supported.
    if (fourcc == CV_FOURCC('J','P','E','G') || fourcc == CV_FOURCC('j','p','e','g') ||
            fourcc == CV_FOURCC('M','J','P','G') || fourcc == CV_FOURCC('m','j','p','g')){
        codec = [AVVideoCodecJPEG copy]; // Use JPEG codec if specified, otherwise H264
    }else if(fourcc == CV_FOURCC('H','2','6','4') || fourcc == CV_FOURCC('a','v','c','1')){
            codec = [AVVideoCodecH264 copy];
// Available since iOS 11
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 110000
    }else if(fourcc == CV_FOURCC('H','2','6','5') || fourcc == CV_FOURCC('h','v','c','1') ||
            fourcc == CV_FOURCC('H','E','V','C') || fourcc == CV_FOURCC('h','e','v','c')){
        if (@available(iOS 11, *)) {
            codec = [AVVideoCodecTypeHEVC copy];
        } else {
            codec = [AVVideoCodecH264 copy];
        }
#endif
    }else{
        codec = [AVVideoCodecH264 copy]; // default canonical H264.
    }

    //NSLog(@"Path: %@", path);

    NSError *error = nil;


    // Make sure the file does not already exist. Necessary to overwrite??
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
        // TODO: error handling, cleanup. Throw exception?
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

    //CGImage -> CVPixelBufferRef conversion
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

#pragma clang diagnostic pop
