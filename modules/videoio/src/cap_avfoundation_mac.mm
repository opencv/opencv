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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <Availability.h>
#import <AVFoundation/AVFoundation.h>

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
 *   been copied before.
 *
 *****************************************************************************/


@interface CaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    NSCondition *mHasNewFrame;
    CVPixelBufferRef mGrabbedPixels;
    CVImageBufferRef mCurrentImageBuffer;
    cv::Mat mOutImage;
    size_t    currSize;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;

- (BOOL)grabImageUntilDate: (NSDate *)limit;
- (bool)updateImage;
- (cv::Mat)getImage;

@end

/*****************************************************************************
 *
 * CvCaptureCAM Declaration.
 *
 * CvCaptureCAM is the instantiation of a capture source for cameras.
 *
 *****************************************************************************/

class CvCaptureCAM : public cv::IVideoCapture {
public:
    CvCaptureCAM(int cameraNum = -1) ;
    ~CvCaptureCAM();
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame(int, cv::OutputArray) CV_OVERRIDE;
    double getProperty(int property_id) const CV_OVERRIDE;
    bool setProperty(int property_id, double value) CV_OVERRIDE;
    int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }
    bool isOpened() const CV_OVERRIDE { return started; }

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

    int started;
};


/*****************************************************************************
 *
 * CvCaptureFile Declaration.
 *
 * CvCaptureFile is the instantiation of a capture source for video files.
 *
 *****************************************************************************/

class CvCaptureFile : public cv::VideoCaptureBase {
public:
    CvCaptureFile(const char* filename) ;
    ~CvCaptureFile();
    bool grabFrame() CV_OVERRIDE;
    bool retrieveFrame_(int, cv::OutputArray) CV_OVERRIDE;
    double getProperty_(int property_id) const CV_OVERRIDE;
    bool setProperty_(int property_id, double value) CV_OVERRIDE;
    int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }
    bool isOpened() const CV_OVERRIDE { return started; }

private:
    AVAsset                  *mAsset;
    AVAssetTrack             *mAssetTrack;
    AVAssetReader            *mAssetReader;
    AVAssetReaderTrackOutput *mTrackOutput;

    CMSampleBufferRef mCurrentSampleBuffer;
    CVImageBufferRef  mGrabbedPixels;
    cv::Mat mOutImage;
    size_t    currSize;
    uint32_t  mMode;
    int       mFormat;

    bool setupReadingAt(CMTime position);
    cv::Mat retrieveFramePixelBuffer();
    int getPreferredOrientationDegrees() const;

    CMTime mFrameTimestamp;
    size_t mFrameNum;

    int started;
};


/*****************************************************************************
 *
 * CvVideoWriter_AVFoundation Declaration.
 *
 * CvVideoWriter_AVFoundation is the instantiation of a video output class.
 *
 *****************************************************************************/

class CvVideoWriter_AVFoundation : public cv::IVideoWriter {
    public:
        CvVideoWriter_AVFoundation(const std::string &filename, int fourcc, double fps, const cv::Size& frame_size, int is_color);
        ~CvVideoWriter_AVFoundation();
        void write(cv::InputArray image) CV_OVERRIDE;
        int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_AVFOUNDATION; }
        bool isOpened() const CV_OVERRIDE
        {
            return is_good;
        }
    private:
        cv::Mat argbimage;

        AVAssetWriter *mMovieWriter;
        AVAssetWriterInput* mMovieWriterInput;
        AVAssetWriterInputPixelBufferAdaptor* mMovieWriterAdaptor;

        NSString* path;
        NSString* codec;
        NSString* fileType;
        double mMovieFPS;
        cv::Size movieSize;
        int movieColor;
        unsigned long mFrameNum;
        bool is_good;
};

/****************** Implementation of interface functions ********************/

cv::Ptr<cv::IVideoCapture> cv::create_AVFoundation_capture_file(const std::string &filename)
{
    cv::Ptr<CvCaptureFile> retval = cv::makePtr<CvCaptureFile>(filename.c_str());
    if(retval->isOpened())
        return retval;
    return NULL;
}

cv::Ptr<cv::IVideoCapture> cv::create_AVFoundation_capture_cam(int index)
{
    cv::Ptr<CvCaptureCAM> retval = cv::makePtr<CvCaptureCAM>(index);
    if (retval->isOpened())
        return retval;
    return NULL;
}

cv::Ptr<cv::IVideoWriter> cv::create_AVFoundation_writer(const std::string& filename, int fourcc,
                                                         double fps, const cv::Size& frameSize,
                                                         const cv::VideoWriterParameters& params)
{
    const bool isColor = params.get(cv::VIDEOWRITER_PROP_IS_COLOR, true);
    cv::Ptr<CvVideoWriter_AVFoundation> wrt = cv::makePtr<CvVideoWriter_AVFoundation>(filename, fourcc, fps, frameSize, isColor);
    if (wrt->isOpened())
        return wrt;
    return NULL;
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

bool CvCaptureCAM::grabFrame() {
    return grabFrame(1);
}

bool CvCaptureCAM::grabFrame(double timeOut) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    bool isGrabbed = false;
    NSDate *limit = [NSDate dateWithTimeIntervalSinceNow: timeOut];
    if ( [mCapture grabImageUntilDate: limit] ) {
        isGrabbed = [mCapture updateImage];
    }

    [localpool drain];
    return isGrabbed;
}

bool CvCaptureCAM::retrieveFrame(int, cv::OutputArray arr) {
    cv::Mat img = [mCapture getImage];
    if (img.empty())
        return false;
    img.copyTo(arr);
    return true;
}

void CvCaptureCAM::stopCaptureDevice() {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    [mCaptureSession stopRunning];

    [mCaptureSession release];
    [mCaptureDeviceInput release];
    // [mCaptureDevice release]; fix #7833

    [mCaptureVideoDataOutput release];
    [mCapture release];

    [localpool drain];
}

int CvCaptureCAM::startCaptureDevice(int cameraNum) {
    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 101400
    if (@available(macOS 10.14, *))
    {
        AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if (status == AVAuthorizationStatusDenied)
        {
            fprintf(stderr, "OpenCV: camera access has been denied. Either run 'tccutil reset Camera' "
                            "command in same terminal to reset application authorization status, "
                            "either modify 'System Preferences -> Security & Privacy -> Camera' "
                            "settings for your application.\n");
            [localpool drain];
            return 0;
        }
        else if (status != AVAuthorizationStatusAuthorized)
        {
            if (!cv::utils::getConfigurationParameterBool("OPENCV_AVFOUNDATION_SKIP_AUTH", false))
            {
                fprintf(stderr, "OpenCV: not authorized to capture video (status %ld), requesting...\n", status);
                [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL) { /* we don't care */}];
                if ([NSThread isMainThread])
                {
                    // we run the main loop for 0.1 sec to show the message
                    [[NSRunLoop mainRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
                }
                else
                {
                    fprintf(stderr, "OpenCV: can not spin main run loop from other thread, set "
                                    "OPENCV_AVFOUNDATION_SKIP_AUTH=1 to disable authorization request "
                                    "and perform it in your application.\n");
                }
            }
            else
            {
                fprintf(stderr, "OpenCV: not authorized to capture video (status %ld), set "
                                "OPENCV_AVFOUNDATION_SKIP_AUTH=0 to enable authorization request or "
                                "perform it in your application.\n", status);
            }
            [localpool drain];
            return 0;
        }
    }
#endif

    // get capture device
    NSArray *devices = [[AVCaptureDevice devicesWithMediaType: AVMediaTypeVideo]
            arrayByAddingObjectsFromArray:[AVCaptureDevice devicesWithMediaType:AVMediaTypeMuxed]];

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

    // Preserve devices ordering on the system
    // see AVCaptureDevice::uniqueID property documentation for more info
    devices = [devices
        sortedArrayUsingComparator:^NSComparisonResult(AVCaptureDevice *d1,
                                                     AVCaptureDevice *d2) {
          return [d1.uniqueID compare:d2.uniqueID];
        }
    ];

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
        case cv::CAP_PROP_FRAME_WIDTH:
            retval = s1.width;
            break;
        case cv::CAP_PROP_FRAME_HEIGHT:
            retval = s1.height;
            break;
        case cv::CAP_PROP_FPS:
            {
                CMTime frameDuration = mCaptureDevice.activeVideoMaxFrameDuration;
                retval = frameDuration.timescale / double(frameDuration.value);
            }
            break;
        case cv::CAP_PROP_FORMAT:
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
        case cv::CAP_PROP_FRAME_WIDTH:
            width = value;
            settingWidth = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth = 0;
                settingHeight = 0;
            }
            isSucceeded = true;
            break;
        case cv::CAP_PROP_FRAME_HEIGHT:
            height = value;
            settingHeight = 1;
            if (settingWidth && settingHeight) {
                setWidthHeight();
                settingWidth = 0;
                settingHeight = 0;
            }
            isSucceeded = true;
            break;
        case cv::CAP_PROP_FPS:
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
 *   been copied before.
 *
 *****************************************************************************/


@implementation CaptureDelegate

- (id)init {
    [super init];
    mHasNewFrame = [[NSCondition alloc] init];
    mCurrentImageBuffer = NULL;
    mGrabbedPixels = NULL;
    currSize = 0;
    return self;
}

-(void)dealloc {
    mOutImage.release();
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

-(bool) updateImage {
    if ( ! mGrabbedPixels ) {
        return false;
    }

    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uchar *baseaddress = reinterpret_cast<uchar*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));

    cv::Size sz { (int)CVPixelBufferGetWidth(mGrabbedPixels), (int)CVPixelBufferGetHeight(mGrabbedPixels) };
    size_t rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);

    bool res = false;
    if (rowBytes != 0 && (pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_422YpCbCr8)) {
        mOutImage.create(sz, CV_8UC3);
        if ( pixelFormat == kCVPixelFormatType_32BGRA ) {
            cv::Mat devImage(sz, CV_8UC4, baseaddress, rowBytes);
            cv::cvtColor(devImage, mOutImage, cv::COLOR_BGRA2BGR);
            res = true;
        } else if ( pixelFormat == kCVPixelFormatType_422YpCbCr8 ) {
            cv::Mat devImage(sz, CV_8UC2, baseaddress, rowBytes);
            cv::cvtColor(devImage, mOutImage, cv::COLOR_YUV2BGR_UYVY);
            res = true;
        }
    } else {
        fprintf(stderr, "OpenCV: rowBytes == 0 or unknown pixel format 0x%08X\n", pixelFormat);
        mOutImage.create(cv::Size(0, 0), mOutImage.type());
    }

    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
    CVBufferRelease(mGrabbedPixels);
    mGrabbedPixels = NULL;

    return res;
}

-(cv::Mat) getImage {
    return mOutImage;
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

    mOutImage.release();
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
        // For CV_CAP_MODE_BGR, read frames as BGRA (AV Foundation's YUV->RGB conversion is slightly faster than OpenCV's cv::COLOR_YUV2BGR_YV12)
        // kCVPixelFormatType_32ABGR is reportedly faster on OS X, but OpenCV doesn't have a CV_ABGR2BGR conversion.
        // kCVPixelFormatType_24RGB is significantly slower than kCVPixelFormatType_32BGRA.
        pixelFormat = kCVPixelFormatType_32BGRA;
        mFormat = CV_8UC3;
    } else if (mMode == CV_CAP_MODE_GRAY) {
        // For CV_CAP_MODE_GRAY, read frames as 420v (faster than 420f or 422 -- at least for H.264 files)
        pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
        mFormat = CV_8UC1;
    } else if (mMode == CV_CAP_MODE_YUYV) {
        // For CV_CAP_MODE_YUYV, read frames directly as 422.
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


cv::Mat CvCaptureFile::retrieveFramePixelBuffer() {
    if ( ! mGrabbedPixels ) {
        return cv::Mat();
    }

    NSAutoreleasePool *localpool = [[NSAutoreleasePool alloc] init];

    CVPixelBufferLockBaseAddress(mGrabbedPixels, 0);
    uchar *baseaddress;
    size_t rowBytes;
    cv::Size sz;

    OSType pixelFormat = CVPixelBufferGetPixelFormatType(mGrabbedPixels);

    if (CVPixelBufferIsPlanar(mGrabbedPixels)) {
        baseaddress = reinterpret_cast<uchar*>(CVPixelBufferGetBaseAddressOfPlane(mGrabbedPixels, 0));
        sz.width = CVPixelBufferGetWidthOfPlane(mGrabbedPixels, 0);
        sz.height = CVPixelBufferGetHeightOfPlane(mGrabbedPixels, 0);
        rowBytes = CVPixelBufferGetBytesPerRowOfPlane(mGrabbedPixels, 0);
    } else {
        baseaddress = reinterpret_cast<uchar*>(CVPixelBufferGetBaseAddress(mGrabbedPixels));
        sz.width = CVPixelBufferGetWidth(mGrabbedPixels);
        sz.height = CVPixelBufferGetHeight(mGrabbedPixels);
        rowBytes = CVPixelBufferGetBytesPerRow(mGrabbedPixels);
    }

    if ( rowBytes == 0 ) {
        fprintf(stderr, "OpenCV: error: rowBytes == 0\n");
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return cv::Mat();
    }

     // Output image parameters.
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
         return cv::Mat();
     }

    // Device image parameters and conversion code.
    // (Not all of these conversions are used in production, but they were all tested to find the fastest options.)
    int deviceChannels;
    int cvtCode;

    if ( pixelFormat == kCVPixelFormatType_32BGRA ) {
        deviceChannels = 4;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = cv::COLOR_BGRA2BGR;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = cv::COLOR_BGRA2RGB;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = cv::COLOR_BGRA2GRAY;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return cv::Mat();
        }
    } else if ( pixelFormat == kCVPixelFormatType_24RGB ) {
        deviceChannels = 3;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = cv::COLOR_RGB2BGR;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = -1;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = cv::COLOR_RGB2GRAY;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return cv::Mat();
        }
    } else if ( pixelFormat == kCVPixelFormatType_422YpCbCr8 ) {    // 422 (2vuy, UYVY)
        deviceChannels = 2;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = cv::COLOR_YUV2BGR_UYVY;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = cv::COLOR_YUV2RGB_UYVY;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = cv::COLOR_YUV2GRAY_UYVY;
        } else if (mMode == CV_CAP_MODE_YUYV) {
            cvtCode = -1;    // Copy
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return cv::Mat();
        }
    } else if ( pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||   // 420v
                pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ) {   // 420f
        // cvtColor(cv::COLOR_YUV2GRAY_420) is expecting a single buffer with both the Y plane and the CrCb planes.
        // So, lie about the height of the buffer.  cvtColor(cv::COLOR_YUV2GRAY_420) will only read the first 2/3 of it.
        sz.height = sz.height * 3 / 2;
        deviceChannels = 1;

        if (mMode == CV_CAP_MODE_BGR) {
            cvtCode = cv::COLOR_YUV2BGR_YV12;
        } else if (mMode == CV_CAP_MODE_RGB) {
            cvtCode = cv::COLOR_YUV2RGB_YV12;
        } else if (mMode == CV_CAP_MODE_GRAY) {
            cvtCode = cv::COLOR_YUV2GRAY_420;
        } else {
            CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
            CVBufferRelease(mGrabbedPixels);
            mGrabbedPixels = NULL;
            fprintf(stderr, "OpenCV: unsupported pixel conversion mode\n");
            return cv::Mat();
        }
    } else {
        fprintf(stderr, "OpenCV: unsupported pixel format 0x%08X\n", pixelFormat);
        CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);
        CVBufferRelease(mGrabbedPixels);
        mGrabbedPixels = NULL;
        return cv::Mat();
    }

    mOutImage.create(sz, CV_MAKE_TYPE(CV_8U, outChannels));
    cv::Mat devImage(sz, CV_MAKE_TYPE(CV_8U, deviceChannels), baseaddress, rowBytes);
    // Convert the device image into the output image.
    if (cvtCode == -1) {
        // Copy.
        devImage.copyTo(mOutImage);
    } else {
        cv::cvtColor(devImage, mOutImage, cvtCode);
    }


    CVPixelBufferUnlockBaseAddress(mGrabbedPixels, 0);

    [localpool drain];

    return mOutImage;
}

int CvCaptureFile::getPreferredOrientationDegrees() const {
    if (mAssetTrack == nil) return 0;

    CGAffineTransform transform = mAssetTrack.preferredTransform;
    double radians = atan2(transform.b, transform.a);
    return static_cast<int>(round(radians * 180 / M_PI));
}

bool CvCaptureFile::retrieveFrame_(int, cv::OutputArray arr) {
    cv::Mat res = retrieveFramePixelBuffer();
    if (res.empty())
        return false;
    res.copyTo(arr);
    return true;
}

double CvCaptureFile::getProperty_(int property_id) const{
    if (mAsset == nil) return 0;

    CMTime t;

    switch (property_id) {
        case cv::CAP_PROP_POS_MSEC:
            return mFrameTimestamp.value * 1000.0 / mFrameTimestamp.timescale;
        case cv::CAP_PROP_POS_FRAMES:
            return mAssetTrack.nominalFrameRate > 0 ? mFrameNum : 0;
        case cv::CAP_PROP_POS_AVI_RATIO:
            t = [mAsset duration];
            return (mFrameTimestamp.value * t.timescale) / double(mFrameTimestamp.timescale * t.value);
        case cv::CAP_PROP_FRAME_WIDTH:
            return mAssetTrack.naturalSize.width;
        case cv::CAP_PROP_FRAME_HEIGHT:
            return mAssetTrack.naturalSize.height;
        case cv::CAP_PROP_FPS:
            return mAssetTrack.nominalFrameRate;
        case cv::CAP_PROP_FRAME_COUNT:
            t = [mAsset duration];
            return round((t.value * mAssetTrack.nominalFrameRate) / double(t.timescale));
        case cv::CAP_PROP_FORMAT:
            return mFormat;
        case cv::CAP_PROP_FOURCC:
            return mMode;
        case cv::CAP_PROP_ORIENTATION_META:
            return getPreferredOrientationDegrees();
        default:
            break;
    }

    return 0;
}

bool CvCaptureFile::setProperty_(int property_id, double value) {
    if (mAsset == nil) return false;

    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    bool retval = false;
    CMTime t;

    switch (property_id) {
        case cv::CAP_PROP_POS_MSEC:
            t = mAsset.duration;
            t.value = value * t.timescale / 1000;
            retval = setupReadingAt(t);
            break;
        case cv::CAP_PROP_POS_FRAMES:
            retval = mAssetTrack.nominalFrameRate > 0 ? setupReadingAt(CMTimeMake(value, mAssetTrack.nominalFrameRate)) : false;
            break;
        case cv::CAP_PROP_POS_AVI_RATIO:
            t = mAsset.duration;
            t.value = round(t.value * value);
            retval = setupReadingAt(t);
            break;
        case cv::CAP_PROP_FOURCC:
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
                        fprintf(stderr, "VIDEOIO ERROR: AVF Mac: Unsupported mode: %d\n", mode);
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
 * CvVideoWriter_AVFoundation Implementation.
 *
 * CvVideoWriter_AVFoundation is the instantiation of a video output class.
 *
 *****************************************************************************/


CvVideoWriter_AVFoundation::CvVideoWriter_AVFoundation(const std::string &filename, int fourcc, double fps, const cv::Size& frame_size, int is_color)
    : mMovieWriter(nil), mMovieWriterInput(nil), mMovieWriterAdaptor(nil), path(nil),
    codec(nil), fileType(nil), mMovieFPS(fps), movieSize(frame_size), movieColor(is_color), mFrameNum(0),
    is_good(true)
{
    if (mMovieFPS <= 0 || movieSize.width <= 0 || movieSize.height <= 0)
    {
        is_good = false;
        return;
    }
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    argbimage.create(movieSize, CV_8UC4);
    path = [[[NSString stringWithUTF8String:filename.c_str()] stringByExpandingTildeInPath] retain];

    NSString *fileExt =[[[path pathExtension] lowercaseString] copy];
    if ([fileExt isEqualToString:@"mov"] || [fileExt isEqualToString:@"qt"]){
        fileType = [AVFileTypeQuickTimeMovie copy];
    }else if ([fileExt isEqualToString:@"mp4"]){
        fileType = [AVFileTypeMPEG4 copy];
    }else if ([fileExt isEqualToString:@"m4v"]){
        fileType = [AVFileTypeAppleM4V copy];
    } else{
        is_good = false;
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
        is_good = false;
    }

    // Three codec supported AVVideoCodecTypeH264 AVVideoCodecTypeJPEG AVVideoCodecTypeHEVC
    // On iPhone 3G H264 is not supported.
    if (fourcc == CV_FOURCC('J','P','E','G') || fourcc == CV_FOURCC('j','p','e','g') ||
            fourcc == CV_FOURCC('M','J','P','G') || fourcc == CV_FOURCC('m','j','p','g')){
        codec = [AVVideoCodecTypeJPEG copy]; // Use JPEG codec if specified, otherwise H264
    }else if(fourcc == CV_FOURCC('H','2','6','4') || fourcc == CV_FOURCC('a','v','c','1')){
            codec = [AVVideoCodecTypeH264 copy];
    // Available since macOS 10.13
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101300
    }else if(fourcc == CV_FOURCC('H','2','6','5') || fourcc == CV_FOURCC('h','v','c','1') ||
            fourcc == CV_FOURCC('H','E','V','C') || fourcc == CV_FOURCC('h','e','v','c')){
        if (@available(macOS 10.13, *)) {
            codec = [AVVideoCodecTypeHEVC copy];
        } else {
            is_good = false;
        }
#endif
    }else{
        is_good = false;
    }

    //NSLog(@"Path: %@", path);

    if (is_good)
    {
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
            NSLog(@"AVF: AVAssetWriter status: %@", [mMovieWriter.error localizedDescription]);
            is_good = false;
        }
    }

    [localpool drain];
}


CvVideoWriter_AVFoundation::~CvVideoWriter_AVFoundation() {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    if (mMovieWriterInput && mMovieWriter && mMovieWriterAdaptor)
    {
        [mMovieWriterInput markAsFinished];
        [mMovieWriter finishWriting];
        [mMovieWriter release];
        [mMovieWriterInput release];
        [mMovieWriterAdaptor release];
    }
    if (path)
        [path release];
    if (codec)
        [codec release];
    if (fileType)
        [fileType release];
    if (!argbimage.empty())
        argbimage.release();

    [localpool drain];

}

static void releaseCallback( void *releaseRefCon, const void * ) {
    CFRelease((CFDataRef)releaseRefCon);
}

void CvVideoWriter_AVFoundation::write(cv::InputArray image) {
    NSAutoreleasePool* localpool = [[NSAutoreleasePool alloc] init];

    // writer status check
    if (mMovieWriter.status !=  AVAssetWriterStatusWriting ) {
        NSLog(@"mMovieWriter.status: %d. Error: %@", (int)mMovieWriter.status, [mMovieWriter.error localizedDescription]);
        [localpool drain];
        return;
    }

    // Make writeFrame() a blocking call.
    while (![mMovieWriterInput isReadyForMoreMediaData]) {
        fprintf(stderr, "OpenCV: AVF: waiting to write video data.\n");
        // Sleep 1 msec.
        usleep(1000);
    }

    BOOL success = FALSE;

    if (image.size().height!=movieSize.height || image.size().width!=movieSize.width){
        fprintf(stderr, "OpenCV: Frame size does not match video size.\n");
        [localpool drain];
        return;
    }

    if (movieColor) {
        cv::cvtColor(image, argbimage, cv::COLOR_BGR2BGRA);
    }else{
        cv::cvtColor(image, argbimage, cv::COLOR_GRAY2BGRA);
    }
    //IplImage -> CGImage conversion
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSData *nsData = [NSData dataWithBytes:argbimage.data length:argbimage.total() * argbimage.elemSize()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)nsData);
    CGImageRef cgImage = CGImageCreate(argbimage.size().width, argbimage.size().height,
            8, 32, argbimage.step[0],
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
            &releaseCallback,
            (void *)cfData,
            NULL,
            &pixelBuffer);
    if(status == kCVReturnSuccess){
        success = [mMovieWriterAdaptor appendPixelBuffer:pixelBuffer
            withPresentationTime:CMTimeMake(mFrameNum, mMovieFPS)];
    }

    //cleanup
    CVPixelBufferRelease(pixelBuffer);
    CGImageRelease(cgImage);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    [localpool drain];

    if (success) {
        mFrameNum ++;
        //NSLog(@"Frame #%d", mFrameNum);
    }else{
        NSLog(@"Frame appendPixelBuffer failed.");
    }

}

#pragma clang diagnostic pop
