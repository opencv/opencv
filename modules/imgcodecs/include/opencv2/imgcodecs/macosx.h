
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#import <AppKit/AppKit.h>
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>
#include "opencv2/core.hpp"

//! @addtogroup imgcodecs_macosx
//! @{

CV_EXPORTS CGImageRef MatToCGImage(const cv::Mat& image);
CV_EXPORTS void CGImageToMat(const CGImageRef image, cv::Mat& m, bool alphaExist = false);
CV_EXPORTS NSImage* MatToNSImage(const cv::Mat& image);
CV_EXPORTS void NSImageToMat(const NSImage* image, cv::Mat& m, bool alphaExist = false);

//! @}
