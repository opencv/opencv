// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGCODECS_APPLE_CONVERSIONS_H
#define OPENCV_IMGCODECS_APPLE_CONVERSIONS_H

#include "opencv2/core.hpp"
#import <Accelerate/Accelerate.h>

// Guard AVFoundation import based on CMake configuration
#ifdef HAVE_AVFOUNDATION
#import <AVFoundation/AVFoundation.h>
#endif

// Guard ImageIO import: available on iOS and macOS 10.8+
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE || (defined(MAC_OS_X_VERSION_MAX_ALLOWED) && MAC_OS_X_VERSION_MAX_ALLOWED >= 1080)
#import <ImageIO/ImageIO.h>
#endif

CV_EXPORTS CGImageRef MatToCGImage(const cv::Mat& image) CF_RETURNS_RETAINED;
CV_EXPORTS void CGImageToMat(const CGImageRef image, cv::Mat& m, bool alphaExist);

#endif // OPENCV_IMGCODECS_APPLE_CONVERSIONS_H