// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/core.hpp"
#import <Accelerate/Accelerate.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>

CV_EXPORTS CGImageRef MatToCGImage(const cv::Mat& image) CF_RETURNS_RETAINED;
CV_EXPORTS void CGImageToMat(const CGImageRef image, cv::Mat& m, bool alphaExist);
