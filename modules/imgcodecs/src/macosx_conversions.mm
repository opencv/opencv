// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "apple_conversions.h"
#import <AppKit/AppKit.h>

CV_EXPORTS NSImage* MatToNSImage(const cv::Mat& image);
CV_EXPORTS void NSImageToMat(const NSImage* image, cv::Mat& m, bool alphaExist);

NSImage* MatToNSImage(const cv::Mat& image) {
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = MatToCGImage(image);

    // Getting NSImage from CGImage
    NSImage *nsImage = [[NSImage alloc] initWithCGImage:imageRef size:CGSizeMake(CGImageGetWidth(imageRef), CGImageGetHeight(imageRef))];
    CGImageRelease(imageRef);

    return nsImage;
}

void NSImageToMat(const NSImage* image, cv::Mat& m, bool alphaExist) {
    CGImageRef imageRef = [image CGImageForProposedRect:NULL context:NULL hints:NULL];
    CGImageToMat(imageRef, m, alphaExist);
}
