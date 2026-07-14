//
//  Mat+Converters.h
//
//  Created by Masaya Tsuruta on 2020/10/08.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import "Mat.h"
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface MatConverters : NSObject

+(CGImageRef)convertMatToCGImageRef:(Mat*)mat CF_RETURNS_RETAINED;
+(Mat*)convertCGImageRefToMat:(CGImageRef)image;
+(Mat*)convertCGImageRefToMat:(CGImageRef)image alphaExist:(BOOL)alphaExist;
+(NSImage*)converMatToNSImage:(Mat*)mat;
+(Mat*)convertNSImageToMat:(NSImage*)image;
+(Mat*)convertNSImageToMat:(NSImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
