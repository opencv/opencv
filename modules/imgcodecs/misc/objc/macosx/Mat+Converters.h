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

CV_EXPORTS @interface Mat (Converters)

-(CGImageRef)toCGImage CF_RETURNS_RETAINED;
-(instancetype)initWithCGImage:(CGImageRef)image;
-(instancetype)initWithCGImage:(CGImageRef)image alphaExist:(BOOL)alphaExist;
-(NSImage*)toNSImage;
-(instancetype)initWithNSImage:(NSImage*)image;
-(instancetype)initWithNSImage:(NSImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
