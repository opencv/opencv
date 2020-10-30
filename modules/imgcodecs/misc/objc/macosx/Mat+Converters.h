//
//  Mat+Converters.h
//
//  Created by Masaya Tsuruta on 2020/10/08.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface Mat (Converters)

-(CGImageRef)toCGImage;
-(instancetype)initWithCGImage:(CGImageRef)image;
-(instancetype)initWithCGImage:(CGImageRef)image alphaExist:(BOOL)alphaExist;
-(NSImage*)toNSImage;
-(instancetype)initWithNSImage:(NSImage*)image;
-(instancetype)initWithNSImage:(NSImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
