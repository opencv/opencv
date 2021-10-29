//
//  Mat+Converters.h
//
//  Created by Giles Payne on 2020/03/03.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import "Mat.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface Mat (Converters)

-(CGImageRef)toCGImage CF_RETURNS_RETAINED;
-(instancetype)initWithCGImage:(CGImageRef)image;
-(instancetype)initWithCGImage:(CGImageRef)image alphaExist:(BOOL)alphaExist;
-(UIImage*)toUIImage;
-(instancetype)initWithUIImage:(UIImage*)image;
-(instancetype)initWithUIImage:(UIImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
