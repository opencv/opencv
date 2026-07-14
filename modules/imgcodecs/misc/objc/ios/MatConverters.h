//
//  MatConverters.h
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

CV_EXPORTS @interface MatConverters : NSObject

+(CGImageRef)convertMatToCGImageRef:(Mat*)mat CF_RETURNS_RETAINED;
+(Mat*)convertCGImageRefToMat:(CGImageRef)image;
+(Mat*)convertCGImageRefToMat:(CGImageRef)image alphaExist:(BOOL)alphaExist;
+(UIImage*)converMatToUIImage:(Mat*)mat;
+(Mat*)convertUIImageToMat:(UIImage*)image;
+(Mat*)convertUIImageToMat:(UIImage*)image alphaExist:(BOOL)alphaExist;

@end

NS_ASSUME_NONNULL_END
