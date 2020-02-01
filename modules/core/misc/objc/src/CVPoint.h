//
//  Point.h
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>
#import "ObjectVector.h"

@class CVRect;

NS_ASSUME_NONNULL_BEGIN

@interface CVPoint : NSObject

@property double x;
@property double y;
#ifdef __cplusplus
@property(readonly) cv::Point2d& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2d&)point;
#endif
- (CVPoint*)clone;
- (double)dot:(CVPoint*)point;
- (BOOL)inside:(CVRect*)rect;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
