//
//  RotatedRect.h
//
//  Created by Giles Payne on 2019/12/26.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Point2f;
@class Size2f;
@class Rect2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface RotatedRect : NSObject

@property(assign) Point2f* center;
@property(assign) Size2f* size;
@property double angle;
#ifdef __cplusplus
@property(readonly) cv::RotatedRect& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithCenter:(Point2f*)center size:(Size2f*)size angle:(double)angle;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::RotatedRect&)rotatedRect;
#endif

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

- (NSArray<Point2f*>*)points;
- (Rect2f*)boundingRect;

- (RotatedRect*)clone;
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
