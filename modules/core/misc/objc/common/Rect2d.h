//
//  Rect.h
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

@class Point2d;
@class Size2d;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Rect2d : NSObject

@property double x;
@property double y;
@property double width;
@property double height;
#ifdef __cplusplus
@property(readonly) cv::Rect2d& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y width:(double)width height:(double)height;
- (instancetype)initWithPoint:(Point2d*)point1 point:(Point2d*)point2;
- (instancetype)initWithPoint:(Point2d*)point size:(Size2d*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect2d&)point;
#endif
- (Rect2d*)clone;
- (Point2d*)tl;
- (Point2d*)br;
- (Size2d*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(Point2d*)point;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
