//
//  Rect.h
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Point2f;
@class Size2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Rect2f : NSObject

@property float x;
@property float y;
@property float width;
@property float height;
#ifdef __cplusplus
@property(readonly) cv::Rect2f& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y width:(float)width height:(float)height;
- (instancetype)initWithPoint:(Point2f*)point1 point:(Point2f*)point2;
- (instancetype)initWithPoint:(Point2f*)point size:(Size2f*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect2f&)point;
#endif
- (Rect2f*)clone;
- (Point2f*)tl;
- (Point2f*)br;
- (Size2f*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(Point2f*)point;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
