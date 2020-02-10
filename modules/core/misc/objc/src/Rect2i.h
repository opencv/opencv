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

@class Point2i;
@class Size2i;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Rect2i : NSObject

@property int x;
@property int y;
@property int width;
@property int height;
#ifdef __cplusplus
@property(readonly) cv::Rect2i& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y width:(int)width height:(int)height;
- (instancetype)initWithPoint:(Point2i*)point1 point:(Point2i*)point2;
- (instancetype)initWithPoint:(Point2i*)point size:(Size2i*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect&)point;
#endif
- (Rect2i*)clone;
- (Point2i*)tl;
- (Point2i*)br;
- (Size2i*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(Point2i*)point;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;

@end

NS_ASSUME_NONNULL_END
