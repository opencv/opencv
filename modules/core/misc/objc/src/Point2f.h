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

@class Rect2f;

NS_ASSUME_NONNULL_BEGIN

@interface Point2f : NSObject

@property float x;
@property float y;
#ifdef __cplusplus
@property(readonly) cv::Point2f& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2f&)point;
- (void)update:(cv::Point2f&)point;
#endif
- (Point2f*)clone;
- (double)dot:(Point2f*)point;
- (BOOL)inside:(Rect2f*)rect;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
