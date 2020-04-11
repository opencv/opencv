//
//  Point.h
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Rect2i;

NS_ASSUME_NONNULL_BEGIN

@interface Point2i : NSObject

@property int x;
@property int y;
#ifdef __cplusplus
@property(readonly) cv::Point2i& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2i&)point;
- (void)update:(cv::Point2i&)point;
#endif
- (Point2i*)clone;
- (double)dot:(Point2i*)point;
- (BOOL)inside:(Rect2i*)rect;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

typedef Point2i CvPoint;

NS_ASSUME_NONNULL_END
