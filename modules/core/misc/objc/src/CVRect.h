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

@class CVPoint;
@class CVSize;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CVRect : NSObject


@property int x;
@property int y;
@property int width;
@property int height;
#ifdef __cplusplus
@property(readonly) cv::Rect& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y width:(int)width height:(int)height;
- (instancetype)initWithPoint:(CVPoint*)point1 point:(CVPoint*)point2;
- (instancetype)initWithPoint:(CVPoint*)point size:(CVSize*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect&)point;
#endif
- (CVRect*)clone;
- (CVPoint*)tl;
- (CVPoint*)br;
- (CVSize*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(CVPoint*)point;

- (void)set:(NSArray<NSNumber*>*)vals;
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;

@end

NS_ASSUME_NONNULL_END
