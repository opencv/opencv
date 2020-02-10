//
//  CVSize.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Point2i;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Size2i : NSObject

@property int width;
@property int height;
#ifdef __cplusplus
@property(readonly) cv::Size2i& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithWidth:(int)width height:(int)height;
- (instancetype)initWithPoint:(Point2i*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Size2i&)size;
#endif
+ (instancetype)width:(int)width height:(int)height;

- (double)area;
- (BOOL)empty;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Size2i*)clone;
- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
