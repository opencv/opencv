//
//  Size2d.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Point2d;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Size2d : NSObject

@property double width;
@property double height;
#ifdef __cplusplus
@property(readonly) cv::Size2d& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithWidth:(double)width height:(double)height;
- (instancetype)initWithPoint:(Point2d*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Size2d&)size;
#endif
+ (instancetype)width:(double)width height:(double)height;

- (double)area;
- (BOOL)empty;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Size2d*)clone;
- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
