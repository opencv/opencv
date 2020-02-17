//
//  Size2f.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Point2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Size2f : NSObject

@property float width;
@property float height;
#ifdef __cplusplus
@property(readonly) cv::Size2f& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithWidth:(float)width height:(float)height;
- (instancetype)initWithPoint:(Point2f*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Size2f&)size;
#endif
+ (instancetype)width:(float)width height:(float)height;

- (double)area;
- (BOOL)empty;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Size2f*)clone;
- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
