//
//  Point3d.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class Point2d;

NS_ASSUME_NONNULL_BEGIN

@interface Point3d : NSObject

@property double x;
@property double y;
@property double z;
#ifdef __cplusplus
@property(readonly) cv::Point3d& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y z:(double)z;
- (instancetype)initWithPoint:(Point2d*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Point3d*)clone;
- (double)dot:(Point3d*)point;
- (Point3d*)cross:(Point3d*)point;

- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
