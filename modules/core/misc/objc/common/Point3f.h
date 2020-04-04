//
//  Point3f.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Point2f;

NS_ASSUME_NONNULL_BEGIN

@interface Point3f : NSObject

@property float x;
@property float y;
@property float z;
#ifdef __cplusplus
@property(readonly) cv::Point3f& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y z:(float)z;
- (instancetype)initWithPoint:(Point2f*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Point3f*)clone;
- (double)dot:(Point3f*)point;
- (Point3f*)cross:(Point3f*)point;

- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
