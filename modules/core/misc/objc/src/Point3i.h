//
//  Point3i.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class Point2i;

NS_ASSUME_NONNULL_BEGIN

@interface Point3i : NSObject

@property int x;
@property int y;
@property int z;
#ifdef __cplusplus
@property(readonly) cv::Point3i& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y z:(int)z;
- (instancetype)initWithPoint:(Point2i*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (Point3i*)clone;
- (double)dot:(Point3i*)point;
- (Point3i*)cross:(Point3i*)point;

- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
