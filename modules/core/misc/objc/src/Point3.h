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

@class CVPoint;

NS_ASSUME_NONNULL_BEGIN

@interface Point3 : NSObject

@property double x;
@property double y;
@property double z;

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y z:(double)z;
- (instancetype)initWithPoint:(CVPoint*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (void)set:(NSArray<NSNumber*>*)vals;
- (Point3*)clone;
- (double)dot:(Point3*)point;
- (Point3*)cross:(Point3*)point;

- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
