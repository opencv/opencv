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

@interface Rect2d : NSObject

@property double x;
@property double y;
@property double width;
@property double height;

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y width:(double)width height:(double)height;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (void)set:(NSArray<NSNumber*>*)vals;

- (Rect2d*)clone;
- (CVPoint*)tl;
- (CVPoint*)br;
- (CVSize*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(CVPoint*)point;

- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
