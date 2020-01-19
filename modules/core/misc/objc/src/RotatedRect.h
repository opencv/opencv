//
//  RotatedRect.h
//
//  Created by Giles Payne on 2019/12/26.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class CVPoint;
@class CVSize;
@class CVRect;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface RotatedRect : NSObject

@property CVPoint* center;
@property CVSize* size;
@property double angle;

- (instancetype)init;
- (instancetype)initWithCenter:(CVPoint*)center size:(CVSize*)size angle:(double)angle;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

- (void)set:(NSArray<NSNumber*>*)vals;

- (NSArray<CVPoint*>*)points;
- (CVRect*)boundingRect;

- (RotatedRect*)clone;
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
