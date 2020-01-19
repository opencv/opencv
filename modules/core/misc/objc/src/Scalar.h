//
//  Scalar.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Scalar : NSObject

@property NSMutableArray<NSNumber*>* val;
#ifdef __cplusplus
@property(readonly) cv::Scalar& nativeRef;
#endif

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2;
- (instancetype)initWithV0:(double)v0 v1:(double)v1;
- (instancetype)initWithV0:(double)v0;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Scalar&)nativeScalar;
#endif

- (void)set:(NSArray<NSNumber*>*)vals;
+ (Scalar*)all:(double)v;
- (Scalar*)clone;
- (Scalar*)mul:(Scalar*)it scale:(double)scale;
- (Scalar*)mul:(Scalar*)it;
- (Scalar*)conj;
- (BOOL)isReal;

- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString *)description;

@end

NS_ASSUME_NONNULL_END
