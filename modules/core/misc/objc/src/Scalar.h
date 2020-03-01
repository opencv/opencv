//
//  Scalar.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Scalar : NSObject

@property NSArray<NSNumber*>* val;
#ifdef __cplusplus
@property(readonly) cv::Scalar& nativeRef;
#endif

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3 NS_SWIFT_NAME(init(_:_:_:_:));
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 NS_SWIFT_NAME(init(_:_:_:));
- (instancetype)initWithV0:(double)v0 v1:(double)v1 NS_SWIFT_NAME(init(_:_:));
- (instancetype)initWithV0:(double)v0 NS_SWIFT_NAME(init(_:));
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Scalar&)nativeScalar;
#endif

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
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
