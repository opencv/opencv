//
//  Float6.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Mat;

NS_ASSUME_NONNULL_BEGIN

@interface Float6 : NSObject

@property float v0;
@property float v1;
@property float v2;
@property float v3;
@property float v4;
@property float v5;
#ifdef __cplusplus
@property(readonly) cv::Vec6f& nativeRef;
#endif

-(instancetype)init;
-(instancetype)initWithV0:(float)v0 v1:(float)v1 v2:(float)v2 v3:(float)v3 v4:(float)v4 v5:(float)v5;
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec6f&)vec6f;
#endif

-(void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
-(NSArray<NSNumber*>*)get;
-(BOOL)isEqual:(nullable id)other;

@end

NS_ASSUME_NONNULL_END
