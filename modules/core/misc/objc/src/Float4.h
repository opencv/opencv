//
//  Float4.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class Mat;

@interface Float4 : NSObject

@property float v0;
@property float v1;
@property float v2;
@property float v3;
#ifdef __cplusplus
@property(readonly) cv::Vec4f& nativeRef;
#endif

-(instancetype)init;
-(instancetype)initWithV0:(float)v0 v1:(float)v1 v2:(float)v2 v3:(float)v3;
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec4f&)vec4f;
#endif

-(void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
-(NSArray<NSNumber*>*)get;
-(BOOL)isEqual:(nullable id)other;

@end
