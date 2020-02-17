//
//  Int4.h
//
//  Created by Giles Payne on 2020/02/05.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class Mat;

NS_ASSUME_NONNULL_BEGIN

@interface Int4 : NSObject

@property int v0;
@property int v1;
@property int v2;
@property int v3;
#ifdef __cplusplus
@property(readonly) cv::Vec4i& nativeRef;
#endif

-(instancetype)init;
-(instancetype)initWithV0:(int)v0 v1:(int)v1 v2:(int)v2 v3:(int)v3;
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec4i&)vec4i;
#endif

-(void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
-(NSArray<NSNumber*>*)get;
-(BOOL)isEqual:(nullable id)other;

@end

NS_ASSUME_NONNULL_END
