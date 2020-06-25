//
//  Moments.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

CV_EXPORTS @interface Moments : NSObject

@property double m00;
@property double m10;
@property double m01;
@property double m20;
@property double m11;
@property double m02;
@property double m30;
@property double m21;
@property double m12;
@property double m03;

@property double mu20;
@property double mu11;
@property double mu02;
@property double mu30;
@property double mu21;
@property double mu12;
@property double mu03;

@property double nu20;
@property double nu11;
@property double nu02;
@property double nu30;
@property double nu21;
@property double nu12;
@property double nu03;

#ifdef __cplusplus
@property(readonly) cv::Moments& nativeRef;
#endif

-(instancetype)initWithM00:(double)m00 m10:(double)m10 m01:(double)m01 m20:(double)m20 m11:(double)m11 m02:(double)m02 m30:(double)m30 m21:(double)m21 m12:(double)m12 m03:(double)m03;

-(instancetype)init;

-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+(instancetype)fromNative:(cv::Moments&)moments;
#endif

-(void)set:(NSArray<NSNumber*>*)vals;
-(void)completeState;
-(NSString *)description;

@end

NS_ASSUME_NONNULL_END
