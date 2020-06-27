//
//  Double3.h
//
//  Created by Giles Payne on 2020/05/22.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

@class Mat;

NS_ASSUME_NONNULL_BEGIN

/**
* Simple wrapper for a vector of three `double`
*/
CV_EXPORTS @interface Double3 : NSObject

#pragma mark - Properties

/**
* First vector element
*/
@property double v0;

/**
* Second vector element
*/
@property double v1;

/**
* Third vector element
*/
@property double v2;


#ifdef __cplusplus
/**
* The wrapped vector
*/
@property(readonly) cv::Vec3d& nativeRef;
#endif

#pragma mark - Constructors

/**
* Create zero-initialize vecior
*/
-(instancetype)init;

/**
* Create vector with specified element values
* @param v0 First element
* @param v1 Second element
* @param v2 Third element
*/
-(instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2;

/**
* Create vector with specified element values
* @param vals array of element values
*/
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec3d&)vec3d;
#endif

/**
* Update vector with specified element values
* @param vals array of element values
*/
-(void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

/**
* Get vector as an array
*/
-(NSArray<NSNumber*>*)get;

#pragma mark - Common Methods

/**
* Compare for equality
* @param other Object to compare
*/
-(BOOL)isEqual:(nullable id)other;

@end

NS_ASSUME_NONNULL_END
