//
//  Double2.h
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
* Simple wrapper for a vector of two `double`
*/
CV_EXPORTS @interface Double2 : NSObject

#pragma mark - Properties

/**
* First vector element
*/
@property double v0;

/**
* Second vector element
*/
@property double v1;


#ifdef __cplusplus
/**
* The wrapped vector
*/
@property(readonly) cv::Vec2d& nativeRef;
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
*/
-(instancetype)initWithV0:(double)v0 v1:(double)v1;

/**
* Create vector with specified element values
* @param vals array of element values
*/
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec2d&)vec2d;
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
