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

/**
* Simple wrapper for a vector of six `float`
*/
@interface Float6 : NSObject

#pragma mark - Properties

/**
* First vector element
*/
@property float v0;

/**
* Second vector element
*/
@property float v1;

/**
* Third vector element
*/
@property float v2;

/**
* Fourth vector element
*/
@property float v3;

/**
* Fifth vector element
*/
@property float v4;

/**
* Sixth vector element
*/
@property float v5;

#ifdef __cplusplus
/**
* The wrapped vector
*/
@property(readonly) cv::Vec6f& nativeRef;
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
* @param v3 Fourth element
* @param v4 Fifth element
* @param v5 Sixth element
*/
-(instancetype)initWithV0:(float)v0 v1:(float)v1 v2:(float)v2 v3:(float)v3 v4:(float)v4 v5:(float)v5;

/**
* Create vector with specified element values
* @param vals array of element values
*/
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec6f&)vec6f;
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
