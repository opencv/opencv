//
//  Int4.h
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
* Simple wrapper for a vector of four `int`
*/
@interface Int4 : NSObject

#pragma mark - Properties

/**
* First vector element
*/
@property int v0;

/**
* Second vector element
*/
@property int v1;

/**
* Third vector element
*/
@property int v2;

/**
* Fourth vector element
*/
@property int v3;

#ifdef __cplusplus
/**
* The wrapped vector
*/
@property(readonly) cv::Vec4i& nativeRef;
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
*/
-(instancetype)initWithV0:(int)v0 v1:(int)v1 v2:(int)v2 v3:(int)v3;

/**
* Create vector with specified element values
* @param vals array of element values
*/
-(instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+(instancetype)fromNative:(cv::Vec4i&)vec4i;
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
