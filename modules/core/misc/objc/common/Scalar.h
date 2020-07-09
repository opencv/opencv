//
//  Scalar.h
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

/**
* Represents a four element vector
*/
CV_EXPORTS @interface Scalar : NSObject

#pragma mark - Properties

@property(readonly) NSArray<NSNumber*>* val;
#ifdef __cplusplus
@property(readonly) cv::Scalar& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3 NS_SWIFT_NAME(init(_:_:_:_:));
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 NS_SWIFT_NAME(init(_:_:_:));
- (instancetype)initWithV0:(double)v0 v1:(double)v1 NS_SWIFT_NAME(init(_:_:));
- (instancetype)initWithV0:(double)v0 NS_SWIFT_NAME(init(_:));
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Scalar&)nativeScalar;
#endif

#pragma mark - Methods

/**
* Creates a scalar with all elements of the same value
* @param v The value to set each element to
*/
+ (Scalar*)all:(double)v;

/**
* Calculates per-element product with another Scalar and a scale factor
* @param it The other Scalar
* @param scale The scale factor
*/
- (Scalar*)mul:(Scalar*)it scale:(double)scale;

/**
* Calculates per-element product with another Scalar
* @param it The other Scalar
*/
- (Scalar*)mul:(Scalar*)it;

/**
* Returns (v0, -v1, -v2, -v3)
*/
- (Scalar*)conj;

/**
* Returns true iff v1 == v2 == v3 == 0
*/
- (BOOL)isReal;

#pragma mark - Common Methods

/**
* Clone object
*/
- (Scalar*)clone;

/**
* Compare for equality
* @param other Object to compare
*/
- (BOOL)isEqual:(nullable id)object;

/**
* Calculate hash value for this object
*/
- (NSUInteger)hash;

/**
* Returns a string that describes the contents of the object
*/
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
