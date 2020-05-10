//
//  DoubleVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif

NS_ASSUME_NONNULL_BEGIN

/**
* Utility class to wrap a `std::vector<double>`
*/
@interface DoubleVector : NSObject

#pragma mark - Constructors

/**
* Create DoubleVector and initialize with the  contents of an NSData object
* @param data  NSData containing raw double array
*/
-(instancetype)initWithData:(NSData*)data;

/**
* Create DoubleVector and initialize with the  contents of another DoubleVector object
* @param src  DoubleVector containing data to copy
*/
-(instancetype)initWithVector:(DoubleVector*)src;

#ifdef __OBJC__
/**
* Create DoubleVector from raw C array
* @param array The raw C array
* @elements elements The number of elements in the array
*/
-(instancetype)initWithNativeArray:(double*)array elements:(int)elements;
#endif

#ifdef __cplusplus
/**
* Create DoubleVector from std::vector<double>
* @param src The std::vector<double> object to wrap
*/
-(instancetype)initWithStdVector:(std::vector<double>&)src;
+(instancetype)fromNative:(std::vector<double>&)src;
#endif

#pragma mark - Properties

/**
* Length of the vector
*/
@property(readonly) size_t length;

#ifdef __OBJC__
/**
* Raw C array
*/
@property(readonly) double* nativeArray;
#endif

#ifdef __cplusplus
/**
* The wrapped std::vector<double> object
*/
@property(readonly) std::vector<double>& nativeRef;
#endif

/**
* NSData object containing the raw double data
*/
@property(readonly) NSData* data;

#pragma mark - Accessor method

/**
* Return array element
* @param index Index of the array element to return
*/
-(double)get:(NSInteger)index;

@end
NS_ASSUME_NONNULL_END
