//
//  FloatVector.h
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
* Utility class to wrap a `std::vector<float>`
*/
@interface FloatVector : NSObject

#pragma mark - Constructors

/**
* Create FloatVector and initialize with the  contents of an NSData object
* @param data  NSData containing raw float array
*/
-(instancetype)initWithData:(NSData*)data;

/**
* Create FloatVector and initialize with the  contents of another FloatVector object
* @param src  FloatVector containing data to copy
*/
-(instancetype)initWithVector:(FloatVector*)src;

#ifdef __OBJC__
/**
* Create FloatVector from raw C array
* @param array The raw C array
* @elements elements The number of elements in the array
*/
-(instancetype)initWithNativeArray:(float*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
/**
* Create FloatVector from std::vector<float>
* @param src The std::vector<float> object to wrap
*/
-(instancetype)initWithStdVector:(std::vector<float>&)src;
+(instancetype)fromNative:(std::vector<float>&)src;
#endif

#pragma mark - Properties

/**
* Length of the vector
*/
@property(readonly) NSInteger length;

#ifdef __OBJC__
/**
* Raw C array
*/
@property(readonly) float* nativeArray;
#endif

#ifdef __cplusplus
/**
* The wrapped std::vector<float> object
*/
@property(readonly) std::vector<float>& nativeRef;
#endif

/**
* NSData object containing the raw float data
*/
@property(readonly) NSData* data;

#pragma mark - Accessor method

/**
* Return array element
* @param index Index of the array element to return
*/
-(float)get:(NSInteger)index;

@end
NS_ASSUME_NONNULL_END
