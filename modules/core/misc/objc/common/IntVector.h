//
//  IntVector.h
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
* Utility class to wrap a `std::vector<int>`
*/
@interface IntVector : NSObject

#pragma mark - Constructors

/**
* Create IntVector and initialize with the  contents of an NSData object
* @param data  NSData containing raw int array
*/
-(instancetype)initWithData:(NSData*)data;

/**
* Create IntVector and initialize with the  contents of another IntVector object
* @param src  IntVector containing data to copy
*/
-(instancetype)initWithVector:(IntVector*)src;

#ifdef __OBJC__
/**
* Create IntVector from raw C array
* @param array The raw C array
* @elements elements The number of elements in the array
*/
-(instancetype)initWithNativeArray:(int*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
/**
* Create IntVector from std::vector<int>
* @param src The std::vector<int> object to wrap
*/
-(instancetype)initWithStdVector:(std::vector<int>&)src;
+(instancetype)fromNative:(std::vector<int>&)src;
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
@property(readonly) int* nativeArray;
#endif

#ifdef __cplusplus
/**
* The wrapped std::vector<int> object
*/
@property(readonly) std::vector<int>& nativeRef;
#endif

/**
* NSData object containing the raw int data
*/
@property(readonly) NSData* data;

#pragma mark - Accessor method

/**
* Return array element
* @param index Index of the array element to return
*/
-(int)get:(NSInteger)index;

@end
NS_ASSUME_NONNULL_END
