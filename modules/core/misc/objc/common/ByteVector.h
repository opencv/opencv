//
//  ByteVector.h
//
//  Created by Giles Payne on 2020/01/04.
//

#pragma once

#import <Foundation/Foundation.h>
#ifdef __cplusplus
#import <vector>
#endif
#import "CVObjcUtil.h"

NS_ASSUME_NONNULL_BEGIN

/**
* Utility class to wrap a `std::vector<char>`
*/
CV_EXPORTS @interface ByteVector : NSObject

#pragma mark - Constructors

/**
* Create ByteVector and initialize with the  contents of an NSData object
* @param data  NSData containing raw byte array
*/
-(instancetype)initWithData:(NSData*)data;

/**
* Create ByteVector and initialize with the  contents of another ByteVector object
* @param src  ByteVector containing data to copy
*/
-(instancetype)initWithVector:(ByteVector*)src;

#ifdef __OBJC__
/**
* Create ByteVector from raw C array
* @param array The raw C array
* @elements elements The number of elements in the array
*/
-(instancetype)initWithNativeArray:(char*)array elements:(NSInteger)elements;
#endif

#ifdef __cplusplus
/**
* Create ByteVector from std::vector<char>
* @param src The std::vector<char> object to wrap
*/
-(instancetype)initWithStdVector:(std::vector<char>&)src;
+(instancetype)fromNative:(std::vector<char>&)src;
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
@property(readonly) char* nativeArray;
#endif

#ifdef __cplusplus
/**
* The wrapped std::vector<char> object
*/
@property(readonly) std::vector<char>& nativeRef;
#endif

/**
* NSData object containing the raw byte data
*/
@property(readonly) NSData* data;

#pragma mark - Accessor method

/**
* Return array element
* @param index Index of the array element to return
*/
-(char)get:(NSInteger)index;

@end
NS_ASSUME_NONNULL_END
