//
//  MatOfRect2d.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Rect2d;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of Rect2d objects
*/
@interface MatOfRect2d : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
+ (instancetype)fromNative:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfRect2d from Mat object
* @param mat Mat object from which to create MatOfRect2d
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfRect2d from array
* @param array Array from which to create MatOfRect2d
*/
- (instancetype)initWithArray:(NSArray<Rect2d*>*)array;

#pragma mark - Methods

/**
*  Allocate specified number of elements
* @param elemNumber Number of elements
*/
- (void)alloc:(int)elemNumber;

/**
*  Populate Mat with elements of an array
* @param array Array with which to populate the Mat
*/
- (void)fromArray:(NSArray<Rect2d*>*)array;

/**
*  Output Mat elements as an array of Rect2d objects
*/
- (NSArray<Rect2d*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
