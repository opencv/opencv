//
//  MatOfRotatedRect.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class RotatedRect;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of RotatedRect objects
*/
@interface MatOfRotatedRect : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfRotatedRect from Mat object
* @param mat Mat object from which to create MatOfRotatedRect
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfRotatedRect from array
* @param array Array from which to create MatOfRotatedRect
*/
- (instancetype)initWithArray:(NSArray<RotatedRect*>*)array;

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
- (void)fromArray:(NSArray<RotatedRect*>*)array;

/**
*  Output Mat elements as an array of RotatedRect objects
*/
- (NSArray<RotatedRect*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
