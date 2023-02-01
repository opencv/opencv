//
//  MatOfRect2i.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Rect2i;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of Rect objects
*/
NS_SWIFT_NAME(MatOfRect)
CV_EXPORTS @interface MatOfRect2i : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfRect from Mat object
* @param mat Mat object from which to create MatOfRect
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfRect from array
* @param array Array from which to create MatOfRect
*/
- (instancetype)initWithArray:(NSArray<Rect2i*>*)array;

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
- (void)fromArray:(NSArray<Rect2i*>*)array;

/**
*  Output Mat elements as an array of Rect objects
*/
- (NSArray<Rect2i*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
