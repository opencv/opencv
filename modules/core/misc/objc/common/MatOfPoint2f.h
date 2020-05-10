//
//  MatOfPoint2f.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

@class Point2f;

/**
* Mat representation of an array of Point2f objects
*/
@interface MatOfPoint2f : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfPoint2f from Mat object
* @param mat Mat object from which to create MatOfPoint2f
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfPoint2f from array
* @param array Array from which to create MatOfPoint2f
*/
- (instancetype)initWithArray:(NSArray<Point2f*>*)array;

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
- (void)fromArray:(NSArray<Point2f*>*)array;

/**
*  Output Mat elements as an array of Point2f objects
*/
- (NSArray<Point2f*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
