//
//  MatOfPoint3.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Point3i;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of Point3i objects
*/
CV_EXPORTS @interface MatOfPoint3 : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfPoint3 from Mat object
* @param mat Mat object from which to create MatOfPoint3
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfPoint3 from array
* @param array Array from which to create MatOfPoint3
*/
- (instancetype)initWithArray:(NSArray<Point3i*>*)array;

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
- (void)fromArray:(NSArray<Point3i*>*)array;

/**
*  Output Mat elements as an array of Point3i objects
*/
- (NSArray<Point3i*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
