//
//  MatOfPoint3f.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Point3f;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of Point3f objects
*/
@interface MatOfPoint3f : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfPoint3f from Mat object
* @param mat Mat object from which to create MatOfPoint3f
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfPoint3f from array
* @param array Array from which to create MatOfPoint3f
*/
- (instancetype)initWithArray:(NSArray<Point3f*>*)array;

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
- (void)fromArray:(NSArray<Point3f*>*)array;

/**
*  Output Mat elements as an array of Point3f objects
*/
- (NSArray<Point3f*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
