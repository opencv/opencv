//
//  MatOfKeyPoint.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class KeyPoint;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of KeyPoint objects
*/
@interface MatOfKeyPoint : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfKeyPoint from Mat object
* @param mat Mat object from which to create MatOfKeyPoint
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfKeyPoint from array
* @param array Array from which to create MatOfKeyPoint
*/
- (instancetype)initWithArray:(NSArray<KeyPoint*>*)array;

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
- (void)fromArray:(NSArray<KeyPoint*>*)array;

/**
*  Output Mat elements as an array of KeyPoint objects
*/
- (NSArray<KeyPoint*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
