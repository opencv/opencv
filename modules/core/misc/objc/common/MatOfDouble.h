//
//  MatOfDouble.h
//
//  Created by Giles Payne on 2019/12/26.
//

#pragma once

#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of doubles
*/
CV_EXPORTS @interface MatOfDouble : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfDouble from Mat object
* @param mat Mat object from which to create MatOfDouble
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfDouble from array
* @param array Array from which to create MatOfDouble
*/
- (instancetype)initWithArray:(NSArray<NSNumber*>*)array;

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
- (void)fromArray:(NSArray<NSNumber*>*)array;

/**
*  Output Mat elements as an array
*/
- (NSArray<NSNumber*>*)toArray;


/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
