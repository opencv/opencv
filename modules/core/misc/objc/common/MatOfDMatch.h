//
//  MatOfDMatch.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class DMatch;

NS_ASSUME_NONNULL_BEGIN

/**
* Mat representation of an array of DMatch objects
*/
@interface MatOfDMatch : Mat

#pragma mark - Constructors

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

/**
*  Create MatOfDMatch from Mat object
* @param mat Mat object from which to create MatOfDMatch
*/
- (instancetype)initWithMat:(Mat*)mat;

/**
*  Create MatOfDMatch from array
* @param array Array from which to create MatOfDMatch
*/
- (instancetype)initWithArray:(NSArray<DMatch*>*)array;

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
- (void)fromArray:(NSArray<DMatch*>*)array;

/**
*  Output Mat elements as an array of DMatch objects
*/
- (NSArray<DMatch*>*)toArray;

/**
*  Total number of values in Mat
*/
- (int)length;

@end

NS_ASSUME_NONNULL_END
