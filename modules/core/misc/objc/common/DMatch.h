//
//  DMatch.h
//
//  Created by Giles Payne on 2019/12/25.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Structure for matching: query descriptor index, train descriptor index, train
* image index and distance between descriptors.
*/
@interface DMatch : NSObject

/**
 * Query descriptor index.
 */
@property int queryIdx;

/**
* Train descriptor index.
*/
@property int trainIdx;

/**
* Train image index.
*/
@property int imgIdx;

/**
* Distance
*/
@property float distance;
#ifdef __cplusplus
@property(readonly) cv::DMatch& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx distance:(float)distance;
- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx imgIdx:(int)imgIdx distance:(float)distance;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::DMatch&)dMatch;
#endif

/**
* Distance comparison
* @param it  DMatch object to compare
*/
- (BOOL)lessThan:(DMatch*)it;

/**
* Clone object
*/
- (DMatch*)clone;

/**
* Compare for equality
* @param other Object to compare
*/
- (BOOL)isEqual:(nullable id)other;

/**
* Calculate hash for this object
*/
- (NSUInteger)hash;

/**
* Returns a string that describes the contents of the object
*/
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
