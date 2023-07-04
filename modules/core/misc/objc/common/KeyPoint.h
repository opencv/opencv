//
//  KeyPoint.h
//
//  Created by Giles Payne on 2019/10/08.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

@class Point2f;

NS_ASSUME_NONNULL_BEGIN
/**
*  Object representing a point feature found by one of many available keypoint detectors, such as Harris corner detector, FAST, StarDetector, SURF, SIFT etc.
*/
CV_EXPORTS @interface KeyPoint : NSObject

#pragma mark - Properties

/**
* Coordinates of the keypoint.
*/
@property Point2f* pt;

/**
* Diameter of the useful keypoint adjacent area.
*/
@property float size;

/**
* Computed orientation of the keypoint (-1 if not applicable).
*/
@property float angle;

/**
* The response, by which the strongest keypoints have been selected. Can
* be used for further sorting or subsampling.
*/
@property float response;

/**
* Octave (pyramid layer), from which the keypoint has been extracted.
*/
@property int octave;

/**
* Object ID, that can be used to cluster keypoints by an object they
* belong to.
*/
@property int classId;

#ifdef __cplusplus
@property(readonly) cv::KeyPoint& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave classId:(int)classId;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::KeyPoint&)keyPoint;
#endif

#pragma mark - Common Methods

/**
* Clone object
*/
- (KeyPoint*)clone;

/**
* Compare for equality
* @param other Object to compare
*/
- (BOOL)isEqual:(nullable id)other;

/**
* Calculate hash value for this object
*/
- (NSUInteger)hash;

/**
* Returns a string that describes the contents of the object
*/
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
