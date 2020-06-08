//
//  RotatedRect.h
//
//  Created by Giles Payne on 2019/12/26.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

@class Point2f;
@class Size2f;
@class Rect2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a rotated rectangle on a plane
*/
@interface RotatedRect : NSObject

#pragma mark - Properties

@property Point2f* center;
@property Size2f* size;
@property double angle;
#ifdef __cplusplus
@property(readonly) cv::RotatedRect& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithCenter:(Point2f*)center size:(Size2f*)size angle:(double)angle;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::RotatedRect&)rotatedRect;
#endif

#pragma mark - Methods
/**
* Returns the corner points of the rotated rectangle as an array
*/
- (NSArray<Point2f*>*)points;

/**
* Returns the bounding (non-rotated) rectangle of the rotated rectangle
*/
- (Rect2f*)boundingRect;

/**
* Set the rotated rectangle coordinates, dimensions and angle of rotation from the values of an array
* @param vals The array of values from which to set the rotated rectangle coordinates, dimensions and angle of rotation
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (RotatedRect*)clone;

/**
* Compare for equality
* @param other Object to compare
*/
- (BOOL)isEqual:(nullable id)object;

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
