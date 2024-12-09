//
//  Rect.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

@class Point2d;
@class Size2d;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a rectange the coordinate and dimension values of which are of type `double`
*/
CV_EXPORTS @interface Rect2d : NSObject

#pragma mark - Properties

@property double x;
@property double y;
@property double width;
@property double height;
#ifdef __cplusplus
@property(readonly) cv::Rect2d& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y width:(double)width height:(double)height;
- (instancetype)initWithPoint:(Point2d*)point1 point:(Point2d*)point2;
- (instancetype)initWithPoint:(Point2d*)point size:(Size2d*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect2d&)point;
#endif

#pragma mark - Methods

/**
* Returns the top left coordinate of the rectangle
*/
- (Point2d*)tl;

/**
* Returns the bottom right coordinate of the rectangle
*/
- (Point2d*)br;

/**
* Returns the size of the rectangle
*/
- (Size2d*)size;

/**
* Returns the area of the rectangle
*/
- (double)area;

/**
* Determines if the rectangle is empty
*/
- (BOOL)empty;

/**
* Determines if the rectangle contains a given point
* @param point The point
*/
- (BOOL)contains:(Point2d*)point;

/**
* Set the rectangle coordinates and dimensions from the values of an array
* @param vals The array of values from which to set the rectangle coordinates and dimensions
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (Rect2d*)clone;

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
