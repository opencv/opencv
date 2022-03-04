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

@class Point2f;
@class Size2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a rectange the coordinate and dimension values of which are of type `float`
*/
CV_EXPORTS @interface Rect2f : NSObject

#pragma mark - Properties

@property float x;
@property float y;
@property float width;
@property float height;
#ifdef __cplusplus
@property(readonly) cv::Rect2f& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y width:(float)width height:(float)height;
- (instancetype)initWithPoint:(Point2f*)point1 point:(Point2f*)point2;
- (instancetype)initWithPoint:(Point2f*)point size:(Size2f*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect2f&)point;
#endif

#pragma mark - Methods

/**
* Returns the top left coordinate of the rectangle
*/
- (Point2f*)tl;

/**
* Returns the bottom right coordinate of the rectangle
*/
- (Point2f*)br;

/**
* Returns the size of the rectangle
*/
- (Size2f*)size;

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
- (BOOL)contains:(Point2f*)point;

/**
* Set the rectangle coordinates and dimensions from the values of an array
* @param vals The array of values from which to set the rectangle coordinates and dimensions
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (Rect2f*)clone;

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
