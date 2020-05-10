//
//  Rect2i.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

@class Point2i;
@class Size2i;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a rectange the coordinate and dimension values of which are of type `int`
*/
NS_SWIFT_NAME(Rect)
@interface Rect2i : NSObject

#pragma mark - Properties

@property int x;
@property int y;
@property int width;
@property int height;
#ifdef __cplusplus
@property(readonly) cv::Rect2i& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y width:(int)width height:(int)height;
- (instancetype)initWithPoint:(Point2i*)point1 point:(Point2i*)point2;
- (instancetype)initWithPoint:(Point2i*)point size:(Size2i*)size;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Rect&)point;
#endif

#pragma mark - Methods

/**
* Returns the top left coordinate of the rectangle
*/
- (Point2i*)tl;

/**
* Returns the bottom right coordinate of the rectangle
*/
- (Point2i*)br;

/**
* Returns the size of the rectangle
*/
- (Size2i*)size;

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
- (BOOL)contains:(Point2i*)point;

/**
* Set the rectangle coordinates and dimensions from the values of an array
* @param vals The array of values from which to set the rectangle coordinates and dimensions
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (Rect2i*)clone;

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
