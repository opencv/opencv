//
//  Point2d.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Rect2d;

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a two dimensional point the coordinate values of which are of type `double`
*/
@interface Point2d : NSObject

# pragma mark - Properties

@property double x;
@property double y;
#ifdef __cplusplus
@property(readonly) cv::Point2d& nativeRef;
#endif

# pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2d&)point;
- (void)update:(cv::Point2d&)point;
#endif

# pragma mark - Methods

/**
* Calculate the dot product of this point and another point
* @param point The other point
*/
- (double)dot:(Point2d*)point;

/**
* Determine if the point lies with a specified rectangle
* @param rect The rectangle
*/
- (BOOL)inside:(Rect2d*)rect;

/**
* Set the point coordinates from the values of an array
* @param vals The array of values from which to set the coordinates
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Point2d*)clone;

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
