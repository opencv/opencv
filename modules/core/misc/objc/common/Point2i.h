//
//  Point2i.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Rect2i;

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a two dimensional point the coordinate values of which are of type `int`
*/
NS_SWIFT_NAME(Point)
@interface Point2i : NSObject

# pragma mark - Properties

@property int x;
@property int y;
#ifdef __cplusplus
@property(readonly) cv::Point2i& nativeRef;
#endif

# pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2i&)point;
- (void)update:(cv::Point2i&)point;
#endif

# pragma mark - Methods

/**
* Calculate the dot product of this point and another point
* @param point The other point
*/
- (double)dot:(Point2i*)point;

/**
* Determine if the point lies with a specified rectangle
* @param rect The rectangle
*/
- (BOOL)inside:(Rect2i*)rect;

/**
* Set the point coordinates from the values of an array
* @param vals The array of values from which to set the coordinates
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Point2i*)clone;

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
