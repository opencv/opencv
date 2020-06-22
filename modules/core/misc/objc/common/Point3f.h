//
//  Point3f.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Point2f;

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a three dimensional point the coordinate values of which are of type `float`
*/
@interface Point3f : NSObject

# pragma mark - Properties

@property float x;
@property float y;
@property float z;
#ifdef __cplusplus
@property(readonly) cv::Point3f& nativeRef;
#endif

# pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y z:(float)z;
- (instancetype)initWithPoint:(Point2f*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;


# pragma mark - Methods

/**
* Calculate the dot product of this point and another point
* @param point The other point
*/
- (double)dot:(Point3f*)point;

/**
* Calculate the cross product of this point and another point
* @param point The other point
*/
- (Point3f*)cross:(Point3f*)point;

/**
* Set the point coordinates from the values of an array
* @param vals The array of values from which to set the coordinates
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Point3f*)clone;

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
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
