//
//  Point3d.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

@class Point2d;

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a three dimensional point the coordinate values of which are of type `double`
*/
CV_EXPORTS @interface Point3d : NSObject

# pragma mark - Properties

@property double x;
@property double y;
@property double z;
#ifdef __cplusplus
@property(readonly) cv::Point3d& nativeRef;
#endif

# pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(double)x y:(double)y z:(double)z;
- (instancetype)initWithPoint:(Point2d*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point3d&)point;
- (void)update:(cv::Point3d&)point;
#endif

# pragma mark - Methods

/**
* Calculate the dot product of this point and another point
* @param point The other point
*/
- (double)dot:(Point3d*)point;

/**
* Calculate the cross product of this point and another point
* @param point The other point
*/
- (Point3d*)cross:(Point3d*)point;

/**
* Set the point coordinates from the values of an array
* @param vals The array of values from which to set the coordinates
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Point3d*)clone;

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
