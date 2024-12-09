//
//  Point3i.h
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

@class Point2i;

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a three dimensional point the coordinate values of which are of type `int`
*/
CV_EXPORTS @interface Point3i : NSObject

# pragma mark - Properties

@property int x;
@property int y;
@property int z;
#ifdef __cplusplus
@property(readonly) cv::Point3i& nativeRef;
#endif

# pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y z:(int)z;
- (instancetype)initWithPoint:(Point2i*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point3i&)point;
- (void)update:(cv::Point3i&)point;
#endif

# pragma mark - Methods

/**
* Calculate the dot product of this point and another point
* @param point The other point
*/
- (double)dot:(Point3i*)point;

/**
* Calculate the cross product of this point and another point
* @param point The other point
*/
- (Point3i*)cross:(Point3i*)point;

/**
* Set the point coordinates from the values of an array
* @param vals The array of values from which to set the coordinates
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Point3i*)clone;

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
