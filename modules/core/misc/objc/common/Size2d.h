//
//  Size2d.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

@class Point2d;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents the dimensions of a rectangle the values of which are of type `double`
*/
@interface Size2d : NSObject

#pragma mark - Properties

@property double width;
@property double height;
#ifdef __cplusplus
@property(readonly) cv::Size2d& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithWidth:(double)width height:(double)height;
- (instancetype)initWithPoint:(Point2d*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Size2d&)size;
#endif
+ (instancetype)width:(double)width height:(double)height;

#pragma mark - Methods

/**
* Returns the area of a rectangle with corresponding dimensions
*/
- (double)area;

/**
* Determines if a rectangle with corresponding dimensions has area of zero
*/
- (BOOL)empty;

/**
* Set the dimensions from the values of an array
* @param vals The array of values from which to set the dimensions
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (Size2d*)clone;

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
