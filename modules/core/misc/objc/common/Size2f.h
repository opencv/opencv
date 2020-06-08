//
//  Size2f.h
//
//  Created by Giles Payne on 2019/10/06.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

@class Point2f;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents the dimensions of a rectangle the values of which are of type `float`
*/
@interface Size2f : NSObject

#pragma mark - Properties

@property float width;
@property float height;
#ifdef __cplusplus
@property(readonly) cv::Size2f& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithWidth:(float)width height:(float)height;
- (instancetype)initWithPoint:(Point2f*)point;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Size2f&)size;
#endif
+ (instancetype)width:(float)width height:(float)height;

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
- (Size2f*)clone;

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
