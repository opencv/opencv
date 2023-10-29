//
//  Range.h
//
//  Created by Giles Payne on 2019/10/08.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Represents a range of dimension indices
*/
CV_EXPORTS @interface Range : NSObject

#pragma mark - Properties

@property int start;
@property int end;
#ifdef __cplusplus
@property(readonly) cv::Range& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithStart:(int)start end:(int)end;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Range&)range;
#endif

#pragma mark - Methods

/**
* The size of the range
*/
- (int)size;

/**
* Determines if the range is empty
*/
- (BOOL)empty;

/**
* Creates a range representing all possible indices for a particular dimension
*/
+ (Range*)all;

/**
* Calculates the intersection of the range with another range
* @param r1 The other range
*/
- (Range*)intersection:(Range*)r1;

/**
* Adjusts each of the range limts
* @param delta The amount of the adjustment
*/
- (Range*)shift:(int)delta;

/**
* Set the range limits from the values of an array
* @param vals The array of values from which to set the range limits
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

# pragma mark - Common Methods

/**
* Clone object
*/
- (Range*)clone;

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
