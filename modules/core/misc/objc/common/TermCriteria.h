//
//  TermCriteria.h
//
//  Created by Giles Payne on 2019/10/08.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
* Class representing termination criteria for iterative algorithms.
*/
@interface TermCriteria : NSObject

#pragma mark - Properties

@property(class, readonly) int COUNT;
@property(class, readonly) int EPS;
@property(class, readonly) int MAX_ITER;

@property int type;
@property int maxCount;
@property double epsilon;
#ifdef __cplusplus
@property(readonly) cv::TermCriteria& nativeRef;
#endif

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithType:(int)type maxCount:(int)maxCount epsilon:(double)epsilon;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::TermCriteria&)nativeTermCriteria;
#endif

#pragma mark - Methods

/**
* Set the termination criteria values from the values of an array
* @param vals The array of values from which to set the termination criteria values
*/
- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));

#pragma mark - Common Methods

/**
* Clone object
*/
- (TermCriteria*)clone;

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
