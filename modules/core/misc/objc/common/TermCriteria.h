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

@interface TermCriteria : NSObject

@property(class, readonly) int COUNT;
@property(class, readonly) int EPS;
@property(class, readonly) int MAX_ITER;

@property int type;
@property int maxCount;
@property double epsilon;
#ifdef __cplusplus
@property(readonly) cv::TermCriteria& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithType:(int)type maxCount:(int)maxCount epsilon:(double)epsilon;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::TermCriteria&)nativeTermCriteria;
#endif

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (TermCriteria*)clone;

- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
