//
//  TermCriteria.h
//
//  Created by Giles Payne on 2019/10/08.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface TermCriteria : NSObject

@property int type;
@property int maxCount;
@property double epsilon;
#ifdef __cplusplus
@property(readonly) cv::TermCriteria& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithType:(int)type maxCount:(int)maxCount epsilon:(double)epsilon;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::TermCriteria&)nativeTermCriteria;
#endif

- (void)set:(NSArray<NSNumber*>*)vals;
- (TermCriteria*)clone;

- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
