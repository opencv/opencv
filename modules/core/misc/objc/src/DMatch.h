//
//  DMatch.h
//
//  Created by Giles Payne on 2019/12/25.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface DMatch : NSObject

@property int queryIdx;
@property int trainIdx;
@property int imgIdx;
@property float distance;
#ifdef __cplusplus
@property(readonly) cv::DMatch& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx distance:(float)distance;
- (instancetype)initWithQueryIdx:(int)queryIdx trainIdx:(int)trainIdx imgIdx:(int)imgIdx distance:(float)distance;
#ifdef __cplusplus
+ (instancetype)fromNative:(cv::DMatch&)dMatch;
#endif

- (BOOL)lessThan:(DMatch*)it;

- (DMatch*)clone;
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
