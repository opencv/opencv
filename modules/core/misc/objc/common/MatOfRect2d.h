//
//  MatOfRect2d.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Rect2d;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfRect2d : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
+ (instancetype)fromNative:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<Rect2d*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<Rect2d*>*)array;
- (NSArray<Rect2d*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
