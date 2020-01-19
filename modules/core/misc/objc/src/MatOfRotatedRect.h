//
//  MatOfRotatedRect.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class RotatedRect;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfRotatedRect : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<RotatedRect*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<RotatedRect*>*)array;
- (NSArray<RotatedRect*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
