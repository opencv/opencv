//
//  MatOfKeyPoint.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class KeyPoint;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfKeyPoint : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<KeyPoint*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<KeyPoint*>*)array;
- (NSArray<KeyPoint*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
