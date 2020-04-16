//
//  MatOfRect2i.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Rect2i;

NS_ASSUME_NONNULL_BEGIN

NS_SWIFT_NAME(MatOfRect)
@interface MatOfRect2i : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<Rect2i*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<Rect2i*>*)array;
- (NSArray<Rect2i*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
