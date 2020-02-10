//
//  MatOfPoint2f.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

@class Point2f;

@interface MatOfPoint2f : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<Point2f*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<Point2f*>*)array;
- (NSArray<Point2f*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
