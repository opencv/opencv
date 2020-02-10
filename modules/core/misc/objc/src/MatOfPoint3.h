//
//  MatOfPoint3.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Point3i;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfPoint3 : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<Point3i*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<Point3i*>*)array;
- (NSArray<Point3i*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
