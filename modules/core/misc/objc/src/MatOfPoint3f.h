//
//  MatOfPoint3f.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class Point3f;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfPoint3f : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<Point3f*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<Point3f*>*)array;
- (NSArray<Point3f*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
