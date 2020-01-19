//
//  MatOfPoint2f.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

@interface MatOfPoint2f : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<CVPoint*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<CVPoint*>*)array;
- (NSArray<CVPoint*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
