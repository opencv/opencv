//
//  MatOfDMatch.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class DMatch;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfDMatch : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<DMatch*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<DMatch*>*)array;
- (NSArray<DMatch*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
