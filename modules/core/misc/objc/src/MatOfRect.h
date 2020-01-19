//
//  MatOfDMatch.h
//
//  Created by Giles Payne on 2019/12/27.
//

#pragma once

#import "Mat.h"

@class CVRect;

NS_ASSUME_NONNULL_BEGIN

@interface MatOfRect : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<CVRect*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<CVRect*>*)array;
- (NSArray<CVRect*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
