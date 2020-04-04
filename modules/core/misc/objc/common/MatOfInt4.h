//
//  MatOfInt4.h
//
//  Created by Giles Payne on 2019/12/27.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#import "Mat.h"

NS_ASSUME_NONNULL_BEGIN

@interface MatOfInt4 : Mat

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat;
#endif

- (instancetype)initWithMat:(Mat*)mat;
- (instancetype)initWithArray:(NSArray<NSNumber*>*)array;

- (void)alloc:(int)elemNumber;

- (void)fromArray:(NSArray<NSNumber*>*)array;
- (NSArray<NSNumber*>*)toArray;
- (int)length;

@end

NS_ASSUME_NONNULL_END
