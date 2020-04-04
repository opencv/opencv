//
//  MinMaxLocResult.h
//
//  Created by Giles Payne on 2019/12/28.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Point2i;

NS_ASSUME_NONNULL_BEGIN

@interface MinMaxLocResult : NSObject

@property double minVal;
@property double maxVal;
@property Point2i* minLoc;
@property Point2i* maxLoc;

- (instancetype)init;
- (instancetype)initWithMinval:(double)minVal maxVal:(double)maxVal minLoc:(Point2i*)minLoc maxLoc:(Point2i*)maxLoc;

@end

NS_ASSUME_NONNULL_END
