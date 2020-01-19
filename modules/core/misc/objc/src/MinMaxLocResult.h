//
//  MinMaxLocResult.h
//  StitchApp
//
//  Created by Giles Payne on 2019/12/28.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class CVPoint;

NS_ASSUME_NONNULL_BEGIN

@interface MinMaxLocResult : NSObject

@property double minVal;
@property double maxVal;
@property CVPoint* minLoc;
@property CVPoint* maxLoc;

- (instancetype)init;
- (instancetype)initWithMinval:(double)minVal maxVal:(double)maxVal minLoc:(CVPoint*)minLoc maxLoc:(CVPoint*)maxLoc;

@end

NS_ASSUME_NONNULL_END
