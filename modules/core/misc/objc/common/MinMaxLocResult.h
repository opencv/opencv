//
//  MinMaxLocResult.h
//
//  Created by Giles Payne on 2019/12/28.
//

#pragma once

#ifdef __cplusplus
#import "opencv2/core.hpp"
#else
#define CV_EXPORTS
#endif

#import <Foundation/Foundation.h>

@class Point2i;

NS_ASSUME_NONNULL_BEGIN

/**
* Result of operation to determine global minimum and maximum of an array
*/
CV_EXPORTS @interface MinMaxLocResult : NSObject

#pragma mark - Properties

@property double minVal;
@property double maxVal;
@property Point2i* minLoc;
@property Point2i* maxLoc;

#pragma mark - Constructors

- (instancetype)init;
- (instancetype)initWithMinval:(double)minVal maxVal:(double)maxVal minLoc:(Point2i*)minLoc maxLoc:(Point2i*)maxLoc;

@end

NS_ASSUME_NONNULL_END
