//
//  Imgcodecs.h
//
//  Created by Giles Payne on 2020/01/22.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

@class Mat;

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Imgcodecs : NSObject

+ (Mat*)imread:(NSString*)file flags:(int)flags;
+ (Mat*)imread:(NSString*)file;

@end

NS_ASSUME_NONNULL_END
