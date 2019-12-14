//
//  Point.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef CVPoint_h
#define CVPoint_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

#import <Foundation/Foundation.h>

@class CVRect;

@interface CVPoint : NSObject

@property double x;
@property double y;

- (instancetype)initWithX:(double)x y:(double)y;
- (instancetype)init;
- (CVPoint*)clone;
- (double)dot:(CVPoint*)point;
- (BOOL)inside:(CVRect*)rect;

- (BOOL)isEqual:(id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

#endif /* CVPoint_h */
