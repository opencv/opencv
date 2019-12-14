//
//  Rect.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/09.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef CVRect_h
#define CVRect_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

@class CVPoint;
@class CVSize;

#import <Foundation/Foundation.h>

@interface CVRect : NSObject

@property int x;
@property int y;
@property int width;
@property int height;

- (instancetype)initWithX:(int)x y:(int)y width:(int)width height:(int)height;
- (instancetype)init;

- (CVRect*)clone;
- (CVPoint*)tl;
- (CVPoint*)br;
- (CVSize*)size;
- (double)area;
- (BOOL)empty;
- (BOOL)contains:(CVPoint*)point;
@end

#endif /* CVRect_h */
