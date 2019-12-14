//
//  CVSize.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef CVSize_h
#define CVSize_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

#import <Foundation/Foundation.h>

@interface CVSize : NSObject

@property double width;
@property double height;

- (instancetype)init;
- (instancetype)initWithWidth:(double)width height:(double)height;
- (double)area;
- (BOOL)empty;
- (CVSize*)clone;
- (BOOL)isEqual:(id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

#endif /* CVSize_h */
