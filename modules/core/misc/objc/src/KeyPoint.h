//
//  KeyPoint.h
//
//  Created by Giles Payne on 2019/10/08.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

@class Point2f;

NS_ASSUME_NONNULL_BEGIN

@interface KeyPoint : NSObject

@property Point2f* pt;
@property float size;
@property float angle;
@property float response;
@property int octave;
@property int classId;

- (instancetype)init;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave classId:(int)classId;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response octave:(int)octave;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle response:(float)response;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size angle:(float)angle;
- (instancetype)initWithX:(float)x y:(float)y size:(float)size;

- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
