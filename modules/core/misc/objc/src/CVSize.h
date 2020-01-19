//
//  CVSize.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CVSize : NSObject

@property double width;
@property double height;

- (instancetype)init;
- (instancetype)initWithWidth:(double)width height:(double)height;
- (double)area;
- (BOOL)empty;
- (CVSize*)clone;
- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
