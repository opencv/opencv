//
//  Range.h
//
//  Created by Giles Payne on 2019/10/08.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#pragma once

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#endif

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Range : NSObject

@property int start;
@property int end;

- (instancetype)init;
- (instancetype)initWithStart:(int)start end:(int)end;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

- (void)set:(NSArray<NSNumber*>*)vals;
- (int)size;
- (BOOL)empty;
+ (Range*)all;

- (Range*)intersection:(Range*)r1;
- (Range*)shift:(int)delta;
- (Range*)clone;

- (BOOL)isEqual:(nullable id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

NS_ASSUME_NONNULL_END
