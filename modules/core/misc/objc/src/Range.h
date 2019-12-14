//
//  Range.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/08.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef Range_h
#define Range_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

#import <Foundation/Foundation.h>

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

- (BOOL)isEqual:(id)object;
- (NSUInteger)hash;
- (NSString*)description;

@end

#endif /* Range_h */
