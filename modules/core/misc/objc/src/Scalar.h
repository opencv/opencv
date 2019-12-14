//
//  Scalar.h
//  StitchApp
//
//  Created by Giles Payne on 2019/10/06.
//  Copyright © 2019 Xtravision. All rights reserved.
//

#ifndef Scalar_h
#define Scalar_h

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/xfeatures2d.hpp>
#endif

#import <Foundation/Foundation.h>

@interface Scalar : NSObject

@property NSMutableArray<NSNumber*>* val;

- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2 v3:(double)v3;
- (instancetype)initWithV0:(double)v0 v1:(double)v1 v2:(double)v2;
- (instancetype)initWithV0:(double)v0 v1:(double)v1;
- (instancetype)initWithV0:(double)v0;

- (void)set:(NSArray<NSNumber*>*)vals;
+ (Scalar*)all:(double)v;
- (Scalar*)clone;
- (Scalar*)mul:(Scalar*)it scale:(double)scale;
- (Scalar*)mul:(Scalar*)it;
- (Scalar*)conj;
- (BOOL)isReal;

- (BOOL)isEqual:(id)object;
- (NSUInteger)hash;
- (NSString *)description;

@end

#endif /* Scalar_h */
